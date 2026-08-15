"""Perplexity Engine - Full Live/Agentic RAG Pipeline with Scrape, Chunk, Rerank, Key/Model Rotation, and Citations."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from sophia.cache.cache import QueryCache
from sophia.chunk.chunker import TextChunk, TextChunker
from sophia.chunk.reranker import InMemoryReranker, RankedChunk
from sophia.engine.agentic.agentic_orchestrator import AgenticEngine
from sophia.engine.citation_mapper import (
    Citation,
    assemble_reranked_context,
    map_citations,
)
from sophia.engine.models import PerplexityResponse, PerplexityStreamEvent
from sophia.llm.models import ModelTier
from sophia.llm.router import LLMRouter
from sophia.scrape.scraper import AsyncScraper, ScrapedDocument
from sophia.search.duckduckgo import DuckDuckGoSearcher
from sophia.search.models import SearchResult
from sophia.session.session import ConversationSession, SessionStore

logger = logging.getLogger(__name__)

QUERY_REWRITE_SYSTEM_PROMPT = """You are a Search Query Optimizer.
Your job is to rewrite follow-up questions into standalone, keyword-rich search queries.

Rules:
1. If the user question refers to previous context (e.g., 'how does it work', 'compare them', 'why?'), resolve pronouns and implicit references using the conversation history.
2. If the user question is already standalone, return it as-is.
3. Output ONLY the standalone search query without quotes, explanations, or punctuation.
"""

STRICT_GROUNDED_PROMPT = """You are a grounded AI search assistant (like Perplexity AI).
Your mission is to synthesize a direct, highly accurate, and comprehensive answer to the user's question using ONLY the provided numbered sources.

Strict Rules:
1. Every single factual claim MUST be cited with numeric bracket citations like [1], [2], [1][3] immediately after the claim.
2. NEVER make claims that are not supported by the provided text.
3. Use markdown formatting with bullet points, bold text, or code blocks where helpful for clarity.
4. If sources conflict or information is missing, clearly state so.
"""

FOLLOW_UP_PROMPT = """You are a helpful research assistant.
Based on the question and synthesized answer, suggest 3 relevant, interesting follow-up questions the user might want to explore next.

Output ONLY a valid JSON list of 3 strings.
Example: ["How do I install this?", "What are the performance differences?", "Is there an alternative?"]
"""


class PerplexityEngine:
    """Complete Agentic Live RAG Engine matching Perplexity architecture."""

    def __init__(
        self,
        router: LLMRouter | None = None,
        searcher: DuckDuckGoSearcher | None = None,
        scraper: AsyncScraper | None = None,
        chunker: TextChunker | None = None,
        reranker: InMemoryReranker | None = None,
        cache: QueryCache | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.router = router or LLMRouter(enable_model_rotation=True)
        self.searcher = searcher or DuckDuckGoSearcher(timeout=10)
        self.scraper = scraper or AsyncScraper(timeout=3.0, max_concurrent=10)
        self.chunker = chunker or TextChunker(chunk_size_tokens=600, chunk_overlap_tokens=100)
        self.reranker = reranker or InMemoryReranker()
        self.cache = cache or QueryCache(db_path="cache.db")
        self.session_store = session_store or SessionStore()
        self.agentic_engine = AgenticEngine(router=self.router, reranker=self.reranker)

    async def rewrite_query(self, user_question: str, session: ConversationSession | None = None) -> str:
        """2.1 Query Rewriter: Creates standalone search query taking multi-turn history into account."""
        history = session.get_history_summary(max_turns=4) if session else ""
        if not history:
            return user_question.strip()

        messages = [
            {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Conversation History:\n{history}\n\nLatest User Message: {user_question}",
            },
        ]
        try:
            resp = await self.router.acomplete(
                messages=messages,
                tier=ModelTier.FAST,
                temperature=0.0,
                max_tokens=80,
            )
            rewritten = resp.content.strip().strip('"').strip("'")
            if rewritten:
                return rewritten
        except Exception as e:
            logger.warning("Query rewrite fallback: %s", str(e))

        return user_question.strip()

    async def _generate_follow_ups(self, question: str, answer: str) -> list[str]:
        """Generate related follow-up questions."""
        try:
            resp = await self.router.acomplete(
                messages=[
                    {"role": "system", "content": FOLLOW_UP_PROMPT},
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nAnswer: {answer[:800]}",
                    },
                ],
                tier=ModelTier.FAST,
                temperature=0.3,
                max_tokens=150,
            )
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(q) for q in data[:3]]
        except Exception:
            pass
        return []

    async def run_pipeline(
        self,
        user_question: str,
        session_id: str | None = None,
        use_cache: bool = True,
        mode: str = "quick",
        max_search_results: int = 8,
        top_k_chunks: int = 8,
    ) -> AsyncGenerator[PerplexityStreamEvent, None]:
        """Execute complete streaming Perplexity RAG pipeline with real-time SSE event yielding."""
        start_time = time.perf_counter()
        session = self.session_store.get_or_create(session_id) if session_id else None

        # 1. Check Cache
        cache_key = f"{mode}:{user_question}"
        if use_cache:
            cached_data = self.cache.get(cache_key) or self.cache.get(user_question)
            if cached_data:
                yield PerplexityStreamEvent(event_type="status", data="Loading cached answer...")
                yield PerplexityStreamEvent(event_type="sources", data=cached_data.get("sources", []))
                for token in cached_data.get("answer", "").split(" "):
                    yield PerplexityStreamEvent(event_type="token", data=token + " ")
                    await asyncio.sleep(0.01)
                yield PerplexityStreamEvent(event_type="citations", data=cached_data.get("citations", []))
                yield PerplexityStreamEvent(event_type="follow_ups", data=cached_data.get("follow_ups", []))
                yield PerplexityStreamEvent(event_type="done", data="complete")
                if session:
                    session.add_user_message(user_question)
                    session.add_assistant_message(
                        content=cached_data.get("answer", ""),
                        sources=[SearchResult(**s) if isinstance(s, dict) else s for s in cached_data.get("sources", [])],
                        search_queries=[cached_data.get("standalone_query", user_question)],
                    )
                return

        # 2. Query Rewriting
        yield PerplexityStreamEvent(event_type="status", data="Optimizing search query...")
        standalone_query = await self.rewrite_query(user_question, session=session)
        yield PerplexityStreamEvent(event_type="query_rewritten", data=standalone_query)

        # 3. DuckDuckGo Search (Over-fetch 8-10 results)
        yield PerplexityStreamEvent(event_type="status", data=f"Searching DuckDuckGo for '{standalone_query}'...")
        search_resp = await self.searcher.asearch(query=standalone_query, max_results=max_search_results)
        
        if search_resp.is_empty:
            yield PerplexityStreamEvent(event_type="status", data="No search results found.")
            yield PerplexityStreamEvent(
                event_type="token",
                data="No web sources could be found for your query. Please try rephrasing.",
            )
            yield PerplexityStreamEvent(event_type="done", data="complete")
            return

        yield PerplexityStreamEvent(event_type="sources", data=search_resp.results)

        # 4. Parallel Scraping with Trafilatura
        yield PerplexityStreamEvent(
            event_type="status",
            data=f"Scraping & extracting text from {len(search_resp.results)} pages in parallel...",
        )
        scrape_items = [
            {
                "url": r.url,
                "source_id": r.index,
                "title": r.title,
                "snippet": r.snippet,
            }
            for r in search_resp.results
        ]
        scraped_docs = await self.scraper.scrape_urls_parallel(scrape_items)
        yield PerplexityStreamEvent(
            event_type="status",
            data=f"Extracted clean text from {len(scraped_docs)}/{len(search_resp.results)} pages.",
        )

        # 5. Chunking
        yield PerplexityStreamEvent(event_type="status", data="Chunking extracted web content...")
        all_chunks: list[TextChunk] = self.chunker.chunk_documents(scraped_docs)
        if not all_chunks:
            # Fallback to search snippets as chunks if scraping dropped all
            for r in search_resp.results:
                all_chunks.append(
                    TextChunk(
                        chunk_id=f"src_{r.index}_snippet",
                        source_id=r.index,
                        url=r.url,
                        title=r.title,
                        text=r.snippet,
                        token_count=len(r.snippet.split()),
                    )
                )

        accumulated_tokens: list[str] = []
        citations_data: list[dict] = []

        # 6. Branch: Agentic Deep Mode vs Fast Single-shot Mode
        if mode in ("deep", "agentic"):
            async for agent_event in self.agentic_engine.astream_agentic_report(
                user_question=user_question,
                standalone_query=standalone_query,
                search_results=search_resp.results,
                all_chunks=all_chunks,
            ):
                if agent_event.event_type == "token":
                    accumulated_tokens.append(agent_event.data)
                elif agent_event.event_type == "citations":
                    citations_data = agent_event.data
                yield agent_event
        else:
            # Fast single-shot synthesis
            yield PerplexityStreamEvent(event_type="status", data="Reranking chunks for query relevance...")
            ranked_chunks: list[RankedChunk] = self.reranker.rerank(
                query=standalone_query,
                chunks=all_chunks,
                top_k=top_k_chunks,
            )

            grounded_context = assemble_reranked_context(ranked_chunks, max_chunks=top_k_chunks)
            user_prompt = f"Sources:\n{grounded_context}\n\nUser Question:\n{user_question}"

            messages = [
                {"role": "system", "content": STRICT_GROUNDED_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            yield PerplexityStreamEvent(event_type="status", data="Synthesizing answer with live citations...")

            async for token in self.router.astream(
                messages=messages,
                temperature=0.2,
                max_tokens=1500,
            ):
                accumulated_tokens.append(token)
                yield PerplexityStreamEvent(event_type="token", data=token)

            full_answer_fast = "".join(accumulated_tokens).strip()
            if not full_answer_fast:
                full_answer_fast = "I searched the web for your query, but could not synthesize a complete answer at this time. Please check your network or try asking again."
                yield PerplexityStreamEvent(event_type="token", data=full_answer_fast)

            citations: list[Citation] = map_citations(full_answer_fast, search_resp.results)
            citations_data = [c.model_dump() for c in citations]
            yield PerplexityStreamEvent(event_type="citations", data=citations_data)

        full_answer = "".join(accumulated_tokens).strip()

        # 7. Follow-up Generation
        follow_ups = await self._generate_follow_ups(user_question, full_answer)
        yield PerplexityStreamEvent(event_type="follow_ups", data=follow_ups)

        duration = round(time.perf_counter() - start_time, 2)
        yield PerplexityStreamEvent(event_type="status", data=f"Completed in {duration}s")
        yield PerplexityStreamEvent(event_type="done", data="complete")

        # 8. Write to Cache
        cache_payload = {
            "query": user_question,
            "mode": mode,
            "standalone_query": standalone_query,
            "answer": full_answer,
            "sources": [r.model_dump() for r in search_resp.results],
            "citations": citations_data,
            "follow_ups": follow_ups,
            "duration": duration,
        }
        self.cache.set(cache_key, cache_payload, ttl_seconds=3600)
        self.cache.set(user_question, cache_payload, ttl_seconds=3600)

        # 9. Save to Session
        if session:
            session.add_user_message(user_question)
            session.add_assistant_message(
                content=full_answer,
                sources=search_resp.results,
                search_queries=[standalone_query],
            )

    async def ask(
        self,
        user_question: str | None = None,
        question: str | None = None,
        session_id: str | None = None,
        mode: str = "quick",
        tier: ModelTier | None = None,
        max_sources: int = 8,
        **kwargs: Any,
    ) -> PerplexityResponse:
        """One-shot execution returning complete PerplexityResponse."""
        actual_query = user_question or question or ""
        sources: list[SearchResult] = []
        answer_tokens: list[str] = []
        follow_ups: list[str] = []
        search_queries: list[str] = []
        start_time = time.perf_counter()

        async for event in self.run_pipeline(
            user_question=actual_query,
            session_id=session_id,
            mode=mode,
            max_search_results=max_sources,
        ):
            if event.event_type == "query_rewritten" and isinstance(event.data, str):
                search_queries.append(event.data)
            elif event.event_type == "sources" and isinstance(event.data, list):
                sources = event.data
            elif event.event_type == "token" and isinstance(event.data, str):
                answer_tokens.append(event.data)
            elif event.event_type == "follow_ups" and isinstance(event.data, list):
                follow_ups = event.data

        duration = round(time.perf_counter() - start_time, 2)
        return PerplexityResponse(
            query=actual_query,
            search_queries=search_queries or [actual_query],
            sources=sources,
            answer="".join(answer_tokens),
            follow_up_questions=follow_ups,
            model_used="rotated",
            total_duration_seconds=duration,
        )
