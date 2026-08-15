"""Agentic Orchestrator Engine.

Breaks down search generation into sub-tasks (sections), dispatches specialized sub-agents
with dedicated token limits, and streams a unified, deep research report with citation tracking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Sequence

from sophia.chunk.chunker import TextChunk
from sophia.chunk.reranker import InMemoryReranker
from sophia.engine.agentic.planner import QueryPlanner, ResearchPlan
from sophia.engine.agentic.section_agent import SectionSynthesizer
from sophia.engine.citation_mapper import Citation, map_citations
from sophia.engine.models import PerplexityStreamEvent
from sophia.llm.router import LLMRouter
from sophia.search.models import SearchResult

logger = logging.getLogger(__name__)


class AgenticEngine:
    """Multi-subtask generation engine for deep agentic research."""

    def __init__(self, router: LLMRouter, reranker: InMemoryReranker):
        self.router = router
        self.reranker = reranker
        self.planner = QueryPlanner(router=router)
        self.section_synthesizer = SectionSynthesizer(router=router, reranker=reranker)

    async def astream_agentic_report(
        self,
        user_question: str,
        standalone_query: str,
        search_results: Sequence[SearchResult],
        all_chunks: Sequence[TextChunk],
        max_tokens_per_section: int = 650,
    ) -> AsyncGenerator[PerplexityStreamEvent, None]:
        """Orchestrate planning and multi-section streaming synthesis."""
        start_time = time.perf_counter()

        # 1. Plan Research Outline (Decomposition)
        yield PerplexityStreamEvent(
            event_type="status",
            data="Planning deep research outline with sub-agents...",
        )
        plan: ResearchPlan = await self.planner.plan_research(
            user_question=user_question,
            search_results=search_results,
        )

        # Emit plan outline to UI for checklist rendering
        yield PerplexityStreamEvent(
            event_type="plan",
            data=plan.model_dump(),
        )

        # 2. Sequential / Pipeline Section Synthesis
        accumulated_sections: list[str] = []
        prior_context_snippets = []

        for i, section in enumerate(plan.sections):
            yield PerplexityStreamEvent(
                event_type="status",
                data=f"Sub-agent [{i+1}/{len(plan.sections)}] researching: {section.title}...",
            )
            yield PerplexityStreamEvent(
                event_type="section_start",
                data={"id": section.id, "title": section.title, "index": i + 1, "total": len(plan.sections)},
            )

            # Insert spacing / header between sections if not first
            if i > 0:
                header_break = f"\n\n---\n\n### {section.title}\n\n"
                yield PerplexityStreamEvent(event_type="token", data=header_break)
                accumulated_sections.append(header_break)
            else:
                initial_header = f"### {section.title}\n\n"
                yield PerplexityStreamEvent(event_type="token", data=initial_header)
                accumulated_sections.append(initial_header)

            prior_summary = " ".join(prior_context_snippets[-2:])
            section_tokens: list[str] = []

            async for token in self.section_synthesizer.astream_section(
                section=section,
                all_chunks=all_chunks,
                prior_sections_summary=prior_summary,
                top_k_chunks=6,
                max_tokens_per_section=max_tokens_per_section,
            ):
                section_tokens.append(token)
                yield PerplexityStreamEvent(event_type="token", data=token)

            section_text = "".join(section_tokens).strip()
            accumulated_sections.append(section_text)
            prior_context_snippets.append(f"Section {section.title}: {section_text[:250]}...")

            yield PerplexityStreamEvent(
                event_type="section_done",
                data={"id": section.id, "title": section.title},
            )

        full_answer = "".join(accumulated_sections).strip()
        if not full_answer:
            full_answer = "Could not synthesize deep research report at this time."
            yield PerplexityStreamEvent(event_type="token", data=full_answer)

        # 3. Global Citation Mapping
        citations: list[Citation] = map_citations(full_answer, search_results)
        yield PerplexityStreamEvent(
            event_type="citations",
            data=[c.model_dump() for c in citations],
        )

        duration = round(time.perf_counter() - start_time, 2)
        yield PerplexityStreamEvent(
            event_type="status",
            data=f"Deep research completed across {len(plan.sections)} sub-tasks in {duration}s",
        )
