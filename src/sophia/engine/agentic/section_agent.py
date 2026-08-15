"""Section Synthesizer Agent.

Responsible for generating a focused, high-density section of the research report
under a strict per-prompt token budget, with exact source citations.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Sequence

from sophia.chunk.chunker import TextChunk
from sophia.chunk.reranker import InMemoryReranker, RankedChunk
from sophia.engine.agentic.planner import SectionPlan
from sophia.engine.citation_mapper import assemble_reranked_context
from sophia.llm.models import ModelTier
from sophia.llm.router import LLMRouter

logger = logging.getLogger(__name__)

SECTION_SYSTEM_PROMPT = """You are a specialized Section Synthesizer sub-agent in an Agentic Deep Research system.
Your mission is to write ONLY the assigned section of a larger, authoritative report.

Guidelines:
1. Write in a clear, informative, and analytical tone.
2. Ground every claim using inline numeric bracket citations like [1], [2], [1][3] corresponding to the provided sources.
3. Start directly with the section heading (e.g. `### Section Title`) or direct text. Do NOT add meta commentary, introductory greetings, or conclusions intended for the whole document.
4. Focus strictly on the assigned Section Goal. Do not repeat facts covered in earlier sections unless essential for context.
5. Use markdown formatting (bolding, concise bullet points, or comparison tables where appropriate).
"""


class SectionSynthesizer:
    """Synthesizes an individual section of a multi-part research report."""

    def __init__(self, router: LLMRouter, reranker: InMemoryReranker):
        self.router = router
        self.reranker = reranker

    async def astream_section(
        self,
        section: SectionPlan,
        all_chunks: Sequence[TextChunk],
        prior_sections_summary: str = "",
        top_k_chunks: int = 6,
        max_tokens_per_section: int = 700,
    ) -> AsyncGenerator[str, None]:
        """Synthesize and stream a single section."""
        # 1. Rerank chunks specifically for this section's sub_query & goal
        query_for_section = f"{section.title} {section.sub_query} {section.goal}"
        ranked_chunks: list[RankedChunk] = self.reranker.rerank(
            query=query_for_section,
            chunks=all_chunks,
            top_k=top_k_chunks,
        )

        # 2. Assemble grounded context with source numbers
        section_context = assemble_reranked_context(ranked_chunks, max_chunks=top_k_chunks)

        user_content = f"""Assigned Section: {section.title}
Section Goal: {section.goal}
Specific Focus: {section.sub_query}

Available Grounded Sources:
{section_context}
"""
        if prior_sections_summary:
            user_content += f"\nContext from previous sections already written (do not duplicate):\n{prior_sections_summary[:400]}\n"

        user_content += f"\nWrite the complete, detailed section content for '{section.title}' with inline citations [1], [2]:"

        messages = [
            {"role": "system", "content": SECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # 3. Stream LLM tokens for this section
        async for token in self.router.astream(
            messages=messages,
            tier=ModelTier.BALANCED,
            temperature=0.2,
            max_tokens=max_tokens_per_section,
        ):
            yield token
