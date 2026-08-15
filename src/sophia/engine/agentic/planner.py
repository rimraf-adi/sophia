"""Query Planner / Decomposer Agent.

Analyzes complex user questions and breaks them down into targeted research sub-tasks/sections.
"""

from __future__ import annotations

import json
import logging
from typing import Sequence
from pydantic import BaseModel, Field

from sophia.llm.models import ModelTier
from sophia.llm.router import LLMRouter
from sophia.search.models import SearchResult

logger = logging.getLogger(__name__)


class SectionPlan(BaseModel):
    """Represents a planned section/sub-task in the research outline."""

    id: int = Field(description="Sequential section index starting at 1")
    title: str = Field(description="Concise section title / heading")
    sub_query: str = Field(description="Specific sub-query or focus question for this section")
    goal: str = Field(description="What specific aspect this section should explain or analyze")


class ResearchPlan(BaseModel):
    """Complete research decomposition plan."""

    summary: str = Field(description="High level overview of what will be investigated")
    sections: list[SectionPlan] = Field(default_factory=list, description="Ordered list of sub-task sections")


PLANNER_SYSTEM_PROMPT = """You are an expert Research Planning Agent.
Your job is to analyze a user's question and decompose it into 2 to 4 structured, non-overlapping sub-task sections.

Each section will be researched and synthesized by a dedicated sub-agent to produce an in-depth, comprehensive report.

Requirements:
1. Divide the question logically (e.g. 1. Core Overview & Mechanism, 2. Deep Dive / Architecture, 3. Comparative Analysis / Tradeoffs, 4. Future Outlook / Implications).
2. For simple or brief questions, produce 2 focused sections. For complex, technical, or broad questions, produce 3 to 4 sections.
3. Respond ONLY with valid JSON matching this schema:
{
  "summary": "Brief 1-sentence description of the research angle",
  "sections": [
    {
      "id": 1,
      "title": "Section Title",
      "sub_query": "Key aspect to focus on",
      "goal": "Explain specific details about X"
    }
  ]
}
Do NOT wrap with backticks or add any other text outside the JSON.
"""


class QueryPlanner:
    """Decomposes complex questions into structured multi-step research plans."""

    def __init__(self, router: LLMRouter):
        self.router = router

    async def plan_research(
        self,
        user_question: str,
        search_results: Sequence[SearchResult] | None = None,
    ) -> ResearchPlan:
        """Create a multi-section research plan."""
        context_snippets = ""
        if search_results:
            top_sources = search_results[:5]
            context_snippets = "\n".join(
                [f"- {s.title}: {s.snippet[:150]}" for s in top_sources]
            )

        user_content = f"User Question: {user_question}\n"
        if context_snippets:
            user_content += f"\nInitial Search Discoveries:\n{context_snippets}\n"
        user_content += "\nDecompose this question into 2-4 sub-task sections for in-depth synthesis:"

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            resp = await self.router.acomplete(
                messages=messages,
                tier=ModelTier.FAST,
                temperature=0.2,
                max_tokens=400,
            )
            raw_text = resp.content.strip()
            if raw_text.startswith("```"):
                parts = raw_text.split("```")
                if len(parts) >= 2:
                    raw_text = parts[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
            raw_text = raw_text.strip()

            parsed = json.loads(raw_text)
            return ResearchPlan(**parsed)
        except Exception as e:
            logger.warning("Query planner fallback to default 2-section plan: %s", str(e))
            return self._default_plan(user_question)

    def _default_plan(self, question: str) -> ResearchPlan:
        """Fallback plan if JSON parsing fails."""
        return ResearchPlan(
            summary=f"In-depth research on {question}",
            sections=[
                SectionPlan(
                    id=1,
                    title="Overview & Key Concepts",
                    sub_query=question,
                    goal=f"Comprehensive overview of {question}",
                ),
                SectionPlan(
                    id=2,
                    title="Detailed Analysis & Key Takeaways",
                    sub_query=f"{question} key details and significance",
                    goal="Technical breakdown, practical implications, and future perspective",
                ),
            ],
        )
