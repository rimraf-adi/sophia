"""Data models for the Perplexity Engine pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

from sophia.llm.models import ModelTier
from sophia.search.models import SearchResult


class PerplexityResponse(BaseModel):
    """Complete response returned by the PerplexityEngine."""

    query: str = Field(..., description="Original user question")
    search_queries: list[str] = Field(default_factory=list, description="Targeted queries generated for search")
    sources: list[SearchResult] = Field(default_factory=list, description="Web sources cited")
    answer: str = Field(..., description="Synthesized cited answer")
    follow_up_questions: list[str] = Field(default_factory=list, description="Suggested related follow-up questions")
    model_used: str = Field(..., description="LLM model used for synthesis")
    total_duration_seconds: float = Field(default=0.0, description="End-to-end execution time")


class PerplexityStreamEvent(BaseModel):
    """Event emitted during streaming pipeline execution."""

    event_type: str = Field(..., description="Type of event: 'query_planning', 'searching', 'sources', 'token', 'follow_ups', 'done'")
    data: str | list[str] | list[SearchResult] | None = Field(default=None, description="Payload data")
