"""Pydantic data models for search results and queries."""

from __future__ import annotations

from enum import Enum
from urllib.parse import urlparse
from pydantic import BaseModel, Field, HttpUrl, computed_field


class SearchType(str, Enum):
    """Supported search types."""

    TEXT = "text"
    NEWS = "news"
    INSTANT_ANSWER = "instant_answer"


class SearchResult(BaseModel):
    """Represents a single unified search result."""

    index: int = Field(default=1, description="1-based index / citation number")
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="Source URL")
    snippet: str = Field(default="", description="Snippet / summary text")
    published_date: str | None = Field(default=None, description="Publication timestamp or date string")
    source: str | None = Field(default=None, description="Original source name / provider")
    raw: dict | None = Field(default=None, description="Raw dictionary from underlying provider", exclude=True)

    @computed_field
    @property
    def domain(self) -> str:
        """Extract the hostname/domain from the URL."""
        try:
            parsed = urlparse(self.url)
            return parsed.netloc.replace("www.", "")
        except Exception:
            return ""

    def to_citation_header(self) -> str:
        """Formats source header suitable for Perplexity citations."""
        domain_str = f" ({self.domain})" if self.domain else ""
        return f"[{self.index}] {self.title}{domain_str}"

    def to_context_block(self) -> str:
        """Formats the result into a clean context block for LLM prompting."""
        lines = [
            f"Source [{self.index}]: {self.title}",
            f"URL: {self.url}",
        ]
        if self.published_date:
            lines.append(f"Date: {self.published_date}")
        lines.append(f"Content: {self.snippet}")
        return "\n".join(lines)


class SearchResponse(BaseModel):
    """Unified search response wrapping a batch of results and metadata."""

    query: str = Field(..., description="The search query submitted")
    search_type: SearchType = Field(default=SearchType.TEXT, description="Type of search executed")
    results: list[SearchResult] = Field(default_factory=list, description="List of search results")
    total_count: int = Field(default=0, description="Total number of results returned")
    duration_seconds: float = Field(default=0.0, description="Search latency in seconds")
    error: str | None = Field(default=None, description="Error message if search encountered issues")

    @property
    def is_empty(self) -> bool:
        """Check if response returned zero results."""
        return len(self.results) == 0

    def format_for_llm(self, max_results: int | None = None) -> str:
        """Format all results into an LLM context block with Perplexity citations."""
        target_results = self.results[:max_results] if max_results else self.results
        if not target_results:
            return "No search results found."
        
        blocks = [r.to_context_block() for r in target_results]
        return "\n\n".join(blocks)
