"""Search module for Sophia."""

from sophia.search.base import BaseSearchProvider
from sophia.search.duckduckgo import DuckDuckGoSearcher
from sophia.search.formatter import (
    PERPLEXITY_SYSTEM_PROMPT,
    format_search_results_for_llm,
    format_sources_markdown,
)
from sophia.search.models import SearchResponse, SearchResult, SearchType

__all__ = [
    "BaseSearchProvider",
    "DuckDuckGoSearcher",
    "SearchResult",
    "SearchResponse",
    "SearchType",
    "format_search_results_for_llm",
    "format_sources_markdown",
    "PERPLEXITY_SYSTEM_PROMPT",
]
