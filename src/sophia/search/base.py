"""Abstract base class for modular search engine providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sophia.search.models import SearchResponse, SearchResult


class BaseSearchProvider(ABC):
    """Abstract base class that all search providers (DuckDuckGo, Tavily, Google, etc.) must implement."""

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 5,
        **kwargs: Any,
    ) -> SearchResponse:
        """Synchronously execute a search query.

        Args:
            query: The search query text.
            max_results: Maximum number of results to return.
            **kwargs: Provider-specific options (region, timelimit, safesearch, etc.).

        Returns:
            SearchResponse containing normalized SearchResult items.
        """
        pass

    @abstractmethod
    async def asearch(
        self,
        query: str,
        max_results: int = 5,
        **kwargs: Any,
    ) -> SearchResponse:
        """Asynchronously execute a search query.

        Args:
            query: The search query text.
            max_results: Maximum number of results to return.
            **kwargs: Provider-specific options (region, timelimit, safesearch, etc.).

        Returns:
            SearchResponse containing normalized SearchResult items.
        """
        pass
