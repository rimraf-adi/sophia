"""DuckDuckGo Search Provider implementation for Perp Clone."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from ddgs import DDGS

from sophia.search.base import BaseSearchProvider
from sophia.search.models import SearchResponse, SearchResult, SearchType

logger = logging.getLogger(__name__)


class DuckDuckGoSearcher(BaseSearchProvider):
    """Modular DuckDuckGo Search Client for Perplexity-style web querying and content extraction.

    Supports synchronous and asynchronous querying, news search, and web page content extraction.
    """

    def __init__(
        self,
        default_region: str = "us-en",
        default_safesearch: Literal["on", "moderate", "off"] = "moderate",
        default_timelimit: str | None = None,
        timeout: int = 15,
        proxy: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize the DuckDuckGo Searcher.

        Args:
            default_region: Default region code (e.g. 'us-en', 'wt-wt', 'uk-en').
            default_safesearch: SafeSearch filtering ('on', 'moderate', 'off').
            default_timelimit: Optional time filter ('d' = past day, 'w' = week, 'm' = month, 'y' = year).
            timeout: HTTP request timeout in seconds.
            proxy: Optional proxy string (e.g. 'socks5://localhost:9050' or 'http://user:pass@host:port').
            max_retries: Number of retries on transient errors.
            retry_delay: Delay in seconds between retries (uses exponential backoff).
        """
        self.default_region = default_region
        self.default_safesearch = default_safesearch
        self.default_timelimit = default_timelimit
        self.timeout = timeout
        self.proxy = proxy
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _get_ddgs(self) -> DDGS:
        """Instantiate a DDGS client instance."""
        return DDGS(
            proxy=self.proxy,
            timeout=self.timeout,
        )

    def search(
        self,
        query: str,
        max_results: int = 5,
        region: str | None = None,
        safesearch: Literal["on", "moderate", "off"] | None = None,
        timelimit: str | None = None,
        backend: str = "auto",
        **kwargs: Any,
    ) -> SearchResponse:
        """Execute a text web search synchronously.

        Args:
            query: The search keywords or question.
            max_results: Maximum number of results to fetch.
            region: Optional region override.
            safesearch: Optional safesearch override.
            timelimit: Optional time limit ('d', 'w', 'm', 'y').
            backend: Search backends to use.

        Returns:
            SearchResponse containing normalized SearchResult list.
        """
        start_time = time.perf_counter()
        region = region or self.default_region
        safesearch = safesearch or self.default_safesearch
        timelimit = timelimit or self.default_timelimit

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with self._get_ddgs() as ddgs:
                    raw_results = ddgs.text(
                        query=query,
                        region=region,
                        safesearch=safesearch,
                        timelimit=timelimit,
                        backend=backend,
                        max_results=max_results,
                    )
                    
                    results: list[SearchResult] = []
                    if raw_results:
                        for idx, item in enumerate(raw_results, start=1):
                            url = item.get("href") or item.get("url") or ""
                            if not url:
                                continue
                            
                            results.append(
                                SearchResult(
                                    index=idx,
                                    title=item.get("title") or "Untitled",
                                    url=url,
                                    snippet=item.get("body") or item.get("snippet") or "",
                                    published_date=item.get("date"),
                                    source=item.get("source"),
                                    raw=item,
                                )
                            )

                    duration = time.perf_counter() - start_time
                    return SearchResponse(
                        query=query,
                        search_type=SearchType.TEXT,
                        results=results,
                        total_count=len(results),
                        duration_seconds=round(duration, 3),
                    )

            except Exception as e:
                last_err = e
                logger.warning(
                    "DuckDuckGo search attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries,
                    str(e),
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** (attempt - 1)))

        duration = time.perf_counter() - start_time
        return SearchResponse(
            query=query,
            search_type=SearchType.TEXT,
            results=[],
            total_count=0,
            duration_seconds=round(duration, 3),
            error=str(last_err) if last_err else "Unknown search error",
        )

    def search_news(
        self,
        query: str,
        max_results: int = 5,
        region: str | None = None,
        safesearch: Literal["on", "moderate", "off"] | None = None,
        timelimit: str | None = None,
        **kwargs: Any,
    ) -> SearchResponse:
        """Execute a news search synchronously."""
        start_time = time.perf_counter()
        region = region or self.default_region
        safesearch = safesearch or self.default_safesearch
        timelimit = timelimit or self.default_timelimit

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with self._get_ddgs() as ddgs:
                    raw_results = ddgs.news(
                        query=query,
                        region=region,
                        safesearch=safesearch,
                        timelimit=timelimit,
                        max_results=max_results,
                    )
                    
                    results: list[SearchResult] = []
                    if raw_results:
                        for idx, item in enumerate(raw_results, start=1):
                            url = item.get("url") or item.get("href") or ""
                            if not url:
                                continue
                            
                            results.append(
                                SearchResult(
                                    index=idx,
                                    title=item.get("title") or "Untitled",
                                    url=url,
                                    snippet=item.get("body") or "",
                                    published_date=item.get("date"),
                                    source=item.get("source"),
                                    raw=item,
                                )
                            )

                    duration = time.perf_counter() - start_time
                    return SearchResponse(
                        query=query,
                        search_type=SearchType.NEWS,
                        results=results,
                        total_count=len(results),
                        duration_seconds=round(duration, 3),
                    )

            except Exception as e:
                last_err = e
                logger.warning("DuckDuckGo news attempt %d/%d failed: %s", attempt, self.max_retries, str(e))
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** (attempt - 1)))

        duration = time.perf_counter() - start_time
        return SearchResponse(
            query=query,
            search_type=SearchType.NEWS,
            results=[],
            total_count=0,
            duration_seconds=round(duration, 3),
            error=str(last_err) if last_err else "Unknown news search error",
        )

    def extract_page_content(
        self,
        url: str,
        fmt: Literal["text_markdown", "text_plain", "text_rich", "text"] = "text_markdown",
    ) -> str:
        """Fetch and extract clean readable content from a given web page URL.

        Args:
            url: Target URL to extract content from.
            fmt: Output format (default is 'text_markdown').

        Returns:
            Extracted text content as a string.
        """
        try:
            with self._get_ddgs() as ddgs:
                extracted = ddgs.extract(url=url, fmt=fmt)
                if isinstance(extracted, dict):
                    content = extracted.get("content", "")
                    if isinstance(content, bytes):
                        return content.decode("utf-8", errors="replace")
                    return str(content)
                return str(extracted or "")
        except Exception as e:
            logger.warning("Failed to extract content from %s: %s", url, str(e))
            return ""

    async def asearch(
        self,
        query: str,
        max_results: int = 5,
        region: str | None = None,
        safesearch: Literal["on", "moderate", "off"] | None = None,
        timelimit: str | None = None,
        backend: str = "auto",
        **kwargs: Any,
    ) -> SearchResponse:
        """Execute a text search asynchronously."""
        return await asyncio.to_thread(
            self.search,
            query=query,
            max_results=max_results,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            backend=backend,
            **kwargs,
        )

    async def asearch_news(
        self,
        query: str,
        max_results: int = 5,
        region: str | None = None,
        safesearch: Literal["on", "moderate", "off"] | None = None,
        timelimit: str | None = None,
        **kwargs: Any,
    ) -> SearchResponse:
        """Execute a news search asynchronously."""
        return await asyncio.to_thread(
            self.search_news,
            query=query,
            max_results=max_results,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            **kwargs,
        )

    async def aextract_page_content(
        self,
        url: str,
        fmt: Literal["text_markdown", "text_plain", "text_rich", "text"] = "text_markdown",
    ) -> str:
        """Extract web page content asynchronously."""
        return await asyncio.to_thread(self.extract_page_content, url=url, fmt=fmt)

    async def asearch_and_extract(
        self,
        query: str,
        max_results: int = 3,
        max_extract_chars: int = 2000,
        **kwargs: Any,
    ) -> SearchResponse:
        """Search and concurrently extract in-depth page content for each result (Deep Perplexity mode)."""
        response = await self.asearch(query=query, max_results=max_results, **kwargs)
        if response.is_empty:
            return response

        # Concurrently extract content for top results
        async def _fetch_and_augment(item: SearchResult) -> SearchResult:
            content = await self.aextract_page_content(item.url)
            if content:
                cleaned = content.strip()
                if len(cleaned) > max_extract_chars:
                    cleaned = cleaned[:max_extract_chars] + "\n...[truncated]"
                item.snippet = f"{item.snippet}\n\n[Full Content Excerpt]:\n{cleaned}"
            return item

        augmented_results = await asyncio.gather(
            *[_fetch_and_augment(r) for r in response.results],
            return_exceptions=False,
        )
        response.results = augmented_results
        return response
