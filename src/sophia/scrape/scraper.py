"""Async parallel web page fetcher and content extractor using trafilatura."""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence
from urllib.parse import urlparse
import httpx
from pydantic import BaseModel, Field
import trafilatura

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
}


class ScrapedDocument(BaseModel):
    """Clean scraped content from a web page."""

    source_id: int = Field(default=1, description="1-based index matching search result")
    url: str = Field(..., description="Target URL")
    title: str = Field(default="", description="Page title")
    text: str = Field(..., description="Extracted clean body text")
    raw_snippet: str = Field(default="", description="Original search snippet")

    @property
    def domain(self) -> str:
        try:
            return urlparse(self.url).netloc.replace("www.", "")
        except Exception:
            return ""


class AsyncScraper:
    """Parallel async web fetcher and cleaner with Trafilatura."""

    def __init__(
        self,
        timeout: float = 3.0,
        max_concurrent: int = 10,
        min_text_length: int = 80,
    ) -> None:
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.min_text_length = min_text_length

    async def fetch_and_extract_single(
        self,
        client: httpx.AsyncClient,
        url: str,
        source_id: int,
        title: str = "",
        snippet: str = "",
    ) -> ScrapedDocument | None:
        """Fetch a single URL and extract clean text with Trafilatura."""
        try:
            resp = await client.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=self.timeout,
                follow_redirects=True,
            )
            if resp.status_code != 200:
                logger.debug("Dropped %s: status %d", url, resp.status_code)
                return None

            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                logger.debug("Dropped %s: unsupported content-type %s", url, content_type)
                return None

            html_text = resp.text
            if not html_text:
                return None

            # Clean extraction with trafilatura
            extracted_text = trafilatura.extract(
                html_text,
                url=url,
                include_comments=False,
                include_tables=True,
                include_links=False,
                favor_recall=True,
            )

            if not extracted_text or len(extracted_text.strip()) < self.min_text_length:
                # If extraction returns empty or too short, fallback to snippet if meaningful
                if snippet and len(snippet.strip()) >= 50:
                    extracted_text = snippet
                else:
                    return None

            return ScrapedDocument(
                source_id=source_id,
                url=url,
                title=title or "Untitled",
                text=extracted_text.strip(),
                raw_snippet=snippet,
            )

        except Exception as e:
            logger.debug("Failed to scrape %s: %s", url, str(e))
            # Fallback to search snippet if available
            if snippet and len(snippet.strip()) >= 60:
                return ScrapedDocument(
                    source_id=source_id,
                    url=url,
                    title=title or "Untitled",
                    text=snippet.strip(),
                    raw_snippet=snippet,
                )
            return None

    async def scrape_urls_parallel(
        self,
        items: Sequence[dict[str, Any]],
    ) -> list[ScrapedDocument]:
        """Parallel fetch of list of items: [{'url': ..., 'source_id': ..., 'title': ..., 'snippet': ...}]."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async with httpx.AsyncClient(http2=True, verify=False) as client:
            async def _safe_fetch(item: dict[str, Any]) -> ScrapedDocument | None:
                async with semaphore:
                    return await self.fetch_and_extract_single(
                        client=client,
                        url=item["url"],
                        source_id=item.get("source_id", 1),
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                    )

            tasks = [_safe_fetch(item) for item in items]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_docs: list[ScrapedDocument] = []
            for r in results:
                if isinstance(r, ScrapedDocument):
                    valid_docs.append(r)

            return valid_docs
