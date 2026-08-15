"""Citation parsing, validation, and mapping for Perplexity responses."""

from __future__ import annotations

import re
from typing import Sequence
from pydantic import BaseModel, Field

from sophia.chunk.reranker import RankedChunk
from sophia.search.models import SearchResult


class Citation(BaseModel):
    """A verified citation mapping an inline marker [n] to source metadata."""

    id: int = Field(..., description="1-based citation number")
    title: str = Field(..., description="Source title")
    url: str = Field(..., description="Source URL")
    domain: str = Field(default="", description="Source domain")
    snippet: str = Field(default="", description="Key context snippet from chunk")


def extract_inline_citations(text: str) -> list[int]:
    """Extract all cited numbers from text like [1], [2], [1][3], etc."""
    matches = re.findall(r"\[(\d+)\]", text)
    cited_ids = []
    seen = set()
    for m in matches:
        num = int(m)
        if num not in seen:
            seen.add(num)
            cited_ids.append(num)
    return cited_ids


def map_citations(
    text: str,
    sources: Sequence[Any],
) -> list[Citation]:
    """Map all inline citation markers [n] in generated text to source objects."""
    cited_ids = extract_inline_citations(text)
    
    # Build lookup map
    source_map: dict[int, Citation] = {}
    for s in sources:
        if isinstance(s, SearchResult):
            source_map[s.index] = Citation(
                id=s.index,
                title=s.title,
                url=s.url,
                domain=s.domain,
                snippet=s.snippet,
            )
        elif hasattr(s, "source_id") and hasattr(s, "url"):
            from urllib.parse import urlparse
            domain = urlparse(s.url).netloc.replace("www.", "")
            source_map[s.source_id] = Citation(
                id=s.source_id,
                title=getattr(s, "title", "Untitled"),
                url=s.url,
                domain=domain,
                snippet=getattr(s, "text", "")[:200] if hasattr(s, "text") else getattr(s, "raw_snippet", ""),
            )
        elif isinstance(s, RankedChunk):
            chunk = s.chunk
            if chunk.source_id not in source_map:
                from urllib.parse import urlparse
                domain = urlparse(chunk.url).netloc.replace("www.", "")
                source_map[chunk.source_id] = Citation(
                    id=chunk.source_id,
                    title=chunk.title,
                    url=chunk.url,
                    domain=domain,
                    snippet=chunk.text[:200],
                )

    result_citations: list[Citation] = []
    for cid in cited_ids:
        if cid in source_map:
            result_citations.append(source_map[cid])

    return result_citations


def assemble_reranked_context(
    ranked_chunks: Sequence[RankedChunk],
    max_chunks: int = 10,
) -> str:
    """Format top reranked chunks into grounded prompt context."""
    if not ranked_chunks:
        return "No relevant sources found."

    target = ranked_chunks[:max_chunks]
    blocks = []
    for rc in target:
        c = rc.chunk
        blocks.append(f"[{c.source_id}] {c.title} — {c.url}\n{c.text}")

    return "\n\n".join(blocks)
