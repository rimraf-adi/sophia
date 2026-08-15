"""Text chunking with token awareness and source metadata tagging."""

from __future__ import annotations

from typing import Sequence
from pydantic import BaseModel, Field
import tiktoken

from sophia.scrape.scraper import ScrapedDocument


class TextChunk(BaseModel):
    """A single tagged chunk of text from a scraped source."""

    chunk_id: str = Field(..., description="Unique ID: e.g. source_1_chunk_0")
    source_id: int = Field(..., description="1-based citation source ID")
    url: str = Field(..., description="Source URL")
    title: str = Field(default="", description="Source page title")
    text: str = Field(..., description="Clean chunk text")
    token_count: int = Field(default=0, description="Number of tokens in chunk")


class TextChunker:
    """Chunks documents into 500-800 token slices with ~100 token overlap."""

    def __init__(
        self,
        chunk_size_tokens: int = 600,
        chunk_overlap_tokens: int = 100,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size_tokens
        self.chunk_overlap = chunk_overlap_tokens
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("gpt2")

    def chunk_document(self, doc: ScrapedDocument) -> list[TextChunk]:
        """Split a single document into tagged TextChunk objects."""
        text = doc.text.strip()
        if not text:
            return []

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.chunk_size:
            return [
                TextChunk(
                    chunk_id=f"src_{doc.source_id}_chunk_0",
                    source_id=doc.source_id,
                    url=doc.url,
                    title=doc.title,
                    text=text,
                    token_count=len(tokens),
                )
            ]

        chunks: list[TextChunk] = []
        start_idx = 0
        chunk_idx = 0
        step = self.chunk_size - self.chunk_overlap

        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=f"src_{doc.source_id}_chunk_{chunk_idx}",
                        source_id=doc.source_id,
                        url=doc.url,
                        title=doc.title,
                        text=chunk_text,
                        token_count=len(chunk_tokens),
                    )
                )
                chunk_idx += 1

            if end_idx >= len(tokens):
                break
            start_idx += step

        return chunks

    def chunk_documents(self, docs: Sequence[ScrapedDocument]) -> list[TextChunk]:
        """Chunk a collection of documents."""
        all_chunks: list[TextChunk] = []
        for d in docs:
            all_chunks.extend(self.chunk_document(d))
        return all_chunks
