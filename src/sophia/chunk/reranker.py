"""In-memory cosine similarity and BM25 hybrid reranker for ephemeral search chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence
import numpy as np
from pydantic import BaseModel, Field

from sophia.chunk.chunker import TextChunk


class RankedChunk(BaseModel):
    """Chunk annotated with relevance score."""

    chunk: TextChunk
    score: float = Field(default=0.0, description="Relevance score (higher is better)")


def tokenize_text(text: str) -> list[str]:
    """Tokenize lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())


class InMemoryReranker:
    """Fast, in-memory BM25 + TF-IDF cosine similarity ranker for ephemeral per-query chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def rerank(
        self,
        query: str,
        chunks: Sequence[TextChunk],
        top_k: int = 10,
    ) -> list[RankedChunk]:
        """Rank chunks against query using BM25 scoring and return top_k chunks."""
        if not chunks:
            return []

        query_tokens = tokenize_text(query)
        if not query_tokens:
            # Return first top_k if query has no words
            return [RankedChunk(chunk=c, score=1.0) for c in chunks[:top_k]]

        # Compute document frequencies
        doc_tokens_list = [tokenize_text(c.text + " " + c.title) for c in chunks]
        num_docs = len(chunks)
        avg_doc_len = sum(len(dt) for dt in doc_tokens_list) / max(1, num_docs)

        df = Counter()
        for dt in doc_tokens_list:
            unique_terms = set(dt)
            for term in unique_terms:
                df[term] += 1

        # Calculate IDF
        idf: dict[str, float] = {}
        for term, freq in df.items():
            idf[term] = math.log(1 + (num_docs - freq + 0.5) / (freq + 0.5))

        # Score each chunk
        scored_chunks: list[RankedChunk] = []
        for idx, chunk in enumerate(chunks):
            doc_tokens = doc_tokens_list[idx]
            doc_len = len(doc_tokens)
            doc_term_counts = Counter(doc_tokens)

            bm25_score = 0.0
            for q_term in query_tokens:
                if q_term in doc_term_counts:
                    tf = doc_term_counts[q_term]
                    term_idf = idf.get(q_term, 0.1)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(1.0, avg_doc_len)))
                    bm25_score += term_idf * (numerator / max(0.001, denominator))

            # Bonus for title keyword match
            title_lower = chunk.title.lower()
            for q_term in query_tokens:
                if q_term in title_lower:
                    bm25_score += 0.5

            scored_chunks.append(RankedChunk(chunk=chunk, score=round(float(bm25_score), 4)))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        return scored_chunks[:top_k]
