"""Comprehensive tests for the full Perplexity Clone architecture."""

import pytest
from sophia.cache.cache import QueryCache
from sophia.chunk.chunker import TextChunk, TextChunker
from sophia.chunk.reranker import InMemoryReranker
from sophia.engine.citation_mapper import extract_inline_citations, map_citations
from sophia.scrape.scraper import AsyncScraper, ScrapedDocument
from sophia.session.session import SessionStore


def test_query_cache(tmp_path):
    db_file = str(tmp_path / "test_cache.db")
    cache = QueryCache(db_path=db_file, default_ttl_seconds=2)

    # Miss
    assert cache.get("What is quantum computing?") is None

    # Set
    cache.set("What is quantum computing?", {"answer": "A fast computer."}, ttl_seconds=2)
    cached = cache.get("What is quantum computing?")
    assert cached is not None
    assert cached["answer"] == "A fast computer."

    # Test normalization
    cached_norm = cache.get("  what   is QUANTUM  computing?  ")
    assert cached_norm is not None


def test_session_store():
    store = SessionStore()
    sess = store.get_or_create("user_123")
    sess.add_user_message("Who created Python?")
    sess.add_assistant_message("Guido van Rossum created Python in 1991.")

    summary = sess.get_history_summary()
    assert "User: Who created Python?" in summary
    assert "Guido van Rossum" in summary


def test_chunker_and_reranker():
    chunker = TextChunker(chunk_size_tokens=100, chunk_overlap_tokens=20)
    doc = ScrapedDocument(
        source_id=1,
        url="https://python.org",
        title="Python Overview",
        text="Python is a dynamic programming language. " * 30,
        raw_snippet="Python overview",
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    assert chunks[0].source_id == 1

    # Test reranker
    reranker = InMemoryReranker()
    ranked = reranker.rerank(query="dynamic programming language", chunks=chunks, top_k=2)
    assert len(ranked) >= 1
    assert ranked[0].score > 0


def test_citation_extraction():
    text = "Python was created by Guido [1]. It was released in 1991 [2] and is very popular [1][3]."
    cited = extract_inline_citations(text)
    assert cited == [1, 2, 3]

    dummy_sources = [
        ScrapedDocument(source_id=1, url="https://python.org", title="Python", text="..."),
        ScrapedDocument(source_id=2, url="https://wikipedia.org/python", title="Wiki", text="..."),
    ]
    citations = map_citations(text, dummy_sources)
    assert len(citations) == 2
    assert citations[0].id == 1
    assert citations[0].url == "https://python.org"
    assert citations[1].id == 2
