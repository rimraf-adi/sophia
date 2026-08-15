"""Unit and integration tests for the DuckDuckGo search module."""

import pytest
from sophia.search.duckduckgo import DuckDuckGoSearcher
from sophia.search.formatter import format_search_results_for_llm, format_sources_markdown
from sophia.search.models import SearchResponse, SearchResult, SearchType


def test_models_and_formatting():
    results = [
        SearchResult(
            index=1,
            title="FastAPI Web Framework",
            url="https://fastapi.tiangolo.com/tutorial/",
            snippet="FastAPI is a modern, fast web framework for building APIs with Python.",
        ),
        SearchResult(
            index=2,
            title="Python Official Site",
            url="https://www.python.org",
            snippet="Python is a programming language that lets you work quickly.",
        ),
    ]

    response = SearchResponse(
        query="python fastapi",
        search_type=SearchType.TEXT,
        results=results,
        total_count=2,
        duration_seconds=0.15,
    )

    assert results[0].domain == "fastapi.tiangolo.com"
    assert results[1].domain == "python.org"
    assert not response.is_empty

    llm_context = format_search_results_for_llm(response)
    assert "Source [1]: FastAPI Web Framework" in llm_context
    assert "Source [2]: Python Official Site" in llm_context
    assert "https://fastapi.tiangolo.com/tutorial/" in llm_context

    markdown_sources = format_sources_markdown(response)
    assert "[1]" in markdown_sources
    assert "[2]" in markdown_sources
    assert "`fastapi.tiangolo.com`" in markdown_sources


@pytest.mark.asyncio
async def test_duckduckgo_searcher_async():
    searcher = DuckDuckGoSearcher(timeout=15)
    response = await searcher.asearch(query="Python programming language", max_results=2)

    assert isinstance(response, SearchResponse)
    assert response.query == "Python programming language"
    assert response.search_type == SearchType.TEXT
    if not response.is_empty:
        assert len(response.results) <= 2
        assert response.results[0].title
        assert response.results[0].url.startswith("http")
        assert response.results[0].domain
