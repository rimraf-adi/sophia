"""Tests for Agentic Planner and Multi-Subtask Generation."""

import pytest
from sophia.engine.agentic.planner import QueryPlanner, ResearchPlan
from sophia.engine.agentic.section_agent import SectionSynthesizer
from sophia.engine.agentic.agentic_orchestrator import AgenticEngine
from sophia.chunk.chunker import TextChunk
from sophia.chunk.reranker import InMemoryReranker
from sophia.llm.router import LLMRouter
from sophia.search.models import SearchResult


def test_planner_default_fallback():
    router = LLMRouter()
    planner = QueryPlanner(router=router)
    plan: ResearchPlan = planner._default_plan("Explain Quantum Computing")
    assert len(plan.sections) == 2
    assert plan.sections[0].id == 1
    assert "Overview" in plan.sections[0].title


@pytest.mark.asyncio
async def test_agentic_pipeline_live():
    router = LLMRouter()
    reranker = InMemoryReranker()
    agentic_engine = AgenticEngine(router=router, reranker=reranker)

    test_sources = [
        SearchResult(
            index=1,
            title="Python 3.13 Overview",
            url="https://python.org/release-313",
            snippet="Python 3.13 introduces experimental free-threaded execution without GIL and JIT compiler.",
        ),
        SearchResult(
            index=2,
            title="GIL Removal Details",
            url="https://peps.python.org/pep-0703",
            snippet="PEP 703 makes the Global Interpreter Lock optional in CPython.",
        ),
    ]

    test_chunks = [
        TextChunk(
            chunk_id="chunk_1",
            source_id=1,
            url="https://python.org/release-313",
            title="Python 3.13 Overview",
            text="Python 3.13 introduces experimental free-threaded execution without GIL and JIT compiler.",
            token_count=18,
        ),
        TextChunk(
            chunk_id="chunk_2",
            source_id=2,
            url="https://peps.python.org/pep-0703",
            title="GIL Removal Details",
            text="PEP 703 makes the Global Interpreter Lock optional in CPython.",
            token_count=12,
        ),
    ]

    events = []
    async for event in agentic_engine.astream_agentic_report(
        user_question="What are the key features of Python 3.13?",
        standalone_query="Python 3.13 features nogil JIT",
        search_results=test_sources,
        all_chunks=test_chunks,
        max_tokens_per_section=100,
    ):
        events.append(event)

    event_types = [e.event_type for e in events]
    assert "status" in event_types
    assert "plan" in event_types
    assert "token" in event_types
    assert "citations" in event_types
