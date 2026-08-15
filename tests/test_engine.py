"""Tests for PerplexityEngine pipeline."""

import pytest
from sophia import ModelTier, SophiaEngine
from sophia.engine.models import PerplexityResponse


@pytest.mark.asyncio
async def test_perplexity_engine_live():
    engine = SophiaEngine()
    
    # Fast test query
    response = await engine.ask(
        question="What is the capital of France?",
        tier=ModelTier.FAST,
        max_sources=2,
    )

    assert isinstance(response, PerplexityResponse)
    assert "Paris" in response.answer
    assert response.search_queries
    assert len(response.sources) > 0
    assert response.total_duration_seconds > 0
