"""Unit tests for the LLM Router and KeyPool rotation."""

import pytest
from sophia.llm.key_pool import KeyPool
from sophia.llm.models import ModelTier
from sophia.llm.router import LLMRouter


def test_key_pool_round_robin():
    fake_keys = ["gsk_key_alpha_1111", "gsk_key_bravo_2222", "gsk_key_charlie_3333"]
    pool = KeyPool(fake_keys, provider_name="test")

    # Verify round-robin sequence
    k1 = pool.get_next_key()
    k2 = pool.get_next_key()
    k3 = pool.get_next_key()
    k4 = pool.get_next_key()

    assert k1 == fake_keys[0]
    assert k2 == fake_keys[1]
    assert k3 == fake_keys[2]
    assert k4 == fake_keys[0]  # wrapped around


def test_key_pool_cooldown_on_rate_limit():
    fake_keys = ["gsk_key_1111", "gsk_key_2222"]
    pool = KeyPool(fake_keys, provider_name="test", default_cooldown_seconds=30.0)

    # Key 1 is used and hits rate limit
    k1 = pool.get_next_key()
    pool.report_rate_limit(k1)

    # Next call should skip Key 1 and return Key 2
    k2 = pool.get_next_key()
    assert k2 == fake_keys[1]

    # Another call should still skip Key 1 because it is on cooldown
    k2_again = pool.get_next_key()
    assert k2_again == fake_keys[1]

    stats = pool.get_stats()
    assert len(stats) == 2
    assert stats[0]["cooldown_remaining_sec"] > 0
    assert not stats[0]["is_available"]
    assert stats[1]["is_available"]


@pytest.mark.asyncio
async def test_llm_router_rotation_live():
    # Test real LLMRouter with available environment keys
    router = LLMRouter(enable_model_rotation=True)
    
    # Make 3 consecutive async requests and verify key AND model rotation
    resp1 = await router.acomplete("Count from 1 to 2 in one word.", max_tokens=60)
    resp2 = await router.acomplete("Count from 3 to 4 in one word.", max_tokens=60)
    resp3 = await router.acomplete("Count from 5 to 6 in one word.", max_tokens=60)
    
    assert resp1.content
    assert resp2.content
    assert resp3.content
    
    print(f"\nReq 1 -> Key: {resp1.key_masked}, Model: {resp1.model}")
    print(f"Req 2 -> Key: {resp2.key_masked}, Model: {resp2.model}")
    print(f"Req 3 -> Key: {resp3.key_masked}, Model: {resp3.model}")

    # Check that keys and models cycled
    assert resp1.key_masked != resp2.key_masked
    assert resp1.model != resp2.model

    # Check pool status
    status = router.get_pool_status()
    assert "groq" in status
    assert status["groq"]["total_keys"] >= 1
