"""Data models for LLM requests, responses, routing tiers, and provider configurations."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    """Supported LLM Providers."""

    GROQ = "groq"
    NVIDIA = "nvidia"


class ModelTier(str, Enum):
    """Model tiers for specialized routing tasks in Perplexity pipeline."""

    FAST = "fast"          # Low latency query planning, search term generation (e.g., Llama 3.1 8B)
    BALANCED = "balanced"  # Standard responses, syntheses (e.g., Llama 3.3 70B)
    REASONING = "reasoning"# Deep research, chain-of-thought, complex answers (e.g., DeepSeek R1 / Llama 70B)


class LLMMessage(BaseModel):
    """A chat message in a conversation."""

    role: Literal["system", "user", "assistant"] = "user"
    content: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMResponse(BaseModel):
    """Unified response object returned from any LLM provider."""

    content: str = Field(..., description="Generated text content")
    model: str = Field(..., description="Model name used for generation")
    provider: ProviderType = Field(..., description="Provider that generated the response")
    key_masked: str | None = Field(default=None, description="Masked API key used for request")
    prompt_tokens: int = Field(default=0, description="Tokens used in prompt")
    completion_tokens: int = Field(default=0, description="Tokens generated")
    total_tokens: int = Field(default=0, description="Total tokens used")
    duration_seconds: float = Field(default=0.0, description="Request execution duration")
    raw_response: Any = Field(default=None, exclude=True, description="Raw provider response")


class KeyHealth(BaseModel):
    """Health and statistics for an individual API key."""

    key: str = Field(..., description="The raw API key", exclude=True)
    masked_key: str = Field(..., description="Masked version for logging/monitoring")
    total_requests: int = Field(default=0, description="Total successful requests made")
    failed_requests: int = Field(default=0, description="Total failed requests")
    consecutive_failures: int = Field(default=0, description="Consecutive failure count")
    cooldown_until: float = Field(default=0.0, description="Timestamp when cooldown expires")
    is_active: bool = Field(default=True, description="Whether the key is enabled")
    last_used_timestamp: float = Field(default=0.0, description="Last time key was used")

    @property
    def is_available(self) -> bool:
        """Check if key is active and not cooling down."""
        import time
        return self.is_active and time.time() >= self.cooldown_until
