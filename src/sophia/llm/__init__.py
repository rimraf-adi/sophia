"""LLM Router and Provider package with key and model rotation."""

from sophia.llm.key_pool import KeyPool, mask_key
from sophia.llm.model_pool import DEFAULT_ROTATING_MODELS, ModelPool
from sophia.llm.models import (
    KeyHealth,
    LLMMessage,
    LLMResponse,
    ModelTier,
    ProviderType,
)
from sophia.llm.providers import BaseLLMProvider, GroqProvider, NvidiaNimProvider
from sophia.llm.router import LLMRouter

__all__ = [
    "LLMRouter",
    "KeyPool",
    "KeyHealth",
    "ModelPool",
    "DEFAULT_ROTATING_MODELS",
    "mask_key",
    "LLMMessage",
    "LLMResponse",
    "ModelTier",
    "ProviderType",
    "BaseLLMProvider",
    "GroqProvider",
    "NvidiaNimProvider",
]
