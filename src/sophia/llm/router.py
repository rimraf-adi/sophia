"""LLM Router with Groq key rotation, model tiers, streaming, and multi-provider failover."""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Generator, Sequence

from sophia.llm.key_pool import KeyPool
from sophia.llm.model_pool import DEFAULT_ROTATING_MODELS, ModelPool
from sophia.llm.models import LLMMessage, LLMResponse, ModelTier, ProviderType
from sophia.llm.providers.base import BaseLLMProvider
from sophia.llm.providers.groq import GroqProvider
from sophia.llm.providers.nvidia import NvidiaNimProvider

logger = logging.getLogger(__name__)

# Default model tier mappings mapped to live Groq & NVIDIA models
DEFAULT_TIER_MODELS: dict[ModelTier, dict[ProviderType, str]] = {
    ModelTier.FAST: {
        ProviderType.GROQ: "llama-3.1-8b-instant",
        ProviderType.NVIDIA: "meta/llama-3.1-8b-instruct",
    },
    ModelTier.BALANCED: {
        ProviderType.GROQ: "llama-3.3-70b-versatile",
        ProviderType.NVIDIA: "meta/llama-3.1-70b-instruct",
    },
    ModelTier.REASONING: {
        ProviderType.GROQ: "openai/gpt-oss-120b",
        ProviderType.NVIDIA: "meta/llama-3.1-70b-instruct",
    },
}


def _ensure_messages(prompt_or_messages: str | Sequence[LLMMessage | dict[str, str]]) -> list[LLMMessage]:
    """Helper to convert raw string prompt or messages list to list of LLMMessage."""
    if isinstance(prompt_or_messages, str):
        return [LLMMessage(role="user", content=prompt_or_messages)]
    
    result = []
    for m in prompt_or_messages:
        if isinstance(m, LLMMessage):
            result.append(m)
        elif isinstance(m, dict):
            result.append(LLMMessage(role=m.get("role", "user"), content=m.get("content", "")))
    return result


class LLMRouter:
    """Intelligent LLM Router with round-robin key rotation AND round-robin model rotation."""

    def __init__(
        self,
        groq_api_keys: list[str] | None = None,
        nvidia_api_key: str | None = None,
        default_tier: ModelTier = ModelTier.BALANCED,
        models_to_rotate: list[str] | None = None,
        enable_model_rotation: bool = True,
    ) -> None:
        """Initialize LLMRouter with both key rotation and model rotation.

        Args:
            groq_api_keys: List of Groq API keys (or reads GROQ_API_KEYS from env).
            nvidia_api_key: Optional NVIDIA NIM API key for fallback.
            default_tier: Default ModelTier.
            models_to_rotate: Custom list of model names to rotate across (or defaults to active models).
            enable_model_rotation: If True, cycles models round-robin for each request.
        """
        self.default_tier = default_tier
        self.enable_model_rotation = enable_model_rotation
        self.model_pool = ModelPool(models_to_rotate)

        # Load environment variables if not already loaded
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        # 1. Initialize Groq Provider with Key Pool
        raw_groq_keys = groq_api_keys
        if raw_groq_keys is None:
            env_keys = os.getenv("GROQ_API_KEYS") or os.getenv("GROQ_API_KEY") or ""
            raw_groq_keys = [k.strip() for k in env_keys.split(",") if k.strip()]

        self.groq_provider: GroqProvider | None = None
        if raw_groq_keys:
            key_pool = KeyPool(raw_groq_keys, provider_name="groq")
            self.groq_provider = GroqProvider(key_pool=key_pool)
            logger.info(
                "LLMRouter initialized with %d Groq key(s) and %d rotating model(s).",
                len(key_pool),
                len(self.model_pool),
            )
        else:
            logger.warning("No Groq API keys found. Groq provider will not be available.")

        # 2. Initialize NVIDIA NIM Fallback Provider
        nim_key = nvidia_api_key or os.getenv("NVIDIA_NIM_API_KEY")
        self.nvidia_provider: NvidiaNimProvider | None = None
        if nim_key and nim_key.strip():
            self.nvidia_provider = NvidiaNimProvider(api_key=nim_key)
            logger.info("LLMRouter initialized with NVIDIA NIM fallback provider.")

        if not self.groq_provider and not self.nvidia_provider:
            raise ValueError(
                "LLMRouter requires at least one configured provider (GROQ_API_KEYS or NVIDIA_NIM_API_KEY)."
            )

    def resolve_model(
        self,
        tier: ModelTier | None = None,
        provider: ProviderType = ProviderType.GROQ,
        model_override: str | None = None,
    ) -> str:
        """Resolve exact model identifier using model pool rotation or explicit override."""
        if model_override:
            return model_override

        # If model rotation is enabled, rotate through the model pool
        if self.enable_model_rotation:
            return self.model_pool.get_next_model()

        target_tier = tier or self.default_tier
        return DEFAULT_TIER_MODELS.get(target_tier, {}).get(provider, "llama-3.3-70b-versatile")

    def complete(
        self,
        prompt_or_messages: str | Sequence[LLMMessage | dict[str, str]] | None = None,
        tier: ModelTier | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        messages: Sequence[LLMMessage | dict[str, str]] | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronously route and complete a chat prompt."""
        target_tier = tier or self.default_tier
        input_data = prompt_or_messages or messages or prompt or ""
        msg_list = _ensure_messages(input_data)

        # Try Groq first with rotating keys
        if self.groq_provider:
            try:
                target_model = self.resolve_model(target_tier, ProviderType.GROQ, model)
                return self.groq_provider.generate(
                    messages=msg_list,
                    model=target_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except Exception as e:
                logger.error("Primary Groq provider failed: %s. Attempting failover...", str(e))
                if not self.nvidia_provider:
                    raise

        # Fallback to NVIDIA NIM
        if self.nvidia_provider:
            target_model = self.resolve_model(target_tier, ProviderType.NVIDIA, model)
            logger.info("Routing request to fallback provider: NVIDIA NIM (%s)", target_model)
            return self.nvidia_provider.generate(
                messages=msg_list,
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        raise RuntimeError("All configured LLM providers failed.")

    async def acomplete(
        self,
        prompt_or_messages: str | Sequence[LLMMessage | dict[str, str]] | None = None,
        tier: ModelTier | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        messages: Sequence[LLMMessage | dict[str, str]] | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Asynchronously route and complete a chat prompt."""
        target_tier = tier or self.default_tier
        input_data = prompt_or_messages or messages or prompt or ""
        msg_list = _ensure_messages(input_data)

        # Try Groq first with rotating keys
        if self.groq_provider:
            try:
                target_model = self.resolve_model(target_tier, ProviderType.GROQ, model)
                return await self.groq_provider.agenerate(
                    messages=msg_list,
                    model=target_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            except Exception as e:
                logger.error("Primary Groq provider failed in async: %s. Attempting failover...", str(e))
                if not self.nvidia_provider:
                    raise

        # Fallback to NVIDIA NIM
        if self.nvidia_provider:
            target_model = self.resolve_model(target_tier, ProviderType.NVIDIA, model)
            logger.info("Routing async request to fallback provider: NVIDIA NIM (%s)", target_model)
            return await self.nvidia_provider.agenerate(
                messages=msg_list,
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        raise RuntimeError("All configured LLM providers failed in async.")

    def stream(
        self,
        prompt_or_messages: str | Sequence[LLMMessage | dict[str, str]] | None = None,
        tier: ModelTier | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        messages: Sequence[LLMMessage | dict[str, str]] | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Synchronously stream tokens from routed provider."""
        target_tier = tier or self.default_tier
        input_data = prompt_or_messages or messages or prompt or ""
        msg_list = _ensure_messages(input_data)

        if self.groq_provider:
            try:
                target_model = self.resolve_model(target_tier, ProviderType.GROQ, model)
                yield from self.groq_provider.stream(
                    messages=msg_list,
                    model=target_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return
            except Exception as e:
                logger.error("Groq stream failed: %s. Falling back...", str(e))
                if not self.nvidia_provider:
                    raise

        if self.nvidia_provider:
            target_model = self.resolve_model(target_tier, ProviderType.NVIDIA, model)
            yield from self.nvidia_provider.stream(
                messages=msg_list,
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return

        raise RuntimeError("No LLM provider available for streaming.")

    async def astream(
        self,
        prompt_or_messages: str | Sequence[LLMMessage | dict[str, str]] | None = None,
        tier: ModelTier | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        messages: Sequence[LLMMessage | dict[str, str]] | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Asynchronously stream tokens from routed provider."""
        target_tier = tier or self.default_tier
        input_data = prompt_or_messages or messages or prompt or ""
        msg_list = _ensure_messages(input_data)

        if self.groq_provider:
            try:
                target_model = self.resolve_model(target_tier, ProviderType.GROQ, model)
                async for chunk in self.groq_provider.astream(
                    messages=msg_list,
                    model=target_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    yield chunk
                return
            except Exception as e:
                logger.error("Groq async stream failed: %s. Falling back...", str(e))
                if not self.nvidia_provider:
                    raise

        if self.nvidia_provider:
            target_model = self.resolve_model(target_tier, ProviderType.NVIDIA, model)
            async for chunk in self.nvidia_provider.astream(
                messages=msg_list,
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            ):
                yield chunk
            return

        raise RuntimeError("No LLM provider available for async streaming.")

    def list_groq_models(self) -> list[dict[str, Any]]:
        """Query Groq API using rotating keys to return all currently active models."""
        import requests
        if not self.groq_provider:
            return []
        
        key = self.groq_provider.key_pool.get_next_key()
        try:
            url = "https://api.groq.com/openai/v1/models"
            resp = requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=10)
            data = resp.json()
            models = data.get("data", [])
            self.groq_provider.key_pool.report_success(key)
            return sorted(models, key=lambda x: x.get("id", ""))
        except Exception as e:
            self.groq_provider.key_pool.report_failure(key)
            logger.warning("Failed to list Groq models: %s", str(e))
            return []

    def get_pool_status(self) -> dict[str, Any]:
        """Return diagnostic metrics and health of all keys across providers."""
        status = {}
        if self.groq_provider:
            status["groq"] = {
                "total_keys": len(self.groq_provider.key_pool),
                "keys": self.groq_provider.key_pool.get_stats(),
            }
        if self.nvidia_provider:
            status["nvidia"] = {
                "available": True,
                "model": self.nvidia_provider.default_model,
            }
        return status
