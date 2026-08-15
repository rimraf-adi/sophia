"""Groq LLM Provider implementation with key pool rotation, retry logic, and streaming."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Generator, Sequence

from groq import AsyncGroq, Groq, RateLimitError, AuthenticationError

from sophia.llm.key_pool import KeyPool, mask_key
from sophia.llm.models import LLMMessage, LLMResponse, ProviderType
from sophia.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def _normalize_messages(messages: Sequence[LLMMessage | dict[str, str]]) -> list[dict[str, str]]:
    """Convert mixed LLMMessage / dict objects to standard dict format."""
    normalized = []
    for m in messages:
        if isinstance(m, LLMMessage):
            normalized.append(m.to_dict())
        elif isinstance(m, dict):
            normalized.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    return normalized


class GroqProvider(BaseLLMProvider):
    """Groq LLM Provider utilizing rotating API keys for maximum throughput and reliability."""

    def __init__(
        self,
        key_pool: KeyPool | list[str],
        default_model: str = "llama-3.3-70b-versatile",
        max_retries: int = 5,
    ) -> None:
        """Initialize Groq provider.

        Args:
            key_pool: KeyPool instance or list of API keys.
            default_model: Default Groq model to use.
            max_retries: Max retries across rotating keys.
        """
        if isinstance(key_pool, list):
            self.key_pool = KeyPool(key_pool, provider_name="groq")
        else:
            self.key_pool = key_pool
            
        self.default_model = default_model
        self.max_retries = max(1, min(max_retries, len(self.key_pool) * 2))

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.GROQ

    def generate(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response with automatic key rotation and failover."""
        target_model = model or self.default_model
        formatted_messages = _normalize_messages(messages)
        start_time = time.perf_counter()

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            key = self.key_pool.get_next_key()
            try:
                client = Groq(api_key=key)
                response = client.chat.completions.create(
                    model=target_model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                self.key_pool.report_success(key)
                duration = time.perf_counter() - start_time

                usage = response.usage
                prompt_toks = usage.prompt_tokens if usage else 0
                comp_toks = usage.completion_tokens if usage else 0
                tot_toks = usage.total_tokens if usage else (prompt_toks + comp_toks)

                msg = response.choices[0].message
                content = msg.content or getattr(msg, "reasoning", "") or ""

                return LLMResponse(
                    content=content,
                    model=target_model,
                    provider=self.provider_type,
                    key_masked=mask_key(key),
                    prompt_tokens=prompt_toks,
                    completion_tokens=comp_toks,
                    total_tokens=tot_toks,
                    duration_seconds=round(duration, 3),
                    raw_response=response,
                )

            except RateLimitError as e:
                last_error = e
                logger.warning(
                    "[Groq] Key %s hit rate limit (attempt %d/%d). Rotating key...",
                    mask_key(key),
                    attempt,
                    self.max_retries,
                )
                self.key_pool.report_rate_limit(key)
            except AuthenticationError as e:
                last_error = e
                logger.error("[Groq] Auth error on key %s. Disabling key...", mask_key(key))
                self.key_pool.report_failure(key, is_auth_error=True)
            except Exception as e:
                last_error = e
                logger.warning(
                    "[Groq] Error on key %s: %s (attempt %d/%d). Rotating...",
                    mask_key(key),
                    str(e),
                    attempt,
                    self.max_retries,
                )
                self.key_pool.report_failure(key)

        raise RuntimeError(
            f"Groq generation failed after {self.max_retries} attempts across keys. Last error: {last_error}"
        )

    async def agenerate(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Asynchronously generate response with rotating keys."""
        target_model = model or self.default_model
        formatted_messages = _normalize_messages(messages)
        start_time = time.perf_counter()

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            key = await self.key_pool.aget_next_key()
            try:
                client = AsyncGroq(api_key=key)
                response = await client.chat.completions.create(
                    model=target_model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                self.key_pool.report_success(key)
                duration = time.perf_counter() - start_time

                usage = response.usage
                prompt_toks = usage.prompt_tokens if usage else 0
                comp_toks = usage.completion_tokens if usage else 0
                tot_toks = usage.total_tokens if usage else (prompt_toks + comp_toks)

                msg = response.choices[0].message
                content = msg.content or getattr(msg, "reasoning", "") or ""

                return LLMResponse(
                    content=content,
                    model=target_model,
                    provider=self.provider_type,
                    key_masked=mask_key(key),
                    prompt_tokens=prompt_toks,
                    completion_tokens=comp_toks,
                    total_tokens=tot_toks,
                    duration_seconds=round(duration, 3),
                    raw_response=response,
                )

            except RateLimitError as e:
                last_error = e
                logger.warning(
                    "[Groq Async] Key %s hit rate limit (attempt %d/%d). Rotating key...",
                    mask_key(key),
                    attempt,
                    self.max_retries,
                )
                self.key_pool.report_rate_limit(key)
            except AuthenticationError as e:
                last_error = e
                logger.error("[Groq Async] Auth error on key %s. Disabling key...", mask_key(key))
                self.key_pool.report_failure(key, is_auth_error=True)
            except Exception as e:
                last_error = e
                logger.warning(
                    "[Groq Async] Error on key %s: %s (attempt %d/%d). Rotating...",
                    mask_key(key),
                    str(e),
                    attempt,
                    self.max_retries,
                )
                self.key_pool.report_failure(key)

        raise RuntimeError(
            f"Groq async generation failed after {self.max_retries} attempts across keys. Last error: {last_error}"
        )

    def stream(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Synchronously stream response tokens with key rotation."""
        target_model = model or self.default_model
        formatted_messages = _normalize_messages(messages)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            key = self.key_pool.get_next_key()
            try:
                client = Groq(api_key=key)
                stream_response = client.chat.completions.create(
                    model=target_model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs,
                )
                has_yielded = False
                for chunk in stream_response:
                    delta = chunk.choices[0].delta.content or getattr(chunk.choices[0].delta, "reasoning", "") or ""
                    if delta:
                        has_yielded = True
                        yield delta

                self.key_pool.report_success(key)
                return

            except Exception as e:
                last_error = e
                if isinstance(e, RateLimitError):
                    self.key_pool.report_rate_limit(key)
                elif isinstance(e, AuthenticationError):
                    self.key_pool.report_failure(key, is_auth_error=True)
                else:
                    self.key_pool.report_failure(key)
                logger.warning(
                    "[Groq Stream] Stream failed on key %s (attempt %d/%d). Error: %s",
                    mask_key(key),
                    attempt,
                    self.max_retries,
                    str(e),
                )

        raise RuntimeError(f"Groq streaming failed after retries: {last_error}")

    async def astream(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Asynchronously stream response tokens with key rotation."""
        target_model = model or self.default_model
        formatted_messages = _normalize_messages(messages)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            key = await self.key_pool.aget_next_key()
            try:
                client = AsyncGroq(api_key=key)
                stream_response = await client.chat.completions.create(
                    model=target_model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs,
                )
                async for chunk in stream_response:
                    delta = chunk.choices[0].delta.content or getattr(chunk.choices[0].delta, "reasoning", "") or ""
                    if delta:
                        yield delta

                self.key_pool.report_success(key)
                return

            except Exception as e:
                last_error = e
                if isinstance(e, RateLimitError):
                    self.key_pool.report_rate_limit(key)
                elif isinstance(e, AuthenticationError):
                    self.key_pool.report_failure(key, is_auth_error=True)
                else:
                    self.key_pool.report_failure(key)
                logger.warning(
                    "[Groq Async Stream] Stream failed on key %s (attempt %d/%d). Error: %s",
                    mask_key(key),
                    attempt,
                    self.max_retries,
                    str(e),
                )

        raise RuntimeError(f"Groq async streaming failed after retries: {last_error}")
