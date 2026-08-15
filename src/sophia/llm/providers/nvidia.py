"""NVIDIA NIM LLM Provider implementation for fallback and specialized models."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Generator, Sequence

from openai import AsyncOpenAI, OpenAI

from sophia.llm.models import LLMMessage, LLMResponse, ProviderType
from sophia.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def _normalize_messages(messages: Sequence[LLMMessage | dict[str, str]]) -> list[dict[str, str]]:
    normalized = []
    for m in messages:
        if isinstance(m, LLMMessage):
            normalized.append(m.to_dict())
        elif isinstance(m, dict):
            normalized.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    return normalized


class NvidiaNimProvider(BaseLLMProvider):
    """NVIDIA NIM inference provider using OpenAI-compatible interface."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        default_model: str = "meta/llama-3.1-8b-instruct",
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url
        self.default_model = default_model

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.NVIDIA

    def generate(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        target_model = model or self.default_model
        formatted = _normalize_messages(messages)
        start_time = time.perf_counter()

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        resp = client.chat.completions.create(
            model=target_model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        duration = time.perf_counter() - start_time
        usage = resp.usage

        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=target_model,
            provider=self.provider_type,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            duration_seconds=round(duration, 3),
            raw_response=resp,
        )

    async def agenerate(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        target_model = model or self.default_model
        formatted = _normalize_messages(messages)
        start_time = time.perf_counter()

        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        resp = await client.chat.completions.create(
            model=target_model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        duration = time.perf_counter() - start_time
        usage = resp.usage

        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=target_model,
            provider=self.provider_type,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            duration_seconds=round(duration, 3),
            raw_response=resp,
        )

    def stream(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        target_model = model or self.default_model
        formatted = _normalize_messages(messages)

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        stream_resp = client.chat.completions.create(
            model=target_model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        for chunk in stream_resp:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta

    async def astream(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        target_model = model or self.default_model
        formatted = _normalize_messages(messages)

        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        stream_resp = await client.chat.completions.create(
            model=target_model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        async for chunk in stream_resp:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
