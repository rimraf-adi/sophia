"""Abstract base interface for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Generator, Sequence

from sophia.llm.models import LLMMessage, LLMResponse, ProviderType


class BaseLLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return provider type enum."""
        pass

    @abstractmethod
    def generate(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronously generate text completion."""
        pass

    @abstractmethod
    async def agenerate(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Asynchronously generate text completion."""
        pass

    @abstractmethod
    def stream(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Synchronously stream token chunks."""
        pass

    @abstractmethod
    async def astream(
        self,
        messages: Sequence[LLMMessage | dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Asynchronously stream token chunks."""
        pass
