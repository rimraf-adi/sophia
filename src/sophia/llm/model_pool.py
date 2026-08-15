"""Thread-safe model pool for round-robin rotation across available models."""

from __future__ import annotations

import logging
import threading
from typing import Sequence

logger = logging.getLogger(__name__)

# Default verified active chat models on Groq
DEFAULT_ROTATING_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound",
]


class ModelPool:
    """Manages round-robin rotation across a pool of LLM models."""

    def __init__(self, models: Sequence[str] | None = None) -> None:
        self._models = list(models) if models else list(DEFAULT_ROTATING_MODELS)
        self._current_index = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._models)

    @property
    def models(self) -> list[str]:
        return list(self._models)

    def get_next_model(self) -> str:
        """Get the next model in round-robin sequence."""
        with self._lock:
            if not self._models:
                return "llama-3.3-70b-versatile"
            model = self._models[self._current_index]
            self._current_index = (self._current_index + 1) % len(self._models)
            return model

    def add_model(self, model_name: str) -> None:
        with self._lock:
            if model_name not in self._models:
                self._models.append(model_name)

    def remove_model(self, model_name: str) -> None:
        with self._lock:
            if model_name in self._models:
                self._models.remove(model_name)
                self._current_index = 0
