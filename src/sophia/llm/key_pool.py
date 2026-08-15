"""Thread-safe and Async-safe API Key Pool with automatic rotation and rate-limit cooldowns."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Sequence

from sophia.llm.models import KeyHealth

logger = logging.getLogger(__name__)


def mask_key(key: str) -> str:
    """Mask an API key for safe logging (e.g., gsk_...1234)."""
    if len(key) <= 8:
        return "********"
    return f"{key[:6]}...{key[-4:]}"


class KeyPool:
    """Manages a pool of API keys with round-robin rotation, health checks, and cooldowns.

    Supports automatic failover on 429 rate limits or transient errors.
    """

    def __init__(
        self,
        keys: Sequence[str],
        provider_name: str = "groq",
        default_cooldown_seconds: float = 60.0,
        max_consecutive_failures: int = 5,
    ) -> None:
        """Initialize key pool.

        Args:
            keys: Sequence of raw API keys.
            provider_name: Label for logging/diagnostics.
            default_cooldown_seconds: Cooldown duration on rate limits (429).
            max_consecutive_failures: Number of continuous failures before disabling key.
        """
        clean_keys = [k.strip() for k in keys if k and k.strip()]
        if not clean_keys:
            raise ValueError(f"KeyPool for {provider_name} initialized with zero valid keys.")

        self.provider_name = provider_name
        self.default_cooldown_seconds = default_cooldown_seconds
        self.max_consecutive_failures = max_consecutive_failures

        self._keys: list[KeyHealth] = [
            KeyHealth(key=k, masked_key=mask_key(k)) for k in clean_keys
        ]
        self._current_index = 0
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()

        logger.info(
            "Initialized KeyPool for %s with %d key(s).",
            self.provider_name,
            len(self._keys),
        )

    def __len__(self) -> int:
        return len(self._keys)

    def get_next_key(self) -> str:
        """Get the next healthy API key using round-robin rotation.

        Raises:
            RuntimeError: If all keys in pool are disabled or on cooldown.
        """
        with self._lock:
            now = time.time()
            total_keys = len(self._keys)
            
            # Look for available keys starting from current index
            for _ in range(total_keys):
                candidate = self._keys[self._current_index]
                self._current_index = (self._current_index + 1) % total_keys

                if candidate.is_available:
                    candidate.last_used_timestamp = now
                    return candidate.key

            # If all are cooling down, check if we can pick the one with lowest cooldown
            available_cooldowns = [
                (k.cooldown_until - now, k)
                for k in self._keys
                if k.is_active
            ]
            if available_cooldowns:
                available_cooldowns.sort(key=lambda x: x[0])
                wait_time, best_key = available_cooldowns[0]
                if wait_time <= 0:
                    best_key.cooldown_until = 0.0
                    return best_key.key
                raise RuntimeError(
                    f"All {self.provider_name} keys on rate-limit cooldown. "
                    f"Earliest recovery in {wait_time:.1f}s."
                )

            raise RuntimeError(f"All {self.provider_name} keys are permanently disabled.")

    async def aget_next_key(self) -> str:
        """Asynchronously acquire next available key."""
        # The rotation check is fast CPU work, so get_next_key is safe
        return self.get_next_key()

    def report_success(self, key: str) -> None:
        """Report successful request for a key to reset consecutive failure counters."""
        with self._lock:
            for k in self._keys:
                if k.key == key:
                    k.total_requests += 1
                    k.consecutive_failures = 0
                    break

    def report_rate_limit(self, key: str, custom_cooldown: float | None = None) -> None:
        """Report 429 Rate Limit on a key to trigger cooldown and switch next."""
        cooldown = custom_cooldown or self.default_cooldown_seconds
        with self._lock:
            for k in self._keys:
                if k.key == key:
                    k.failed_requests += 1
                    k.consecutive_failures += 1
                    k.cooldown_until = time.time() + cooldown
                    logger.warning(
                        "[%s] Key %s hit 429 Rate Limit. Cooling down for %.1fs.",
                        self.provider_name,
                        k.masked_key,
                        cooldown,
                    )
                    break

    def report_failure(self, key: str, is_auth_error: bool = False) -> None:
        """Report general failure or authentication error."""
        with self._lock:
            for k in self._keys:
                if k.key == key:
                    k.failed_requests += 1
                    k.consecutive_failures += 1
                    if is_auth_error or k.consecutive_failures >= self.max_consecutive_failures:
                        k.is_active = False
                        logger.error(
                            "[%s] Key %s disabled (auth error: %s, consecutive failures: %d).",
                            self.provider_name,
                            k.masked_key,
                            is_auth_error,
                            k.consecutive_failures,
                        )
                    else:
                        # Soft backoff for general error
                        k.cooldown_until = time.time() + 10.0
                    break

    def get_stats(self) -> list[dict]:
        """Return diagnostic metrics for all keys in the pool."""
        now = time.time()
        with self._lock:
            return [
                {
                    "masked_key": k.masked_key,
                    "is_active": k.is_active,
                    "is_available": k.is_available,
                    "total_requests": k.total_requests,
                    "failed_requests": k.failed_requests,
                    "consecutive_failures": k.consecutive_failures,
                    "cooldown_remaining_sec": max(0.0, round(k.cooldown_until - now, 1)),
                }
                for k in self._keys
            ]
