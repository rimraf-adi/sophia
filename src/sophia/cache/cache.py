"""SQLite persistent cache for query results with TTL."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Any

logger = logging.getLogger(__name__)


def hash_query(query: str) -> str:
    """Normalize and hash query string for cache keying."""
    normalized = " ".join(query.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class QueryCache:
    """SQLite-backed persistent cache for search & synthesis results."""

    def __init__(self, db_path: str = "cache.db", default_ttl_seconds: int = 3600) -> None:
        self.db_path = db_path
        self.default_ttl_seconds = default_ttl_seconds
        self._init_db()

    def _init_db(self) -> None:
        """Create cache table if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    query_raw TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON query_cache (expires_at)")
            conn.commit()

    def get(self, query: str) -> dict[str, Any] | None:
        """Retrieve cached result if not expired."""
        q_hash = hash_query(query)
        now = time.time()

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT result_json, expires_at FROM query_cache WHERE query_hash = ?",
                    (q_hash,),
                )
                row = cursor.fetchone()
                if row:
                    result_json, expires_at = row
                    if expires_at >= now:
                        logger.info("Cache HIT for query: '%s'", query)
                        return json.loads(result_json)
                    else:
                        # Clean up expired
                        cursor.execute("DELETE FROM query_cache WHERE query_hash = ?", (q_hash,))
                        conn.commit()
        except Exception as e:
            logger.warning("Cache read failed: %s", str(e))

        return None

    def set(self, query: str, data: dict[str, Any], ttl_seconds: int | None = None) -> None:
        """Store query result in cache."""
        q_hash = hash_query(query)
        now = time.time()
        ttl = ttl_seconds or self.default_ttl_seconds
        expires_at = now + ttl

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO query_cache
                    (query_hash, query_raw, result_json, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (q_hash, query, json.dumps(data), now, expires_at),
                )
                conn.commit()
                logger.info("Cache SET for query: '%s' (TTL: %ds)", query, ttl)
        except Exception as e:
            logger.warning("Cache write failed: %s", str(e))

    def cleanup_expired(self) -> int:
        """Delete all expired cache entries."""
        now = time.time()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM query_cache WHERE expires_at < ?", (now,))
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.warning("Cache cleanup failed: %s", str(e))
            return 0
