"""Conversation history session store for multi-turn search."""

from __future__ import annotations

import threading
import time
from typing import Sequence
from pydantic import BaseModel, Field

from sophia.search.models import SearchResult


class ChatTurn(BaseModel):
    """A single turn in a conversation."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text content")
    timestamp: float = Field(default_factory=time.time)
    sources: list[SearchResult] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


class ConversationSession(BaseModel):
    """A multi-turn conversation session."""

    session_id: str
    turns: list[ChatTurn] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def add_user_message(self, content: str) -> None:
        self.turns.append(ChatTurn(role="user", content=content))
        self.updated_at = time.time()

    def add_assistant_message(
        self,
        content: str,
        sources: list[SearchResult] | None = None,
        search_queries: list[str] | None = None,
    ) -> None:
        self.turns.append(
            ChatTurn(
                role="assistant",
                content=content,
                sources=sources or [],
                search_queries=search_queries or [],
            )
        )
        self.updated_at = time.time()

    def get_history_summary(self, max_turns: int = 6) -> str:
        """Format recent turns for Query Rewriter context."""
        recent = self.turns[-max_turns:]
        if not recent:
            return ""
        
        lines = []
        for t in recent:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)


class SessionStore:
    """In-memory session manager with thread safety."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> ConversationSession:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationSession(session_id=session_id)
            return self._sessions[session_id]

    def get(self, session_id: str) -> ConversationSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
