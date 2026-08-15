"""Sophia - AI Search & Discovery Engine."""

from sophia.cache import QueryCache, hash_query
from sophia.chunk import InMemoryReranker, RankedChunk, TextChunk, TextChunker
from sophia.engine import (
    PerplexityEngine,
    PerplexityResponse,
    PerplexityStreamEvent,
)
from sophia.llm import (
    DEFAULT_ROTATING_MODELS,
    KeyPool,
    LLMMessage,
    LLMResponse,
    LLMRouter,
    ModelPool,
    ModelTier,
    ProviderType,
)
from sophia.scrape import AsyncScraper, ScrapedDocument
from sophia.search import (
    BaseSearchProvider,
    DuckDuckGoSearcher,
    SearchResponse,
    SearchResult,
    SearchType,
    format_search_results_for_llm,
    format_sources_markdown,
)
from sophia.session import ChatTurn, ConversationSession, SessionStore

# Alias PerplexityEngine as SophiaEngine
SophiaEngine = PerplexityEngine

__all__ = [
    # Engine
    "SophiaEngine",
    "PerplexityEngine",
    "PerplexityResponse",
    "PerplexityStreamEvent",
    # Search
    "BaseSearchProvider",
    "DuckDuckGoSearcher",
    "SearchResult",
    "SearchResponse",
    "SearchType",
    "format_search_results_for_llm",
    "format_sources_markdown",
    # Scraping & Chunking
    "AsyncScraper",
    "ScrapedDocument",
    "TextChunk",
    "TextChunker",
    "InMemoryReranker",
    "RankedChunk",
    # LLM
    "LLMRouter",
    "KeyPool",
    "ModelPool",
    "DEFAULT_ROTATING_MODELS",
    "LLMMessage",
    "LLMResponse",
    "ModelTier",
    "ProviderType",
    # Cache & Session
    "QueryCache",
    "hash_query",
    "ChatTurn",
    "ConversationSession",
    "SessionStore",
]
