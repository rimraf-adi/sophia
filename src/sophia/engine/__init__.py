"""Engine package for Sophia."""

from sophia.engine.agentic import AgenticEngine, QueryPlanner, ResearchPlan, SectionPlan, SectionSynthesizer
from sophia.engine.citation_mapper import Citation, assemble_reranked_context, map_citations
from sophia.engine.models import PerplexityResponse, PerplexityStreamEvent
from sophia.engine.perplexity import PerplexityEngine

SophiaEngine = PerplexityEngine
SophiaResponse = PerplexityResponse
SophiaStreamEvent = PerplexityStreamEvent

__all__ = [
    "SophiaEngine",
    "SophiaResponse",
    "SophiaStreamEvent",
    "PerplexityEngine",
    "PerplexityResponse",
    "PerplexityStreamEvent",
    "Citation",
    "assemble_reranked_context",
    "map_citations",
    "AgenticEngine",
    "QueryPlanner",
    "ResearchPlan",
    "SectionPlan",
    "SectionSynthesizer",
]
