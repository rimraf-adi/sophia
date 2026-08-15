"""Agentic multi-subtask generation package."""

from sophia.engine.agentic.agentic_orchestrator import AgenticEngine
from sophia.engine.agentic.planner import QueryPlanner, ResearchPlan, SectionPlan
from sophia.engine.agentic.section_agent import SectionSynthesizer

__all__ = [
    "AgenticEngine",
    "QueryPlanner",
    "ResearchPlan",
    "SectionPlan",
    "SectionSynthesizer",
]
