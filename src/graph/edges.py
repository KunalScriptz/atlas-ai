"""Conditional routing logic for LangGraph edges."""

from __future__ import annotations

from src.graph.state import AtlasState
from src.utils.logger import get_logger

log = get_logger(__name__)


def after_research_route(state: AtlasState) -> str:
    """After each market_research node returns, check if we should proceed.

    Called once per Send() return. If no results accumulated, route to report (fail fast).
    Otherwise route to aggregate_results which will batch-check completion.
    """
    if state.get("errors") and not state.get("agent_results"):
        log.warning("All agents failed — aborting to report")
        return "report"
    return "aggregate_results"


def after_synthesis_route(state: AtlasState) -> str:
    """After synthesis, route to critique if confidence is borderline."""
    synthesis = state.get("synthesis", {})
    confidence = synthesis.get("confidence", 0.0)

    if confidence < 0.5:
        log.warning("Very low confidence (%.2f) — routing to critique", confidence)
    return "critique"


def after_critique_route(state: AtlasState) -> str:
    """After critique, check if re-research is needed."""
    critique = state.get("critique", {})
    if critique.get("refuted"):
        log.warning("Recommendation refuted — consider re-research")
        # Phase 3: could loop back to research with refined context
        # For now, proceed to report with critique noted
    return "report"
