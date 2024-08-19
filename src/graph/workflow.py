"""Compiled LangGraph StateGraph — the main workflow.

Flow:
  parse_input → [fan-out → market_research × N] → aggregate_results
  → synthesis → critique → report
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.graph.edges import after_critique_route, after_research_route, after_synthesis_route
from src.graph.nodes import (
    aggregate_results_node,
    continue_to_market_research,
    critique_node,
    market_research_node,
    parse_input_node,
    report_node,
    synthesis_node,
)
from src.graph.state import AtlasState
from src.utils.logger import get_logger

log = get_logger(__name__)


def build_graph() -> StateGraph:
    """Build and return the compiled Atlas AI workflow graph."""
    workflow = StateGraph(AtlasState)

    # Nodes
    workflow.add_node("parse_input", parse_input_node)  # type: ignore[arg-type]
    workflow.add_node("market_research", market_research_node)  # type: ignore[arg-type]
    workflow.add_node("aggregate_results", aggregate_results_node)  # type: ignore[arg-type]
    workflow.add_node("synthesis", synthesis_node)  # type: ignore[arg-type]
    workflow.add_node("critique", critique_node)  # type: ignore[arg-type]
    workflow.add_node("report", report_node)  # type: ignore[arg-type]

    # Edges
    workflow.set_entry_point("parse_input")

    # Fan-out: parse_input → multiple market_research in parallel
    workflow.add_conditional_edges(
        "parse_input",
        continue_to_market_research,
        path_map=["market_research"],
    )

    # After each market_research: check completion then aggregate
    workflow.add_conditional_edges(
        "market_research",
        after_research_route,
        path_map={"aggregate_results": "aggregate_results", "report": "report"},
    )

    # Sequential pipeline
    workflow.add_edge("aggregate_results", "synthesis")
    workflow.add_conditional_edges(
        "synthesis",
        after_synthesis_route,
        path_map={"critique": "critique"},
    )
    workflow.add_conditional_edges(
        "critique",
        after_critique_route,
        path_map={"report": "report"},
    )
    workflow.add_edge("report", END)

    return workflow.compile()


# Module-level compiled app
app = build_graph()
