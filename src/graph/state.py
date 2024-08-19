"""AtlasState — the TypedDict that flows through every LangGraph node."""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def _merge_agent_results(left: dict, right: dict) -> dict:
    """Reducer: deep-merge agent_results from concurrent Send() nodes.
    Each node writes {market: {agent_type: result}} — no key conflicts.
    """
    merged = dict(left or {})
    for market, agents in (right or {}).items():
        if market not in merged:
            merged[market] = {}
        merged[market].update(agents)
    return merged


def _merge_lists(left: list, right: list) -> list:
    """Reducer: concatenate errors lists."""
    return (left or []) + (right or [])


class AtlasState(TypedDict):
    """Shared state flowing through the agent swarm.

    Fields evolve as the graph progresses:
    - parse_input populates user_input + markets
    - market_research fans out and writes to agent_results
    - synthesis reads agent_results, writes synthesis
    - critique reads synthesis, writes critique
    - report writes final_report
    """

    # ── Input (set by parse_input) ──
    user_input: dict[str, Any]  # Parsed AnalyzeRequest fields
    markets: list[str]          # Target market names

    # ── Agent results (accumulated by market_research nodes) ──
    # {market: {agent_type: result_dict}}
    agent_results: Annotated[dict[str, dict[str, Any]], _merge_agent_results]

    # ── Synthesis + Critique ──
    synthesis: dict[str, Any] | None
    critique: dict[str, Any] | None

    # ── Output ──
    final_report: dict[str, Any] | None

    # ── Send() fan-out fields (injected per invocation) ──
    agent_type: str | None      # Which agent type to run
    market: str | None          # Which market to research

    # ── Tracking ──
    errors: Annotated[list[str], _merge_lists]
    status: str                 # parsing | researching | synthesizing | critiquing | done | failed
    messages: Annotated[list, add_messages]
