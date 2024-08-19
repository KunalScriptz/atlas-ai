"""LangGraph nodes — each node is a step in the agent swarm pipeline.

All 9 agents active. 6 research agents fan out per market via Send().
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from src.agents.competitive import CompetitiveAgent
from src.agents.corporate import CorporateAgent
from src.agents.cultural import CulturalAgent
from src.agents.devils_advocate import DevilsAdvocateAgent
from src.agents.economic import EconomicAgent
from src.agents.regulatory import RegulatoryAgent
from src.agents.synthesis import SynthesisAgent
from src.agents.talent import TalentAgent
from src.graph.state import AtlasState
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── All 6 research agent types deployed per market ──
RESEARCH_AGENT_TYPES = [
    "regulatory",
    "corporate",
    "cultural",
    "competitive",
    "talent",
    "economic",
]

# Agent class registry — maps type string to class
_AGENT_REGISTRY = {
    "regulatory": RegulatoryAgent,
    "corporate": CorporateAgent,
    "cultural": CulturalAgent,
    "competitive": CompetitiveAgent,
    "talent": TalentAgent,
    "economic": EconomicAgent,
}


async def parse_input_node(state: AtlasState) -> dict[str, Any]:
    """Validate + enrich user input. Sets markets and user_input."""
    ui = state.get("user_input", {})
    markets = state.get("markets", [])

    if not markets:
        return {"errors": ["No markets specified"], "status": "failed"}

    if not ui.get("product"):
        return {"errors": ["Product description required"], "status": "failed"}

    enriched_input = {
        "product": ui.get("product", ""),
        "product_description": ui.get("product", ""),
        "home_country": ui.get("home_country", ""),
        "industry": ui.get("industry", "Technology"),
        "budget": ui.get("budget", "100K-500K"),
        "concerns": ", ".join(ui.get("specific_concerns", [])),
        "priorities": ui.get("priorities", ["speed", "cost", "talent", "regulatory", "stability"]),
    }

    log.info("Parsed input: product=%s, markets=%s", enriched_input["product"], markets)
    return {
        "user_input": enriched_input,
        "markets": markets,
        "status": "researching",
        "agent_results": {},
        "errors": [],
        "synthesis": None,
        "critique": None,
        "final_report": None,
    }


async def market_research_node(state: AtlasState) -> dict[str, Any]:
    """Run a single agent for a single market. Called via Send() fan-out.

    18 of these run in parallel (6 agent types × N markets).
    Each invocation is fully independent — no shared mutable state.

    The `agent_type` and `market` fields are injected by Send() into state.
    """
    agent_type = state.get("agent_type", "")
    market = state.get("market", "")

    if not agent_type or not market:
        log.error("market_research_node missing agent_type=%s market=%s", agent_type, market)
        return {"errors": [f"Missing agent_type or market in Send()"]}

    log.info("Agent %s researching %s", agent_type, market)

    context = {
        **state["user_input"],
        "market": market,
    }

    try:
        agent_cls = _AGENT_REGISTRY.get(agent_type)
        if agent_cls is None:
            return {"errors": [f"Unknown agent type: {agent_type}"]}

        agent = agent_cls()
        result = await agent.run(context)

        result_dict = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        result_dict["market"] = market

        # Merge into agent_results (LangGraph handles state merging)
        current = dict(state.get("agent_results", {}))
        if market not in current:
            current[market] = {}
        current[market][agent_type] = result_dict

        return {"agent_results": current}

    except Exception as e:
        log.error("Agent %s failed for %s: %s", agent_type, market, e)
        return {"errors": [f"{agent_type}/{market}: {e}"]}


async def aggregate_results_node(state: AtlasState) -> dict[str, Any]:
    """Check if all agents completed. Transition to synthesis."""
    markets = state.get("markets", [])
    results = state.get("agent_results", {})

    completed = sum(
        1 for m in markets
        for a in RESEARCH_AGENT_TYPES
        if m in results and a in results[m]
    )
    expected = len(markets) * len(RESEARCH_AGENT_TYPES)

    log.info("Aggregation: %d/%d agents completed across %d markets", completed, expected, len(markets))

    if state.get("errors"):
        log.warning("Errors during research: %s", state["errors"])

    return {"status": "synthesizing"}


async def synthesis_node(state: AtlasState) -> dict[str, Any]:
    """Run the real SynthesisAgent — cross-market comparison, weighted scoring."""
    log.info("Running synthesis agent...")

    context = {
        "product": state["user_input"].get("product", ""),
        "home_country": state["user_input"].get("home_country", ""),
        "industry": state["user_input"].get("industry", "Technology"),
        "budget": state["user_input"].get("budget", "100K-500K"),
        "priorities": state["user_input"].get("priorities", []),
        "concerns": state["user_input"].get("concerns", ""),
        "markets": state.get("markets", []),
        "agent_results": state.get("agent_results", {}),
    }

    try:
        agent = SynthesisAgent()
        result = await agent.run(context)
        return {"synthesis": result.model_dump(), "status": "critiquing"}
    except Exception as e:
        log.error("Synthesis failed: %s", e)
        # Fallback: basic synthesis from raw scores
        return _fallback_synthesis(state)


async def critique_node(state: AtlasState) -> dict[str, Any]:
    """Run the real DevilsAdvocateAgent — challenges the synthesis."""
    log.info("Running devil's advocate critique...")

    context = {
        "synthesis": state.get("synthesis", {}),
        "product": state["user_input"].get("product", ""),
        "home_country": state["user_input"].get("home_country", ""),
        "industry": state["user_input"].get("industry", "Technology"),
        "budget": state["user_input"].get("budget", "100K-500K"),
        "priorities": state["user_input"].get("priorities", []),
        "concerns": state["user_input"].get("concerns", ""),
    }

    try:
        agent = DevilsAdvocateAgent()
        result = await agent.run(context)
        if result is not None:
            critique_data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            return {"critique": critique_data, "status": "done"}
        else:
            log.warning("Critique LLM returned None — using fallback")
            return _fallback_critique(state)
    except Exception as e:
        log.error("Critique failed: %s", e)
        return _fallback_critique(state)


async def report_node(state: AtlasState) -> dict[str, Any]:
    """Assemble final report from all agent results, synthesis, and critique."""
    report = {
        "request": state["user_input"],
        "markets_analyzed": state["markets"],
        "agent_results": state.get("agent_results", {}),
        "synthesis": state.get("synthesis", {}),
        "critique": state.get("critique", {}),
        "errors": state.get("errors", []),
    }

    log.info("Report generated for %d markets", len(state.get("markets", [])))
    return {"final_report": report, "status": "done"}


# ── Fallback helpers (if LLM call fails) ──

def _fallback_synthesis(state: AtlasState) -> dict[str, Any]:
    """Compute basic synthesis from raw agent scores without LLM."""
    from src.schemas import SynthesisResult

    results = state.get("agent_results", {})
    markets = state.get("markets", [])
    priorities = state["user_input"].get("priorities", [])

    market_scores = {}
    for market in markets:
        scores = []
        for atype in RESEARCH_AGENT_TYPES:
            agent_result = results.get(market, {}).get(atype, {})
            # Extract any score field
            for key in agent_result:
                if key.endswith("_score") or key == "ease_of_entry_score":
                    score_val = agent_result.get(key, 5)
                    if isinstance(score_val, (int, float)):
                        scores.append(score_val)
        market_scores[market] = sum(scores) / len(scores) if scores else 5

    ranked = sorted(market_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "synthesis": SynthesisResult(
            ranked_markets=[{"market": m, "total_score": round(s, 1)} for m, s in ranked],
            dimension_comparison={},
            recommendation=f"Based on {len(RESEARCH_AGENT_TYPES)} dimensions, {ranked[0][0]} is recommended.",
            confidence=0.6,
            phased_roadmap=[f"Phase 1: Enter {ranked[0][0]}"] if ranked else [],
            executive_summary="Computed synthesis (LLM fallback). Review before decisions.",
        ).model_dump(),
        "status": "critiquing",
    }


def _fallback_critique(state: AtlasState) -> dict[str, Any]:
    """Basic critique without LLM."""
    from src.schemas import CritiqueResult

    return {
        "critique": CritiqueResult(
            refuted=False,
            confidence_adjustment=-0.1,
            flagged_concerns=["LLM synthesis failed — using computed fallback"],
            missing_data_gaps=["Synthesis may lack qualitative insight"],
            alternative_view="Re-run with working LLM for full analysis.",
        ).model_dump(),
        "status": "done",
    }


# ── Fan-out helper ──

def continue_to_market_research(state: AtlasState) -> list[Send]:
    """Fan-out: for each market × agent_type, spawn a research node via Send().

    IMPORTANT: Send() passes ONLY its kwargs dict as the node's state — it does
    NOT merge with the parent state. Every field the target node needs MUST be
    included explicitly.
    """
    markets = state.get("markets", [])
    if state.get("errors"):
        return []

    sends = []
    for market in markets:
        for agent_type in RESEARCH_AGENT_TYPES:
            sends.append(
                Send(
                    "market_research",
                    {
                        "agent_type": agent_type,
                        "market": market,
                        "user_input": state.get("user_input", {}),
                        "agent_results": state.get("agent_results", {}),
                    },
                )
            )
    log.info("Fan-out: dispatching %d research agents (%d markets × %d types)",
             len(sends), len(markets), len(RESEARCH_AGENT_TYPES))
    return sends
