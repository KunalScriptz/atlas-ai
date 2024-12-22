"""Synthesis & Strategy Agent — cross-market comparison and recommendation."""

from __future__ import annotations

import json
from typing import Any

from src.agents.base import BaseAgent
from src.schemas import SynthesisResult


class SynthesisAgent(BaseAgent):
    prompt_file = "synthesis.yaml"
    output_schema = SynthesisResult

    def build_messages(self, context: dict[str, Any]) -> list:
        """Augment context with formatted agent results summary."""
        agent_results = context.get("agent_results", {})

        # Summarize results per market for the synthesis prompt
        summary_parts = []
        for market, agents in agent_results.items():
            summary_parts.append(f"\n### {market}")
            for agent_type, result in agents.items():
                if isinstance(result, dict):
                    score_fields = [k for k in result if k.endswith("_score") or k == "ease_of_entry_score"]
                    scores = ", ".join(f"{k}={result.get(k)}" for k in score_fields if result.get(k) is not None)
                    summary_parts.append(f"  {agent_type}: {scores}")

        context["agent_results_summary"] = "\n".join(summary_parts)
        context["markets"] = ", ".join(context.get("markets", []))
        context["priorities"] = ", ".join(context.get("priorities", []))
        context["concerns"] = context.get("concerns", "none")

        return super().build_messages(context)
