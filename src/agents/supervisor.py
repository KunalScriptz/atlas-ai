"""Supervisor Agent — lightweight orchestrator, routing, and error handling."""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent


class SupervisorAgent(BaseAgent):
    prompt_file = "supervisor.yaml"
    # Supervisor uses tool-calling (routing decisions), not structured output
    output_schema = None

    def build_messages(self, context: dict[str, Any]) -> list:
        context["user_input_summary"] = str(context.get("user_input", {}))
        return super().build_messages(context)
