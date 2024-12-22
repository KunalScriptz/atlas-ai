"""Devil's Advocate Agent — challenges the synthesis recommendation."""

from __future__ import annotations

import json
from typing import Any

from src.agents.base import BaseAgent
from src.schemas import CritiqueResult


class DevilsAdvocateAgent(BaseAgent):
    prompt_file = "devils_advocate.yaml"
    output_schema = CritiqueResult

    def build_messages(self, context: dict[str, Any]) -> list:
        """Format synthesis + user context for critique."""
        synthesis = context.get("synthesis", {})
        context["synthesis"] = json.dumps(synthesis, indent=2)[:8000]
        context["concerns"] = context.get("concerns", "none")
        return super().build_messages(context)
