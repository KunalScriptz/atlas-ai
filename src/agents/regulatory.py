"""Regulatory Navigator Agent — first agent proving the pattern."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.schemas import RegulatoryResult


class RegulatoryAgent(BaseAgent):
    prompt_file = "regulatory.yaml"
    output_schema = RegulatoryResult
    domain = "trade_laws"
