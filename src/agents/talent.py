"""Talent & Workforce Agent."""

from src.agents.base import BaseAgent
from src.schemas import TalentResult


class TalentAgent(BaseAgent):
    prompt_file = "talent.yaml"
    output_schema = TalentResult
