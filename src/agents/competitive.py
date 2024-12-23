"""Competitive Intelligence Agent."""

from src.agents.base import BaseAgent
from src.schemas import CompetitiveResult


class CompetitiveAgent(BaseAgent):
    prompt_file = "competitive.yaml"
    output_schema = CompetitiveResult
    domain = "competitive"
