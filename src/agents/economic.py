"""Economic & Political Risk Agent."""

from src.agents.base import BaseAgent
from src.schemas import EconomicResult


class EconomicAgent(BaseAgent):
    prompt_file = "economic.yaml"
    output_schema = EconomicResult
    domain = "economic"
