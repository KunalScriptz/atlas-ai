"""Cultural Intelligence Agent."""

from src.agents.base import BaseAgent
from src.schemas import CulturalResult


class CulturalAgent(BaseAgent):
    prompt_file = "cultural.yaml"
    output_schema = CulturalResult
    domain = "cultural"
