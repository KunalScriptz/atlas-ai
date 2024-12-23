"""Corporate Structuring Agent."""

from src.agents.base import BaseAgent
from src.schemas import CorporateResult


class CorporateAgent(BaseAgent):
    prompt_file = "corporate.yaml"
    output_schema = CorporateResult
    domain = "tax_corporate"
