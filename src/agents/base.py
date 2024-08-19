"""Base agent class — YAML prompt loading, tool binding, structured output.

Every agent inherits from this. Zero hardcoded prompts — all from YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.llm.factory import get_llm
from src.tools.web_search import DuckDuckGoSearchTool
from src.utils.logger import get_logger

log = get_logger(__name__)


class BaseAgent:
    """Base for all agents. Handles prompt loading, LLM binding, tool registration.

    Subclass and call `self.run(context)` — structured output is automatic.
    """

    prompt_file: str = ""  # Set in subclass, e.g. "regulatory.yaml"
    output_schema: type | None = None  # Pydantic model for structured output

    def __init__(self, redis_url: str | None = None):
        if not self.prompt_file:
            raise ValueError(f"{self.__class__.__name__} must set `prompt_file`")

        self.llm: BaseChatModel = get_llm()
        self.prompts: dict[str, str] = self._load_prompts()
        self.search_tool = DuckDuckGoSearchTool()
        self.tools: list[BaseTool] = [self.search_tool.as_tool()]

    def _load_prompts(self) -> dict[str, str]:
        prompt_path = Path(__file__).parent.parent / "prompts" / self.prompt_file
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with open(prompt_path) as f:
            return yaml.safe_load(f)

    def build_messages(self, context: dict[str, Any]) -> list:
        """Build system + human messages from YAML templates."""
        system_template = self.prompts.get("system", "")
        human_template = self.prompts.get("human", "")

        # Format with context. Fallback empty string only for keys NOT in context —
        # passing the same key twice (in context AND fallback) is a TypeError.
        missing = {k: "" for k in self._expected_keys() - set(context.keys())}
        system_msg = system_template.format(**context, **missing)
        human_msg = human_template.format(**context, **missing)

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": human_msg},
        ]

    def _expected_keys(self) -> set[str]:
        """Keys expected in context dict — overridable by subclasses."""
        return {
            "product", "product_description", "market", "home_country",
            "industry", "budget", "concerns",
        }

    async def run(self, context: dict[str, Any]) -> Any:
        """Execute the agent — returns structured output if output_schema is set."""
        messages = self.build_messages(context)

        if self.output_schema:
            structured_llm = self.llm.with_structured_output(self.output_schema)
            result = await structured_llm.ainvoke(messages)
            return result
        else:
            # Bind tools and let the agent decide when to use them
            llm_with_tools = self.llm.bind_tools(self.tools)
            result = await llm_with_tools.ainvoke(messages)
            return result
