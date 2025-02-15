"""Base agent class — YAML prompt loading, tool binding, structured output.

Every agent inherits from this. Zero hardcoded prompts — all from YAML.
Uses PydanticOutputParser for reliable structured output across any LLM provider.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import BaseTool

from src.llm.factory import get_llm
from src.tools.web_search import DuckDuckGoSearchTool
from src.utils.logger import get_logger

log = get_logger(__name__)


class BaseAgent:
    """Base for all agents. Handles prompt loading, LLM binding, tool registration.

    Subclass and call `self.run(context)` — structured output is automatic.
    Uses PydanticOutputParser (text parsing) instead of with_structured_output()
    for better compatibility across LLM providers including DeepSeek.
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

        # Set up output parser from schema
        self.parser: PydanticOutputParser | None = None
        if self.output_schema:
            self.parser = PydanticOutputParser(pydantic_object=self.output_schema)

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

        # Append format instructions from the output parser
        if self.parser:
            format_instructions = self.parser.get_format_instructions()
            system_msg = f"{system_msg}\n\n{format_instructions}"

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

    async def run(self, context: dict[str, Any], max_retries: int = 2) -> Any:
        """Execute the agent — uses PydanticOutputParser for structured output."""
        messages = self.build_messages(context)

        if self.parser:
            for attempt in range(max_retries + 1):
                result = await self.llm.ainvoke(messages)
                text = result.content if hasattr(result, "content") else str(result)
                try:
                    parsed = self.parser.parse(text)
                    if parsed is not None:
                        return parsed
                except Exception as e:
                    if attempt < max_retries:
                        log.warning("Agent %s parse failed (attempt %d/%d): %s",
                                    self.__class__.__name__, attempt + 1, max_retries, e)
                    else:
                        log.error("Agent %s failed after %d retries: %s",
                                  self.__class__.__name__, max_retries + 1, e)
                        raise
            return None
        else:
            llm_with_tools = self.llm.bind_tools(self.tools)
            result = await llm_with_tools.ainvoke(messages)
            return result
