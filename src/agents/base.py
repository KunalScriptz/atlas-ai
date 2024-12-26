"""Base agent class — YAML prompt loading, tool binding, structured output.

Every agent inherits from this. Zero hardcoded prompts — all from YAML.
Uses PydanticOutputParser for reliable structured output across any LLM provider.
Before each LLM call, runs Milvus RAG retrieval + DuckDuckGo web search in parallel
and injects both as context into the prompt.
"""

from __future__ import annotations

import asyncio
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
    """Base for all agents. Handles prompt loading, LLM binding, tool registration,
    and pre-LLM context retrieval (RAG + web search).

    Subclass and call `self.run(context)` — structured output is automatic.
    Uses PydanticOutputParser (text parsing) instead of with_structured_output()
    for better compatibility across LLM providers including DeepSeek.

    Attributes:
        prompt_file: YAML prompt filename (e.g. "regulatory.yaml")
        output_schema: Pydantic model for structured output (None = tool-calling mode)
        domain: Milvus domain name for RAG retrieval. Empty string = RAG skipped.
                Research agents set this; synthesis/critique/supervisor leave it empty.
    """

    prompt_file: str = ""  # Set in subclass, e.g. "regulatory.yaml"
    output_schema: type | None = None  # Pydantic model for structured output
    domain: str = ""  # Milvus domain — set in research agent subclasses

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

    # ── Search query construction ──

    def _build_search_query(self, context: dict[str, Any]) -> str:
        """Build a targeted search query from the agent context."""
        product = context.get("product", "")
        market = context.get("market", "")
        industry = context.get("industry", "Technology")
        home = context.get("home_country", "")

        # Domain-specific query prefixes for better search results
        prefixes = {
            "trade_laws": f"business license foreign ownership regulations {market}",
            "tax_corporate": f"corporate tax rate entity setup cost {market} 2024",
            "cultural": f"business culture etiquette negotiation {market} foreign companies",
            "competitive": f"{industry} competitors market share {market}",
            "talent": f"{industry} salary benchmark hiring visa {market}",
            "economic": f"{market} currency inflation sovereign rating political risk 2024",
        }
        prefix = prefixes.get(self.domain, f"{industry} {product} market entry {market}")
        return prefix

    # ── Context retrieval (RAG + Web Search in parallel) ──

    async def _retrieve_context(self, query: str, market: str = "") -> tuple[str, str]:
        """Run Milvus retriever + DuckDuckGo web search concurrently.

        Args:
            query: Search query string
            market: Optional market filter for RAG retrieval (e.g. "UAE", "Germany")

        Returns:
            (rag_context, web_search_context) — formatted strings ready for prompt injection.
            Returns placeholder text if retrieval fails or no results found.
        """
        rag_coro = None
        web_coro = self.search_tool.search(query, max_results=5)

        # Only use RAG if this agent has a domain (research agents only)
        if self.domain:
            try:
                from src.rag.retrievers import get_retriever

                retriever = get_retriever(self.domain, market=market)
                rag_coro = retriever.ainvoke(query)
            except Exception as e:
                log.debug("RAG retriever unavailable for %s: %s", self.domain, e)

        # Run both tasks concurrently
        coros = []
        if rag_coro:
            coros.append(rag_coro)
        coros.append(web_coro)

        results = await asyncio.gather(*coros, return_exceptions=True)

        # Unpack results
        idx = 0
        docs = []
        if rag_coro:
            raw = results[idx]
            if not isinstance(raw, Exception):
                docs = raw
            else:
                log.debug("RAG retrieval error: %s", raw)
            idx += 1

        web_results = []
        raw_web = results[idx]
        if not isinstance(raw_web, Exception):
            web_results = raw_web
        else:
            log.debug("Web search error: %s", raw_web)

        # Format RAG context
        rag_text = ""
        if docs:
            lines = ["### Internal Knowledge Base"]
            for i, d in enumerate(docs[:5], 1):
                src = d.metadata.get("source", d.metadata.get("file_path", "document"))
                content = d.page_content[:600].replace("\n", " ")
                lines.append(f"{i}. [{src}] {content}")
            rag_text = "\n".join(lines)
        else:
            rag_text = "### Internal Knowledge Base\nNo internal documents found for this query."

        # Format web search context
        web_text = ""
        if web_results:
            lines = ["### Live Web Search Results"]
            for i, r in enumerate(web_results[:5], 1):
                lines.append(f"{i}. **{r.title}**\n   URL: {r.url}\n   {r.snippet}")
            web_text = "\n".join(lines)
        else:
            web_text = "### Live Web Search Results\nNo web results found."

        return rag_text, web_text

    # ── Prompt building ──

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
            "product",
            "product_description",
            "market",
            "home_country",
            "industry",
            "budget",
            "concerns",
            "rag_context",
            "web_search_context",
        }

    # ── Main execution ──

    async def run(self, context: dict[str, Any], max_retries: int = 2) -> Any:
        """Execute the agent.

        1. Run RAG retrieval + web search in parallel (if domain is set).
        2. Inject results as {rag_context} and {web_search_context} into the prompt.
        3. Call LLM with PydanticOutputParser for structured output.
        """
        # ── Step 1: Retrieve context (RAG + Web Search) ──
        query = self._build_search_query(context)
        market = context.get("market", "")
        rag_context, web_context = await self._retrieve_context(query, market=market)

        context["rag_context"] = rag_context
        context["web_search_context"] = web_context

        log.debug(
            "Retrieved context for %s: RAG=%d chars, Web=%d chars",
            self.__class__.__name__,
            len(rag_context),
            len(web_context),
        )

        # ── Step 2: Build messages with retrieved context injected ──
        messages = self.build_messages(context)

        # ── Step 3: LLM call with structured output parsing ──
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
                        log.warning(
                            "Agent %s parse failed (attempt %d/%d): %s",
                            self.__class__.__name__,
                            attempt + 1,
                            max_retries,
                            e,
                        )
                    else:
                        log.error(
                            "Agent %s failed after %d retries: %s",
                            self.__class__.__name__,
                            max_retries + 1,
                            e,
                        )
                        raise
            return None
        else:
            # Tool-calling mode (used by supervisor, agents without output_schema)
            llm_with_tools = self.llm.bind_tools(self.tools)
            result = await llm_with_tools.ainvoke(messages)
            return result
