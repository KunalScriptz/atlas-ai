"""DuckDuckGo web search tool — free, zero API key.

Registered as a LangChain tool for agent use.
Uses duckduckgo-search library with async support.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

log = get_logger(__name__)


class SearchInput(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(default=5, ge=1, le=10, description="Number of results")


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


def _cache_key(query: str) -> str:
    return f"search:{hashlib.sha256(query.encode()).hexdigest()[:16]}"


class DuckDuckGoSearchTool:
    """Async DuckDuckGo search tool with optional Redis caching."""

    def __init__(self, redis_client: Any = None):
        self._redis = redis_client

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        # Check cache first
        if self._redis:
            cached = await self._redis.get(_cache_key(query))
            if cached:
                import json

                log.debug("Cache hit for search: %s", query[:50])
                return [SearchResult(**r) for r in json.loads(cached)]

        # Execute search
        from duckduckgo_search import DDGS

        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        SearchResult(
                            title=r.get("title", ""),
                            url=r.get("href", ""),
                            snippet=r.get("body", ""),
                        )
                    )
        except Exception as e:
            log.warning("DuckDuckGo search error: %s", e)

        # Cache results
        if self._redis and results:
            import json

            await self._redis.setex(
                _cache_key(query),
                3600,  # 1 hour TTL
                json.dumps([r.model_dump() for r in results]),
            )

        return results

    def as_tool(self):
        """Return as LangChain tool for agent binding."""

        search_self = self

        @tool(args_schema=SearchInput)
        async def duckduckgo_web_search(query: str, max_results: int = 5) -> str:
            """Search the web using DuckDuckGo. Use for current information, facts, news."""
            results = await search_self.search(query, max_results)
            if not results:
                return "No results found."
            return "\n\n".join(
                f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}"
                for r in results
            )

        return duckduckgo_web_search


# Default instance (no Redis cache)
_duckduckgo_search_tool = DuckDuckGoSearchTool()

# LangChain tool for direct use
duckduckgo_search_tool = _duckduckgo_search_tool.as_tool()
