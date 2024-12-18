"""Web loader — DuckDuckGo search → scrape top results → chunk documents.

All free, no API keys. Rate-limited to be respectful.
"""

from __future__ import annotations

import asyncio
import hashlib

from langchain_core.documents import Document

from src.tools.web_search import DuckDuckGoSearchTool
from src.utils.cache import get_cache
from src.utils.logger import get_logger

log = get_logger(__name__)


async def fetch_page(url: str) -> str | None:
    """Fetch a single page's HTML with caching."""
    import httpx
    from bs4 import BeautifulSoup

    cache = get_cache()

    # Check cache
    cached = await cache.get_web(url)
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers={
            "User-Agent": "AtlasAI/1.0 (Market Research; +https://github.com/atlas-ai)",
        }) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            # Extract text content
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            # Basic cleanup: collapse whitespace
            import re
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text[:10000]  # Cap at 10K chars per page

            # Cache
            await cache.set_web(url, text)
            return text

    except Exception as e:
        log.warning("Failed to fetch %s: %s", url, e)
        return None


async def web_search_and_load(
    query: str,
    domain: str,
    max_results: int = 3,
) -> list[Document]:
    """Search DuckDuckGo, fetch top results, return LangChain Documents.

    Args:
        query: Search query
        domain: Domain label for metadata (trade_laws, cultural, etc.)
        max_results: Number of search results to fetch and scrape

    Returns:
        List of Documents ready for chunking and embedding
    """
    search = DuckDuckGoSearchTool()
    results = await search.search(query, max_results=max_results)

    if not results:
        log.info("No search results for: %s", query[:80])
        return []

    # Fetch pages concurrently
    tasks = [fetch_page(r.url) for r in results]
    pages = await asyncio.gather(*tasks, return_exceptions=True)

    docs = []
    for result, page_text in zip(results, pages):
        if isinstance(page_text, Exception) or not page_text:
            continue

        doc = Document(
            page_content=page_text,
            metadata={
                "source": result.url,
                "title": result.title,
                "domain": domain,
                "query": query,
                "loader": "duckduckgo_web",
            },
        )
        docs.append(doc)

    log.info("Web loaded: %d docs from query '%s'", len(docs), query[:60])
    return docs
