"""Free API data loaders — World Bank, no keys required."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.documents import Document

from src.tools.economic_data import COUNTRY_CODES as _COUNTRY_CODES
from src.tools.economic_data import get_country_code, get_economic_indicators
from src.utils.logger import get_logger

log = get_logger(__name__)


async def load_economic_data(market: str) -> list[Document]:
    """Fetch World Bank economic indicators for a market and return as Documents.

    Use as RAG source for the economic agent.
    """
    code = get_country_code(market)
    indicators = await get_economic_indicators(code)

    # Format as readable text
    lines = [f"Economic indicators for {market} ({code}):"]
    for label, data in indicators.items():
        if label == "country_code":
            continue
        if data and data.get("value") is not None:
            lines.append(f"  {label}: {data['value']} (year: {data.get('year', 'N/A')})")
        else:
            lines.append(f"  {label}: data unavailable")

    text = "\n".join(lines)

    return [
        Document(
            page_content=text,
            metadata={
                "source": "World Bank API",
                "domain": "economic",
                "market": market,
                "country_code": code,
                "loader": "world_bank_api",
            },
        )
    ]


async def load_all_economic_data(markets: list[str]) -> list[Document]:
    """Fetch economic indicators for multiple markets concurrently."""
    tasks = [load_economic_data(m) for m in markets]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    docs = []
    for market, result in zip(markets, results):
        if isinstance(result, Exception):
            log.warning("Economic data failed for %s: %s", market, result)
        else:
            docs.extend(result)

    return docs
