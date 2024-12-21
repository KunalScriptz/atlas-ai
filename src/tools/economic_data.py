"""Free World Bank API data tool — no API key required."""

from __future__ import annotations

from typing import Any

import httpx

from src.utils.logger import get_logger

log = get_logger(__name__)

# World Bank API base — completely free, no auth
WB_BASE = "https://api.worldbank.org/v2"


async def get_economic_indicators(country_code: str) -> dict[str, Any]:
    """Fetch key economic indicators for a country from World Bank API.

    Args:
        country_code: ISO 2-letter country code (e.g., 'AE' for UAE, 'DE' for Germany)

    Returns:
        Dict with GDP, inflation, ease of doing business, etc.
    """
    indicators = {
        "NY.GDP.MKTP.CD": "GDP (current US$)",
        "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
        "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
        "SL.UEM.TOTL.ZS": "Unemployment, total (% of labor force)",
        "BX.KLT.DINV.WD.GD.ZS": "Foreign direct investment, net inflows (% of GDP)",
        "IC.BUS.EASE.XQ": "Ease of doing business score",
    }

    results: dict[str, Any] = {"country_code": country_code}

    async with httpx.AsyncClient(timeout=15.0) as client:
        for indicator_code, label in indicators.items():
            url = f"{WB_BASE}/country/{country_code}/indicator/{indicator_code}?format=json&per_page=1"
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                if len(data) > 1 and data[1]:
                    value = data[1][0].get("value")
                    year = data[1][0].get("date")
                    results[label] = {"value": value, "year": year}
            except Exception as e:
                log.warning("World Bank API error for %s/%s: %s", country_code, indicator_code, e)
                results[label] = None

    return results


# Country code mapping for common markets
COUNTRY_CODES = {
    "UAE": "AE",
    "Germany": "DE",
    "Singapore": "SG",
    "France": "FR",
    "Netherlands": "NL",
    "Saudi Arabia": "SA",
    "Japan": "JP",
    "India": "IN",
    "China": "CN",
    "United Kingdom": "GB",
    "Ireland": "IE",
    "Estonia": "EE",
    "Finland": "FI",
}


def get_country_code(market_name: str) -> str:
    """Map a market name to its ISO 2-letter code."""
    return COUNTRY_CODES.get(market_name, market_name[:2].upper())
