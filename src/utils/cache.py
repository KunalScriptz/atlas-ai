"""Redis caching layer for web scraping, embeddings, and search results."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class CacheClient:
    """Async Redis cache client.

    Three cache patterns:
    - web:{sha256(url)} — scraped HTML, 24h TTL
    - embed:{sha256(chunk)} — embedding vectors, permanent
    - search:{sha256(query)} — DuckDuckGo results, 1h TTL
    """

    def __init__(self, redis_url: str | None = None):
        self._redis_url = redis_url or settings.redis_uri
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    # ── Web cache ──

    @staticmethod
    def web_key(url: str) -> str:
        return f"web:{hashlib.sha256(url.encode()).hexdigest()[:16]}"

    async def get_web(self, url: str) -> str | None:
        r = await self._get_client()
        return await r.get(self.web_key(url))

    async def set_web(self, url: str, html: str, ttl: int = 86400) -> None:
        r = await self._get_client()
        await r.setex(self.web_key(url), ttl, html)

    # ── Embedding cache ──

    @staticmethod
    def embed_key(text: str) -> str:
        return f"embed:{hashlib.sha256(text.encode()).hexdigest()[:16]}"

    async def get_embed(self, text: str) -> list[float] | None:
        r = await self._get_client()
        raw = await r.get(self.embed_key(text))
        if raw:
            return json.loads(raw)
        return None

    async def set_embed(self, text: str, vector: list[float]) -> None:
        r = await self._get_client()
        await r.set(self.embed_key(text), json.dumps(vector))

    # ── Search cache ──

    @staticmethod
    def search_key(query: str) -> str:
        return f"search:{hashlib.sha256(query.encode()).hexdigest()[:16]}"

    async def get_search(self, query: str) -> list[dict] | None:
        r = await self._get_client()
        raw = await r.get(self.search_key(query))
        if raw:
            return json.loads(raw)
        return None

    async def set_search(self, query: str, results: list[dict], ttl: int = 3600) -> None:
        r = await self._get_client()
        await r.setex(self.search_key(query), ttl, json.dumps(results))

    # ── Lifecycle ──

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


# Module-level singleton
_cache: CacheClient | None = None


def get_cache() -> CacheClient:
    global _cache
    if _cache is None:
        _cache = CacheClient()
    return _cache
