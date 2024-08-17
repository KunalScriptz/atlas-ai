"""Milvus client wrapper — collection management and connection."""

from __future__ import annotations

from functools import lru_cache

from pymilvus import MilvusClient, connections

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# Per-domain collections for clean isolation
DOMAINS = [
    "trade_laws",
    "tax_corporate",
    "cultural",
    "talent",
    "economic",
    "competitive",
]

VECTOR_DIM = 1024  # BGE-M3


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:
    """Return cached Milvus client connected to Docker instance."""
    log.info("Connecting to Milvus at %s", settings.milvus_uri)
    client = MilvusClient(uri=settings.milvus_uri)
    return client


def ensure_collections(client: MilvusClient | None = None) -> None:
    """Create collections for all domains if they don't exist."""
    client = client or get_client()
    existing = client.list_collections()

    for domain in DOMAINS:
        collection_name = f"{settings.milvus_collection}_{domain}"
        if collection_name not in existing:
            log.info("Creating collection: %s", collection_name)
            client.create_collection(
                collection_name=collection_name,
                dimension=VECTOR_DIM,
                metric_type="COSINE",
                auto_id=True,
            )


def get_collection_name(domain: str) -> str:
    """Return the full collection name for a domain."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain}. Must be one of {DOMAINS}")
    return f"{settings.milvus_collection}_{domain}"
