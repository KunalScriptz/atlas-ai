"""BGE-M3 embeddings via sentence-transformers — local, multilingual, 1024-dim."""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return cached BGE-M3 embeddings instance.

    Downloads from HuggingFace on first call (~2GB), cached to disk after.
    Subsequent calls return the in-memory singleton.
    """
    log.info("Loading embedding model: %s", settings.embedding_model)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )
