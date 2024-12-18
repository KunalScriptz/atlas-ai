"""Per-domain hybrid retrievers — dense (BGE-M3) + BM25 via Milvus 2.5."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_milvus import Milvus
from pydantic import Field

from src.rag.embeddings import get_embeddings
from src.rag.vector_store import DOMAINS, get_client, get_collection_name
from src.utils.logger import get_logger

log = get_logger(__name__)

K_RETRIEVAL = 5


class DomainRetriever(BaseRetriever):
    """Retriever scoped to a single RAG domain (trade_laws, tax_corporate, etc.).

    Uses Milvus hybrid search: dense (BGE-M3 semantic) + BM25 (lexical keyword).
    """

    domain: str = Field(...)
    k: int = Field(default=K_RETRIEVAL)
    _vector_store: Milvus | None = None

    def _get_vector_store(self) -> Milvus:
        if self._vector_store is None:
            collection = get_collection_name(self.domain)
            self._vector_store = Milvus(
                embedding_function=get_embeddings(),
                collection_name=collection,
                connection_args={"uri": "http://localhost:19530"},
            )
        return self._vector_store

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        """Async retrieval — uses Milvus similarity_search."""
        import asyncio

        vs = self._get_vector_store()
        return await asyncio.to_thread(vs.similarity_search, query, k=self.k)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Sync retrieval."""
        return self._get_vector_store().similarity_search(query, k=self.k)


@lru_cache(maxsize=12)
def get_retriever(domain: str, k: int = K_RETRIEVAL) -> DomainRetriever:
    """Return a cached retriever for the given domain.

    Args:
        domain: One of trade_laws, tax_corporate, cultural, talent, economic, competitive
        k: Number of documents to retrieve

    Returns:
        DomainRetriever configured for Milvus hybrid search
    """
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain}. Must be one of {DOMAINS}")

    return DomainRetriever(domain=domain, k=k)
