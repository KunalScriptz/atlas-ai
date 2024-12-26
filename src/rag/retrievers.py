"""Per-domain hybrid retrievers — dense (BGE-M3) + market-filtered via Milvus 2.5."""

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
    """Retriever scoped to a single RAG domain with optional market filter.

    Uses Milvus similarity search with scalar filtering:
    - Dense semantic search via BGE-M3 (1024d COSINE)
    - Market filter: `market == "UAE" or market == "global"`
      Documents tagged with a specific market only match that market.
      Documents tagged "global" match all markets.
    """

    domain: str = Field(...)
    k: int = Field(default=K_RETRIEVAL)
    market: str = Field(default="")
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

    def _build_filter_expr(self) -> str | None:
        """Build Milvus scalar filter expression for market-aware retrieval.

        Returns:
            Filter expression string, or None if no market filter is set.
            - market="" or no filter → return all documents (None)
            - market="UAE" → 'market == "UAE" or market == "global"'
        """
        if not self.market:
            return None
        return f'market == "{self.market}" or market == "global"'

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        """Async retrieval with optional market filter."""
        import asyncio

        vs = self._get_vector_store()
        filter_expr = self._build_filter_expr()

        # Milvus similarity_search supports expr= for scalar filtering
        kwargs = {"query": query, "k": self.k}
        if filter_expr:
            kwargs["expr"] = filter_expr

        # Milvus similarity_search is sync, so run in thread
        results = await asyncio.to_thread(
            lambda: vs.similarity_search(**kwargs)
        )
        return results

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Sync retrieval with optional market filter."""
        vs = self._get_vector_store()
        filter_expr = self._build_filter_expr()

        kwargs = {"query": query, "k": self.k}
        if filter_expr:
            kwargs["expr"] = filter_expr

        return vs.similarity_search(**kwargs)


@lru_cache(maxsize=12)
def get_retriever(domain: str, market: str = "", k: int = K_RETRIEVAL) -> DomainRetriever:
    """Return a cached retriever for the given domain + market combination.

    Args:
        domain: One of trade_laws, tax_corporate, cultural, talent, economic, competitive
        market: Optional market filter (e.g. "UAE", "Germany"). Empty = no filter.
        k: Number of documents to retrieve

    Returns:
        DomainRetriever configured for Milvus similarity search with scalar filter
    """
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain: {domain}. Must be one of {DOMAINS}")

    return DomainRetriever(domain=domain, market=market, k=k)
