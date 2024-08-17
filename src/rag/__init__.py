"""RAG package — ingestion, embeddings, retrievers, vector store, loaders."""

from src.rag.embeddings import get_embeddings
from src.rag.vector_store import DOMAINS, ensure_collections, get_client, get_collection_name

__all__ = [
    "get_embeddings",
    "get_client",
    "ensure_collections",
    "get_collection_name",
    "DOMAINS",
]
