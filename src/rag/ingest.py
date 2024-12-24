"""RAG ingestion pipeline — Docling parsing → chunk → embed → Milvus upsert."""

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from src.rag.embeddings import get_embeddings
from src.rag.loaders.docling_loader import load_documents
from src.rag.vector_store import DOMAINS, ensure_collections, get_client, get_collection_name
from src.utils.logger import get_logger

log = get_logger(__name__)

# Chunking: 1000 tokens, 200 overlap — good balance for BGE-M3 (max 8192 tokens)
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)


async def ingest_single_file(file_path: str | Path, domain: str) -> int:
    """Ingest a single file into a Milvus domain collection.

    Pipeline: Docling parse → chunk → embed → Milvus insert.

    Args:
        file_path: Path to a PDF/DOCX/PPTX/HTML file
        domain: Target Milvus domain (trade_laws, tax_corporate, etc.)

    Returns:
        Number of chunks ingested
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    log.info("Ingesting single file '%s' into domain '%s'", file_path.name, domain)

    # Parse with Docling
    docs = await load_documents([str(file_path)])
    if not docs:
        log.warning("Docling parsed zero documents from %s", file_path.name)
        return 0

    # Add domain + source metadata
    for doc in docs:
        doc.metadata["domain"] = domain
        doc.metadata["source"] = file_path.name
        doc.metadata["chunk_hash"] = hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]

    # Chunk
    chunks = _text_splitter.split_documents(docs)
    log.info("Split '%s': %d docs → %d chunks", file_path.name, len(docs), len(chunks))

    if not chunks:
        return 0

    # Embed + insert in batches
    embeddings = get_embeddings()
    client = get_client()
    collection_name = get_collection_name(domain)

    ensure_collections(client)

    batch_size = 32
    total = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        metadatas = [c.metadata for c in batch]

        vectors = await asyncio.to_thread(embeddings.embed_documents, texts)

        data = [
            {"vector": v, "text": t, **m}
            for v, t, m in zip(vectors, texts, metadatas)
        ]
        await asyncio.to_thread(client.insert, collection_name=collection_name, data=data)

        total += len(batch)

    log.info("Ingested %d chunks from '%s' into %s", total, file_path.name, collection_name)
    return total


async def ingest_domain(domain: str, source_dir: str | Path) -> int:
    """Ingest all documents in a domain directory into Milvus.

    Pipeline: Docling parse → chunk → embed → Milvus insert.

    Args:
        domain: One of trade_laws, tax_corporate, cultural, talent, economic, competitive
        source_dir: Directory containing PDF/DOCX/PPTX/HTML files

    Returns:
        Number of chunks ingested
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        log.warning("Source directory not found: %s — skipping", source_dir)
        return 0

    files = list(source_dir.glob("*"))
    files = [f for f in files if f.suffix.lower() in {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".txt"}]
    if not files:
        log.warning("No documents found in %s", source_dir)
        return 0

    log.info("Ingesting %d files into domain '%s'", len(files), domain)

    # Parse all docs with Docling
    docs = await load_documents([str(f) for f in files])
    if not docs:
        return 0

    # Add domain metadata
    for doc in docs:
        doc.metadata["domain"] = domain
        doc.metadata["chunk_hash"] = hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]

    # Chunk
    chunks = _text_splitter.split_documents(docs)
    log.info("Split %d docs → %d chunks", len(docs), len(chunks))

    if not chunks:
        return 0

    # Embed + insert in batches
    embeddings = get_embeddings()
    client = get_client()
    collection_name = get_collection_name(domain)

    ensure_collections(client)

    batch_size = 32
    total_ingested = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task(
            f"[cyan]Embedding + inserting {domain}...", total=len(chunks)
        )

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.page_content for c in batch]
            metadatas = [c.metadata for c in batch]

            # Embed (CPU-bound)
            vectors = await asyncio.to_thread(embeddings.embed_documents, texts)

            # Insert into Milvus
            data = [
                {"vector": v, "text": t, **m}
                for v, t, m in zip(vectors, texts, metadatas)
            ]
            await asyncio.to_thread(client.insert, collection_name=collection_name, data=data)

            total_ingested += len(batch)
            progress.advance(task, len(batch))

    log.info("Ingested %d chunks into %s", total_ingested, collection_name)
    return total_ingested


async def ingest_all(data_dir: str | Path) -> dict[str, int]:
    """Ingest all domains from the data directory.

    Expects: data_dir/{trade_laws,tax_corporate,cultural,talent,economic,competitive}/

    Returns:
        {domain: chunk_count}
    """
    data_dir = Path(data_dir)
    results = {}

    tasks = [
        ingest_domain(domain, data_dir / domain)
        for domain in DOMAINS
    ]
    counts = await asyncio.gather(*tasks, return_exceptions=True)

    for domain, count in zip(DOMAINS, counts):
        if isinstance(count, Exception):
            log.error("Ingest failed for %s: %s", domain, count)
            results[domain] = 0
        else:
            results[domain] = count

    total = sum(results.values())
    log.info("Total ingested across all domains: %d chunks", total)
    return results


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    asyncio.run(ingest_all(data_dir))
