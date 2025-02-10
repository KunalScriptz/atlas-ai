"""Atlas AI — FastAPI entry point."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    log.info("Atlas AI starting on port %s", settings.api_port)

    # Ensure Milvus collections exist
    try:
        from src.rag.vector_store import ensure_collections
        ensure_collections()
        log.info("Milvus collections verified")
    except Exception as e:
        log.warning("Milvus not available at startup: %s", e)

    yield

    log.info("Atlas AI shutting down")


app = FastAPI(
    title="Atlas AI",
    description="Cross-border market entry intelligence — multi-agent RAG swarm",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=settings.api_port, reload=True)
