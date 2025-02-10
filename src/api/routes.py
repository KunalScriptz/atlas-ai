"""FastAPI routes — analyze, status, report, health."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.graph.workflow import app
from src.schemas import AnalyzeRequest, AnalyzeResponse, JobStatus
from src.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter()

# In-memory job store (replace with DB for production)
_jobs: dict[str, dict[str, Any]] = {}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Submit a market entry analysis. Returns job_id immediately.

    The agent swarm runs asynchronously. Poll GET /status/{job_id} for progress.
    """
    job_id = uuid.uuid4().hex[:12]

    job = {
        "job_id": job_id,
        "status": "queued",
        "request": request.model_dump(),
        "result": None,
        "errors": [],
        "progress_pct": 0.0,
        "current_agent": None,
        "agents_completed": 0,
        "agents_total": 21,
    }
    _jobs[job_id] = job

    # Launch async — don't await here, let it run in background
    import asyncio
    asyncio.create_task(_run_analysis(job_id))

    log.info("Job %s queued: %s → %s", job_id, request.home_country, request.markets)
    return AnalyzeResponse(job_id=job_id, status="queued")


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str) -> JobStatus:
    """Get current job progress."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress_pct=job.get("progress_pct", 0.0),
        current_agent=job.get("current_agent"),
        agents_completed=job.get("agents_completed", 0),
        agents_total=job.get("agents_total", 21),
        elapsed_seconds=job.get("elapsed_seconds", 0.0),
        errors=job.get("errors", []),
    )


@router.get("/report/{job_id}")
async def get_report(job_id: str) -> dict[str, Any]:
    """Get the final analysis report (JSON)."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job not complete. Status: {job['status']}")

    return {
        "job_id": job_id,
        "request": job["request"],
        "result": job.get("result", {}),
    }


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check — verifies Milvus and Redis connectivity."""
    status = {"api": "ok"}

    # Check Milvus
    try:
        from src.rag.vector_store import get_client
        client = get_client()
        collections = client.list_collections()
        status["milvus"] = f"ok ({len(collections)} collections)"
    except Exception as e:
        status["milvus"] = f"error: {e}"

    # Check Redis
    try:
        from src.utils.cache import get_cache
        cache = get_cache()
        r = await cache._get_client()
        await r.ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"error: {e}"

    return status


async def _run_analysis(job_id: str) -> None:
    """Execute the full agent swarm workflow in background."""
    import time
    job = _jobs[job_id]
    t_start = time.monotonic()

    try:
        job["status"] = "running"

        state = {
            "user_input": job["request"],
            "markets": job["request"].get("markets", []),
        }

        result = await app.ainvoke(state)

        # Track progress from result
        agent_results = result.get("agent_results", {})
        total_completed = sum(
            len(agents) for agents in agent_results.values()
        )
        job["agents_completed"] = total_completed
        job["progress_pct"] = (
            total_completed / job["agents_total"] * 60  # 60% for research
            + 20  # +20% for synthesis
            + 20  # +20% for critique + report
            if result.get("status") == "done" else total_completed / job["agents_total"] * 60
        )

        job["result"] = result.get("final_report", result)
        job["status"] = "done" if result.get("status") == "done" else "failed"
        job["errors"] = result.get("errors", [])

    except Exception as e:
        log.error("Job %s failed: %s", job_id, e)
        job["status"] = "failed"
        job["errors"].append(str(e))

    finally:
        job["elapsed_seconds"] = time.monotonic() - t_start
        job["progress_pct"] = 100.0 if job["status"] == "done" else job.get("progress_pct", 0)
        log.info("Job %s finished: %s (%.1fs)", job_id, job["status"], job["elapsed_seconds"])
