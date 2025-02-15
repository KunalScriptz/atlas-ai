"""Streamlit UI — Atlas AI Market Entry Intelligence Platform."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to path so `src` is importable when run via `streamlit run`
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src.ui.components.input_form import render_input_form
from src.ui.components.progress import render_progress
from src.ui.components.report_view import render_report

st.set_page_config(
    page_title="Atlas AI — Market Entry Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ──
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "result" not in st.session_state:
    st.session_state.result = None


API_URL = __import__("os").environ.get("API_URL", "http://localhost:9734")

async def run_analysis(request: dict) -> None:
    """Submit analysis to API and poll for results."""
    import httpx

    async with httpx.AsyncClient(timeout=600) as client:
        # Submit
        resp = await client.post(
            f"{API_URL}/analyze",
            json=request,
        )
        resp.raise_for_status()
        job = resp.json()
        st.session_state.job_id = job["job_id"]
        st.session_state.analysis_running = True

        # Poll
        with st.spinner("Agent swarm analyzing markets..."):
            while True:
                status_resp = await client.get(
                    f"{API_URL}/status/{job['job_id']}"
                )
                job_status = status_resp.json()

                if job_status["status"] in ("done", "failed"):
                    break

                st.session_state.progress = job_status
                await __import__("asyncio").sleep(2)

        # Fetch report
        if job_status["status"] == "done":
            report_resp = await client.get(
                f"{API_URL}/report/{job['job_id']}"
            )
            st.session_state.result = report_resp.json()

    st.session_state.analysis_running = False


# ── UI Layout ──
st.title("🌍 Atlas AI")
st.caption("Cross-Border Market Entry Intelligence — Multi-Agent RAG Swarm")

col1, col2 = st.columns([1, 2])

with col1:
    submitted, request_data = render_input_form()

    if submitted:
        st.session_state.result = None
        import asyncio
        asyncio.run(run_analysis(request_data))

with col2:
    if st.session_state.analysis_running:
        render_progress(st.session_state.get("progress", {}))
    elif st.session_state.result:
        render_report(st.session_state.result)
    else:
        st.info("👈 Fill in the form and click **Run Analysis** to deploy the agent swarm across your target markets.")
        with st.expander("What happens when I click Run?"):
            st.markdown("""
            1. **Supervisor** parses your request
            2. **18 research agents** (6 types × 3 markets) run in parallel
            3. Each agent uses RAG + web search to research its dimension
            4. **Synthesis Agent** cross-compares all markets
            5. **Devil's Advocate** challenges the recommendation
            6. **Final Report** with scored comparison and roadmap

            ⏱ Typical runtime: 3–8 minutes
            """)
