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

# Domain mapping for auto-categorizing uploaded files
_DOMAIN_FILE_KEYWORDS = {
    "trade_laws": ["law", "regulation", "compliance", "license", "gazette", "data protection", "gdpr"],
    "tax_corporate": ["tax", "treaty", "dtaa", "free zone", "entity", "incorporation", "corporate"],
    "cultural": ["culture", "etiquette", "negotiation", "hofstede", "business practice", "localization"],
    "talent": ["salary", "visa", "labor", "talent", "workforce", "hiring", "employment"],
    "economic": ["economic", "gdp", "inflation", "currency", "sovereign", "imf", "world bank"],
    "competitive": ["competitor", "market share", "pricing", "industry report", "g2", "crunchbase"],
}


def _guess_domain(filename: str) -> str:
    """Guess the RAG domain from filename keywords. Falls back to 'trade_laws'."""
    lower = filename.lower()
    for domain, keywords in _DOMAIN_FILE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return domain
    return "trade_laws"


async def run_analysis(request: dict) -> None:
    """Upload any attached files, then submit analysis to API and poll for results."""
    import httpx

    uploaded_files = request.pop("uploaded_files", [])
    upload_market = request.pop("upload_market", "")

    async with httpx.AsyncClient(timeout=600) as client:
        # Step 1: Upload files (if any)
        if uploaded_files:
            upload_count = 0
            market_label = upload_market or "global"
            with st.spinner(f"Uploading and indexing {len(uploaded_files)} document(s) [{market_label}]..."):
                for uf in uploaded_files:
                    domain = _guess_domain(uf.name)
                    try:
                        files_data = {"file": (uf.name, uf.getvalue(), uf.type)}
                        upload_resp = await client.post(
                            f"{API_URL}/upload",
                            data={"domain": domain, "market": upload_market},
                            files=files_data,
                        )
                        if upload_resp.status_code == 200:
                            result = upload_resp.json()
                            upload_count += 1
                            st.toast(
                                f"✅ {result['filename']} → {result['domain']} "
                                f"({result['chunks_ingested']} chunks)"
                            )
                    except Exception as e:
                        st.toast(f"⚠️ Upload failed for {uf.name}: {e}")

            if upload_count:
                st.success(f"Indexed {upload_count} document(s) for RAG retrieval")

        # Step 2: Submit analysis
        resp = await client.post(
            f"{API_URL}/analyze",
            json=request,
        )
        resp.raise_for_status()
        job = resp.json()
        st.session_state.job_id = job["job_id"]
        st.session_state.analysis_running = True

        # Step 3: Poll
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

        # Step 4: Fetch report
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
