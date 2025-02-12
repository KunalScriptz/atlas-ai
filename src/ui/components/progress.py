"""Live progress display component."""

from __future__ import annotations

import streamlit as st


def render_progress(job_status: dict) -> None:
    """Render agent-by-agent progress during analysis."""
    st.subheader("⏳ Agent Swarm in Progress")

    pct = job_status.get("progress_pct", 0)
    agents_done = job_status.get("agents_completed", 0)
    agents_total = job_status.get("agents_total", 21)

    st.progress(pct / 100)
    st.caption(f"{agents_done}/{agents_total} agents completed ({pct:.0f}%)")

    if job_status.get("errors"):
        with st.expander("⚠️ Warnings"):
            for err in job_status["errors"]:
                st.warning(err)
