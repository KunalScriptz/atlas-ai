"""Final report rendering component."""

from __future__ import annotations

import streamlit as st
import pandas as pd


def render_report(result: dict) -> None:
    """Render the analysis results as an interactive report."""
    st.subheader("📊 Market Entry Analysis Report")

    report = result.get("result", result)
    synthesis = report.get("synthesis") or {}
    critique = report.get("critique") or {}
    agent_results = report.get("agent_results", {})
    markets = report.get("markets_analyzed", [])

    # ── Executive Summary ──
    if synthesis and synthesis.get("executive_summary"):
        st.markdown("### 🎯 Executive Summary")
        st.info(synthesis["executive_summary"])

    # ── Recommendation ──
    if synthesis.get("recommendation"):
        st.markdown("### 🏆 Recommendation")
        st.success(synthesis["recommendation"])

    # ── Confidence ──
    confidence = synthesis.get("confidence", 0)
    adj = critique.get("confidence_adjustment", 0)
    final_conf = confidence + adj
    st.metric("Confidence", f"{final_conf:.0%}", delta=f"{adj:+.0%}" if adj else None)

    # ── Market Comparison Table ──
    if synthesis.get("ranked_markets"):
        st.markdown("### 📈 Market Ranking")
        df = pd.DataFrame(synthesis["ranked_markets"])
        st.dataframe(df, width="stretch", hide_index=True)

    # ── Per-Market Agent Results ──
    if agent_results:
        st.markdown("### 🔍 Per-Market Research")
        for market in markets:
            market_data = agent_results.get(market, {})
            if not market_data:
                continue

            with st.expander(f"**{market}** — {len(market_data)} dimensions analyzed"):
                for agent_type, result in market_data.items():
                    st.markdown(f"**{agent_type.replace('_', ' ').title()}**")

                    # Extract score fields
                    score_fields = {k: v for k, v in result.items()
                                    if k.endswith("_score") or k == "ease_of_entry_score"}
                    if score_fields:
                        cols = st.columns(len(score_fields))
                        for col, (label, val) in zip(cols, score_fields.items()):
                            col.metric(label.replace("_score", "").replace("_", " ").title(), f"{val}/10")

                    # Summary
                    summary = result.get("summary", "")
                    if summary:
                        st.caption(summary[:300])

                    st.divider()

    # ── Critique ──
    if critique.get("flagged_concerns"):
        st.markdown("### ⚠️ Devil's Advocate")
        for concern in critique["flagged_concerns"]:
            st.warning(concern)
        if critique.get("alternative_view"):
            st.info(f"**Alternative view:** {critique['alternative_view']}")

    # ── Roadmap ──
    if synthesis.get("phased_roadmap"):
        st.markdown("### 🗺️ Phased Roadmap")
        for i, phase in enumerate(synthesis["phased_roadmap"], 1):
            st.markdown(f"{i}. {phase}")

    # ── Errors ──
    if report.get("errors"):
        with st.expander("🐞 Debug Info"):
            st.json(report["errors"])
