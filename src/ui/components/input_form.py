"""User input form component."""

from __future__ import annotations

import streamlit as st

MARKETS_OPTIONS = [
    "UAE (Dubai)", "Germany (Berlin)", "Singapore",
    "Saudi Arabia (Riyadh)", "Netherlands (Amsterdam)", "Japan (Tokyo)",
    "India (Bangalore)", "France (Paris)", "United Kingdom (London)",
    "Ireland (Dublin)", "Estonia (Tallinn)", "China (Shanghai)",
]

BUDGET_OPTIONS = ["< €100K", "€100K–€500K", "€500K–€2M", "€2M+"]

PRIORITY_OPTIONS = [
    "speed_to_market",
    "low_setup_cost",
    "regulatory_ease",
    "talent_access",
    "long_term_stability",
    "tax_optimization",
    "ip_protection",
]

CONCERN_OPTIONS = [
    "data_residency",
    "foreign_ownership_limits",
    "ip_protection",
    "fintech_licensing",
    "hiring_engineering_talent",
    "currency_risk",
    "political_stability",
    "language_barrier",
]


def render_input_form() -> tuple[bool, dict]:
    """Render the analysis input form. Returns (submitted, request_data)."""
    with st.form("analysis_form"):
        st.subheader("📋 Analysis Setup")

        product = st.text_input(
            "What's your product/service? *",
            placeholder="e.g., AI-powered HR analytics platform for SMEs",
            help="Describe what your company does in 1–2 sentences.",
        )

        home_country = st.text_input(
            "Where are you based now? *",
            placeholder="e.g., Helsinki, Finland",
        )

        markets = st.multiselect(
            "Which markets are you considering? *",
            options=MARKETS_OPTIONS,
            default=["UAE (Dubai)", "Germany (Berlin)", "Singapore"],
            help="Select 2–5 target markets to research.",
        )

        industry = st.selectbox(
            "Industry",
            options=["Technology", "FinTech", "Healthcare", "Manufacturing", "Retail", "Energy", "Logistics", "Other"],
            index=0,
        )

        budget = st.selectbox(
            "Expansion budget",
            options=BUDGET_OPTIONS,
            index=1,
        )

        priorities = st.multiselect(
            "What matters most? (ranked)",
            options=PRIORITY_OPTIONS,
            default=["speed_to_market", "low_setup_cost", "talent_access", "regulatory_ease"],
            help="Drag to reorder in your preferred ranking.",
        )

        specific_concerns = st.multiselect(
            "Any specific concerns?",
            options=CONCERN_OPTIONS,
            default=[],
        )

        uploaded_files = st.file_uploader(
            "Upload documents (optional)",
            type=["pdf", "docx", "pptx", "html", "md", "txt"],
            accept_multiple_files=True,
            help="Upload trade laws, tax guides, salary surveys, or any market research documents. "
                 "These will be parsed and used by the RAG agents for deeper analysis.",
        )

        upload_market = st.selectbox(
            "Documents apply to which market?",
            options=["All Markets (global)"] + [m for m in MARKETS_OPTIONS],
            index=0,
            help="Filter uploaded docs to a specific market, or leave as global to match all.",
            disabled=not bool(uploaded_files),
        )

        submitted = st.form_submit_button(
            "🔍 Run Analysis",
            type="primary",
            use_container_width=True,
        )

    if submitted and product and home_country and markets:
        # Strip region labels for API
        clean_markets = [m.split(" (")[0] if " (" in m else m for m in markets]

        # Parse market filter
        upload_market_value = ""
        if upload_market and upload_market != "All Markets (global)":
            upload_market_value = upload_market.split(" (")[0] if " (" in upload_market else upload_market

        return True, {
            "product": product,
            "home_country": home_country,
            "markets": clean_markets,
            "industry": industry,
            "budget": budget,
            "priorities": priorities,
            "specific_concerns": specific_concerns,
            "uploaded_files": uploaded_files or [],
            "upload_market": upload_market_value,
        }

    if submitted:
        st.error("Please fill in all required fields (*).")

    return False, {}
