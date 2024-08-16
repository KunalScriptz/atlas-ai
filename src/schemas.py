"""ALL Pydantic models — API request/response + agent structured outputs.

Single source of truth for every data contract in the system.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ──────────────────────────────────────────────
# Base model with auto-parse for LLM string quirks
# ──────────────────────────────────────────────

class AgentResult(BaseModel):
    """Base for all agent structured outputs.

    Handles a common LLM quirk: DeepSeek (and others) sometimes return JSON-encoded
    strings for list/dict fields — e.g. '["a", "b"]' instead of ["a", "b"].
    This validator auto-parses those.
    """

    @model_validator(mode="before")
    @classmethod
    def parse_json_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        result = dict(data)
        for field_name, field_info in cls.model_fields.items():
            origin = getattr(field_info.annotation, "__origin__", None)
            # Only auto-parse list/dict fields
            if origin not in (list, dict):
                continue
            val = result.get(field_name)
            if isinstance(val, str) and val.strip().startswith(("[", "{")):
                try:
                    result[field_name] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass  # Keep original string if parsing fails
        return result


# ──────────────────────────────────────────────
# API Request / Response
# ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """User submits this to start a market entry analysis."""

    product: str = Field(..., description="Product or service description", min_length=5)
    home_country: str = Field(..., description="Company's current country of operation")
    markets: list[str] = Field(
        ..., description="Target markets to analyze (2-5 recommended)", min_length=1, max_length=5
    )
    industry: str = Field(default="Technology", description="Industry vertical")
    budget: str = Field(default="100K-500K", description="Expansion budget range")
    priorities: list[str] = Field(
        default_factory=lambda: ["speed", "cost", "talent", "regulatory", "stability"],
        description="Ranked list of what matters most",
    )
    specific_concerns: list[str] = Field(
        default_factory=list,
        description="Flagged concerns e.g. 'data_residency', 'fintech_licensing'",
    )


class AnalyzeResponse(BaseModel):
    """Immediate response after submitting analysis."""

    job_id: str
    status: str = "queued"
    estimated_seconds: int = 480


class JobStatus(BaseModel):
    """Streaming progress update."""

    job_id: str
    status: str  # queued | running | completed | failed
    progress_pct: float = 0.0
    current_agent: str | None = None
    agents_completed: int = 0
    agents_total: int = 21
    elapsed_seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Agent Structured Outputs
# ──────────────────────────────────────────────

class ScoreDimension(BaseModel):
    """A single scored dimension within an agent's output."""

    label: str
    score: int = Field(..., ge=1, le=10)
    summary: str
    source_urls: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class RegulatoryResult(AgentResult):
    """Regulatory Navigator agent output."""

    market: str
    ease_of_entry_score: int = Field(..., ge=1, le=10)
    dimensions: list[ScoreDimension]
    total_estimated_weeks: int
    critical_blockers: list[str]
    summary: str


class CorporateResult(AgentResult):
    """Corporate Structuring agent output."""

    market: str
    recommended_entity: str
    setup_cost_estimate: str
    setup_timeline_weeks: int
    corporate_tax_rate: float
    free_zone_benefits: list[str]
    dimensions: list[ScoreDimension]
    summary: str


class CulturalResult(AgentResult):
    """Cultural Intelligence agent output."""

    market: str
    cultural_distance_score: int = Field(..., ge=1, le=10)
    business_language: str
    negotiation_style: str
    localization_requirements: list[str]
    critical_pitfalls: list[str]
    summary: str


class CompetitiveResult(AgentResult):
    """Competitive Intelligence agent output."""

    market: str
    market_saturation: str  # low | medium | high
    top_competitors: list[dict[str, str]]  # [{name, website, strength}]
    market_share_estimate: str
    differentiation_opportunities: list[str]
    summary: str


class TalentResult(AgentResult):
    """Talent & Workforce agent output."""

    market: str
    talent_pool_score: int = Field(..., ge=1, le=10)
    avg_salary_band: str
    visa_complexity: str  # easy | moderate | complex
    localization_quotas: str | None
    key_roles_available: list[str]
    summary: str


class EconomicResult(AgentResult):
    """Economic & Political Risk agent output."""

    market: str
    stability_score: int = Field(..., ge=1, le=10)
    currency_risk: str  # low | medium | high
    inflation_rate: float
    sovereign_rating: str
    geopolitical_flags: list[str]
    summary: str


class SynthesisResult(AgentResult):
    """Synthesis agent — cross-market comparison and final recommendation."""

    ranked_markets: list[dict[str, Any]]  # [{market, total_score, breakdown}]
    dimension_comparison: dict[str, list[dict[str, Any]]]  # {dimension: [{market, score}]}
    recommendation: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    phased_roadmap: list[str]
    executive_summary: str


class CritiqueResult(AgentResult):
    """Devil's Advocate — challenges the synthesis recommendation."""

    refuted: bool
    confidence_adjustment: float = Field(default=0.0, ge=-0.5, le=0.5)
    flagged_concerns: list[str]
    missing_data_gaps: list[str]
    alternative_view: str


# ──────────────────────────────────────────────
# Final Report
# ──────────────────────────────────────────────

class AtlasReport(BaseModel):
    """Complete analysis output."""

    job_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    request: AnalyzeRequest
    agent_results: dict[str, dict[str, Any]]  # {market: {agent_type: result}}
    synthesis: SynthesisResult
    critique: CritiqueResult
    pdf_url: str | None = None
