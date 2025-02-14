"""PDF report generation via ReportLab."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from src.utils.logger import get_logger

log = get_logger(__name__)


async def generate_pdf(report: dict, output_path: str | None = None) -> bytes | str:
    """Generate a PDF report from the analysis results.

    Args:
        report: Full analysis report dict
        output_path: If provided, save to file. Otherwise return bytes.

    Returns:
        PDF bytes if no output_path, else the file path as string
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Atlas AI — Market Entry Report")
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Atlas AI — Market Entry Intelligence Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    story.append(Spacer(1, 24))

    # Request summary
    req = report.get("request", {})
    story.append(Paragraph("Analysis Request", styles["Heading2"]))
    story.append(Paragraph(f"Product: {req.get('product', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"Home Country: {req.get('home_country', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"Markets: {', '.join(report.get('markets_analyzed', []))}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Synthesis
    synthesis = report.get("synthesis", {})
    if synthesis:
        story.append(Paragraph("Recommendation", styles["Heading2"]))
        story.append(Paragraph(synthesis.get("recommendation", "N/A"), styles["Normal"]))
        story.append(Paragraph(f"Confidence: {synthesis.get('confidence', 0):.0%}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Ranked markets table
        ranked = synthesis.get("ranked_markets", [])
        if ranked:
            story.append(Paragraph("Market Ranking", styles["Heading3"]))
            table_data = [["Rank", "Market", "Total Score"]]
            for i, m in enumerate(ranked, 1):
                table_data.append([str(i), m.get("market", ""), str(m.get("total_score", ""))])
            story.append(Table(table_data))
            story.append(Spacer(1, 12))

        # Roadmap
        roadmap = synthesis.get("phased_roadmap", [])
        if roadmap:
            story.append(Paragraph("Phased Roadmap", styles["Heading2"]))
            for phase in roadmap:
                story.append(Paragraph(f"• {phase}", styles["Normal"]))
            story.append(Spacer(1, 12))

    # Executive Summary
    exec_summary = synthesis.get("executive_summary", "")
    if exec_summary:
        story.append(Paragraph("Executive Summary", styles["Heading2"]))
        story.append(Paragraph(exec_summary, styles["Normal"]))
        story.append(Spacer(1, 12))

    # Critique
    critique = report.get("critique", {})
    if critique:
        story.append(Paragraph("Devil's Advocate Review", styles["Heading2"]))
        for concern in critique.get("flagged_concerns", []):
            story.append(Paragraph(f"⚠ {concern}", styles["Normal"]))
        alt = critique.get("alternative_view", "")
        if alt:
            story.append(Paragraph(f"Alternative view: {alt}", styles["Normal"]))

    # Build
    doc.build(story)

    pdf_bytes = buf.getvalue()
    buf.close()

    if output_path:
        output_path = Path(output_path)
        output_path.write_bytes(pdf_bytes)
        log.info("PDF saved to %s", output_path)
        return str(output_path)

    return pdf_bytes
