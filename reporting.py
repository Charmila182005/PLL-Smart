"""On-demand engineering PDF report generation."""

from __future__ import annotations

import csv
import json
import math
import textwrap
import unicodedata
from datetime import datetime, timezone
from io import BytesIO, StringIO
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#10233f")
CYAN = colors.HexColor("#12aeca")
MUTED = colors.HexColor("#58677c")
PALE = colors.HexColor("#edf5fa")
AMBER = colors.HexColor("#8f5a00")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat(timespec="seconds")
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (ValueError, TypeError):
            pass
    return str(value)


def build_engineering_report_exports(
    payload: Mapping[str, Any],
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create JSON, PDF, and CSV downloads from one normalized report snapshot."""
    timestamp = generated_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    report_data = _json_safe(payload)
    report_data["generated_at_utc"] = timestamp
    json_bytes = json.dumps(
        report_data,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

    csv_rows: list[dict[str, Any]] = []
    csv_rows.append(
        {
            "generated_at_utc": timestamp,
            "section": "report",
            "metric": "generated_at_utc",
            "value": timestamp,
        }
    )
    for section_name, section_data in report_data.items():
        if section_name == "generated_at_utc":
            continue
        if isinstance(section_data, Mapping):
            values = section_data.items()
        elif isinstance(section_data, list):
            values = ((f"{section_name}_{index + 1}", value) for index, value in enumerate(section_data))
        else:
            values = ((section_name, section_data),)
        for metric, value in values:
            if isinstance(value, (Mapping, list)):
                rendered = json.dumps(value, ensure_ascii=False, allow_nan=False)
            elif value is None:
                rendered = ""
            elif isinstance(value, bool):
                rendered = str(value).lower()
            else:
                rendered = value
            csv_rows.append(
                {
                    "generated_at_utc": timestamp,
                    "section": section_name,
                    "metric": str(metric),
                    "value": rendered,
                }
            )
    csv_buffer = StringIO(newline="")
    csv_writer = csv.DictWriter(
        csv_buffer,
        fieldnames=["generated_at_utc", "section", "metric", "value"],
        extrasaction="raise",
    )
    csv_writer.writeheader()
    for row in csv_rows:
        for column, value in row.items():
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                row[column] = f"'{value}"
        csv_writer.writerow(row)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    pdf_payload = dict(report_data)
    for section_name in (
        "dataset_info",
        "model_performance",
        "locktime_analysis",
    ):
        if not isinstance(pdf_payload.get(section_name), Mapping):
            pdf_payload[section_name] = {}
    for section_name in ("input_measurements", "prediction_results", "actual_comparison"):
        if not isinstance(pdf_payload.get(section_name), Mapping):
            pdf_payload[section_name] = None
    for section_name in ("warnings", "recommendations"):
        if not isinstance(pdf_payload.get(section_name), (list, tuple)):
            pdf_payload[section_name] = []
    pdf_payload["generated_at_utc"] = timestamp
    pdf_bytes = build_engineering_pdf_report(**pdf_payload)
    return {
        "payload": report_data,
        "json": json_bytes,
        "pdf": pdf_bytes,
        "csv": csv_bytes,
    }


def build_engineering_pdf_report(
    *,
    dataset_info: Mapping[str, Any],
    input_measurements: Mapping[str, Any] | None,
    prediction_results: Mapping[str, Any] | None,
    actual_comparison: Mapping[str, Any] | None,
    model_performance: Mapping[str, Any],
    locktime_analysis: Mapping[str, Any],
    warnings: Sequence[str],
    recommendations: Sequence[str],
    generated_at_utc: str | None = None,
) -> bytes:
    """Build a self-contained PDF report without writing workspace files."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="PLL-Smart Engineering Report",
        author="PLL-Smart",
    )
    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PLLTitle",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=27,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "PLLSubtitle",
        parent=base["Normal"],
        fontSize=9,
        leading=13,
        textColor=MUTED,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "PLLHeading",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "PLLBody",
        parent=base["BodyText"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#233148"),
        spaceAfter=4,
    )
    warning_style = ParagraphStyle(
        "PLLWarning",
        parent=body_style,
        textColor=AMBER,
        leftIndent=8,
        borderColor=colors.HexColor("#e9c57b"),
        borderWidth=0.5,
        borderPadding=6,
        backColor=colors.HexColor("#fff8e7"),
        spaceAfter=7,
    )
    header_style = ParagraphStyle(
        "PLLTableHeader",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
    )

    def paragraph(text: Any, style: ParagraphStyle = body_style) -> Paragraph:
        ascii_text = (
            unicodedata.normalize("NFKD", str(text))
            .encode("ascii", "backslashreplace")
            .decode("ascii")
        )
        return Paragraph(escape(ascii_text), style)

    def make_table(rows: list[list[Any]]) -> Table:
        table = Table(
            [
                [
                    paragraph(cell, header_style if row_index == 0 else body_style)
                    for cell in row
                ]
                for row_index, row in enumerate(rows)
            ],
            colWidths=[48 * mm, 112 * mm],
            repeatRows=1,
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (-1, -1), PALE),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALE, colors.white]),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d5e0eb")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return table

    def section(title: str, data: Mapping[str, Any] | None) -> list[Any]:
        story: list[Any] = [paragraph(title, heading_style)]
        if not data:
            story.append(paragraph("No data available for this section.", body_style))
            return story
        rows: list[list[Any]] = [["Measure", "Value"]]
        for key, value in data.items():
            if isinstance(value, (dict, list, tuple)):
                rendered = str(value)
            if value is None:
                rendered = "Not available"
            elif isinstance(value, float):
                rendered = f"{value:.6g}"
            elif not isinstance(value, (dict, list, tuple)):
                rendered = value
            value_lines = textwrap.wrap(
                str(rendered),
                width=100,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            for line_index, line in enumerate(value_lines):
                rows.append(
                    [
                        str(key).replace("_", " ").title() if line_index == 0 else "",
                        line,
                    ]
                )
        story.append(make_table(rows))
        return story

    report_date = generated_at_utc or datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    story: list[Any] = [
        paragraph("PLL-Smart Engineering Report", title_style),
        paragraph(f"Generated {report_date} | Synthetic-data research prototype", subtitle_style),
        HRFlowable(width="100%", thickness=1.2, color=CYAN, spaceAfter=8),
        paragraph(
            "Demonstration analysis only. This report does not authorize removal or "
            "reduction of production tests.",
            warning_style,
        ),
    ]
    story.extend(
        section(
            "1. Dataset information",
            {
                "Dataset": dataset_info.get("name", "Not specified"),
                "Rows": dataset_info.get("rows", "Not specified"),
                "Columns": dataset_info.get("columns", "Not specified"),
                "Invalid rows": dataset_info.get("invalid_rows", "Not specified"),
                "Source": dataset_info.get("source", "Not specified"),
            },
        )
    )
    story.extend(section("2. Input measurements", input_measurements))
    story.extend(section("3. Prediction results", prediction_results))
    if actual_comparison and actual_comparison.get("sufficient"):
        story.extend(
            section(
                "4. Actual versus predicted comparison",
                {
                    key: actual_comparison.get(key)
                    for key in ("count", "mae_ps", "rmse_ps", "r2", "signed_error_mean_ps")
                },
            )
        )
    else:
        story.extend(
            section(
                "4. Actual versus predicted comparison",
                {
                    "Status": (
                        actual_comparison.get("reason")
                        if actual_comparison
                        else "No measured actual jitter was provided."
                    )
                },
            )
        )
    story.extend(section("5. Original model performance", model_performance))
    story.extend(section("6. Adaptive lock-time analysis", locktime_analysis))

    story.append(paragraph("7. Safety warnings", heading_style))
    if warnings:
        story.extend(paragraph(f"• {warning}", warning_style) for warning in warnings)
    else:
        story.append(paragraph("No additional warnings supplied.", body_style))

    story.append(paragraph("8. Engineering recommendations", heading_style))
    if recommendations:
        story.extend(
            paragraph(f"{index}. {recommendation}", body_style)
            for index, recommendation in enumerate(recommendations, start=1)
        )
    else:
        story.append(paragraph("No recommendations supplied.", body_style))

    story.extend(
        [
            Spacer(1, 9),
            HRFlowable(width="100%", thickness=0.7, color=CYAN, spaceAfter=7),
            KeepTogether(
                [
                    paragraph(
                        "Synthetic-data / non-production disclaimer",
                        heading_style,
                    ),
                    paragraph(
                        "The source data and thresholds are synthetic or illustrative. "
                        "The current illustrative jitter rule admitted 52 measured failing "
                        "dies among 96,121 held-out candidates. Keep the full jitter test "
                        "enabled. No production qualification or DPPM guarantee is claimed.",
                        warning_style,
                    ),
                ]
            ),
        ]
    )

    def draw_page(canvas: Any, _document: Any) -> None:
        canvas.saveState()
        width, _ = A4
        canvas.setStrokeColor(colors.HexColor("#d5e0eb"))
        canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 9 * mm, "PLL-Smart | Engineering prototype | Synthetic data")
        canvas.drawRightString(width - 18 * mm, 9 * mm, f"Page {_document.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return buffer.getvalue()
