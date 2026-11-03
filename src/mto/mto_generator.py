"""
MTO (Material Take-Off) Generator.

Aggregates extracted P&ID components into a structured Excel spreadsheet,
grouped by category (Equipment, Valves, Instruments, Piping Specialties).

An MTO is a *procurement* document: it answers "how many of each item do I
buy?". Items that are identical apart from their tag are therefore written
as a single line with a real quantity, with the individual tags preserved
in the Tag Number column.
"""

import logging
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from config.settings import settings
from src.graph.entity_mapper import EngineeringEntity
from src.mto.templates import MTO_CATEGORIES, MTO_COLUMNS, STYLES

logger = logging.getLogger(__name__)

# Characters Excel interprets as the start of a formula.
_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _sanitize_filename_component(value: str, max_length: int = 60) -> str:
    """
    Turn arbitrary text into a safe filename component.

    The project name arrives from an API query string and was interpolated
    straight into the output filename, so '../../evil' wrote outside the
    output directory and 'a/b' raised an uncaught error.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    slug = slug.strip("._-")
    return slug[:max_length] or "untitled"


def _sanitize_cell_value(value):
    """
    Neutralise spreadsheet formula injection.

    openpyxl writes any string starting with '=' as a formula. OCR text is
    attacker-controllable through a crafted P&ID, so it must be stored as
    literal text.
    """
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


class MTOGenerator:
    """Generates Material Take-Off spreadsheets from P&ID extraction data."""

    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: Directory to save MTO files.
        """
        self.output_dir = Path(
            output_dir if output_dir is not None else settings.output_dir
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        entities: List[EngineeringEntity],
        graph: nx.DiGraph,
        project_name: str = "P&ID Extraction",
        sheet_ref: str = "Sheet 1",
        validation_report: Optional[dict] = None,
        rule_report: Optional[dict] = None,
    ) -> str:
        """
        Generate an MTO Excel file from extracted entities.

        Args:
            entities: List of engineering entities.
            graph: P&ID relationship graph.
            project_name: Project name for the MTO header.
            sheet_ref: Drawing sheet reference.
            validation_report: LLM validation results to include.
            rule_report: Deterministic rule validation results to include.

        Returns:
            Path to the generated Excel file.
        """
        wb = Workbook()
        ws = wb.active
        ws.title = "Material Take-Off"

        # Group entities by category, then aggregate identical items.
        grouped = self._group_entities(entities)
        aggregated = self._aggregate(grouped)

        # Write header section
        current_row = self._write_header(ws, project_name, sheet_ref, len(entities))
        current_row += 1

        # Write column headers
        current_row = self._write_column_headers(ws, current_row)

        # Write data grouped by category
        item_no = 1
        last_data_row = current_row
        for category_key, category_info in self._category_order():
            rows = aggregated.get(category_key, [])
            if not rows:
                continue

            # Write category header
            current_row = self._write_category_header(
                ws, current_row, category_info["name"],
                sum(r["quantity"] for r in rows), category_info["color"],
            )

            # Write item rows
            for i, row_data in enumerate(rows):
                row_data["item_no"] = item_no
                row_data["sheet_ref"] = sheet_ref
                self._write_data_row(
                    ws, current_row, row_data,
                    alt=(i % 2 == 1),
                    low_conf=(row_data["confidence"] < 0.6),
                )
                item_no += 1
                current_row += 1
                last_data_row = current_row

            current_row += 1  # Blank row between categories

        # Write summary section
        current_row = self._write_summary(ws, current_row, grouped, entities)

        # Add validation results sheets
        if rule_report:
            self._add_rule_sheet(wb, rule_report)
        if validation_report:
            self._add_validation_sheet(wb, validation_report)

        # Set column widths
        for col_idx, col_def in enumerate(MTO_COLUMNS, 1):
            ws.column_dimensions[get_column_letter(col_idx)].width = col_def.width

        # Auto-filter -- must stop at the last DATA row, not the summary block
        ws.auto_filter.ref = (
            f"A5:{get_column_letter(len(MTO_COLUMNS))}{max(last_data_row - 1, 5)}"
        )

        # Freeze panes (header visible while scrolling)
        ws.freeze_panes = "A6"

        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _sanitize_filename_component(project_name)
        filename = f"MTO_{slug}_{timestamp}.xlsx"
        output_path = self.output_dir / filename
        wb.save(str(output_path))

        logger.info(
            f"MTO generated: {output_path} "
            f"({len(entities)} items in {item_no - 1} rows)"
        )
        return str(output_path)

    # ------------------------------------------------------------------ #
    # Grouping and aggregation                                              #
    # ------------------------------------------------------------------ #

    def _category_order(self) -> List[Tuple[str, dict]]:
        """
        Return (bucket_key, category_info) pairs in display order.

        The body and the summary must be driven by the SAME key set, or an
        entity can be counted in the header and the summary while having no
        row of its own.
        """
        return [(c["filter_type"], c) for c in MTO_CATEGORIES]

    def _group_entities(
        self,
        entities: List[EngineeringEntity],
    ) -> "OrderedDict[str, List[EngineeringEntity]]":
        """
        Group entities by MTO category.

        Categories outside the canonical list are folded into 'other' so
        that every entity lands in a section of the body. Previously an
        unrecognised category was counted in the header and the summary but
        had no row of its own.
        """
        known = {c["filter_type"] for c in MTO_CATEGORIES}
        grouped: "OrderedDict[str, List[EngineeringEntity]]" = OrderedDict()

        for entity in entities:
            category = entity.component_category or "other"
            if category not in known:
                category = "other"
            grouped.setdefault(category, []).append(entity)

        # Sort within each group by tag_id
        for category in grouped:
            grouped[category].sort(
                key=lambda e: e.tag_id or "ZZZ"  # Untagged items last
            )

        return grouped

    def _aggregate(
        self,
        grouped: "OrderedDict[str, List[EngineeringEntity]]",
    ) -> "OrderedDict[str, List[dict]]":
        """
        Collapse identical items into a single row with a real quantity.

        Two entities are the same purchasable item when they share a
        category, description, size and specification. Their tags differ
        (P-101A vs P-101B) so the tags are joined into one cell.
        """
        aggregated: "OrderedDict[str, List[dict]]" = OrderedDict()

        for category, entities in grouped.items():
            buckets: "OrderedDict[tuple, List[EngineeringEntity]]" = OrderedDict()
            for entity in entities:
                key = (
                    category,
                    entity.description or entity.entity_type or "Unknown",
                    entity.size or "",
                    entity.specification or "",
                )
                buckets.setdefault(key, []).append(entity)

            rows = []
            for (cat, description, size, spec), members in buckets.items():
                rows.append(self._members_to_row(cat, description, size, spec,
                                                 members))
            rows.sort(key=lambda r: (r["description"], r["tag_id"]))
            aggregated[category] = rows

        return aggregated

    def _members_to_row(
        self,
        category: str,
        description: str,
        size: str,
        spec: str,
        members: List[EngineeringEntity],
    ) -> dict:
        """Build one MTO row from a group of identical entities."""
        tags = [e.tag_id for e in members if e.tag_id]
        tag_cell = ", ".join(tags) if tags else "—"

        confidences = [e.confidence for e in members]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        notes = ""
        if avg_confidence < 0.6:
            notes = "⚠ Low confidence — needs review"
        elif not tags:
            notes = "Tag not identified"

        line_numbers = {e.line_number for e in members if e.line_number}
        line_number = ", ".join(sorted(line_numbers)) if line_numbers else "—"

        return {
            "item_no": 0,  # assigned at write time
            "tag_id": tag_cell,
            "description": description,
            "entity_type": self._pretty_type(category),
            "size": size or "—",
            "specification": spec or "—",
            "quantity": len(members),
            "line_number": line_number,
            "sheet_ref": "",
            "confidence": round(avg_confidence, 4),
            "notes": notes,
        }

    @staticmethod
    def _pretty_type(category: str) -> str:
        """Render a category key as a human-readable type."""
        special = {
            "valves": "Valve",
            "instruments": "Instrument",
            "piping": "Piping",
        }
        if category in special:
            return special[category]
        return category.replace("_", " ").title()

    # ------------------------------------------------------------------ #
    # Sheet writing                                                         #
    # ------------------------------------------------------------------ #

    def _write_header(self, ws, project_name: str, sheet_ref: str, total: int) -> int:
        """Write the MTO title and metadata header."""
        row = 1

        # Title
        ws.merge_cells(f"A{row}:{get_column_letter(len(MTO_COLUMNS))}{row}")
        cell = ws.cell(row=row, column=1)
        cell.value = f"MATERIAL TAKE-OFF (MTO) — {project_name}"
        cell.font = Font(size=16, bold=True, color="1a1a2e")
        cell.alignment = Alignment(horizontal="center")
        row += 1

        # Subtitle
        ws.merge_cells(f"A{row}:{get_column_letter(len(MTO_COLUMNS))}{row}")
        cell = ws.cell(row=row, column=1)
        cell.value = (
            f"Auto-extracted from P&ID using AI/ML Pipeline | "
            f"Sheet: {sheet_ref} | "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
            f"Total Items: {total}"
        )
        cell.font = Font(size=10, color="666666")
        cell.alignment = Alignment(horizontal="center")
        row += 1

        # Blank row
        row += 1

        return row

    def _write_column_headers(self, ws, row: int) -> int:
        """Write the column header row."""
        header_fill = PatternFill(start_color="1a1a2e", end_color="1a1a2e", fill_type="solid")
        header_font = Font(size=10, bold=True, color="FFFFFF")
        thin_border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        for col_idx, col_def in enumerate(MTO_COLUMNS, 1):
            cell = ws.cell(row=row, column=col_idx)
            cell.value = col_def.name
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border

        return row + 1

    def _write_category_header(
        self, ws, row: int, name: str, count: int, color: str
    ) -> int:
        """Write a category section header."""
        ws.merge_cells(f"A{row}:{get_column_letter(len(MTO_COLUMNS))}{row}")
        cell = ws.cell(row=row, column=1)
        cell.value = f"  {name} ({count} items)"
        cell.font = Font(size=11, bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
        cell.alignment = Alignment(horizontal="left", vertical="center")
        return row + 1

    def _write_data_row(
        self, ws, row: int, row_data: dict, alt: bool = False, low_conf: bool = False
    ):
        """Write a single MTO data row."""
        thin_border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        # Choose fill based on confidence
        if low_conf:
            fill = PatternFill(start_color="FFF3CD", end_color="FFF3CD", fill_type="solid")
        elif alt:
            fill = PatternFill(start_color="F5F5F5", end_color="F5F5F5", fill_type="solid")
        else:
            fill = PatternFill(fill_type=None)

        for col_idx, col_def in enumerate(MTO_COLUMNS, 1):
            cell = ws.cell(row=row, column=col_idx)
            value = row_data.get(col_def.header_key, "")
            cell.value = _sanitize_cell_value(value)
            cell.font = Font(size=9)
            cell.border = thin_border
            cell.fill = fill
            cell.alignment = Alignment(vertical="center")
            if col_def.header_key == "confidence":
                cell.number_format = "0%"
            elif col_def.header_key in ("quantity", "item_no"):
                cell.number_format = "0"
                cell.alignment = Alignment(horizontal="center", vertical="center")

    def _write_summary(
        self, ws, row: int, grouped: dict, entities: list
    ) -> int:
        """Write a summary section at the bottom."""
        row += 1  # Blank row

        summary_fill = PatternFill(start_color="E3F2FD", end_color="E3F2FD", fill_type="solid")
        bold_font = Font(size=10, bold=True)

        ws.merge_cells(f"A{row}:C{row}")
        cell = ws.cell(row=row, column=1)
        cell.value = "SUMMARY"
        cell.font = Font(size=12, bold=True, color="1a1a2e")
        row += 1

        # Use the same key set as the body so the two always agree.
        known = set()
        for key, _info in self._category_order():
            known.add(key)
            items = grouped.get(key, [])
            if not items:
                continue
            ws.cell(row=row, column=1).value = key.upper()
            ws.cell(row=row, column=1).font = bold_font
            ws.cell(row=row, column=2).value = len(items)
            ws.cell(row=row, column=2).fill = summary_fill
            row += 1

        # Any category not in the canonical list still gets counted.
        for category, items in grouped.items():
            if category in known:
                continue
            ws.cell(row=row, column=1).value = category.upper()
            ws.cell(row=row, column=1).font = bold_font
            ws.cell(row=row, column=2).value = len(items)
            ws.cell(row=row, column=2).fill = summary_fill
            row += 1

        ws.cell(row=row, column=1).value = "TOTAL"
        ws.cell(row=row, column=1).font = Font(size=10, bold=True, color="1a1a2e")
        ws.cell(row=row, column=2).value = len(entities)
        ws.cell(row=row, column=2).font = bold_font
        ws.cell(row=row, column=2).fill = summary_fill

        return row + 1

    def _add_validation_sheet(self, wb: Workbook, validation_report: dict):
        """Add a separate sheet with LLM validation results."""
        ws = wb.create_sheet("LLM Report")

        row = 1
        ws.cell(row=row, column=1).value = "P&ID Extraction — LLM Validation Report"
        ws.cell(row=row, column=1).font = Font(size=14, bold=True)
        row += 2

        if validation_report.get("degraded"):
            ws.cell(row=row, column=1).value = (
                "⚠ LLM validation did not run — see the note below."
            )
            ws.cell(row=row, column=1).font = Font(size=11, bold=True, color="B26B00")
            row += 2

        # Write the report text
        report_text = validation_report.get("report", "No report available")
        for line in report_text.split("\n"):
            ws.cell(row=row, column=1).value = _sanitize_cell_value(line)
            if line.startswith("#"):
                ws.cell(row=row, column=1).font = Font(bold=True, size=11)
            row += 1

        ws.column_dimensions["A"].width = 80

    def _add_rule_sheet(self, wb: Workbook, rule_report: dict):
        """
        Add a sheet with the deterministic rule findings.

        These are the only machine-readable validation results; without this
        sheet the workbook contained no structured validation data at all.
        """
        ws = wb.create_sheet("Rule Validation")

        header_font = Font(size=10, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1a1a2e", end_color="1a1a2e", fill_type="solid")
        thin_border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        headers = ["Rule ID", "Severity", "Entity", "Message", "Suggestion"]
        for col_idx, name in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.value = name
            cell.font = header_font
            cell.fill = header_fill
            cell.border = thin_border

        summary = rule_report.get("summary", {})
        row = 2
        ws.cell(row=row, column=1).value = "SUMMARY"
        ws.cell(row=row, column=1).font = Font(size=11, bold=True)
        row += 1
        for key, value in summary.items():
            ws.cell(row=row, column=1).value = key
            ws.cell(row=row, column=2).value = value
            row += 1

        row += 1
        for issue in rule_report.get("issues", []):
            ws.cell(row=row, column=1).value = _sanitize_cell_value(issue.get("rule_id"))
            ws.cell(row=row, column=2).value = _sanitize_cell_value(issue.get("severity"))
            ws.cell(row=row, column=3).value = _sanitize_cell_value(issue.get("entity_id"))
            ws.cell(row=row, column=4).value = _sanitize_cell_value(issue.get("message"))
            ws.cell(row=row, column=5).value = _sanitize_cell_value(issue.get("suggestion"))
            for col_idx in range(1, 6):
                ws.cell(row=row, column=col_idx).border = thin_border
            row += 1

        for col_idx, width in enumerate([12, 12, 12, 70, 60], 1):
            ws.column_dimensions[get_column_letter(col_idx)].width = width
        ws.freeze_panes = "A2"
