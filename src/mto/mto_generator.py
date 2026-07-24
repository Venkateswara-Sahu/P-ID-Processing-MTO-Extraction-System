"""
MTO (Material Take-Off) Generator.

Traverses the P&ID graph and aggregates components into a structured
Excel spreadsheet, grouped by category (Equipment, Valves, Instruments,
Piping Specialties). Produces a professional, formatted .xlsx file.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from config.settings import settings
from src.graph.entity_mapper import EngineeringEntity
from src.mto.templates import MTO_CATEGORIES, MTO_COLUMNS, STYLES

logger = logging.getLogger(__name__)


class MTOGenerator:
    """Generates Material Take-Off spreadsheets from P&ID extraction data."""

    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: Directory to save MTO files.
        """
        self.output_dir = Path(output_dir or settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        entities: List[EngineeringEntity],
        graph: nx.DiGraph,
        project_name: str = "P&ID Extraction",
        sheet_ref: str = "Sheet 1",
        validation_report: Optional[dict] = None,
    ) -> str:
        """
        Generate an MTO Excel file from extracted entities.

        Args:
            entities: List of engineering entities.
            graph: P&ID relationship graph.
            project_name: Project name for the MTO header.
            sheet_ref: Drawing sheet reference.
            validation_report: Optional validation results to include.

        Returns:
            Path to the generated Excel file.
        """
        wb = Workbook()
        ws = wb.active
        ws.title = "Material Take-Off"

        # Group entities by category
        grouped = self._group_entities(entities)

        # Write header section
        current_row = self._write_header(ws, project_name, sheet_ref, len(entities))
        current_row += 1

        # Write column headers
        current_row = self._write_column_headers(ws, current_row)

        # Write data grouped by category
        item_no = 1
        for category_info in MTO_CATEGORIES:
            cat_name = category_info["name"]
            cat_filter = category_info["filter_type"]
            cat_color = category_info["color"]

            cat_entities = grouped.get(cat_filter, [])
            if not cat_entities:
                continue

            # Write category header
            current_row = self._write_category_header(
                ws, current_row, cat_name, len(cat_entities), cat_color
            )

            # Write entity rows
            for i, entity in enumerate(cat_entities):
                row_data = self._entity_to_row(entity, item_no, sheet_ref)
                self._write_data_row(
                    ws, current_row, row_data,
                    alt=(i % 2 == 1),
                    low_conf=(entity.confidence < 0.6),
                )
                item_no += 1
                current_row += 1

            current_row += 1  # Blank row between categories

        # Write summary section
        current_row = self._write_summary(ws, current_row, grouped, entities)

        # Add validation results sheet if available
        if validation_report:
            self._add_validation_sheet(wb, validation_report)

        # Set column widths
        for col_idx, col_def in enumerate(MTO_COLUMNS, 1):
            ws.column_dimensions[get_column_letter(col_idx)].width = col_def.width

        # Auto-filter
        ws.auto_filter.ref = f"A5:{get_column_letter(len(MTO_COLUMNS))}{current_row}"

        # Freeze panes (header visible while scrolling)
        ws.freeze_panes = "A6"

        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"MTO_{project_name.replace(' ', '_')}_{timestamp}.xlsx"
        output_path = self.output_dir / filename
        wb.save(str(output_path))

        logger.info(f"MTO generated: {output_path} ({item_no - 1} items)")
        return str(output_path)

    def _group_entities(
        self,
        entities: List[EngineeringEntity],
    ) -> Dict[str, List[EngineeringEntity]]:
        """Group entities by MTO category."""
        grouped: Dict[str, List[EngineeringEntity]] = {}

        for entity in entities:
            category = entity.component_category or "other"
            if category not in grouped:
                grouped[category] = []
            grouped[category].append(entity)

        # Sort within each group by tag_id
        for category in grouped:
            grouped[category].sort(
                key=lambda e: e.tag_id or "ZZZ"  # Untagged items last
            )

        return grouped

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
            cell.value = value
            cell.font = Font(size=9)
            cell.border = thin_border
            cell.fill = fill
            cell.alignment = Alignment(vertical="center")

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

        for category, items in grouped.items():
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
        """Add a separate sheet with validation results."""
        ws = wb.create_sheet("Validation Report")

        row = 1
        ws.cell(row=row, column=1).value = "P&ID Extraction Validation Report"
        ws.cell(row=row, column=1).font = Font(size=14, bold=True)
        row += 2

        # Write the report text
        report_text = validation_report.get("report", "No report available")
        for line in report_text.split("\n"):
            ws.cell(row=row, column=1).value = line
            if line.startswith("#"):
                ws.cell(row=row, column=1).font = Font(bold=True, size=11)
            row += 1

        ws.column_dimensions["A"].width = 80

    @staticmethod
    def _entity_to_row(
        entity: EngineeringEntity,
        item_no: int,
        sheet_ref: str,
    ) -> dict:
        """Convert an entity to a flat MTO row dict."""
        # Generate notes based on confidence
        notes = ""
        if entity.confidence < 0.6:
            notes = "⚠ Low confidence — needs review"
        elif not entity.tag_id:
            notes = "Tag not identified"

        return {
            "item_no": item_no,
            "tag_id": entity.tag_id or "—",
            "description": entity.description or entity.entity_type or "Unknown",
            "entity_type": (entity.entity_type or "unknown").replace("_", " ").title(),
            "size": entity.size or "—",
            "specification": entity.specification or "—",
            "quantity": 1,
            "line_number": entity.line_number or "—",
            "sheet_ref": sheet_ref,
            "confidence": f"{entity.confidence:.0%}",
            "notes": notes,
        }
