"""
MTO Excel Templates for Material Take-Off output.

Defines the column structure, formatting, and styles
for professional MTO spreadsheets.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MTOColumn:
    """Definition of a column in the MTO spreadsheet."""

    name: str
    width: int  # Excel column width
    header_key: str  # Key in entity data dict


# Standard MTO columns
MTO_COLUMNS = [
    MTOColumn("Item No.", 10, "item_no"),
    MTOColumn("Tag Number", 18, "tag_id"),
    MTOColumn("Description", 35, "description"),
    MTOColumn("Type", 20, "entity_type"),
    MTOColumn("Size", 12, "size"),
    MTOColumn("Rating/Spec", 15, "specification"),
    MTOColumn("Quantity", 10, "quantity"),
    MTOColumn("Line Number", 22, "line_number"),
    MTOColumn("Sheet Ref.", 12, "sheet_ref"),
    MTOColumn("Confidence", 12, "confidence"),
    MTOColumn("Notes", 25, "notes"),
]

# Category headers for grouped MTO
MTO_CATEGORIES = [
    {
        "name": "EQUIPMENT",
        "filter_type": "equipment",
        "color": "FF4444",
    },
    {
        "name": "VALVES",
        "filter_type": "valves",
        "color": "44AA44",
    },
    {
        "name": "INSTRUMENTS",
        "filter_type": "instruments",
        "color": "FF9900",
    },
    {
        "name": "PIPING SPECIALTIES",
        "filter_type": "piping",
        "color": "4488FF",
    },
    {
        "name": "OTHER COMPONENTS",
        "filter_type": "other",
        "color": "888888",
    },
]

# Excel style definitions
STYLES = {
    "title": {
        "font_size": 16,
        "font_bold": True,
        "font_color": "1a1a2e",
    },
    "subtitle": {
        "font_size": 11,
        "font_color": "666666",
    },
    "header": {
        "font_size": 10,
        "font_bold": True,
        "font_color": "FFFFFF",
        "bg_color": "1a1a2e",
        "border": True,
    },
    "category_header": {
        "font_size": 11,
        "font_bold": True,
        "font_color": "FFFFFF",
        "border": True,
    },
    "data": {
        "font_size": 9,
        "border": True,
    },
    "data_alt": {
        "font_size": 9,
        "border": True,
        "bg_color": "F5F5F5",
    },
    "low_confidence": {
        "font_size": 9,
        "border": True,
        "bg_color": "FFF3CD",
    },
    "summary": {
        "font_size": 10,
        "font_bold": True,
        "border": True,
        "bg_color": "E3F2FD",
    },
}
