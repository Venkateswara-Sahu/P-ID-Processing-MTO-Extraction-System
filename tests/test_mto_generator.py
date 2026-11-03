"""
MTO generator tests.

Covers the security and data-integrity properties of the Excel output:
formula injection, filename sanitisation, category completeness, quantity
aggregation, and number formats.
"""
import re
from pathlib import Path

import openpyxl
import pytest

from src.detection.symbol_detector import Detection
from src.graph.entity_mapper import EngineeringEntity
from src.mto.mto_generator import MTOGenerator


def _entity(entity_id, entity_type, category, description, tag_id=None,
            confidence=0.9, x=100.0, size=None, spec=None, line_number=None):
    det = Detection(
        class_id=0, class_name=description, confidence=0.9,
        bbox_xyxy=(x - 10, 90, x + 10, 110), center=(x, 100.0),
    )
    return EngineeringEntity(
        entity_id=entity_id, symbol_detection=det, entity_type=entity_type,
        component_category=category, description=description, tag_id=tag_id,
        position=(x, 100.0), confidence=confidence, size=size,
        specification=spec, line_number=line_number,
    )


def _graph_for(entities):
    import networkx as nx
    g = nx.DiGraph()
    for e in entities:
        g.add_node(e.entity_id, entity_type=e.entity_type,
                   category=e.component_category, position=e.position,
                   confidence=e.confidence, tag_id=e.tag_id)
    return g


def _read(path):
    return openpyxl.load_workbook(path)


# --------------------------------------------------------------------------- #
# C6: Excel formula injection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("payload", [
    '=HYPERLINK("http://evil.test","Click me")',
    '=1+1',
    '=cmd|\'/c calc\'!A0',
])
def test_ocr_text_cannot_become_a_live_formula(payload, project_root):
    """
    openpyxl writes any string starting with '=' as a formula. OCR text is
    attacker-controllable via a crafted P&ID, so it must be stored as text.
    """
    entities = [_entity("E0001", "valve", "valves", payload, tag_id=payload)]
    path = MTOGenerator().generate(
        entities, _graph_for(entities), project_name="INJ", sheet_ref="S1"
    )
    wb = _read(path)
    formula_cells = [
        c.coordinate
        for ws in wb.worksheets
        for row in ws.iter_rows()
        for c in row
        if c.data_type == "f"
    ]
    assert not formula_cells, (
        f"OCR text became live formula cells: {formula_cells}"
    )


def test_formula_payload_is_still_visible_as_text(payload=None):
    """The text must be preserved (not silently dropped) once neutralised."""
    payload = '=HYPERLINK("http://evil.test","Click me")'
    entities = [_entity("E0001", "valve", "valves", payload, tag_id=payload)]
    path = MTOGenerator().generate(
        entities, _graph_for(entities), project_name="INJ2", sheet_ref="S1"
    )
    wb = _read(path)
    values = [
        str(c.value)
        for ws in wb.worksheets
        for row in ws.iter_rows()
        for c in row
        if c.value is not None
    ]
    assert any("HYPERLINK" in v for v in values), (
        "the neutralised text should still be present in the sheet"
    )


# --------------------------------------------------------------------------- #
# C12: project_name must not escape the output directory
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("project_name", [
    "../../evil",
    "a/b",
    "..\\..\\evil",
    "bad:name",
    "x" * 300,
])
def test_project_name_cannot_escape_the_output_directory(project_name, project_root):
    """
    project_name came straight from the API query string and was interpolated
    into the output filename, so '../../evil' wrote outside the output dir.
    """
    out_dir = project_root / "data" / "output"
    before = set(out_dir.glob("*.xlsx"))

    gen = MTOGenerator()
    path = Path(gen.generate([], _graph_for([]), project_name=project_name,
                               sheet_ref="S1"))
    after = set(out_dir.glob("*.xlsx"))

    created = after - before
    try:
        assert path.parent.resolve() == out_dir.resolve(), (
            f"output escaped the output directory: {path}"
        )
        assert path.name == Path(path.name).name
    finally:
        for p in created:
            p.unlink(missing_ok=True)


def test_project_name_is_sanitised_to_a_readable_slug(project_root):
    """A normal project name should survive mostly intact."""
    gen = MTOGenerator()
    path = Path(gen.generate([], _graph_for([]), project_name="Plant A Expansion",
                               sheet_ref="S1"))
    try:
        assert "Plant" in path.name
        assert path.suffix == ".xlsx"
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# C15: every entity must appear as a row
# --------------------------------------------------------------------------- #

def test_every_entity_appears_as_a_row(project_root):
    """An unknown category was counted in the header but had no row."""
    entities = [
        _entity("E0001", "valve", "valves", "Gate Valve", tag_id="HV-1"),
        _entity("E0002", "weird", "weird_category", "Odd Thing", tag_id="X-1"),
    ]
    path = MTOGenerator().generate(
        entities, _graph_for(entities), project_name="CAT", sheet_ref="S1"
    )
    wb = _read(path)
    ws = wb.active
    item_rows = [
        r for r in ws.iter_rows(values_only=True)
        if r and r[0] is not None and isinstance(r[0], int)
    ]
    assert len(item_rows) == len(entities), (
        f"expected {len(entities)} item rows, found {len(item_rows)}"
    )


def test_header_total_matches_the_number_of_rows(project_root):
    entities = [
        _entity("E0001", "valve", "valves", "Gate Valve", tag_id="HV-1"),
        _entity("E0002", "weird", "weird_category", "Odd Thing", tag_id="X-1"),
    ]
    path = MTOGenerator().generate(
        entities, _graph_for(entities), project_name="CAT2", sheet_ref="S1"
    )
    ws = _read(path).active
    subtitle = next(
        str(r[0]) for r in ws.iter_rows(values_only=True)
        if r and r[0] and "Total Items" in str(r[0])
    )
    stated = int(re.search(r"Total Items:\s*(\d+)", subtitle).group(1))
    item_rows = [
        r for r in ws.iter_rows(values_only=True)
        if r and r[0] is not None and isinstance(r[0], int)
    ]
    assert stated == len(item_rows), (
        f"header says {stated} items but the sheet has {len(item_rows)} rows"
    )


# --------------------------------------------------------------------------- #
# S13: the MTO must aggregate identical items into real quantities
# --------------------------------------------------------------------------- #

def test_identical_items_are_aggregated_with_a_quantity(project_root):
    """
    An MTO is a procurement document. Three identical pumps must become one
    line with quantity 3, not three lines with quantity 1.
    """
    entities = [
        _entity("E0001", "equipment", "equipment", "Centrifugal Pump", "P-101A", x=10),
        _entity("E0002", "equipment", "equipment", "Centrifugal Pump", "P-101B", x=50),
        _entity("E0003", "equipment", "equipment", "Centrifugal Pump", "P-102A", x=90),
        _entity("E0004", "valve", "valves", "Gate Valve", "HV-1001", x=200),
    ]
    path = MTOGenerator().generate(
        entities, _graph_for(entities), project_name="AGG", sheet_ref="S1"
    )
    ws = _read(path).active

    rows = [
        r for r in ws.iter_rows(values_only=True)
        if r and r[0] is not None and isinstance(r[0], int)
    ]
    assert len(rows) == 2, f"expected 2 aggregated rows, got {len(rows)}"

    quantities = {str(r[2]): r[6] for r in rows}
    assert quantities["Centrifugal Pump"] == 3, (
        f"3 pumps should aggregate to quantity 3, got {quantities}"
    )
    assert quantities["Gate Valve"] == 1


def test_aggregated_row_lists_every_member_tag(project_root):
    """The individual tags must not be lost when items are aggregated."""
    entities = [
        _entity("E0001", "equipment", "equipment", "Centrifugal Pump", "P-101A", x=10),
        _entity("E0002", "equipment", "equipment", "Centrifugal Pump", "P-101B", x=50),
    ]
    path = MTOGenerator().generate(
        entities, _graph_for(entities), project_name="AGG2", sheet_ref="S1"
    )
    ws = _read(path).active
    all_text = " ".join(
        str(c.value) for row in ws.iter_rows() for c in row if c.value is not None
    )
    assert "P-101A" in all_text and "P-101B" in all_text, (
        "member tags were lost during aggregation"
    )


def test_total_item_count_is_preserved_after_aggregation(project_root):
    """The summary must still account for every individual item."""
    entities = [
        _entity("E0001", "equipment", "equipment", "Centrifugal Pump", "P-101A", x=10),
        _entity("E0002", "equipment", "equipment", "Centrifugal Pump", "P-101B", x=50),
        _entity("E0003", "equipment", "equipment", "Centrifugal Pump", "P-102A", x=90),
    ]
    path = MTOGenerator().generate(
        entities, _graph_for(entities), project_name="AGG3", sheet_ref="S1"
    )
    ws = _read(path).active
    total_row = next(
        r for r in ws.iter_rows(values_only=True)
        if r and r[0] == "TOTAL"
    )
    assert total_row[1] == 3, (
        f"the summary total should be 3 items, got {total_row[1]}"
    )


# --------------------------------------------------------------------------- #
# Number formats: confidence must be a number, not a string
# --------------------------------------------------------------------------- #

def test_confidence_is_a_number_not_a_text_string(project_root):
    """A text '51%' cannot be sorted, filtered or charted."""
    entities = [_entity("E0001", "valve", "valves", "Gate Valve", tag_id="HV-1")]
    path = MTOGenerator().generate(
        entities, _graph_for(entities), project_name="FMT", sheet_ref="S1"
    )
    ws = _read(path).active
    header = [c.value for c in ws[5]]
    conf_col = header.index("Confidence") + 1
    data_row = next(
        r for r in ws.iter_rows(min_row=6, values_only=True)
        if r and r[0] is not None and isinstance(r[0], int)
    )
    value = data_row[conf_col - 1]
    assert isinstance(value, (int, float)), (
        f"confidence should be numeric, got {value!r} ({type(value).__name__})"
    )


def test_quantity_column_is_numeric(project_root):
    entities = [
        _entity("E0001", "equipment", "equipment", "Centrifugal Pump", "P-1A", x=10),
        _entity("E0002", "equipment", "equipment", "Centrifugal Pump", "P-1B", x=50),
    ]
    path = MTOGenerator().generate(
        entities, _graph_for(entities), project_name="QTY", sheet_ref="S1"
    )
    ws = _read(path).active
    header = [c.value for c in ws[5]]
    qty_col = header.index("Quantity") + 1
    row = next(
        r for r in ws.iter_rows(min_row=6, values_only=True)
        if r and r[0] is not None and isinstance(r[0], int)
    )
    assert isinstance(row[qty_col - 1], int), (
        f"quantity should be an int, got {row[qty_col - 1]!r}"
    )


# --------------------------------------------------------------------------- #
# Autofilter must not swallow the summary block
# --------------------------------------------------------------------------- #

def test_autofilter_range_stops_at_the_last_data_row(project_root):
    entities = [_entity("E0001", "valve", "valves", "Gate Valve", tag_id="HV-1")]
    path = MTOGenerator().generate(
        entities, _graph_for(entities), project_name="AF", sheet_ref="S1"
    )
    ws = _read(path).active
    ref = ws.auto_filter.ref
    last_row = int(re.search(r"(\d+)$", ref).group(1))

    summary_row = next(
        r[0].row for r in ws.iter_rows()
        if r[0].value == "SUMMARY"
    )
    assert last_row < summary_row, (
        f"the autofilter range {ref} includes the summary block at row "
        f"{summary_row}"
    )
