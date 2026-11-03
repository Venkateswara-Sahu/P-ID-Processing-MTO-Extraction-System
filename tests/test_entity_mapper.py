"""
Entity mapper tests.

Covers spatial text->symbol association, class->category classification,
and the ordering-sensitivity of the assignment.
"""
import pytest

from src.detection.symbol_detector import Detection
from src.graph.class_map import (
    EQUIPMENT, INSTRUMENTS, OTHER, PIPING, VALVES,
    classify_class_name, coverage_report, entity_type_for,
)
from src.graph.entity_mapper import EntityMapper
from src.ocr.text_extractor import TextRegion


def _det(class_name="Gate Valve", conf=0.9, x=100.0, y=100.0, class_id=0):
    return Detection(
        class_id=class_id, class_name=class_name, confidence=conf,
        bbox_xyxy=(x - 10, y - 10, x + 10, y + 10), center=(x, y),
    )


def _txt(text="FIC-101", conf=0.9, x=100.0, y=100.0):
    return TextRegion(
        text=text, confidence=conf,
        bbox=(x - 20, y - 6, x + 20, y + 6), center=(x, y),
        polygon=[],
    )


# --------------------------------------------------------------------------- #
# Class -> category mapping
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("class_name,expected", [
    ("Gate Valve", VALVES),
    ("Control_valve", VALVES),
    ("3 Way Gate Valve", VALVES),
    ("PSV", VALVES),
    ("PRV", VALVES),
    ("Relife PRV", VALVES),
    ("Safety PSV", VALVES),
    ("Solenoid", VALVES),
    ("Centrifugal Pump", EQUIPMENT),
    ("Turbine Pump", EQUIPMENT),
    ("Heat Exchanger", EQUIPMENT),
    ("Vessel", EQUIPMENT),
    ("Agitator or Mixer", EQUIPMENT),
    ("Electric Motor", EQUIPMENT),
    ("Flow Transmitter", INSTRUMENTS),
    ("Temp Indicator", INSTRUMENTS),
    ("Pressure Gauge", INSTRUMENTS),
    ("Flowmeter", INSTRUMENTS),
    ("Flange", PIPING),
    ("Socket Weld", PIPING),
])
def test_real_model_class_names_are_categorised(class_name, expected):
    """These are actual class names from models/best.pt."""
    assert classify_class_name(class_name) == expected


def test_safety_valves_are_not_lost():
    """PSV and PRV are the highest-value items in a P&ID MTO."""
    for name in ("PSV", "PRV", "Safety PSV", "Relife PRV"):
        assert classify_class_name(name) == VALVES, f"{name} was misfiled"


def test_turbine_is_not_matched_by_a_substring_of_bin():
    """
    A 'bin' keyword matched inside 'Turbine' and filed every turbine as
    equipment via the wrong rule.
    """
    assert classify_class_name("Turbine Pump") == EQUIPMENT


def test_entity_type_labels():
    assert entity_type_for(VALVES) == "valve"
    assert entity_type_for(INSTRUMENTS) == "instrument"
    assert entity_type_for(EQUIPMENT) == "equipment"
    assert entity_type_for(PIPING) == "piping_component"
    assert entity_type_for(OTHER) == "unknown"


def test_coverage_of_the_real_model_vocabulary():
    """Coverage against the trained model's 229 classes must be high."""
    from ultralytics import YOLO
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    model = root / "models" / "best.pt"
    if not model.exists():
        pytest.skip("trained weights not available")

    names = YOLO(str(model)).names
    classes = [names[i] for i in range(len(names))]
    report = coverage_report(classes)

    assert report["coverage_pct"] >= 75.0, (
        f"only {report['coverage_pct']}% of the model's classes are categorised; "
        f"unmapped: {report['unmapped'][:20]}"
    )


# --------------------------------------------------------------------------- #
# Classification through the mapper
# --------------------------------------------------------------------------- #

def test_mapper_uses_the_class_map():
    mapper = EntityMapper()
    entities = mapper.map_entities([_det(class_name="Centrifugal Pump")], [])
    assert entities[0].component_category == EQUIPMENT
    assert entities[0].entity_type == "equipment"


def test_mapper_classifies_a_safety_valve():
    mapper = EntityMapper()
    entities = mapper.map_entities([_det(class_name="Safety PSV")], [])
    assert entities[0].component_category == VALVES


# --------------------------------------------------------------------------- #
# Assignment must not be order-dependent
# --------------------------------------------------------------------------- #

def test_a_nearby_label_is_not_stolen_by_a_distant_symbol():
    """
    The greedy assignment gave a symbol up to five nearest texts, so a
    symbol 290px away could take a label that belonged to a symbol 10px
    away.
    """
    symbol_a = _det(x=100.0, y=100.0)
    symbol_b = _det(x=400.0, y=100.0)
    text_a = _txt("FIC-101", x=120.0, y=100.0)   # 20px from A
    text_b = _txt("FIC-201", x=390.0, y=100.0)   # 10px from B

    entities = EntityMapper().map_entities(
        [symbol_a, symbol_b], [text_a, text_b]
    )
    by_position = {e.position[0]: e for e in entities}

    assert "FIC-201" in [t.text for t in by_position[400.0].text_regions], (
        "symbol B did not get the label that is 10px away from it"
    )


def test_assignment_is_independent_of_input_order():
    """Reversing the detection order must not change who gets which label."""
    symbol_a = _det(x=100.0, y=100.0)
    symbol_b = _det(x=400.0, y=100.0)
    text_a = _txt("FIC-101", x=120.0, y=100.0)
    text_b = _txt("FIC-201", x=390.0, y=100.0)

    def assign(order):
        mapper = EntityMapper()
        ents = mapper.map_entities(
            [symbol_a, symbol_b] if order == "ab" else [symbol_b, symbol_a],
            [text_a, text_b],
        )
        return {
            round(e.position[0]): sorted(t.text for t in e.text_regions)
            for e in ents
        }

    assert assign("ab") == assign("ba"), (
        "the assignment changed when the input order changed"
    )


def test_each_label_is_assigned_to_at_most_one_symbol():
    """A text region must not be attached to two different symbols."""
    symbols = [_det(x=x, y=100.0) for x in (100.0, 200.0, 300.0)]
    texts = [_txt(f"FIC-{i}", x=100.0 + i * 60, y=100.0) for i in range(3)]

    entities = EntityMapper().map_entities(symbols, texts)
    assigned = [t.text for e in entities for t in e.text_regions]
    assert len(assigned) == len(set(assigned)), (
        f"a label was assigned more than once: {assigned}"
    )


# --------------------------------------------------------------------------- #
# Size / spec must not be inherited from a line label
# --------------------------------------------------------------------------- #

def test_a_line_label_does_not_give_a_valve_a_pipe_size():
    """
    '6\"' is the size of the PIPE on the line, not the size of the valve.
    Inheriting it made the MTO claim a 6-inch valve that is not there.
    """
    symbol = _det(class_name="Gate Valve", x=50.0, y=10.0)
    line_label = _txt('6"-PA-1001-A1A-N', x=50.0, y=10.0)

    entities = EntityMapper().map_entities([symbol], [line_label])
    valve = entities[0]
    assert valve.size is None, (
        f"the valve inherited size={valve.size!r} from a line label"
    )
    assert valve.specification is None


# --------------------------------------------------------------------------- #
# LINE_NUMBER is not a procurable item
# --------------------------------------------------------------------------- #

def test_a_line_number_does_not_become_a_procurable_entity():
    """A line number is a property of a line, not something you buy."""
    line_label = _txt('6"-PA-1001-A1A-N', x=50.0, y=10.0)

    entities = EntityMapper().map_entities([], [line_label])
    for e in entities:
        assert e.entity_type not in ("line", "unknown"), (
            f"a line number became a {e.entity_type} entity"
        )


# --------------------------------------------------------------------------- #
# Entity ids
# --------------------------------------------------------------------------- #

def test_entity_ids_are_unique_within_a_run():
    symbols = [_det(x=x, y=100.0) for x in (100.0, 200.0, 300.0)]
    entities = EntityMapper().map_entities(symbols, [])
    ids = [e.entity_id for e in entities]
    assert len(ids) == len(set(ids))
