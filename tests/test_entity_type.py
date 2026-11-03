"""
Entity type vs tag consistency.

A symbol the model classified as a valve may carry a tag that says it is
something else entirely ("E-1003" = heat exchanger). The tag is stronger
evidence than the symbol class, so the entity type must follow the tag.
"""
import pytest

from src.detection.symbol_detector import Detection
from src.graph.entity_mapper import EntityMapper
from src.ocr.text_extractor import TextRegion


def _det(class_name="Gate_Valve", x=100.0, y=100.0):
    return Detection(
        class_id=0, class_name=class_name, confidence=0.9,
        bbox_xyxy=(x - 10, y - 10, x + 10, y + 10), center=(x, y),
    )


def _txt(text, x=100.0, y=100.0):
    return TextRegion(
        text=text, confidence=0.9,
        bbox=(x - 20, y - 6, x + 20, y + 6), center=(x, y), polygon=[],
    )


def test_an_equipment_tag_overrides_a_misclassified_symbol():
    """
    The model called it a valve; the drawing says E-1003, a heat exchanger.
    The tag is stronger evidence, so the MTO must say Equipment.
    """
    entities = EntityMapper().map_entities(
        [_det(class_name="Gate_Valve")], [_txt("E-1003")]
    )
    entity = entities[0]

    assert entity.tag_id == "E-1003"
    assert entity.entity_type == "equipment", (
        f"the entity was typed as {entity.entity_type!r} despite an "
        "equipment tag"
    )
    assert entity.component_category == "equipment"


def test_an_instrument_tag_types_the_entity_as_an_instrument():
    entities = EntityMapper().map_entities(
        [_det(class_name="Valve")], [_txt("FIC-1001")]
    )
    entity = entities[0]

    assert entity.tag_id == "FIC-1001"
    assert entity.entity_type == "instrument"
    assert entity.component_category == "instruments"


def test_an_untagged_symbol_keeps_its_detected_type():
    """Without a tag, the symbol class is the only evidence there is."""
    entities = EntityMapper().map_entities(
        [_det(class_name="Centrifugal Pump")], []
    )
    assert entities[0].entity_type == "equipment"
    assert entities[0].component_category == "equipment"


def test_a_non_tag_label_does_not_change_the_type():
    """'Methanol' is a service label, not a tag; the symbol type stands."""
    entities = EntityMapper().map_entities(
        [_det(class_name="Gate_Valve")], [_txt("Methanol")]
    )
    assert entities[0].entity_type == "valve"
    assert entities[0].component_category == "valves"
