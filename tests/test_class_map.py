"""
The class taxonomy must be exhaustive over the model's real vocabulary.

A keyword heuristic left 25 of the model's 229 classes uncategorised
("Axical COMP", "Counterflow Forced Draft", "Treaded", "lmi", ...).
The explicit taxonomy must place every one of them, and must place the
canonicalised names that tools/repair_dataset.py produces identically.
"""
import pytest

from src.graph.class_map import (
    CLASS_CATEGORY,
    EQUIPMENT,
    INSTRUMENTS,
    OTHER,
    PIPING,
    VALVES,
    classify_class_name,
    coverage_report,
)


@pytest.fixture(scope="module")
def model_class_names():
    """The real 229 class names from the shipped checkpoint."""
    from pathlib import Path

    from ultralytics import YOLO

    root = Path(__file__).resolve().parents[1]
    model = YOLO(str(root / "models" / "best.pt"))
    return [model.names[i] for i in range(len(model.names))]


def test_every_model_class_is_in_the_explicit_taxonomy(model_class_names):
    missing = [n for n in model_class_names if n not in CLASS_CATEGORY]
    assert not missing, (
        f"{len(missing)} model classes are not in the taxonomy: {missing}"
    )


# The only classes allowed to land in 'other': these are annotations and
# control-panel labels on the drawing, not physical items you can buy.
ALLOWED_OTHER_CLASSES = {
    "On Contral Control Panel",
    "On Control Panel",
    "On Local Control Panel",
    "pneumatic",
    "tag",
}


def test_only_non_symbol_annotations_fall_through_to_other(model_class_names):
    """
    'other' means the item lands in the MTO's "OTHER COMPONENTS" bucket.
    Only genuinely non-symbol annotations may go there; every physical
    item must be placed.
    """
    report = coverage_report(model_class_names)
    unexpected = [
        n for n in report["unmapped"] if n not in ALLOWED_OTHER_CLASSES
    ]
    assert not unexpected, (
        f"{len(unexpected)} physical items fell through to 'other': {unexpected}"
    )
    assert set(report["unmapped"]) <= ALLOWED_OTHER_CLASSES, (
        f"unexpected 'other' classes: {report['unmapped']}"
    )
    # Every physical class is placed. Only the known annotations sit in
    # 'other', so the vocabulary is 100% accounted for.
    assert len(report["unmapped"]) == len(
        [n for n in model_class_names if n in ALLOWED_OTHER_CLASSES]
    )
    assert report["counts"].get(OTHER, 0) == len(ALLOWED_OTHER_CLASSES) - (
        1 if "On Control Panel" not in model_class_names else 0
    )


def test_the_taxonomy_covers_the_whole_vocabulary_not_just_the_shipped_model():
    """The repaired dataset's canonical names must also be covered."""
    from tools.repair_dataset import build_canonical_map

    canonical = build_canonical_map(list(CLASS_CATEGORY.keys()))[1]
    missing = [n for n in canonical if n not in CLASS_CATEGORY]
    assert not missing, f"canonical names missing from taxonomy: {missing}"


# --------------------------------------------------------------------------- #
# The 25 classes the old keyword heuristic could not place
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "class_name,expected",
    [
        ("Axical COMP", EQUIPMENT),
        ("Counterflow Forced Draft", EQUIPMENT),
        ("Counterflow Natural Draft", EQUIPMENT),
        ("Crossflow Inducted", EQUIPMENT),
        ("Fluid Cooking", EQUIPMENT),
        ("Liquid Ring Vacuum", EQUIPMENT),
        ("Positive Displacement", EQUIPMENT),
        ("Boom Loader", EQUIPMENT),
        ("Tubular", EQUIPMENT),
        ("Sampler", EQUIPMENT),
        ("Angle Blowdown", VALVES),
        ("Treaded", PIPING),
        ("Manhole", PIPING),
        ("lmi", INSTRUMENTS),
        ("radio link", INSTRUMENTS),
        ("Mangnetic", INSTRUMENTS),
    ],
)
def test_previously_unmappable_classes_now_resolve(class_name, expected):
    assert classify_class_name(class_name) == expected


# --------------------------------------------------------------------------- #
# Duplicate spellings must resolve to the SAME category
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "a,b",
    [
        ("Pneumatic - Diaphragm Gate Valve", "Pneumatic-DiaphragmGate Valve"),
        ("Pneumatic - Diaphragm Globe Valve", "Pneumatic-Diaphragm Globe Valve"),
        ("Pneumatic - Diaphragm Angle Valve", "Pneumatic-Diaphragm Angle Valve"),
        ("Pneumatic - Diaphragm Butterfly Valve", "Pneumatic-Diaphragm Butterfly Valve"),
        ("Back Pressure Regulator", "BackPressure Regulator"),
        ("Hand Operated Gate Valve", "Hand OperatedGate Valve"),
        ("Hand Operated Gate Valve", "Hand_operated_gate_valve"),
        ("Reformer", "reformer"),
    ],
)
def test_duplicate_spellings_resolve_to_the_same_category(a, b):
    assert classify_class_name(a) == classify_class_name(b)


# --------------------------------------------------------------------------- #
# Safety-critical items must never land in 'other'
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "class_name",
    [
        "PSV", "PRV", "Relief Valve", "Relife PRV", "Safety PSV",
        "pressure safety Valve", "Pressure Regulator", "BackPressure Regulator",
    ],
)
def test_safety_valves_are_never_other(class_name):
    assert classify_class_name(class_name) == VALVES


# --------------------------------------------------------------------------- #
# Misspellings resolve to the same category as their corrected form
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "typo,correct,expected",
    [
        ("bolier", "Boiler", EQUIPMENT),
        ("Fluidzed Recactor", "Fluidized Reactor", EQUIPMENT),
        ("Level Trasmitter", "Level Transmitter", INSTRUMENTS),
        ("temp trasmitter", "Temperature Transmitter", INSTRUMENTS),
    ],
)
def test_misspellings_match_their_corrected_form(typo, correct, expected):
    assert classify_class_name(typo) == expected
    assert classify_class_name(correct) == expected


def test_an_unknown_class_falls_back_to_the_keyword_heuristic():
    """A future retrain may add classes; they must not crash the mapper."""
    assert classify_class_name("Some Brand New Widget Valve") == VALVES
    assert classify_class_name("Some Brand New Widget Pump") == EQUIPMENT
    assert classify_class_name("Totally Unknown Thing") == OTHER


def test_empty_class_name_is_other():
    assert classify_class_name("") == OTHER
