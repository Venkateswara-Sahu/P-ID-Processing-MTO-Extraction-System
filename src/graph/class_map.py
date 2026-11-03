"""
Symbol class -> MTO category mapping.

The model in ``models/best.pt`` has 229 human-readable class names
("Gate Valve", "Turbine Pump", "Temp Indicator"). A previous substring
heuristic classified 43.7% of them as "unknown" and filed them under
"OTHER COMPONENTS" -- including safety-critical items like PSV and PRV.

This module maps the model's own vocabulary onto the four MTO categories.

Design note
-----------
The mapping is an EXPLICIT taxonomy, not a keyword guess. The dataset
audit (see AUDIT_REPORT.md) found the model's vocabulary contains:

  * 7 groups of class names that are the same symbol spelled differently
    ("Pneumatic - Diaphragm Gate Valve" / "Pneumatic-DiaphragmGate Valve"),
  * 7 misspellings ("bolier", "Fluidzed Recactor", "temp trasmitter"),
  * 25 names that no substring heuristic can place ("Axical COMP",
    "Counterflow Forced Draft", "Treaded", "lmi").

A keyword list either misses those 25 or accidentally places them. So the
taxonomy below is explicit and EXHAUSTIVE over the model's 229 names, and
the keyword heuristic survives only as a fallback for a class name the
taxonomy has never seen (a future retrain with a new vocabulary).

Both the original names and the canonicalised names that
``tools/repair_dataset.py`` produces resolve to the same category, so
swapping in a retrained model cannot silently change the MTO categories.
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# MTO categories. The keys match src/mto/templates.py MTO_CATEGORIES.
VALVES = "valves"
INSTRUMENTS = "instruments"
EQUIPMENT = "equipment"
PIPING = "piping"
OTHER = "other"

# --------------------------------------------------------------------------- #
# EXPLICIT TAXONOMY
#
# Every one of the model's 229 class names, assigned to a category.
# Names that differ only in punctuation/spacing/case are grouped onto one
# line so the canonical name and its variants stay in step.
# --------------------------------------------------------------------------- #

_TAXONOMY: List[Tuple[str, str]] = [
    # ---- Valves ---------------------------------------------------------- #
    ("3 Way Ball Valve", VALVES),
    ("3 Way Gate Valve", VALVES),
    ("3 Way Globe Valve", VALVES),
    ("3 Way Valve", VALVES),
    ("4 Way Ball Valve", VALVES),
    ("4 Way Gate Valve", VALVES),
    ("4 Way Valve", VALVES),
    ("Angle Blowdown", VALVES),
    ("Angle Globe Valve", VALVES),
    ("Angle GlobeValve", VALVES),
    ("Angle Valve", VALVES),
    ("Back Pressure Regulator", VALVES),
    ("BackPressure Regulator", VALVES),
    ("Balance Diaphragm Gate Valve", VALVES),
    ("Ball Valve", VALVES),
    ("Ball_valve", VALVES),
    ("Bleeder Valve", VALVES),
    ("Butterfly Valve", VALVES),
    ("Check Valve", VALVES),
    ("Closed Ball Valve", VALVES),
    ("Closed Gate Valve", VALVES),
    ("Cock Globe Valve", VALVES),
    ("Cock Valve", VALVES),
    ("Control Valve", VALVES),
    ("Control_valve", VALVES),
    ("Diaphragm Valve", VALVES),
    ("Electric operator gate valve", VALVES),
    ("Fail Closed Fail Safe Position", VALVES),
    ("Fail Closed Safe Position", VALVES),
    ("Fail Indeterminate Safe Position", VALVES),
    ("Fail Lock Safe Position", VALVES),
    ("Fail Open Fail Safe Position", VALVES),
    ("Fail Open Safe Position", VALVES),
    ("Flanged Ball", VALVES),
    ("Flanged Butterfly", VALVES),
    ("Flanged Cock", VALVES),
    ("Flanged Globe", VALVES),
    ("Float Operated Valve", VALVES),
    ("Gate Valve", VALVES),
    ("Gate_Valve", VALVES),
    ("Globe Valve", VALVES),
    ("Globe_valve", VALVES),
    ("Hand Operated Angle Valve", VALVES),
    ("Hand Operated Ball Valve", VALVES),
    ("Hand Operated Gate Closed Valve", VALVES),
    ("Hand Operated Gate Valve", VALVES),
    ("Hand OperatedGate Valve", VALVES),
    ("Hand Operated Globe Valve", VALVES),
    ("Hand_operated_gate_valve", VALVES),
    ("Hydraulic Ball Valve", VALVES),
    ("Hydraulic Closed Valve", VALVES),
    ("Hydraulic Gate Valve", VALVES),
    ("Hydraulic Globe Valve", VALVES),
    ("Hydraulic Operated Ball Valve", VALVES),
    ("Hydraulic Operated Globe Valve", VALVES),
    ("Integrated Block Valve", VALVES),
    ("Knife Valve", VALVES),
    ("Motor Angle Valve", VALVES),
    ("Motor Ball Valve", VALVES),
    ("Motor Diaphragm Valve", VALVES),
    ("Motor Gate Valve", VALVES),
    ("Motor Globe Valve", VALVES),
    ("Motor Operated Angle Valve", VALVES),
    ("Motor Operated Butterfly Valve", VALVES),
    ("Motor Operated Diaphragm Valve", VALVES),
    ("Motor Operated Gate Valve", VALVES),
    ("Motor Operated Globe Valve", VALVES),
    ("Motor Operated Plug Valve", VALVES),
    ("Motor Plug Valve", VALVES),
    ("Motor closed Gate Valve", VALVES),
    ("Needle Valve", VALVES),
    ("Pinch Valve", VALVES),
    ("Piston Operated Valve", VALVES),
    ("Pneumatic - Diaphragm Angle Valve", VALVES),
    ("Pneumatic - Diaphragm Butterfly Valve", VALVES),
    ("Pneumatic - Diaphragm Gate Valve", VALVES),
    ("Pneumatic - Diaphragm Globe Valve", VALVES),
    ("Pneumatic - DiaphragmGate Valve", VALVES),
    ("Pneumatic-Diaphragm 3 Way Valve", VALVES),
    ("Pneumatic-Diaphragm Angle Valve", VALVES),
    ("Pneumatic-Diaphragm Ball Valve", VALVES),
    ("Pneumatic-Diaphragm Butterfly Valve", VALVES),
    ("Pneumatic-Diaphragm Gate Valve", VALVES),
    ("Pneumatic-Diaphragm Globe Valve", VALVES),
    ("Plug Valve", VALVES),
    ("Powered Gate Valve", VALVES),
    ("Powered Valve", VALVES),
    ("PRV", VALVES),
    ("Pressure Regulator", VALVES),
    ("pressure safety Valve", VALVES),
    ("PSV", VALVES),
    ("Quarter Turn Valve Double Acting", VALVES),
    ("Quarter Turn Valve Spring Acting", VALVES),
    ("Relief Valve", VALVES),
    ("Relief PRV", VALVES),
    ("Relife PRV", VALVES),
    ("Rotary Piston-Pneumatic Butterfly Valve", VALVES),
    ("Rotary Piston-Pneumatic Closed Gate Valve", VALVES),
    ("Rotary Piston-Pneumatic Gate Valve", VALVES),
    ("Rotary Valve", VALVES),
    ("Safety PSV", VALVES),
    ("Slide Valve", VALVES),
    ("Solenoid", VALVES),
    ("Solenoid Closed Valve", VALVES),
    ("Solenoid Operated Gate Valve", VALVES),
    ("Solenoid Valve", VALVES),
    ("Stop Valve", VALVES),
    ("Valve", VALVES),

    # ---- Instruments ----------------------------------------------------- #
    ("Flow Controller", INSTRUMENTS),
    ("Flow Recorder", INSTRUMENTS),
    ("Flow Transmitter", INSTRUMENTS),
    ("Flowmeter", INSTRUMENTS),
    ("Gauge", INSTRUMENTS),
    ("Gauge Equipment", INSTRUMENTS),
    ("Indicator", INSTRUMENTS),
    ("Level Alarm", INSTRUMENTS),
    ("Level Alarm High", INSTRUMENTS),
    ("Level Alarm Low", INSTRUMENTS),
    ("Level Controller", INSTRUMENTS),
    ("Level Indicator", INSTRUMENTS),
    ("Level Meter", INSTRUMENTS),
    ("Level Recorder", INSTRUMENTS),
    ("Level Transmitter", INSTRUMENTS),
    ("Level Trasmitter", INSTRUMENTS),
    ("lmi", INSTRUMENTS),
    ("Magnetic Level Indicator", INSTRUMENTS),
    ("Mangnetic", INSTRUMENTS),
    ("Orifice", INSTRUMENTS),
    ("Orifice Equipment", INSTRUMENTS),
    ("Pressure Balance Diaphragm", INSTRUMENTS),
    ("Pressure Controller", INSTRUMENTS),
    ("Pressure Gauge", INSTRUMENTS),
    ("Pressure Indicating Controller", INSTRUMENTS),
    ("Pressure Indicator", INSTRUMENTS),
    ("Pressure Recorder", INSTRUMENTS),
    ("Pressure Recording Controller", INSTRUMENTS),
    ("Pressure Transmitter", INSTRUMENTS),
    ("Programmable Indicator", INSTRUMENTS),
    ("radio link", INSTRUMENTS),
    ("Rotameter Equipment", INSTRUMENTS),
    ("Rotary Meter", INSTRUMENTS),
    ("Rotometer", INSTRUMENTS),
    ("Shared Indicator", INSTRUMENTS),
    ("Temp Indicator", INSTRUMENTS),
    ("Temperature Transmitter", INSTRUMENTS),
    ("temp controller", INSTRUMENTS),
    ("temp recorder", INSTRUMENTS),
    ("temp trasmitter", INSTRUMENTS),
    ("Odometer", INSTRUMENTS),
    ("Pitot Tube Type Flow Meter", INSTRUMENTS),
    ("Quick Change Meter", INSTRUMENTS),

    # ---- Equipment ------------------------------------------------------- #
    ("Agitator or Mixer", EQUIPMENT),
    ("Alkyalation", EQUIPMENT),
    ("Automatic Stoker", EQUIPMENT),
    ("Axial Compressor", EQUIPMENT),
    ("Axical COMP", EQUIPMENT),
    ("Boiler", EQUIPMENT),
    ("bolier", EQUIPMENT),
    ("Boom Loader", EQUIPMENT),
    ("Cavity Pump", EQUIPMENT),
    ("Centrifugal Blower", EQUIPMENT),
    ("Centrifugal Compressor", EQUIPMENT),
    ("Centrifugal Pump", EQUIPMENT),
    ("Chimney Tower", EQUIPMENT),
    ("Compressor Turbine", EQUIPMENT),
    ("Conveyor", EQUIPMENT),
    ("Counterflow Forced Draft", EQUIPMENT),
    ("Counterflow Natural Draft", EQUIPMENT),
    ("Crossflow Inducted", EQUIPMENT),
    ("Cylinder", EQUIPMENT),
    ("Diesel Motor", EQUIPMENT),
    ("Double Flow Turbine", EQUIPMENT),
    ("Drum", EQUIPMENT),
    ("Ejector", EQUIPMENT),
    ("Electric Motor", EQUIPMENT),
    ("Elevator", EQUIPMENT),
    ("Fan Blades", EQUIPMENT),
    ("Fluid Catalytic Cracking", EQUIPMENT),
    ("Fluid Cooking", EQUIPMENT),
    ("Fluidized Reactor", EQUIPMENT),
    ("Fluidzed Recactor", EQUIPMENT),
    ("Furnace", EQUIPMENT),
    ("Gear Pump", EQUIPMENT),
    ("Heat Exchanger", EQUIPMENT),
    ("Hoist", EQUIPMENT),
    ("Horizontal Pump", EQUIPMENT),
    ("Horizontal Vessel", EQUIPMENT),
    ("Hydrocracking", EQUIPMENT),
    ("Hydrodesulferization", EQUIPMENT),
    ("Liquid Ring Vacuum", EQUIPMENT),
    ("Mixer", EQUIPMENT),
    ("Mixing Reactor", EQUIPMENT),
    ("Motor", EQUIPMENT),
    ("Motor Driven Turbine", EQUIPMENT),
    ("Not Gate", EQUIPMENT),
    ("OR Gate", EQUIPMENT),
    ("Oil Burner", EQUIPMENT),
    ("Overhead Conveyor", EQUIPMENT),
    ("Packed Tower", EQUIPMENT),
    ("Plate Tower", EQUIPMENT),
    ("Positive Displacement", EQUIPMENT),
    ("Progressive Cavity Pump", EQUIPMENT),
    ("Pump", EQUIPMENT),
    ("Reactor", EQUIPMENT),
    ("Reciprocation Compressor", EQUIPMENT),
    ("Reciprocation  Compressor", EQUIPMENT),
    ("Reformer", EQUIPMENT),
    ("reformer", EQUIPMENT),
    ("Rotary Compressor", EQUIPMENT),
    ("Screw Conveyor", EQUIPMENT),
    ("Screw Pump", EQUIPMENT),
    ("Scraper Conveyor", EQUIPMENT),
    ("Skip Hoist", EQUIPMENT),
    ("Sampler", EQUIPMENT),
    ("Sump Pump", EQUIPMENT),
    ("Tank", EQUIPMENT),
    ("Triple Fan Blade", EQUIPMENT),
    ("Tubular", EQUIPMENT),
    ("Turbine Driver", EQUIPMENT),
    ("Turbine Pump", EQUIPMENT),
    ("Vaccum Pump", EQUIPMENT),
    ("Vane Pump", EQUIPMENT),
    ("Vertical Pump", EQUIPMENT),
    ("Vertical Vessel", EQUIPMENT),
    ("vessel", EQUIPMENT),

    # ---- Piping components ----------------------------------------------- #
    ("FlangePair", PIPING),
    ("Flanged", PIPING),
    ("Flanged End Connection", PIPING),
    ("Flanged Ends", PIPING),
    ("Funnel", PIPING),
    ("Manhole", PIPING),
    ("Socket Ends", PIPING),
    ("Socket Weld", PIPING),
    ("Socket Weld End Connection", PIPING),
    ("Threaded", PIPING),
    ("Treaded", PIPING),
    ("Welded", PIPING),
    ("Welded End Connection", PIPING),

    # ---- Not a P&ID symbol (annotations, logic, process names) ----------- #
    ("On Contral Control Panel", OTHER),
    ("On Control Panel", OTHER),
    ("On Local Control Panel", OTHER),
    ("pneumatic", OTHER),
    ("tag", OTHER),
]


def _key(name: str) -> str:
    """Normalise a class name: lowercase, punctuation stripped."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


CLASS_CATEGORY: Dict[str, str] = {name: cat for name, cat in _TAXONOMY}
CLASS_CATEGORY_BY_KEY: Dict[str, str] = {}
for _name, _cat in _TAXONOMY:
    CLASS_CATEGORY_BY_KEY.setdefault(_key(_name), _cat)

# Keyword -> category, in priority order. Kept as a FALLBACK for class
# names the explicit taxonomy does not know (e.g. after a retrain that
# adds classes). Earlier entries win.
CATEGORY_KEYWORDS: List[Tuple[str, str]] = [
    ("valve", VALVES),
    ("stop cock", VALVES),
    ("stopcock", VALVES),
    ("blowdown", VALVES),
    ("psv", VALVES),
    ("prv", VALVES),
    ("relief", VALVES),
    ("solenoid", VALVES),
    ("regulator", VALVES),
    ("fail open", VALVES),
    ("fail closed", VALVES),
    ("fail lock", VALVES),
    ("fail indeterminate", VALVES),

    ("instrument", INSTRUMENTS),
    ("transmitter", INSTRUMENTS),
    ("trasmitter", INSTRUMENTS),
    ("transducer", INSTRUMENTS),
    ("indicator", INSTRUMENTS),
    ("controller", INSTRUMENTS),
    ("recorder", INSTRUMENTS),
    ("gauge", INSTRUMENTS),
    ("meter", INSTRUMENTS),
    ("alarm", INSTRUMENTS),
    ("switch", INSTRUMENTS),
    ("thermocouple", INSTRUMENTS),
    ("orifice", INSTRUMENTS),
    ("flowmeter", INSTRUMENTS),
    ("flow meter", INSTRUMENTS),

    ("pump", EQUIPMENT),
    ("compressor", EQUIPMENT),
    ("exchanger", EQUIPMENT),
    ("vessel", EQUIPMENT),
    ("reactor", EQUIPMENT),
    ("recactor", EQUIPMENT),
    ("tank", EQUIPMENT),
    ("tower", EQUIPMENT),
    ("column", EQUIPMENT),
    ("boiler", EQUIPMENT),
    ("bolier", EQUIPMENT),
    ("heater", EQUIPMENT),
    ("cooler", EQUIPMENT),
    ("chiller", EQUIPMENT),
    ("condenser", EQUIPMENT),
    ("furnace", EQUIPMENT),
    ("kiln", EQUIPMENT),
    ("cyclone", EQUIPMENT),
    ("agitator", EQUIPMENT),
    ("mixer", EQUIPMENT),
    ("motor", EQUIPMENT),
    ("blower", EQUIPMENT),
    ("fan", EQUIPMENT),
    ("drum", EQUIPMENT),
    ("filter", EQUIPMENT),
    ("strainer", EQUIPMENT),
    ("elevator", EQUIPMENT),
    ("conveyor", EQUIPMENT),
    ("hoist", EQUIPMENT),
    ("bottle", EQUIPMENT),
    ("silo", EQUIPMENT),
    ("engine", EQUIPMENT),
    ("turbine", EQUIPMENT),
    ("generator", EQUIPMENT),
    ("cylinder", EQUIPMENT),
    ("funnel", EQUIPMENT),
    ("ejector", EQUIPMENT),
    ("stoker", EQUIPMENT),
    ("chute", EQUIPMENT),
    ("burner", EQUIPMENT),
    ("reformer", EQUIPMENT),
    ("cracking", EQUIPMENT),
    ("hydrocracking", EQUIPMENT),

    ("reducer", PIPING),
    ("tee", PIPING),
    ("elbow", PIPING),
    ("flange", PIPING),
    ("spectacle", PIPING),
    ("expansion joint", PIPING),
    ("gasket", PIPING),
    ("coupling", PIPING),
    ("nipple", PIPING),
    ("cap", PIPING),
    ("weld", PIPING),
    ("socket", PIPING),
    ("thread", PIPING),
    ("blind", PIPING),
    ("hose", PIPING),
    ("tray", PIPING),
    ("manhole", PIPING),
    ("funnel", PIPING),
]

# Entity type labels, per category.
ENTITY_TYPE_BY_CATEGORY = {
    VALVES: "valve",
    INSTRUMENTS: "instrument",
    EQUIPMENT: "equipment",
    PIPING: "piping_component",
    OTHER: "unknown",
}


def classify_class_name(class_name: str) -> str:
    """
    Return the MTO category for a model class name.

    Lookup order:
      1. exact name in the explicit taxonomy
      2. normalised name (case/punctuation insensitive) in the taxonomy
      3. keyword heuristic (fallback for unseen vocabularies)
      4. "other"

    Args:
        class_name: A class name from the model's ``names`` mapping.

    Returns:
        One of "valves", "instruments", "equipment", "piping", "other".
    """
    if not class_name:
        return OTHER

    category = CLASS_CATEGORY.get(class_name)
    if category is not None:
        return category

    category = CLASS_CATEGORY_BY_KEY.get(_key(class_name))
    if category is not None:
        return category

    lowered = class_name.lower()
    for keyword, kw_category in CATEGORY_KEYWORDS:
        if keyword in lowered:
            return kw_category
    return OTHER


def entity_type_for(category: str) -> str:
    """Return the entity type label for an MTO category."""
    return ENTITY_TYPE_BY_CATEGORY.get(category, "unknown")


def build_class_map(class_names: List[str]) -> Dict[str, str]:
    """
    Build a class_name -> category mapping for a list of model class names.

    Args:
        class_names: The model's class names.

    Returns:
        Dict mapping each class name to its MTO category.
    """
    return {name: classify_class_name(name) for name in class_names}


def coverage_report(class_names: List[str]) -> Dict[str, object]:
    """
    Summarise how many classes land in each category.

    Useful for checking the mapping against the real model vocabulary.
    """
    mapping = build_class_map(class_names)
    counts: Dict[str, int] = {}
    for category in mapping.values():
        counts[category] = counts.get(category, 0) + 1
    unmapped = [n for n, c in mapping.items() if c == OTHER]
    total = len(class_names)
    mapped = total - len(unmapped)
    return {
        "total": total,
        "counts": counts,
        "unmapped_count": len(unmapped),
        "unmapped": unmapped,
        "coverage_pct": round(100.0 * mapped / total, 1) if total else 0.0,
    }
