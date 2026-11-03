"""
Tag parser behaviour tests.

These encode the P&ID tagging rules the parser is supposed to follow. They were
written before the fixes (RED) and describe the CORRECT behaviour, not the
current (buggy) behaviour.

Domain rules being encoded:
  - 1-letter tag prefixes (E, P, V, R, H, T, ...) are EQUIPMENT tags.
  - 2-4 letter prefixes (FIC, TT, PSV, ...) are INSTRUMENT tags.
  - Pipe sizes carry an explicit inch mark; bare numbers are stream numbers.
  - Words are not tags.
"""
import pytest

from src.ocr.tag_parser import TagParser, TagType


# --------------------------------------------------------------------------- #
# C2: equipment tags must not be shadowed by the instrument parser
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected_type,expected_desc", [
    ("E-1001", TagType.EQUIPMENT_TAG, "Heat Exchanger"),
    ("R-1001", TagType.EQUIPMENT_TAG, "Reactor"),
    ("P-101A", TagType.EQUIPMENT_TAG, "Pump"),
    ("V-3040", TagType.EQUIPMENT_TAG, "Vessel"),
    ("H-100",  TagType.EQUIPMENT_TAG, "Heater"),
    ("T-1001", TagType.EQUIPMENT_TAG, "Tank/Tower"),
    ("C-201",  TagType.EQUIPMENT_TAG, "Compressor"),
    ("B-301",  TagType.EQUIPMENT_TAG, "Boiler"),
    ("M-401",  TagType.EQUIPMENT_TAG, "Mixer"),
    ("D-501",  TagType.EQUIPMENT_TAG, "Drum"),
])
def test_single_letter_prefix_is_equipment(tag_parser, text, expected_type, expected_desc):
    """A one-letter prefix is an equipment tag, not an instrument tag."""
    parsed = tag_parser.parse(text)
    assert parsed.tag_type == expected_type, f"{text} -> {parsed.tag_type} ({parsed.parsed_description})"
    assert parsed.parsed_description == expected_desc


@pytest.mark.parametrize("text,expected_tag_id", [
    ("E-1001", "E-1001"),
    ("R-1001", "R-1001"),
    ("P-101A", "P-101A"),
    ("V-3040", "V-3040"),
    ("H-100", "H-100"),
    ("E-201A/B", "E-201A/B"),
])
def test_equipment_tag_id_is_well_formed(tag_parser, text, expected_tag_id):
    """The tag id must not contain a doubled separator or drop the suffix."""
    parsed = tag_parser.parse(text)
    assert parsed.tag_id == expected_tag_id
    assert "--" not in parsed.tag_id


@pytest.mark.parametrize("text,expected_desc", [
    ("FIC-1001", "Flow Indicator/Controller"),
    ("FIC-101",  "Flow Indicator/Controller"),
    ("PSV-201",  "Pressure Switch/Valve"),
    ("PSHH",     "Pressure Switch/High/High"),
    ("TT-301",   "Temperature Transmitter"),
    ("TI-1102",  "Temperature Indicator"),
    ("PT-202",   "Pressure Transmitter"),
])
def test_multi_letter_prefix_is_instrument(tag_parser, text, expected_desc):
    """A 2-4 letter ISA prefix is an instrument tag."""
    parsed = tag_parser.parse(text)
    assert parsed.tag_type == TagType.INSTRUMENT_TAG
    assert parsed.parsed_description == expected_desc


def test_equipment_parse_order_wins_over_instrument(tag_parser):
    """
    Regression guard for the shadowing bug: the instrument parser used to be
    tried first and matched 1-letter equipment prefixes.
    """
    parsed = tag_parser.parse("P-101A")
    assert parsed.tag_type == TagType.EQUIPMENT_TAG
    assert parsed.equipment_type == "Pump"
    assert parsed.parsed_description != "Pressure"


# --------------------------------------------------------------------------- #
# C3: words must not be manufactured into tags
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text", ["PUMP 101", "LINE 12", "FLOW 200", "VALVE 100", "HEATER 2"])
def test_ordinary_words_are_not_tags(tag_parser, text):
    """A word followed by a number is prose, not an engineering tag."""
    parsed = tag_parser.parse(text)
    assert parsed.tag_type != TagType.INSTRUMENT_TAG, f"{text} became {parsed.tag_id}"
    assert parsed.tag_type != TagType.EQUIPMENT_TAG, f"{text} became {parsed.tag_id}"


def test_word_with_number_does_not_get_a_measure_description(tag_parser):
    parsed = tag_parser.parse("PUMP 101")
    assert parsed.parsed_description != "Pressure U/M/P"


# --------------------------------------------------------------------------- #
# C8: pipe sizes need an explicit inch mark
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected_size", [
    ('2"', "2\""),
    ('6"', "6\""),
    ('10"', "10\""),
    ('3/4"', "3/4\""),
])
def test_pipe_size_with_inch_mark(tag_parser, text, expected_size):
    parsed = tag_parser.parse(text)
    assert parsed.tag_type == TagType.PIPE_SIZE
    assert parsed.pipe_size == expected_size


@pytest.mark.parametrize("text", ["12", "5", "9"])
def test_bare_number_is_not_a_pipe_size(tag_parser, text):
    """
    Stream numbers are bare 1-2 digit numbers and are the most common text on a
    P&ID. Treating them as pipe sizes fabricates valve sizes.
    """
    parsed = tag_parser.parse(text)
    assert parsed.tag_type != TagType.PIPE_SIZE, f"{text} became a {parsed.pipe_size} pipe"
    assert parsed.pipe_size is None


# --------------------------------------------------------------------------- #
# S18 / S19: legitimate ISA prefixes and material grades
# --------------------------------------------------------------------------- #

def test_temperature_element_prefix_is_preserved(tag_parser):
    """'TE' is a valid ISA prefix (Temperature Element); the OCR rule broke it."""
    parsed = tag_parser.parse("TE101")
    assert parsed.tag_type == TagType.INSTRUMENT_TAG
    assert parsed.tag_id == "TE-101"


@pytest.mark.parametrize("text", ["A105", "SS304", "A312", "CS"])
def test_material_grades_are_not_instruments(tag_parser, text):
    """ASTM grades and material codes are not instrument tags."""
    parsed = tag_parser.parse(text)
    assert parsed.tag_type != TagType.INSTRUMENT_TAG, f"{text} became {parsed.tag_id}"


# --------------------------------------------------------------------------- #
# Preserved behaviour - the parser must keep working
# --------------------------------------------------------------------------- #

def test_line_number_still_parses(tag_parser):
    parsed = tag_parser.parse('6"-PA-1001-A1A-N')
    assert parsed.tag_type == TagType.LINE_NUMBER
    assert parsed.pipe_size == "6\""
    assert parsed.pipe_spec == "A1A"
    assert parsed.tag_id == '6"-PA-1001-A1A-N'


def test_specification_still_parses(tag_parser):
    parsed = tag_parser.parse("150#")
    assert parsed.tag_type == TagType.SPECIFICATION
    assert parsed.pipe_spec == "150#"


def test_slash_separated_loop_keeps_both_numbers(tag_parser):
    """'TT-301/302' names two elements; the second must not be dropped."""
    parsed = tag_parser.parse("TT-301/302")
    assert parsed.tag_type == TagType.INSTRUMENT_TAG
    assert "301" in parsed.tag_id
    assert "302" in parsed.tag_id


def test_single_digit_loop_number_is_accepted(tag_parser):
    """'F-1' is a legitimate small tag; the pattern required 2+ digits."""
    parsed = tag_parser.parse("F-1")
    assert parsed.tag_type != TagType.UNKNOWN


def test_free_text_stays_a_description(tag_parser):
    parsed = tag_parser.parse("Methanol")
    assert parsed.tag_type == TagType.DESCRIPTION
    assert parsed.parsed_description == "Methanol"


def test_parse_batch_returns_one_result_per_input(tag_parser):
    results = tag_parser.parse_batch(["E-1001", "FIC-1001", "Methanol"])
    assert len(results) == 3
    assert results[0].tag_type == TagType.EQUIPMENT_TAG
    assert results[1].tag_type == TagType.INSTRUMENT_TAG
