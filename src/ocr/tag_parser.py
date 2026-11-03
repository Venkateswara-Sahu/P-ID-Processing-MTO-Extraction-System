"""
Engineering Tag Parser for P&ID text.

Parses OCR-extracted text into structured engineering entities
following ISA (International Society of Automation) standards
and common P&ID naming conventions.

Tag grammar implemented here:

===================  ==========================================  ==========
Form                 Example                                     Type
===================  ==========================================  ==========
``X-101``            ``P-101A``, ``E-1001``, ``R-1001``          Equipment
``XX-101``           ``FIC-1001``, ``PSV-201``, ``TT-301/302``   Instrument
``6"-PA-1001-A1A-N`` line numbers                              Line number
``6"``               pipe sizes (inch mark required)            Pipe size
``150#``             ratings / material grades                   Specification
===================  ==========================================  ==========

The key distinction is **prefix length**: a one-letter prefix is equipment
(``P`` pump, ``E`` exchanger, ``V`` vessel), while a two-to-four letter
prefix is an ISA instrument code (``FIC``, ``PSV``). Trying the instrument
pattern first silently converted every equipment tag into an instrument tag
(``P-101A`` became "Pressure"), so the two are matched on length.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from config.settings import INSTRUMENT_TAG_MAP, INSTRUMENT_FUNCTION_MAP

logger = logging.getLogger(__name__)


class TagType(str, Enum):
    """Classification of text found on P&IDs."""

    INSTRUMENT_TAG = "instrument_tag"       # e.g., FV-101, PI-202, TT-301
    EQUIPMENT_TAG = "equipment_tag"         # e.g., P-101, E-201, V-301
    LINE_NUMBER = "line_number"             # e.g., 6"-PA-1001-A1A-N
    PIPE_SIZE = "pipe_size"                 # e.g., 2", 6", 10"
    SPECIFICATION = "specification"         # e.g., 150#, 300#, CS, SS
    DESCRIPTION = "description"             # General text labels
    UNKNOWN = "unknown"


@dataclass
class ParsedTag:
    """Structured representation of a parsed engineering tag."""

    raw_text: str
    tag_type: TagType
    tag_id: str                              # Full tag identifier
    primary_variable: Optional[str] = None   # F=Flow, P=Pressure, etc.
    function: Optional[str] = None           # T=Transmitter, I=Indicator, etc.
    loop_number: Optional[str] = None        # Numeric loop identifier
    suffix: Optional[str] = None             # A/B for redundant equipment
    equipment_type: Optional[str] = None     # Pump, Exchanger, Vessel, etc.
    pipe_size: Optional[str] = None          # Pipe diameter
    pipe_spec: Optional[str] = None          # Pipe specification/class
    service_code: Optional[str] = None       # Process service (PA, CW, etc.)
    parsed_description: Optional[str] = None # Human-readable description

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class TagParser:
    """Parses engineering text from P&IDs into structured tags."""

    # ---- Regex Patterns ----

    # Instrument tags: 2-4 letters + loop number + optional suffix.
    #   FIC-1001, PSV-201, TT-301/302, PIC-201A
    INSTRUMENT_PATTERN = re.compile(
        r"^([A-Z]{2,4})[- ]?(\d{1,5})([A-Z])?(?:\s*/\s*(\d{1,5}))?$",
    )

    # Letter-only instrument codes: PSHH, LSH, TSH.
    INSTRUMENT_LETTERS_ONLY_PATTERN = re.compile(r"^([A-Z]{2,4})$")

    # Single-letter instrument tags: A-101 (analysis), D-101 (density).
    # Only applies to letters that are NOT equipment letters.
    SINGLE_LETTER_INSTRUMENT_PATTERN = re.compile(
        r"^([A-Z])([- ]?)(\d{1,5})$",
    )

    # Equipment tags: one letter + loop number + optional suffix.
    #   P-101A, E-1001, R-1001, V-3040, E-201 A/B
    EQUIPMENT_PATTERN = re.compile(
        r"^([A-Z])([- ]?)(\d{1,5})\s*([A-Z](?:\s*/\s*[A-Z])?)?$",
    )

    # Line numbers: 6"-PA-1001-A1A-N, 2"-CW-2001-B1B
    LINE_NUMBER_PATTERN = re.compile(
        r'^(\d{1,2})["\u201d]?\s*[-]?\s*([A-Z]{1,4})\s*[-]\s*(\d{3,5})\s*[-]?\s*([A-Z0-9]{1,4})?\s*[-]?\s*([A-Z])?$',
        re.IGNORECASE,
    )

    # Pipe size: 2", 6", 10", 3/4"  -- the inch mark is REQUIRED.
    # Bare numbers are stream numbers, which are far more common on a P&ID
    # and were previously being read as pipe diameters.
    PIPE_SIZE_PATTERN = re.compile(
        r'^(\d{1,2}(?:/\d{1,2})?)\s*["\u201d]$',
    )

    # Specification: 150#, 300# RF, CS, SS304
    SPEC_PATTERN = re.compile(
        r"^(\d{2,4})\s*#|^(CS|SS|A105|A106|A312|A234|ASTM)\s*\d*",
        re.IGNORECASE,
    )

    # Equipment type prefixes
    EQUIPMENT_TYPE_MAP = {
        "P": "Pump",
        "E": "Heat Exchanger",
        "V": "Vessel",
        "T": "Tank/Tower",
        "R": "Reactor",
        "C": "Compressor",
        "D": "Drum",
        "F": "Filter",
        "K": "Compressor",
        "M": "Mixer",
        "B": "Boiler",
        "H": "Heater",
        "S": "Strainer/Screen",
        "X": "Special Equipment",
    }

    # Service codes
    SERVICE_CODE_MAP = {
        "PA": "Process Air",
        "CW": "Cooling Water",
        "IA": "Instrument Air",
        "ST": "Steam",
        "CD": "Condensate",
        "FW": "Firewater",
        "DW": "Drinking Water",
        "NG": "Natural Gas",
        "FG": "Fuel Gas",
        "N2": "Nitrogen",
        "HC": "Hydrocarbon",
        "EF": "Effluent",
        "DR": "Drain",
        "VT": "Vent",
    }

    # Two-letter ISA instrument prefixes that are real codes. Used to decide
    # whether an OCR'd trailing vowel is noise ("Re1001" -> R-1001) or part of
    # a legitimate prefix ("TE101" -> Temperature Element, TE-101).
    KNOWN_INSTRUMENT_PREFIXES = frozenset({
        "AE", "AI", "AR", "AS", "AT", "AV",
        "CE", "CI", "CR", "CS", "CT", "CV",
        "FE", "FI", "FC", "FQ", "FS", "FT", "FV",
        "HE", "HI", "HK", "HS", "HV",
        "IE", "II", "IR", "IS", "IT",
        "LE", "LI", "LR", "LS", "LT", "LV",
        "ME", "MF", "MI", "MR", "MS", "MT",
        "PE", "PI", "PQ", "PR", "PS", "PT", "PV",
        "SE", "SI", "SR", "SS", "ST", "SV",
        "TE", "TI", "TR", "TS", "TT", "TV",
        "VE", "VI", "VR", "VS", "VT",
        "XE", "XI", "XR",
    })

    # Common OCR character confusions in engineering tag context.
    # Note: there is deliberately NO rule stripping a trailing vowel from a
    # prefix ('Re' -> 'R'). It destroyed the legitimate ISA prefix 'TE'
    # (Temperature Element) and 'PS' (Pressure Switch).
    _OCR_CORRECTIONS = [
        # 'O' confused with '0', 'I'/'l' confused with '1'
        (re.compile(r'(?<=\d)O(?=\d|$)'), '0'),
        (re.compile(r'(?<=\d)I(?=\d|$)'), '1'),
        (re.compile(r'(?<=\d)l(?=\d|$)'), '1'),
        # Leading '+' or stray chars before known tag prefixes
        (re.compile(r'^[+\-./]+([A-Z])'), r'\1'),
        # Clean up spaces around dashes in tags
        (re.compile(r'([A-Z])\s+(\d)'), r'\1-\2'),
    ]

    def _normalize_ocr(self, text: str) -> str:
        """Apply known OCR correction rules for engineering tag text."""
        t = text.strip()
        for pattern, repl in self._OCR_CORRECTIONS:
            t = pattern.sub(repl, t)
        return t

    def parse(self, text: str) -> 'ParsedTag':
        """
        Parse a single text string into a structured engineering tag.

        Args:
            text: Raw OCR text to parse.

        Returns:
            ParsedTag with classification and structured fields.
        """
        text = self._normalize_ocr(text.strip())

        # Line numbers first: they contain digits and a service code, and
        # would otherwise be mis-read as a spec or a description.
        result = self._try_line_number(text)
        if result:
            return result

        # Material grades and ratings (CS, SS304, A105, 150#) before
        # instruments: "A105" otherwise parses as an analyser loop tag.
        result = self._try_specification(text)
        if result:
            return result

        # Equipment: a ONE letter prefix, and that letter must be a known
        # equipment letter. This is what un-shadows P/E/R/V/H/T tags.
        result = self._try_equipment_tag(text)
        if result:
            return result

        # Instruments: 2-4 letter ISA prefix.
        result = self._try_instrument_tag(text)
        if result:
            return result

        result = self._try_pipe_size(text)
        if result:
            return result

        # Default: classify as description or unknown
        if len(text) > 3:
            return ParsedTag(
                raw_text=text,
                tag_type=TagType.DESCRIPTION,
                tag_id=text,
                parsed_description=text,
            )

        return ParsedTag(
            raw_text=text,
            tag_type=TagType.UNKNOWN,
            tag_id=text,
        )

    def parse_batch(self, texts: List[str]) -> List[ParsedTag]:
        """Parse a list of text strings."""
        results = []
        for text in texts:
            results.append(self.parse(text))

        # Log summary
        type_counts = {}
        for r in results:
            type_counts[r.tag_type] = type_counts.get(r.tag_type, 0) + 1
        logger.info(f"Parsed {len(results)} tags: {dict(type_counts)}")

        return results

    def _try_instrument_tag(self, text: str) -> Optional[ParsedTag]:
        """
        Try to parse as an instrument tag (e.g., FIC-1001, PSV-201, PSHH).

        The prefix must be 2-4 letters, the first letter must be a known ISA
        measured variable, and every remaining letter must be a known ISA
        function code. That last rule is what stops prose such as "PUMP 101"
        or "LINE 12" from being manufactured into a tag.
        """
        match = self.INSTRUMENT_PATTERN.match(text.upper())
        if match:
            letters, number, suffix, second = (
                match.group(1), match.group(2), match.group(3), match.group(4)
            )
        else:
            # Letter-only codes: PSHH, LSH, TSH
            match = self.INSTRUMENT_LETTERS_ONLY_PATTERN.match(text.upper())
            if match:
                letters, number, suffix, second = match.group(1), None, None, None
            else:
                # Single-letter instrument tags: A-101 (analysis).
                match = self.SINGLE_LETTER_INSTRUMENT_PATTERN.match(text.upper())
                if not match:
                    return None
                letters, number, suffix, second = (
                    match.group(1), match.group(3), None, None
                )

        if not self._is_valid_instrument_prefix(letters):
            return None

        # First letter = primary measured variable
        primary = INSTRUMENT_TAG_MAP.get(letters[0], "Unknown")

        # Remaining letters = function(s)
        functions = []
        for char in letters[1:]:
            func = INSTRUMENT_FUNCTION_MAP.get(char)
            if func is None:
                return None
            functions.append(func)

        function_str = "/".join(functions) if functions else None
        description = f"{primary} {function_str}" if function_str else primary

        tag_id = f"{letters}-{number}" if number else letters
        if suffix:
            tag_id += suffix
        if second:
            tag_id += f"/{second}"

        return ParsedTag(
            raw_text=text,
            tag_type=TagType.INSTRUMENT_TAG,
            tag_id=tag_id,
            primary_variable=primary,
            function=function_str,
            loop_number=number,
            suffix=suffix,
            parsed_description=description,
        )

    def _is_valid_instrument_prefix(self, letters: str) -> bool:
        """
        Return True if `letters` is a plausible ISA instrument prefix.

        - 1 letter: a known measured variable that is NOT an equipment letter
          (e.g. ``A`` for analysis). Equipment letters are handled earlier.
        - 2+ letters: first letter a known measured variable, every remaining
          letter a known ISA function code.
        """
        if not letters:
            return False

        if letters[0] not in INSTRUMENT_TAG_MAP:
            return False

        if len(letters) == 1:
            # A single letter is ambiguous with equipment; only accept it when
            # the letter is not an equipment letter.
            return letters[0] not in self.EQUIPMENT_TYPE_MAP

        for char in letters[1:]:
            if char not in INSTRUMENT_FUNCTION_MAP:
                return False
        return True

    def _try_equipment_tag(self, text: str) -> Optional[ParsedTag]:
        """
        Try to parse as an equipment tag (e.g., P-101A, E-201 A/B).

        Only a single-letter prefix is accepted, and that letter must be a
        known equipment letter. Longer prefixes are ISA instrument codes and
        are handled by :meth:`_try_instrument_tag`.
        """
        match = self.EQUIPMENT_PATTERN.match(text.upper())
        if not match:
            return None

        # group(1)=prefix letter, group(2)=separator, group(3)=number,
        # group(4)=optional A/B suffix
        prefix, number, suffix = match.group(1), match.group(3), match.group(4)

        if prefix not in self.EQUIPMENT_TYPE_MAP:
            return None

        equip_type = self.EQUIPMENT_TYPE_MAP[prefix]
        tag_id = f"{prefix}-{number}"
        if suffix:
            tag_id += suffix.replace(" ", "")

        return ParsedTag(
            raw_text=text,
            tag_type=TagType.EQUIPMENT_TAG,
            tag_id=tag_id,
            equipment_type=equip_type,
            loop_number=number,
            suffix=suffix.replace(" ", "") if suffix else None,
            parsed_description=equip_type,
        )

    def _try_line_number(self, text: str) -> Optional[ParsedTag]:
        """Try to parse as a line number (e.g., 6"-PA-1001-A1A-N)."""
        match = self.LINE_NUMBER_PATTERN.match(text)
        if not match:
            return None

        size = match.group(1)
        service = match.group(2)
        number = match.group(3)
        spec_class = match.group(4)
        insulation = match.group(5)

        service_desc = self.SERVICE_CODE_MAP.get(
            service.upper(), f"Service ({service})"
        )

        tag_parts = [f'{size}"', service, number]
        if spec_class:
            tag_parts.append(spec_class)
        if insulation:
            tag_parts.append(insulation)

        return ParsedTag(
            raw_text=text,
            tag_type=TagType.LINE_NUMBER,
            tag_id="-".join(tag_parts),
            pipe_size=f'{size}"',
            service_code=service.upper(),
            loop_number=number,
            pipe_spec=spec_class,
            parsed_description=f'{size}" {service_desc} Line {number}',
        )

    def _try_pipe_size(self, text: str) -> Optional[ParsedTag]:
        """Try to parse as a pipe size (e.g., 2", 6")."""
        match = self.PIPE_SIZE_PATTERN.match(text)
        if not match:
            return None

        return ParsedTag(
            raw_text=text,
            tag_type=TagType.PIPE_SIZE,
            tag_id=text,
            pipe_size=f'{match.group(1)}"',
            parsed_description=f'{match.group(1)} inch pipe',
        )

    def _try_specification(self, text: str) -> Optional[ParsedTag]:
        """Try to parse as a specification (e.g., 150#, CS)."""
        match = self.SPEC_PATTERN.match(text)
        if not match:
            return None

        return ParsedTag(
            raw_text=text,
            tag_type=TagType.SPECIFICATION,
            tag_id=text,
            pipe_spec=text,
            parsed_description=f"Specification: {text}",
        )
