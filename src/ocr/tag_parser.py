"""
Engineering Tag Parser for P&ID text.

Parses OCR-extracted text into structured engineering entities
following ISA (International Society of Automation) standards
and common P&ID naming conventions.
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

    # Instrument tags: FV-101, PI-202A, TT-301/302, FIC-401
    INSTRUMENT_PATTERN = re.compile(
        r"^([A-Z]{1,4})[- ]?(\d{2,5})([A-Z])?(?:\s*/\s*(\d{2,5}))?$",
        re.IGNORECASE,
    )

    # Equipment tags: P-101, E-201 A/B, V-301, R-101
    EQUIPMENT_PATTERN = re.compile(
        r"^([A-Z]{1,3})[- ]?(\d{2,5})\s*([A-Z](?:\s*/\s*[A-Z])?)?$",
        re.IGNORECASE,
    )

    # Line numbers: 6"-PA-1001-A1A-N, 2"-CW-2001-B1B
    LINE_NUMBER_PATTERN = re.compile(
        r'^(\d{1,2})["\u201d]?\s*[-]?\s*([A-Z]{1,4})\s*[-]\s*(\d{3,5})\s*[-]?\s*([A-Z0-9]{1,4})?\s*[-]?\s*([A-Z])?$',
        re.IGNORECASE,
    )

    # Pipe size: 2", 6", 10", 3/4"
    PIPE_SIZE_PATTERN = re.compile(
        r'^(\d{1,2}(?:/\d{1,2})?)\s*["\u201d]?$',
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

    # Common OCR character confusions in engineering tag context
    _OCR_CORRECTIONS = [
        # Extra vowel appended to single-letter prefix: 'Re' → 'R', 'Ee' → 'E' etc.
        (re.compile(r'^([RTEPCVDBFKHMSX])e(\d)', re.IGNORECASE), r'\1\2'),
        (re.compile(r'^([RTEPCVDBFKHMSX])o(\d)', re.IGNORECASE), r'\1\2'),
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

        # Try each pattern in order of specificity
        result = self._try_instrument_tag(text)
        if result:
            return result


        result = self._try_line_number(text)
        if result:
            return result

        result = self._try_equipment_tag(text)
        if result:
            return result

        result = self._try_pipe_size(text)
        if result:
            return result

        result = self._try_specification(text)
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
        """Try to parse as an instrument tag (e.g., FV-101, PI-202)."""
        match = self.INSTRUMENT_PATTERN.match(text.upper())
        if not match:
            return None

        letters = match.group(1)
        number = match.group(2)
        suffix = match.group(3)

        # First letter = primary measured variable
        primary = INSTRUMENT_TAG_MAP.get(letters[0], "Unknown")

        # Remaining letters = function(s)
        functions = []
        for char in letters[1:]:
            func = INSTRUMENT_FUNCTION_MAP.get(char, char)
            functions.append(func)

        function_str = "/".join(functions) if functions else None
        description = f"{primary} {function_str}" if function_str else primary

        return ParsedTag(
            raw_text=text,
            tag_type=TagType.INSTRUMENT_TAG,
            tag_id=f"{letters}-{number}{suffix or ''}",
            primary_variable=primary,
            function=function_str,
            loop_number=number,
            suffix=suffix,
            parsed_description=description,
        )

    def _try_equipment_tag(self, text: str) -> Optional[ParsedTag]:
        """Try to parse as an equipment tag (e.g., P-101, E-201 A/B)."""
        match = self.EQUIPMENT_PATTERN.match(text.upper())
        if not match:
            return None

        prefix = match.group(1)
        number = match.group(2)
        suffix = match.group(3)

        # Don't confuse with instrument tags (multi-letter prefixes)
        if len(prefix) > 2:
            return None

        equip_type = self.EQUIPMENT_TYPE_MAP.get(prefix, f"Equipment ({prefix})")

        return ParsedTag(
            raw_text=text,
            tag_type=TagType.EQUIPMENT_TAG,
            tag_id=f"{prefix}-{number}{' ' + suffix if suffix else ''}",
            equipment_type=equip_type,
            loop_number=number,
            suffix=suffix,
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
