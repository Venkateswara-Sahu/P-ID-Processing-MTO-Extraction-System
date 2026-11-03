"""
Entity Mapper for P&ID components.

Associates detected symbols with their nearest OCR text regions
using spatial proximity matching. Creates unified engineering
entities that combine visual detection with text identification.

The text->symbol assignment is **global**, not greedy: every candidate
(symbol, label) pair is scored by distance and the globally closest pairs
win. A greedy pass let a symbol 290px away claim a label that belonged to
a symbol 10px away, purely because it was processed first.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.detection.symbol_detector import Detection
from src.graph.class_map import classify_class_name, entity_type_for
from src.ocr.text_extractor import TextRegion
from src.ocr.tag_parser import TagParser, ParsedTag, TagType

logger = logging.getLogger(__name__)


@dataclass
class EngineeringEntity:
    """
    A unified engineering entity combining detected symbol and OCR text.
    Represents a single component on the P&ID (valve, instrument, pump, etc.)
    """

    entity_id: str                          # Unique identifier
    symbol_detection: Optional[Detection]   # Visual detection data
    text_regions: List[TextRegion] = field(default_factory=list)
    parsed_tags: List[ParsedTag] = field(default_factory=list)

    # Derived fields
    tag_id: Optional[str] = None            # Primary tag (e.g., "FV-101")
    entity_type: Optional[str] = None       # valve, instrument, equipment, etc.
    description: Optional[str] = None       # Human-readable description
    position: Optional[Tuple[float, float]] = None  # Center position on drawing
    confidence: float = 0.0                 # Overall extraction confidence

    # MTO-relevant fields
    component_category: Optional[str] = None  # For MTO grouping
    size: Optional[str] = None               # Pipe/equipment size
    specification: Optional[str] = None      # Material/pressure spec
    line_number: Optional[str] = None        # Associated line number

    def to_dict(self) -> dict:
        result = {
            "entity_id": self.entity_id,
            "tag_id": self.tag_id,
            "entity_type": self.entity_type,
            "description": self.description,
            "position": self.position,
            "confidence": round(self.confidence, 4),
            "component_category": self.component_category,
            "size": self.size,
            "specification": self.specification,
            "line_number": self.line_number,
        }
        if self.symbol_detection:
            result["symbol"] = self.symbol_detection.to_dict()
        result["text_count"] = len(self.text_regions)
        return {k: v for k, v in result.items() if v is not None}


class EntityMapper:
    """Maps detected symbols to OCR text using spatial proximity."""

    def __init__(self, max_distance: float = 350.0, max_labels_per_symbol: int = 3):
        """
        Args:
            max_distance: Maximum pixel distance between a symbol center
                         and a text region center for association.
                         NOTE: This is in the coordinate space of the processed
                         (upscaled) image. At 3x upscale a 100px gap becomes 300px.
            max_labels_per_symbol: How many labels one symbol may carry.
                         Labels are attached closest-first, and only after the
                         global assignment has decided who owns them.
        """
        self.max_distance = max_distance
        self.max_labels_per_symbol = max_labels_per_symbol
        self.tag_parser = TagParser()
        self._entity_counter = 0

    def map_entities(
        self,
        detections: List[Detection],
        text_regions: List[TextRegion],
    ) -> List[EngineeringEntity]:
        """
        Create unified engineering entities by matching symbols with text.

        Args:
            detections: Symbol detections from YOLOv8.
            text_regions: Text regions from the OCR stage.

        Returns:
            List of EngineeringEntity objects.
        """
        logger.info(
            f"Mapping {len(detections)} symbols with {len(text_regions)} text regions"
        )

        # The mapper is reused across pipeline runs, so reset the counter to
        # keep entity ids starting at E0001 each time.
        self._entity_counter = 0

        # Parse all text regions
        parsed_tags = [self.tag_parser.parse(tr.text) for tr in text_regions]

        # Global nearest-neighbour assignment: build every candidate pair,
        # sort by distance, and let the closest pairs claim their labels.
        assignments = self._assign_text_to_symbols(detections, text_regions)

        entities = []
        for det_idx, det in enumerate(detections):
            entity = self._create_entity_from_detection(det)

            for text_idx in assignments.get(det_idx, []):
                text_region = text_regions[text_idx]
                parsed_tag = parsed_tags[text_idx]
                entity.text_regions.append(text_region)
                entity.parsed_tags.append(parsed_tag)

                # Use the first instrument/equipment tag as the primary tag
                if entity.tag_id is None and parsed_tag.tag_type in (
                    TagType.INSTRUMENT_TAG,
                    TagType.EQUIPMENT_TAG,
                ):
                    entity.tag_id = parsed_tag.tag_id
                    entity.description = parsed_tag.parsed_description

                    # The tag is stronger evidence than the symbol class.
                    # The model called the symbol "Gate_Valve" but the
                    # drawing says E-1003, which is a heat exchanger; the
                    # MTO must not call it a valve.
                    if parsed_tag.tag_type == TagType.EQUIPMENT_TAG:
                        entity.entity_type = "equipment"
                        entity.component_category = "equipment"
                    else:
                        entity.entity_type = "instrument"
                        entity.component_category = "instruments"

                # Size and spec come from the component's OWN label. A line
                # number carries the size of the PIPE on that line, not the
                # size of the valve sitting on it, so it is never inherited.
                if parsed_tag.pipe_size and entity.size is None \
                        and parsed_tag.tag_type == TagType.PIPE_SIZE:
                    entity.size = parsed_tag.pipe_size
                if parsed_tag.pipe_spec and entity.specification is None \
                        and parsed_tag.tag_type == TagType.SPECIFICATION:
                    entity.specification = parsed_tag.pipe_spec

            entity.confidence = self._calculate_confidence(entity)
            entities.append(entity)

        # Step 2: Create entities for unmatched text (standalone labels).
        # A LINE_NUMBER is a property of a line, not a procurable item, so
        # it is never turned into an MTO row.
        assigned_text = {
            text_idx
            for indices in assignments.values()
            for text_idx in indices
        }
        for i, (text_region, parsed_tag) in enumerate(zip(text_regions, parsed_tags)):
            if i in assigned_text:
                continue
            if parsed_tag.tag_type in (
                TagType.INSTRUMENT_TAG,
                TagType.EQUIPMENT_TAG,
            ):
                entities.append(self._create_entity_from_text(text_region, parsed_tag))

        logger.info(
            f"Created {len(entities)} engineering entities "
            f"({sum(1 for e in entities if e.tag_id is not None)} with tags)"
        )

        return entities

    # ------------------------------------------------------------------ #
    # Global assignment                                                      #
    # ------------------------------------------------------------------ #

    def _assign_text_to_symbols(
        self,
        detections: List[Detection],
        text_regions: List[TextRegion],
    ) -> Dict[int, List[int]]:
        """
        Assign text regions to symbols globally, closest pairs first.

        Returns:
            Dict mapping detection index -> list of text region indices
            (ordered closest-first).
        """
        # Every candidate (symbol, label) pair within range.
        candidates = []
        for det_idx, det in enumerate(detections):
            for text_idx, text_region in enumerate(text_regions):
                distance = self._euclidean_distance(det.center, text_region.center)
                if distance <= self.max_distance:
                    candidates.append((distance, det_idx, text_idx))

        # Closest pairs first. Ties break on index so the result is
        # deterministic and independent of the input ordering.
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))

        assignments: Dict[int, List[int]] = {}
        used_symbols: set = set()
        used_texts: set = set()

        # First pass: one label per symbol, the globally closest.
        for distance, det_idx, text_idx in candidates:
            if det_idx in used_symbols or text_idx in used_texts:
                continue
            assignments.setdefault(det_idx, []).append(text_idx)
            used_symbols.add(det_idx)
            used_texts.add(text_idx)

        # Second pass: allow a symbol to take additional nearby labels
        # (e.g. a tag and a size on the same line), still closest-first.
        for distance, det_idx, text_idx in candidates:
            if text_idx in used_texts:
                continue
            current = assignments.setdefault(det_idx, [])
            if len(current) >= self.max_labels_per_symbol:
                continue
            current.append(text_idx)
            used_texts.add(text_idx)

        # Keep each symbol's labels in distance order.
        for det_idx, indices in assignments.items():
            indices.sort(
                key=lambda ti: self._euclidean_distance(
                    detections[det_idx].center, text_regions[ti].center
                )
            )

        return assignments

    # ------------------------------------------------------------------ #
    # Entity construction                                                   #
    # ------------------------------------------------------------------ #

    def _create_entity_from_detection(self, detection: Detection) -> EngineeringEntity:
        """Create an entity from a symbol detection."""
        self._entity_counter += 1
        entity_id = f"E{self._entity_counter:04d}"

        # Classify using the model's own vocabulary. A previous substring
        # heuristic left 43.7% of the 229 real class names as 'unknown'.
        category = classify_class_name(detection.class_name)
        entity_type = entity_type_for(category)

        return EngineeringEntity(
            entity_id=entity_id,
            symbol_detection=detection,
            entity_type=entity_type,
            component_category=category,
            position=detection.center,
            description=detection.class_name,
        )

    def _create_entity_from_text(
        self,
        text_region: TextRegion,
        parsed_tag: ParsedTag,
    ) -> EngineeringEntity:
        """Create an entity from standalone text (no associated symbol)."""
        self._entity_counter += 1
        entity_id = f"E{self._entity_counter:04d}"

        entity_type = "unknown"
        category = "other"

        if parsed_tag.tag_type == TagType.INSTRUMENT_TAG:
            entity_type = "instrument"
            category = "instruments"
        elif parsed_tag.tag_type == TagType.EQUIPMENT_TAG:
            entity_type = "equipment"
            category = "equipment"

        return EngineeringEntity(
            entity_id=entity_id,
            symbol_detection=None,
            text_regions=[text_region],
            parsed_tags=[parsed_tag],
            tag_id=parsed_tag.tag_id,
            entity_type=entity_type,
            component_category=category,
            description=parsed_tag.parsed_description,
            position=text_region.center,
            confidence=text_region.confidence * 0.7,  # Lower confidence without symbol
            size=parsed_tag.pipe_size,
            specification=parsed_tag.pipe_spec,
        )

    @staticmethod
    def _euclidean_distance(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def _calculate_confidence(entity: EngineeringEntity) -> float:
        """Calculate overall extraction confidence for an entity."""
        scores = []

        # Symbol detection confidence
        if entity.symbol_detection:
            scores.append(entity.symbol_detection.confidence)

        # OCR confidence (average)
        if entity.text_regions:
            ocr_avg = sum(tr.confidence for tr in entity.text_regions) / len(
                entity.text_regions
            )
            scores.append(ocr_avg)

        # Tag parsing bonus (successful parse = higher confidence)
        if entity.tag_id:
            scores.append(0.9)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)
