"""
Entity Mapper for P&ID components.

Associates detected symbols with their nearest OCR text regions
using spatial proximity matching. Creates unified engineering
entities that combine visual detection with text identification.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from src.detection.symbol_detector import Detection
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

    def __init__(self, max_distance: float = 350.0):
        """
        Args:
            max_distance: Maximum pixel distance between a symbol center
                         and a text region center for association.
                         NOTE: This is in the coordinate space of the processed
                         (upscaled) image. At 3x upscale a 100px gap becomes 300px.
        """
        self.max_distance = max_distance
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
            text_regions: Text regions from PaddleOCR.

        Returns:
            List of EngineeringEntity objects.
        """
        logger.info(
            f"Mapping {len(detections)} symbols with {len(text_regions)} text regions"
        )

        # Parse all text regions
        parsed_tags = [self.tag_parser.parse(tr.text) for tr in text_regions]

        # Track which text regions have been assigned
        assigned_text = set()

        entities = []

        # Step 1: Create entities for each detected symbol
        for det in detections:
            entity = self._create_entity_from_detection(det)

            # Find nearest text regions
            nearest_texts = self._find_nearest_texts(
                det.center, text_regions, parsed_tags, assigned_text
            )

            for idx, (text_region, parsed_tag, distance) in enumerate(nearest_texts):
                entity.text_regions.append(text_region)
                entity.parsed_tags.append(parsed_tag)
                assigned_text.add(id(text_region))

                # Use the first instrument/equipment tag as the primary tag
                if entity.tag_id is None and parsed_tag.tag_type in (
                    TagType.INSTRUMENT_TAG,
                    TagType.EQUIPMENT_TAG,
                ):
                    entity.tag_id = parsed_tag.tag_id
                    entity.description = parsed_tag.parsed_description

                # Extract size and spec info
                if parsed_tag.pipe_size and entity.size is None:
                    entity.size = parsed_tag.pipe_size
                if parsed_tag.pipe_spec and entity.specification is None:
                    entity.specification = parsed_tag.pipe_spec

            # Calculate confidence
            entity.confidence = self._calculate_confidence(entity)
            entities.append(entity)

        # Step 2: Create entities for unmatched text (standalone labels)
        for i, (text_region, parsed_tag) in enumerate(zip(text_regions, parsed_tags)):
            if id(text_region) not in assigned_text:
                if parsed_tag.tag_type in (
                    TagType.INSTRUMENT_TAG,
                    TagType.EQUIPMENT_TAG,
                    TagType.LINE_NUMBER,
                ):
                    entity = self._create_entity_from_text(text_region, parsed_tag)
                    entities.append(entity)

        logger.info(
            f"Created {len(entities)} engineering entities "
            f"({sum(1 for e in entities if e.tag_id is not None)} with tags)"
        )

        return entities

    def _create_entity_from_detection(self, detection: Detection) -> EngineeringEntity:
        """Create an entity from a symbol detection."""
        self._entity_counter += 1
        entity_id = f"E{self._entity_counter:04d}"

        # Determine entity type from class name
        class_name = detection.class_name.lower()
        entity_type = "unknown"
        category = "other"

        if "valve" in class_name:
            entity_type = "valve"
            category = "valves"
        elif "instrument" in class_name or any(
            kw in class_name for kw in ["transmitter", "indicator", "controller"]
        ):
            entity_type = "instrument"
            category = "instruments"
        elif any(
            kw in class_name
            for kw in ["pump", "compressor", "exchanger", "vessel", "tank", "reactor"]
        ):
            entity_type = "equipment"
            category = "equipment"
        elif any(
            kw in class_name
            for kw in ["reducer", "tee", "elbow", "flange", "strainer"]
        ):
            entity_type = "piping_component"
            category = "piping"

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
        elif parsed_tag.tag_type == TagType.LINE_NUMBER:
            entity_type = "line"
            category = "piping"

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

    def _find_nearest_texts(
        self,
        symbol_center: Tuple[float, float],
        text_regions: List[TextRegion],
        parsed_tags: List[ParsedTag],
        assigned: set,
        max_results: int = 5,
    ) -> List[Tuple[TextRegion, ParsedTag, float]]:
        """
        Find the nearest unassigned text regions to a symbol.

        Returns:
            List of (TextRegion, ParsedTag, distance) tuples, sorted by distance.
        """
        candidates = []

        for text_region, parsed_tag in zip(text_regions, parsed_tags):
            if id(text_region) in assigned:
                continue

            distance = self._euclidean_distance(symbol_center, text_region.center)
            if distance <= self.max_distance:
                candidates.append((text_region, parsed_tag, distance))

        # Sort by distance (nearest first)
        candidates.sort(key=lambda x: x[2])

        return candidates[:max_results]

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
