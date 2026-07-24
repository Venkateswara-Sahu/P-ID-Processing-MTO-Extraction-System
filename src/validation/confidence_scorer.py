"""
Confidence Scorer for P&ID extraction quality.

Computes per-entity and overall extraction confidence scores
based on detection quality, OCR accuracy, and tag parsing success.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from src.graph.entity_mapper import EngineeringEntity

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBreakdown:
    """Detailed confidence breakdown for an entity."""

    entity_id: str
    tag_id: str
    detection_score: float
    ocr_score: float
    tag_parse_score: float
    overall_score: float
    needs_review: bool

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "tag_id": self.tag_id or "N/A",
            "detection": round(self.detection_score, 3),
            "ocr": round(self.ocr_score, 3),
            "tag_parse": round(self.tag_parse_score, 3),
            "overall": round(self.overall_score, 3),
            "needs_review": self.needs_review,
        }


class ConfidenceScorer:
    """Calculates extraction confidence scores for entities."""

    def __init__(
        self,
        detection_weight: float = 0.4,
        ocr_weight: float = 0.35,
        parse_weight: float = 0.25,
        review_threshold: float = 0.6,
    ):
        """
        Args:
            detection_weight: Weight for symbol detection confidence.
            ocr_weight: Weight for OCR text confidence.
            parse_weight: Weight for tag parsing success.
            review_threshold: Entities below this score are flagged for review.
        """
        self.detection_weight = detection_weight
        self.ocr_weight = ocr_weight
        self.parse_weight = parse_weight
        self.review_threshold = review_threshold

    def score_entities(
        self,
        entities: List[EngineeringEntity],
    ) -> List[ConfidenceBreakdown]:
        """
        Calculate confidence scores for all entities.

        Args:
            entities: List of engineering entities.

        Returns:
            List of ConfidenceBreakdown objects.
        """
        breakdowns = []

        for entity in entities:
            breakdown = self._score_entity(entity)
            breakdowns.append(breakdown)

            # Update entity confidence
            entity.confidence = breakdown.overall_score

        # Log summary
        avg_confidence = (
            sum(b.overall_score for b in breakdowns) / len(breakdowns)
            if breakdowns
            else 0
        )
        needs_review = sum(1 for b in breakdowns if b.needs_review)

        logger.info(
            f"Confidence scoring: avg={avg_confidence:.3f}, "
            f"{needs_review}/{len(breakdowns)} need review"
        )

        return breakdowns

    def _score_entity(self, entity: EngineeringEntity) -> ConfidenceBreakdown:
        """Calculate confidence for a single entity."""
        # Detection score
        detection_score = 0.0
        if entity.symbol_detection:
            detection_score = entity.symbol_detection.confidence

        # OCR score (average of text regions)
        ocr_score = 0.0
        if entity.text_regions:
            ocr_score = sum(tr.confidence for tr in entity.text_regions) / len(
                entity.text_regions
            )

        # Tag parse score
        parse_score = 0.0
        if entity.tag_id:
            parse_score = 0.9  # Successfully parsed
        elif entity.text_regions:
            parse_score = 0.3  # Has text but couldn't parse

        # Weighted overall score
        if entity.symbol_detection and entity.text_regions:
            # Full pipeline (best case)
            overall = (
                self.detection_weight * detection_score
                + self.ocr_weight * ocr_score
                + self.parse_weight * parse_score
            )
        elif entity.symbol_detection:
            # Detection only
            overall = detection_score * 0.6
        elif entity.text_regions:
            # Text only
            overall = ocr_score * 0.5 + parse_score * 0.2
        else:
            overall = 0.0

        return ConfidenceBreakdown(
            entity_id=entity.entity_id,
            tag_id=entity.tag_id,
            detection_score=detection_score,
            ocr_score=ocr_score,
            tag_parse_score=parse_score,
            overall_score=overall,
            needs_review=overall < self.review_threshold,
        )

    def get_summary_stats(
        self,
        breakdowns: List[ConfidenceBreakdown],
    ) -> Dict:
        """Get summary statistics for confidence scores."""
        if not breakdowns:
            return {"total": 0}

        scores = [b.overall_score for b in breakdowns]

        return {
            "total_entities": len(breakdowns),
            "avg_confidence": round(sum(scores) / len(scores), 3),
            "min_confidence": round(min(scores), 3),
            "max_confidence": round(max(scores), 3),
            "needs_review": sum(1 for b in breakdowns if b.needs_review),
            "high_confidence": sum(1 for s in scores if s >= 0.8),
            "medium_confidence": sum(1 for s in scores if 0.6 <= s < 0.8),
            "low_confidence": sum(1 for s in scores if s < 0.6),
        }
