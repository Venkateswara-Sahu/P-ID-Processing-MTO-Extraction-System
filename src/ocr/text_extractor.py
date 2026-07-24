"""
Text Extractor for P&ID drawings using EasyOCR.

Extracts all text regions from P&ID images including instrument tags,
line numbers, equipment labels, notes, and specifications.
Returns structured text data with bounding boxes and confidence scores.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """A single text region detected by OCR."""

    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]
    polygon: List[List[float]]  # 4-point polygon

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox],
            "center": [round(v, 1) for v in self.center],
        }


class TextExtractor:
    """EasyOCR-based text extractor for P&ID drawings."""

    def __init__(
        self,
        language: str = None,
        confidence_threshold: float = None,
        use_gpu: bool = False,
    ):
        """
        Initialize the text extractor.

        Args:
            language: OCR language (default: 'en').
            confidence_threshold: Minimum OCR confidence to keep.
            use_gpu: Whether to use GPU (False for Intel Iris Xe).
        """
        self.language = language or settings.ocr_language
        self.confidence_threshold = confidence_threshold or settings.ocr_confidence_threshold
        self.use_gpu = use_gpu
        self._reader = None

    def _init_ocr(self):
        """Lazy-initialize EasyOCR engine."""
        import easyocr

        self._reader = easyocr.Reader(
            ["en"],
            gpu=self.use_gpu,
            verbose=False,
        )
        logger.info("EasyOCR initialized successfully")

    def extract(self, image: np.ndarray) -> List[TextRegion]:
        """
        Extract all text regions from an image.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            List of TextRegion objects.
        """
        if self._reader is None:
            self._init_ocr()

        # EasyOCR accepts BGR numpy arrays directly
        results = self._reader.readtext(
            image,
            paragraph=False,        # Keep individual text boxes
            min_size=10,            # Minimum text region size in pixels
            text_threshold=0.6,     # Text confidence threshold
            link_threshold=0.3,     # Link adjacent characters
            low_text=0.3,           # Low-bound text score
            width_ths=0.7,          # Width threshold for merging boxes
        )

        text_regions = []
        for bbox_points, text, confidence in results:
            if confidence < self.confidence_threshold:
                continue

            text = text.strip()
            if not text:
                continue

            # EasyOCR returns 4 corner points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            text_regions.append(
                TextRegion(
                    text=text,
                    confidence=float(confidence),
                    bbox=bbox,
                    center=center,
                    polygon=[[float(x), float(y)] for x, y in bbox_points],
                )
            )

        logger.info(f"Extracted {len(text_regions)} text regions (EasyOCR)")
        return text_regions

    def draw_text_regions(
        self,
        image: np.ndarray,
        text_regions: List[TextRegion],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 1,
        font_scale: float = 0.4,
    ) -> np.ndarray:
        """Draw text bounding boxes and labels on the image."""
        result = image.copy()

        for tr in text_regions:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            # Draw text label above the box
            label = f"{tr.text} ({tr.confidence:.2f})"
            cv2.putText(
                result,
                label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
            )

        return result
