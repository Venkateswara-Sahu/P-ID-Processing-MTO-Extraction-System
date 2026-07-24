"""
Image Enhancement for P&ID drawings.

Engineering drawings often have low contrast, noise from scanning,
and inconsistent line weights. This module provides preprocessing
to improve detection and OCR quality.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """Enhances P&ID images for better detection and OCR accuracy."""

    def __init__(
        self,
        clip_limit: float = None,
        grid_size: int = None,
    ):
        """
        Initialize the image enhancer.

        Args:
            clip_limit: CLAHE clip limit for contrast enhancement.
            grid_size: CLAHE grid size.
        """
        self.clip_limit = clip_limit or settings.clahe_clip_limit
        self.grid_size = grid_size or settings.clahe_grid_size

    def enhance(
        self,
        image: np.ndarray,
        apply_clahe: bool = True,
        apply_denoise: bool = True,
        apply_sharpen: bool = False,
        apply_binarize: bool = False,
    ) -> np.ndarray:
        """
        Apply a chain of enhancements to the image.

        Args:
            image: Input BGR image.
            apply_clahe: Apply contrast-limited adaptive histogram equalization.
            apply_denoise: Apply non-local means denoising.
            apply_sharpen: Apply unsharp mask sharpening.
            apply_binarize: Apply adaptive binarization (for OCR).

        Returns:
            Enhanced image as numpy array.
        """
        result = image.copy()

        if apply_clahe:
            result = self._apply_clahe(result)

        if apply_denoise:
            result = self._apply_denoise(result)

        if apply_sharpen:
            result = self._apply_sharpen(result)

        if apply_binarize:
            result = self._apply_adaptive_binarize(result)

        return result

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        logger.debug("Applying CLAHE contrast enhancement")

        # Convert to LAB color space for luminance-only enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.grid_size, self.grid_size),
        )
        l_enhanced = clahe.apply(l_channel)

        # Merge back and convert to BGR
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return result

    def _apply_denoise(self, image: np.ndarray, strength: int = 6) -> np.ndarray:
        """Apply fast non-local means denoising."""
        logger.debug("Applying denoising")

        # fastNlMeansDenoisingColored is effective for scanned documents
        # Use positional args for OpenCV 4.10+ compatibility
        result = cv2.fastNlMeansDenoisingColored(
            image, None, strength, strength, 7, 21
        )

        return result

    def _apply_sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply unsharp mask sharpening to enhance line edges."""
        logger.debug("Applying sharpening")

        # Gaussian blur to create the "unsharp" version
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)

        # Unsharp mask: original + (original - blurred) * amount
        result = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

        return result

    def _apply_adaptive_binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive binarization for cleaner OCR.
        Converts to grayscale and applies adaptive thresholding.
        """
        logger.debug("Applying adaptive binarization")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold works better than global for P&IDs
        # because different areas may have different background brightness
        binary = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )

        # Convert back to 3-channel for consistency
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        return result

    @staticmethod
    def resize_for_display(
        image: np.ndarray,
        max_dim: int = 1280,
    ) -> np.ndarray:
        """
        Resize image for display while maintaining aspect ratio.

        Args:
            image: Input image.
            max_dim: Maximum dimension (width or height).

        Returns:
            Resized image.
        """
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image

        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def draw_grid_overlay(
        image: np.ndarray,
        grid_size: int = 640,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1,
    ) -> np.ndarray:
        """Draw a grid overlay showing tile boundaries (for debugging)."""
        result = image.copy()
        h, w = result.shape[:2]

        for x in range(0, w, grid_size):
            cv2.line(result, (x, 0), (x, h), color, thickness)
        for y in range(0, h, grid_size):
            cv2.line(result, (0, y), (w, y), color, thickness)

        return result
