"""
PDF to Image Converter for P&ID documents.

Converts multi-page P&ID PDFs into high-resolution PNG images
suitable for downstream symbol detection and OCR processing.
Uses PyMuPDF (fitz) as the primary engine for speed and quality.
"""

import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from config.settings import settings

logger = logging.getLogger(__name__)


class PDFConverter:
    """Converts P&ID PDF documents to high-resolution images."""

    def __init__(self, dpi: int = None):
        """
        Initialize the PDF converter.

        Args:
            dpi: Resolution for rendering. Higher DPI = better quality but larger images.
                 Default uses config setting (300 DPI).
        """
        self.dpi = dpi or settings.pdf_dpi
        # PyMuPDF uses a zoom matrix based on DPI (72 DPI is the base)
        self.zoom = self.dpi / 72.0

    def convert(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """
        Convert a PDF file to PNG images (one per page).

        Args:
            pdf_path: Path to the input PDF file.
            output_dir: Directory to save images. If None, saves alongside the PDF.

        Returns:
            List of paths to the generated PNG images.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if output_dir is None:
            output_dir = pdf_path.parent / f"{pdf_path.stem}_images"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting PDF: {pdf_path} at {self.dpi} DPI")

        image_paths = []
        doc = fitz.open(str(pdf_path))

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Render page at specified DPI
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Save as PNG
            output_path = output_dir / f"{pdf_path.stem}_page_{page_num + 1}.png"
            pix.save(str(output_path))

            image_paths.append(str(output_path))
            logger.info(
                f"  Page {page_num + 1}/{len(doc)}: "
                f"{pix.width}x{pix.height}px → {output_path.name}"
            )

        doc.close()
        logger.info(f"Converted {len(image_paths)} pages from {pdf_path.name}")
        return image_paths

    def convert_to_numpy(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert a PDF directly to numpy arrays (no disk write).

        Args:
            pdf_path: Path to the input PDF file.

        Returns:
            List of numpy arrays (BGR format, compatible with OpenCV).
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        images = []
        doc = fitz.open(str(pdf_path))

        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Convert pixmap to numpy array (RGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            # Convert RGB → BGR for OpenCV compatibility
            img_bgr = img[:, :, ::-1].copy()
            images.append(img_bgr)

        doc.close()
        return images

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Load a single image file as a numpy array.

        Args:
            image_path: Path to the image file (PNG, JPG, TIFF, etc.)

        Returns:
            Numpy array in BGR format (OpenCV compatible).
        """
        import cv2

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        logger.info(f"Loaded image: {image_path.name} ({img.shape[1]}x{img.shape[0]}px)")
        return img
