"""
Image Tiler for large P&ID drawings.

P&ID drawings are typically A1/A0 size with very small symbols.
This module tiles large images into overlapping patches suitable
for YOLOv8 detection, then merges results back to original coordinates.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    """Represents a single tile cropped from the original image."""

    image: np.ndarray
    x_offset: int  # Top-left x in original image
    y_offset: int  # Top-left y in original image
    tile_width: int
    tile_height: int
    row: int
    col: int


@dataclass
class TilingResult:
    """Result of tiling an image."""

    tiles: List[Tile] = field(default_factory=list)
    original_width: int = 0
    original_height: int = 0
    tile_size: int = 0
    overlap: int = 0
    num_rows: int = 0
    num_cols: int = 0


class ImageTiler:
    """Tiles large P&ID images into overlapping patches for detection."""

    def __init__(self, tile_size: int = None, overlap: int = None):
        """
        Initialize the image tiler.

        Args:
            tile_size: Size of each tile in pixels (square tiles).
            overlap: Overlap between adjacent tiles in pixels.
        """
        self.tile_size = tile_size or settings.tile_size
        self.overlap = overlap or settings.tile_overlap
        self.stride = self.tile_size - self.overlap

    def tile(self, image: np.ndarray) -> TilingResult:
        """
        Split a large image into overlapping tiles.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            TilingResult containing all tiles and metadata.
        """
        h, w = image.shape[:2]
        logger.info(f"Tiling image ({w}x{h}) into {self.tile_size}x{self.tile_size} patches "
                     f"with {self.overlap}px overlap")

        # If image is smaller than tile size, return as single tile
        if w <= self.tile_size and h <= self.tile_size:
            logger.info("Image fits in single tile — no tiling needed")
            tile = Tile(
                image=image.copy(),
                x_offset=0,
                y_offset=0,
                tile_width=w,
                tile_height=h,
                row=0,
                col=0,
            )
            return TilingResult(
                tiles=[tile],
                original_width=w,
                original_height=h,
                tile_size=self.tile_size,
                overlap=self.overlap,
                num_rows=1,
                num_cols=1,
            )

        tiles = []
        row_idx = 0

        y = 0
        while y < h:
            col_idx = 0
            x = 0

            while x < w:
                # Calculate crop boundaries
                x_end = min(x + self.tile_size, w)
                y_end = min(y + self.tile_size, h)
                x_start = max(0, x_end - self.tile_size)
                y_start = max(0, y_end - self.tile_size)

                # Crop tile
                tile_img = image[y_start:y_end, x_start:x_end].copy()

                # Pad if tile is smaller than expected (edge case)
                if tile_img.shape[0] < self.tile_size or tile_img.shape[1] < self.tile_size:
                    padded = np.zeros(
                        (self.tile_size, self.tile_size, image.shape[2]),
                        dtype=image.dtype,
                    )
                    padded[: tile_img.shape[0], : tile_img.shape[1]] = tile_img
                    tile_img = padded

                tile = Tile(
                    image=tile_img,
                    x_offset=x_start,
                    y_offset=y_start,
                    tile_width=x_end - x_start,
                    tile_height=y_end - y_start,
                    row=row_idx,
                    col=col_idx,
                )
                tiles.append(tile)

                col_idx += 1
                x += self.stride
                if x_end >= w:
                    break

            row_idx += 1
            y += self.stride
            if y_end >= h:
                break

        result = TilingResult(
            tiles=tiles,
            original_width=w,
            original_height=h,
            tile_size=self.tile_size,
            overlap=self.overlap,
            num_rows=row_idx,
            num_cols=col_idx,
        )

        logger.info(f"Generated {len(tiles)} tiles ({result.num_rows} rows × {result.num_cols} cols)")
        return result

    @staticmethod
    def tile_to_original_coords(
        bbox: Tuple[float, float, float, float],
        tile: Tile,
    ) -> Tuple[float, float, float, float]:
        """
        Convert bounding box from tile coordinates to original image coordinates.

        Args:
            bbox: Bounding box in tile space (x1, y1, x2, y2).
            tile: The tile the detection came from.

        Returns:
            Bounding box in original image coordinates (x1, y1, x2, y2).
        """
        x1, y1, x2, y2 = bbox
        return (
            x1 + tile.x_offset,
            y1 + tile.y_offset,
            x2 + tile.x_offset,
            y2 + tile.y_offset,
        )

    @staticmethod
    def stitch_visualization(
        tiles: List[Tile],
        original_width: int,
        original_height: int,
    ) -> np.ndarray:
        """
        Stitch tiles back into the original image (for visualization).

        Args:
            tiles: List of Tile objects.
            original_width: Width of the original image.
            original_height: Height of the original image.

        Returns:
            Stitched image as numpy array.
        """
        canvas = np.zeros((original_height, original_width, 3), dtype=np.uint8)

        for tile in tiles:
            h = tile.tile_height
            w = tile.tile_width
            canvas[
                tile.y_offset: tile.y_offset + h,
                tile.x_offset: tile.x_offset + w,
            ] = tile.image[:h, :w]

        return canvas
