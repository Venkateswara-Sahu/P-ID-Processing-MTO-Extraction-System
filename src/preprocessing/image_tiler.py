"""
Image Tiler for large P&ID drawings.

P&ID drawings are typically A1/A0 size with very small symbols.
This module tiles large images into overlapping patches suitable
for YOLOv8 detection, then merges results back to original coordinates.
"""

import logging
import math
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
    # True when the corresponding tile edge is an INTERNAL cut rather than
    # the image border. A symbol touching an internal edge is sliced in
    # half by the boundary, so the detector sees a partial shape - which
    # is a false positive, not a detection.
    cut_left: bool = False
    cut_right: bool = False
    cut_top: bool = False
    cut_bottom: bool = False

    def cut_edges(self) -> Tuple[float, float, float, float]:
        """
        The tile's edge coordinates in ORIGINAL image space.

        Returns (left, top, right, bottom) where each value is the
        coordinate of that edge in the original image, or None-equivalent
        (a negative value) for edges that are not cuts.
        """
        left = self.x_offset if self.cut_left else -1.0
        top = self.y_offset if self.cut_top else -1.0
        right = (self.x_offset + self.tile_width) if self.cut_right else -1.0
        bottom = (self.y_offset + self.tile_height) if self.cut_bottom else -1.0
        return (left, top, right, bottom)


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
        self.tile_size = tile_size if tile_size is not None else settings.tile_size
        self.overlap = overlap if overlap is not None else settings.tile_overlap
        if not (0 <= self.overlap < self.tile_size):
            raise ValueError(
                f"overlap ({self.overlap}) must be in [0, tile_size={self.tile_size}); "
                "otherwise the tiling stride is zero or negative and tiling never ends"
            )
        self.stride = self.tile_size - self.overlap

    def _axis_starts(self, extent: int) -> List[int]:
        """
        Compute the tile start offsets along one axis.

        The number of tiles is derived from the stride, and the tiles are
        then spaced EVENLY across the axis. The previous approach kept
        the stride for every tile and back-shifted the final one to end
        flush with the image, which made that tile overlap the previous one
        by up to 639px -- roughly 90% redundant inference, and it doubled
        the tile count for some widths.

        Even spacing keeps the overlap uniform and never larger than the
        configured one by more than rounding.
        """
        if extent <= self.tile_size:
            return [0]

        # How many tiles are needed at the configured stride?
        n_tiles = math.ceil((extent - self.tile_size) / self.stride) + 1
        n_tiles = max(2, n_tiles)

        # Space them evenly from 0 to the flush position.
        flush = extent - self.tile_size
        spacing = flush / (n_tiles - 1)

        starts = []
        for i in range(n_tiles):
            start = int(round(i * spacing))
            if start not in starts:
                starts.append(start)
        return starts

    def tile(self, image: np.ndarray) -> TilingResult:
        """
        Split a large image into overlapping tiles for detection.

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
                # A single tile has no internal boundaries: every edge is
                # the image border, so nothing is ever "cut".
                cut_left=False,
                cut_right=False,
                cut_top=False,
                cut_bottom=False,
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

        x_starts = self._axis_starts(w)
        y_starts = self._axis_starts(h)

        tiles = []
        for row_idx, y_start in enumerate(y_starts):
            y_end = min(y_start + self.tile_size, h)
            for col_idx, x_start in enumerate(x_starts):
                x_end = min(x_start + self.tile_size, w)

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
                    # An edge is a CUT when the tile does not reach the
                    # image border on that side.
                    cut_left=x_start > 0,
                    cut_right=x_end < w,
                    cut_top=y_start > 0,
                    cut_bottom=y_end < h,
                )
                tiles.append(tile)

        result = TilingResult(
            tiles=tiles,
            original_width=w,
            original_height=h,
            tile_size=self.tile_size,
            overlap=self.overlap,
            num_rows=len(y_starts),
            num_cols=len(x_starts),
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
