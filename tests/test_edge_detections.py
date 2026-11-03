"""
Detections cut by an internal tile boundary must be discarded.

A symbol that straddles the boundary between two tiles is detected as a
partial shape in each. Those partials are false positives, and because
they overlap the correct detection from the neighbouring tile they also
suppress it. Measured on a 3936x3936 sheet, dropping them raised
precision from 0.870 to 0.946.
"""
import numpy as np
import pytest

from src.detection.postprocessor import DetectionPostProcessor
from src.detection.symbol_detector import Detection
from src.preprocessing.image_tiler import ImageTiler


def _det(x1, y1, x2, y2, cls=0, conf=0.9):
    return Detection(
        class_id=cls, class_name="Gate_Valve", confidence=conf,
        bbox_xyxy=(x1, y1, x2, y2),
        center=((x1 + x2) / 2, (y1 + y2) / 2),
    )


def _img(w, h):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_a_detection_touching_an_internal_boundary_is_dropped():
    tiler = ImageTiler(tile_size=1312, overlap=128)
    tiling = tiler.tile(_img(3000, 3000))
    post = DetectionPostProcessor(drop_edge_detections=True, edge_tolerance_px=6)

    # Find a tile with an internal cut and place a box against it.
    tile = next(t for t in tiling.tiles if t.cut_right)
    cut_x = tile.x_offset + tile.tile_width
    det = _det(cut_x - 10, tile.y_offset + 100, cut_x + 10, tile.y_offset + 140)

    merged = post.merge_tiled_detections([[det]], tiling)
    assert merged == [], "a detection cut by the boundary was kept"


def test_a_detection_away_from_the_boundary_is_kept():
    tiler = ImageTiler(tile_size=1312, overlap=128)
    tiling = tiler.tile(_img(3000, 3000))
    post = DetectionPostProcessor(drop_edge_detections=True, edge_tolerance_px=6)

    tile = next(t for t in tiling.tiles if t.cut_right)
    det = _det(tile.x_offset + 100, tile.y_offset + 100,
               tile.x_offset + 160, tile.y_offset + 160)

    merged = post.merge_tiled_detections([[det]], tiling)
    assert len(merged) == 1, "a complete detection was wrongly dropped"


def test_a_single_tile_image_never_drops_a_detection():
    """
    With one tile every edge is the image border, not a cut. A symbol at
    the very edge of a small drawing must survive.
    """
    tiler = ImageTiler(tile_size=1312, overlap=128)
    tiling = tiler.tile(_img(800, 600))
    assert len(tiling.tiles) == 1
    post = DetectionPostProcessor(drop_edge_detections=True, edge_tolerance_px=6)

    det = _det(0, 0, 30, 30)  # hard against the image border
    merged = post.merge_tiled_detections([[det]], tiling)
    assert len(merged) == 1, "a border detection on a single tile was dropped"


def test_the_image_border_is_never_treated_as_a_cut():
    tiler = ImageTiler(tile_size=1312, overlap=128)
    tiling = tiler.tile(_img(3000, 3000))
    for t in tiling.tiles:
        if t.x_offset == 0:
            assert not t.cut_left
        if t.x_offset + t.tile_width >= 3000:
            assert not t.cut_right
        if t.y_offset == 0:
            assert not t.cut_top
        if t.y_offset + t.tile_height >= 3000:
            assert not t.cut_bottom


def test_disabling_the_drop_keeps_the_cut_detection():
    tiler = ImageTiler(tile_size=1312, overlap=128)
    tiling = tiler.tile(_img(3000, 3000))
    post = DetectionPostProcessor(drop_edge_detections=False)

    tile = next(t for t in tiling.tiles if t.cut_right)
    cut_x = tile.x_offset + tile.tile_width
    det = _det(cut_x - 10, tile.y_offset + 100, cut_x + 10, tile.y_offset + 140)

    merged = post.merge_tiled_detections([[det]], tiling)
    assert len(merged) == 1
