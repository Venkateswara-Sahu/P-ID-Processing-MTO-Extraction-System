"""
Detection post-processor tests.

Covers the geometry of merging tiled detections back into whole-image
coordinates: clipping, degenerate boxes, and class-aware NMS.
"""
import numpy as np
import pytest

from src.detection.postprocessor import DetectionPostProcessor
from src.detection.symbol_detector import Detection
from src.preprocessing.image_tiler import Tile, TilingResult


def _det(cls=0, name="Gate Valve", conf=0.9, box=(0, 0, 60, 60)):
    x1, y1, x2, y2 = box
    return Detection(
        class_id=cls, class_name=name, confidence=conf,
        bbox_xyxy=box, center=((x1 + x2) / 2, (y1 + y2) / 2),
    )


def _tiling(width=500, height=500, offsets=((0, 0),)):
    tiles = [
        Tile(image=np.zeros((640, 640, 3), np.uint8), x_offset=ox, y_offset=oy,
             tile_width=640, tile_height=640, row=i, col=0)
        for i, (ox, oy) in enumerate(offsets)
    ]
    return TilingResult(
        tiles=tiles, original_width=width, original_height=height,
        tile_size=640, overlap=64, num_rows=len(tiles), num_cols=1,
    )


# --------------------------------------------------------------------------- #
# C9: clipping must not invert a box
# --------------------------------------------------------------------------- #

def test_detection_in_the_padded_region_does_not_invert():
    """
    The upper bound was clipped but the lower bound was not, so a detection
    in a tile's zero-padded region produced a box with negative width.
    """
    tiling = _tiling(width=500, height=500)
    det = _det(box=(520.0, 10.0, 600.0, 50.0))

    merged = DetectionPostProcessor().merge_tiled_detections([[det]], tiling)

    for m in merged:
        x1, y1, x2, y2 = m.bbox_xyxy
        assert x2 > x1, f"inverted box: width={x2 - x1}"
        assert y2 > y1, f"inverted box: height={y2 - y1}"
        assert m.area > 0, f"negative area: {m.area}"


def test_degenerate_boxes_are_dropped_entirely():
    """A box with no area cannot be a real symbol; it must not become one."""
    tiling = _tiling(width=500, height=500)
    det = _det(box=(520.0, 10.0, 600.0, 50.0))

    merged = DetectionPostProcessor().merge_tiled_detections([[det]], tiling)
    assert merged == [], (
        f"a fully-clipped detection should be dropped, got {merged}"
    )


def test_boxes_are_clipped_to_the_image_bounds():
    tiling = _tiling(width=500, height=500)
    det = _det(box=(400.0, 400.0, 900.0, 900.0))

    merged = DetectionPostProcessor().merge_tiled_detections([[det]], tiling)
    assert len(merged) == 1
    x1, y1, x2, y2 = merged[0].bbox_xyxy
    assert 0 <= x1 and 0 <= y1
    assert x2 <= 500 and y2 <= 500


def test_tile_offset_is_applied():
    tiling = _tiling(width=1200, height=500, offsets=((0, 0), (576, 0)))
    det = _det(box=(10.0, 10.0, 50.0, 50.0))

    merged = DetectionPostProcessor().merge_tiled_detections(
        [[], [det]], tiling
    )
    assert len(merged) == 1
    x1, y1, x2, y2 = merged[0].bbox_xyxy
    assert x1 == 576 + 10.0
    assert x2 == 576 + 50.0


# --------------------------------------------------------------------------- #
# C13: the same physical symbol must not survive as two MTO line items
# --------------------------------------------------------------------------- #

def test_class_agnostic_nms_merges_a_symbol_classified_two_ways():
    """
    The model has 82 near-synonymous valve classes, so a symbol on a tile
    boundary can be classified differently in each tile. Class-aware NMS
    alone keeps both, producing two MTO line items for one valve.
    """
    pp = DetectionPostProcessor()
    a = _det(cls=10, name="Gate_Valve", conf=0.9, box=(0, 0, 60, 60))
    b = _det(cls=11, name="Valve", conf=0.85, box=(2, 2, 62, 62))

    assert pp._compute_iou(a.bbox_xyxy, b.bbox_xyxy) > 0.8

    kept = pp._class_agnostic_nms([a, b])
    assert len(kept) == 1, (
        f"the same symbol was kept twice: {[d.class_name for d in kept]}"
    )
    # The higher-confidence detection should win.
    assert kept[0].confidence == 0.9


def test_class_agnostic_nms_keeps_genuinely_different_symbols():
    pp = DetectionPostProcessor()
    a = _det(cls=0, name="Gate_Valve", conf=0.9, box=(0, 0, 60, 60))
    b = _det(cls=1, name="Centrifugal Pump", conf=0.9, box=(200, 200, 260, 260))

    kept = pp._class_agnostic_nms([a, b])
    assert len(kept) == 2


def test_class_agnostic_nms_does_not_merge_adjacent_symbols():
    """Two valves side by side must stay two valves."""
    pp = DetectionPostProcessor()
    a = _det(cls=0, name="Gate_Valve", conf=0.9, box=(0, 0, 40, 40))
    b = _det(cls=1, name="Valve", conf=0.9, box=(45, 0, 85, 40))

    kept = pp._class_agnostic_nms([a, b])
    assert len(kept) == 2


def test_merge_applies_both_nms_passes():
    """
    The merge must run a class-agnostic pass before the class-aware one, so
    a boundary symbol classified two ways yields one detection.
    """
    # Two tiles overlapping in x: tile 0 covers 0-640, tile 1 covers 576-1216.
    # A symbol at global x=580-640 is fully visible in both. The same
    # physical symbol is detected with a slight jitter in each tile.
    tiling = _tiling(width=1200, height=500, offsets=((0, 0), (576, 0)))
    a = _det(cls=10, name="Gate_Valve", conf=0.9, box=(580.0, 100.0, 640.0, 160.0))
    b = _det(cls=11, name="Valve", conf=0.85, box=(4.0, 102.0, 64.0, 162.0))

    merged = DetectionPostProcessor().merge_tiled_detections([[a], [b]], tiling)
    assert len(merged) == 1, (
        f"expected 1 merged detection, got {len(merged)}: "
        f"{[d.class_name for d in merged]}"
    )
    assert merged[0].class_name == "Gate_Valve", (
        "the more confident detection should survive"
    )


# --------------------------------------------------------------------------- #
# NMS IoU math
# --------------------------------------------------------------------------- #

def test_iou_of_identical_boxes_is_one():
    pp = DetectionPostProcessor()
    box = (0.0, 0.0, 60.0, 60.0)
    assert pp._compute_iou(box, box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    pp = DetectionPostProcessor()
    assert pp._compute_iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0
