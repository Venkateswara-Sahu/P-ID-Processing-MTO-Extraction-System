"""
Image tiler tests.

Covers grid geometry, coverage, and the overlap validation that prevents
an infinite loop.
"""
import math

import numpy as np
import pytest

from src.preprocessing.image_tiler import ImageTiler


def _img(w, h):
    return np.zeros((h, w, 3), np.uint8)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_overlap_must_be_smaller_than_the_tile():
    """overlap == tile_size makes the stride zero and tiling never ends."""
    with pytest.raises(ValueError):
        ImageTiler(tile_size=640, overlap=640)


def test_overlap_greater_than_the_tile_is_rejected():
    with pytest.raises(ValueError):
        ImageTiler(tile_size=640, overlap=700)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError):
        ImageTiler(tile_size=640, overlap=-1)


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("w,h", [
    (64, 64), (640, 640), (641, 641), (1217, 800), (1920, 1163),
    (2369, 1500), (4000, 3000), (5000, 400),
])
def test_every_pixel_is_covered(w, h):
    """The union of the tiles must cover the whole image."""
    tiler = ImageTiler()
    result = tiler.tile(_img(w, h))

    covered = np.zeros((h, w), dtype=bool)
    for t in result.tiles:
        y0 = max(0, t.y_offset)
        x0 = max(0, t.x_offset)
        y1 = min(h, t.y_offset + t.tile_height)
        x1 = min(w, t.x_offset + t.tile_width)
        if y1 > y0 and x1 > x0:
            covered[y0:y1, x0:x1] = True

    uncovered = int((~covered).sum())
    assert uncovered == 0, f"{uncovered} pixels are not covered by any tile"


def test_a_small_image_is_a_single_tile():
    result = ImageTiler().tile(_img(100, 100))
    assert len(result.tiles) == 1


def test_the_last_tile_reaches_the_far_edge():
    result = ImageTiler().tile(_img(1920, 1163))
    max_x = max(t.x_offset + t.tile_width for t in result.tiles)
    max_y = max(t.y_offset + t.tile_height for t in result.tiles)
    assert max_x >= 1920
    assert max_y >= 1163


# --------------------------------------------------------------------------- #
# Grid geometry
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("w", [1217, 1920, 2369, 4000, 5000])
def test_no_tile_is_a_near_duplicate_of_its_neighbour(w):
    """
    The final tile used to be back-shifted to end flush with the image,
    which made it overlap the previous tile by up to 639px -- a near-copy
    that doubled the inference cost for no new coverage.
    """
    tiler = ImageTiler()
    result = tiler.tile(_img(w, 500))

    starts = sorted({t.x_offset for t in result.tiles if t.row == 0})
    if w <= tiler.tile_size:
        assert len(starts) == 1
        return
    assert len(starts) >= 2
    for a, b in zip(starts, starts[1:]):
        overlap = tiler.tile_size - (b - a)
        assert overlap < tiler.tile_size * 0.75, (
            f"tiles at x={a} and x={b} overlap by {overlap}px -- one is "
            f"largely a duplicate of the other"
        )


@pytest.mark.parametrize("w", [1217, 1920, 2369, 4000, 9933])
def test_the_tile_count_is_minimal(w):
    """
    Tiles are 640px wide with a 64px overlap (stride 576). Covering the
    width with that stride needs a specific number of tiles; using more
    is wasted inference.
    """
    tiler = ImageTiler()
    result = tiler.tile(_img(w, 500))

    if w <= tiler.tile_size:
        assert len(result.tiles) == 1
        return

    span = w - tiler.tile_size
    minimum = math.ceil(span / tiler.stride) + 1
    actual = len({t.x_offset for t in result.tiles if t.row == 0})
    assert actual == minimum, (
        f"used {actual} tiles for width {w}, minimum is {minimum}"
    )


def test_no_two_tiles_start_at_the_same_offset():
    result = ImageTiler().tile(_img(2369, 500))
    starts = [t.x_offset for t in result.tiles if t.row == 0]
    assert len(starts) == len(set(starts)), "duplicate tile starts"


def test_num_rows_and_cols_match_the_actual_grid():
    result = ImageTiler().tile(_img(1920, 1163))
    actual_rows = len({t.row for t in result.tiles})
    actual_cols = len({t.col for t in result.tiles})
    assert result.num_rows == actual_rows
    assert result.num_cols == actual_cols


def test_tiles_are_the_expected_size():
    tiler = ImageTiler()
    result = tiler.tile(_img(1920, 1163))
    for t in result.tiles:
        assert t.image.shape[0] == tiler.tile_size
        assert t.image.shape[1] == tiler.tile_size


def test_the_default_tile_size_matches_the_models_native_resolution():
    """
    The model was trained at imgsz=640 on 640x640 and 1312x1312 images.
    A 1312 tile at imgsz=640 reproduces the training scale exactly;
    tiling at 640 measured F1 0.773 against 0.970 for whole-image
    detection on the dataset's own val split.
    """
    from config.settings import settings

    assert settings.tile_size == 1312
    assert settings.yolo_image_size == 640
    assert settings.tile_overlap < settings.tile_size


def test_zero_overlap_is_allowed():
    """Zero overlap is a legitimate configuration (no padding of the model)."""
    tiler = ImageTiler(tile_size=640, overlap=0)
    result = tiler.tile(_img(1280, 640))
    assert len(result.tiles) == 2
