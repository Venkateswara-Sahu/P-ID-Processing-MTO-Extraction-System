"""
End-to-end pipeline tests.

These exercise the real pipeline on the real sample drawing. They are
slower than the unit tests but they are the only place the coordinate
space, the OCR stage and the MTO generation are checked together.
"""
import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)


@pytest.fixture(scope="module")
def pipeline_result(sample_pid):
    from src.pipeline import PIDPipeline

    return PIDPipeline().process(
        str(sample_pid),
        project_name="TEST",
        run_llm_validation=False,
        generate_mto=True,
    )


# --------------------------------------------------------------------------- #
# C7: the reported frame must match the coordinate space
# --------------------------------------------------------------------------- #

def test_reported_image_size_matches_the_coordinate_space(pipeline_result):
    """
    Every entity position must fall inside the reported image size. They did
    not before: the size was the pre-upscale original while the coordinates
    were in the upscaled space.
    """
    width, height = pipeline_result.image_size
    assert width > 0 and height > 0

    for entity in pipeline_result.entities:
        if entity.position is None:
            continue
        x, y = entity.position
        assert 0 <= x <= width, (
            f"{entity.entity_id} x={x:.0f} is outside the reported width {width}"
        )
        assert 0 <= y <= height, (
            f"{entity.entity_id} y={y:.0f} is outside the reported height {height}"
        )


def test_scale_factor_is_reported(pipeline_result):
    """The upscale factor must be available so consumers can map back."""
    assert pipeline_result.scale_factor >= 1.0
    assert "scale_factor" in pipeline_result.to_summary()


def test_annotated_image_matches_the_reported_size(pipeline_result):
    """The annotated overlay must be in the same frame as the coordinates."""
    assert pipeline_result.annotated_image is not None
    h, w = pipeline_result.annotated_image.shape[:2]
    assert (w, h) == tuple(pipeline_result.image_size), (
        f"annotated image {w}x{h} does not match reported size "
        f"{pipeline_result.image_size}"
    )


# --------------------------------------------------------------------------- #
# The pipeline must produce a usable MTO
# --------------------------------------------------------------------------- #

def test_pipeline_produces_an_mto(pipeline_result):
    assert pipeline_result.mto_path
    assert Path(pipeline_result.mto_path).exists()


def test_mto_has_rows(pipeline_result):
    import openpyxl

    ws = openpyxl.load_workbook(pipeline_result.mto_path).active
    item_rows = [
        r for r in ws.iter_rows(values_only=True)
        if r and r[0] is not None and isinstance(r[0], int)
    ]
    assert item_rows, "the MTO contains no item rows"


def test_mto_quantities_are_integers(pipeline_result):
    import openpyxl

    ws = openpyxl.load_workbook(pipeline_result.mto_path).active
    header = [c.value for c in ws[5]]
    qty_col = header.index("Quantity")
    for row in ws.iter_rows(min_row=6, values_only=True):
        if row and row[0] is not None and isinstance(row[0], int):
            assert isinstance(row[qty_col], int), (
                f"quantity {row[qty_col]!r} is not an integer"
            )


def test_no_entity_position_is_nan(pipeline_result):
    """A NaN position would poison the graph and the visualisation."""
    import math

    for entity in pipeline_result.entities:
        if entity.position is None:
            continue
        x, y = entity.position
        assert not math.isnan(x) and not math.isnan(y), (
            f"{entity.entity_id} has a NaN position"
        )


def test_pipeline_is_repeatable_on_the_same_input(sample_pid):
    """Running twice must not leak state between runs."""
    from src.pipeline import PIDPipeline

    pipe = PIDPipeline()
    first = pipe.process(str(sample_pid), run_llm_validation=False,
                         generate_mto=False)
    second = pipe.process(str(sample_pid), run_llm_validation=False,
                          generate_mto=False)

    assert len(first.entities) == len(second.entities)
    assert len(first.detections) == len(second.detections)
    # Entity ids must restart, not continue from the previous run.
    assert first.entities[0].entity_id == second.entities[0].entity_id


def test_the_model_is_loaded_only_once(sample_pid):
    """
    The 22MB YOLOv8 weights were re-read from disk on every process()
    call, because _init_components() rebuilt the detector each time.
    """
    from src.pipeline import PIDPipeline

    pipe = PIDPipeline()
    pipe.process(str(sample_pid), run_llm_validation=False, generate_mto=False)
    model_after_first = pipe._detector.model
    assert model_after_first is not None, "the model was not loaded"

    pipe.process(str(sample_pid), run_llm_validation=False, generate_mto=False)
    assert pipe._detector.model is model_after_first, (
        "the model was reloaded on the second run"
    )


# --------------------------------------------------------------------------- #
# C5: multi-page PDFs
# --------------------------------------------------------------------------- #

def test_every_pdf_page_is_processed(project_root):
    """
    Only the first page was processed; the rest of the P&ID package was
    discarded with no warning.
    """
    import fitz

    from src.pipeline import PIDPipeline

    pdf_path = project_root / "data" / "sample_pids" / "multi_page_test.pdf"
    if not pdf_path.exists():
        pytest.skip("multi-page test PDF not generated")

    doc = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    result = PIDPipeline().process(
        str(pdf_path), run_llm_validation=False, generate_mto=False
    )

    assert len(result.per_page) == page_count, (
        f"only {len(result.per_page)} of {page_count} pages were processed"
    )
    assert [p["page"] for p in result.per_page] == list(range(1, page_count + 1))


def test_max_pages_limits_the_pages_processed(project_root):
    from src.pipeline import PIDPipeline

    pdf_path = project_root / "data" / "sample_pids" / "multi_page_test.pdf"
    if not pdf_path.exists():
        pytest.skip("multi-page test PDF not generated")

    result = PIDPipeline().process(
        str(pdf_path), run_llm_validation=False, generate_mto=False, max_pages=1
    )
    assert len(result.per_page) == 1


def test_entity_ids_are_unique_across_pages(project_root):
    """Each page's mapper restarts its counter, so ids could collide."""
    from src.pipeline import PIDPipeline

    pdf_path = project_root / "data" / "sample_pids" / "multi_page_test.pdf"
    if not pdf_path.exists():
        pytest.skip("multi-page test PDF not generated")

    result = PIDPipeline().process(
        str(pdf_path), run_llm_validation=False, generate_mto=False
    )
    ids = [e.entity_id for e in result.entities]
    assert len(ids) == len(set(ids)), "entity ids collided across pages"

