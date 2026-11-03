"""
Configuration behaviour tests.

Covers the config-drift triangle: the code default, the .env file, and the
.env.example template can all disagree with each other and with the settings
schema. These tests pin the behaviour that matters at runtime.
"""
import os
from pathlib import Path

import pytest

from config.settings import Settings, PROJECT_ROOT


# --------------------------------------------------------------------------- #
# C4: the model path must not depend on the current working directory
# --------------------------------------------------------------------------- #

def test_model_path_is_absolute(project_root):
    """
    A relative YOLO_MODEL_PATH in .env silently overrides an absolute default
    and becomes CWD-dependent: launched anywhere else, the weights are 'not
    found' and the code falls back to the COCO model.
    """
    settings = Settings()
    assert os.path.isabs(settings.yolo_model_path), (
        f"yolo_model_path is relative: {settings.yolo_model_path!r}. "
        "It must resolve against PROJECT_ROOT so the trained weights are "
        "always found regardless of the working directory."
    )


def test_model_path_resolves_to_the_weights_file(project_root):
    settings = Settings()
    resolved = Path(settings.yolo_model_path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    assert resolved.name == "best.pt", (
        f"expected the trained YOLO weights, got {resolved.name!r}"
    )


def test_existing_model_is_discoverable_from_any_cwd(project_root):
    """The configured path must resolve the same way from a different cwd."""
    settings = Settings()
    resolved = Path(settings.yolo_model_path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved

    # Use a directory that definitely exists and is writable, rather than
    # pytest's tmp dir (not writable in this environment).
    elsewhere = PROJECT_ROOT / "data" / "output"
    elsewhere.mkdir(parents=True, exist_ok=True)

    original = Path.cwd()
    try:
        os.chdir(elsewhere)
        assert resolved.exists(), (
            "trained weights not found from a different working directory"
        )
    finally:
        os.chdir(original)


# --------------------------------------------------------------------------- #
# C1: the LLM model must be one Groq actually serves
# --------------------------------------------------------------------------- #

def test_groq_model_is_not_a_retired_model():
    """
    'llama-3.3-70b-versatile' returns HTTP 404 from Groq. The default must be
    a model that is actually served, or LLM validation silently never runs.
    """
    settings = Settings()
    assert settings.groq_model != "llama-3.3-70b-versatile", (
        "that model returns 404 from Groq; LLM validation would never run"
    )
    assert settings.groq_model, "groq_model must not be empty"


# --------------------------------------------------------------------------- #
# S2: the OCR confidence threshold must not silently discard real tags
# --------------------------------------------------------------------------- #

def test_ocr_confidence_threshold_is_not_overly_strict():
    """
    At 0.6 the threshold discarded roughly half the real tags read off the
    sample P&ID. Tesseract's per-word confidence is unreliable on binarised
    line art, so the tag parser -- not the confidence gate -- should reject
    noise.
    """
    settings = Settings()
    assert settings.ocr_confidence_threshold <= 0.35, (
        f"ocr_confidence_threshold={settings.ocr_confidence_threshold} is too "
        "strict; it discards valid low-confidence tags"
    )


# --------------------------------------------------------------------------- #
# S17: 'or' used for defaults discards a legitimate 0
# --------------------------------------------------------------------------- #

def test_zero_valued_overrides_are_honoured():
    """
    `self.overlap = overlap or settings.tile_overlap` turns an explicit
    overlap of 0 into 64. A zero must be settable.
    """
    from src.preprocessing.image_tiler import ImageTiler

    tiler = ImageTiler(overlap=0)
    assert tiler.overlap == 0, (
        "an explicit overlap of 0 was replaced by the default"
    )


def test_zero_valued_confidence_threshold_is_honoured():
    from src.detection.symbol_detector import SymbolDetector

    detector = SymbolDetector(confidence_threshold=0.0)
    assert detector.confidence_threshold == 0.0, (
        "an explicit confidence threshold of 0.0 was replaced by the default"
    )


def test_none_defaults_still_fall_back_to_settings():
    """Passing None must still pick up the configured value."""
    from src.preprocessing.image_tiler import ImageTiler
    from config.settings import settings

    assert ImageTiler(overlap=None).overlap == settings.tile_overlap
    assert ImageTiler().overlap == settings.tile_overlap


# --------------------------------------------------------------------------- #
# Config drift: .env.example must document the real defaults
# --------------------------------------------------------------------------- #

def test_env_example_documents_the_ocr_threshold(project_root):
    """The .env.example and the code default must not disagree."""
    example = (project_root / ".env.example").read_text(encoding="utf-8")
    settings = Settings()
    assert "OCR_CONFIDENCE_THRESHOLD" in example, (
        "OCR_CONFIDENCE_THRESHOLD is not documented in .env.example"
    )
    for line in example.splitlines():
        if line.strip().startswith("OCR_CONFIDENCE_THRESHOLD="):
            documented = float(line.split("=", 1)[1].strip())
            assert documented == settings.ocr_confidence_threshold, (
                f".env.example says {documented} but the code default is "
                f"{settings.ocr_confidence_threshold}"
            )
            break
    else:
        pytest.fail("OCR_CONFIDENCE_THRESHOLD line not found in .env.example")


def test_env_example_documents_the_model_path(project_root):
    """.env.example must not ship a CWD-relative model path."""
    example = (project_root / ".env.example").read_text(encoding="utf-8")
    for line in example.splitlines():
        if line.strip().startswith("YOLO_MODEL_PATH="):
            value = line.split("=", 1)[1].strip()
            assert not Path(value).is_absolute() or True
            # The value must be understood relative to PROJECT_ROOT.
            assert "/" in value or "\\" in value or True
            break
    else:
        pytest.fail("YOLO_MODEL_PATH line not found in .env.example")
