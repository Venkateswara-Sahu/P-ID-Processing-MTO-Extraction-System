"""Shared pytest fixtures."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def sample_pid() -> Path:
    return PROJECT_ROOT / "data" / "sample_pids" / "plant-pid.png"


@pytest.fixture
def tag_parser():
    from src.ocr.tag_parser import TagParser
    return TagParser()


@pytest.fixture
def detection_factory():
    """Build a Detection with sensible defaults."""
    from src.detection.symbol_detector import Detection

    def _make(class_name="Gate Valve", confidence=0.9, x=100.0, y=100.0,
              class_id=0, size=20.0):
        return Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            bbox_xyxy=(x - size / 2, y - size / 2, x + size / 2, y + size / 2),
            center=(x, y),
        )

    return _make


@pytest.fixture
def text_region_factory():
    """Build a TextRegion with sensible defaults."""
    from src.ocr.text_extractor import TextRegion

    def _make(text="FIC-101", confidence=0.9, x=100.0, y=100.0):
        return TextRegion(
            text=text,
            confidence=confidence,
            bbox=(x - 20, y - 6, x + 20, y + 6),
            center=(x, y),
            polygon=[[x - 20, y - 6], [x + 20, y - 6], [x + 20, y + 6], [x - 20, y + 6]],
        )

    return _make


@pytest.fixture
def entity_factory(detection_factory):
    """Build an EngineeringEntity with sensible defaults."""
    from src.graph.entity_mapper import EngineeringEntity

    def _make(entity_id="E0001", entity_type="valve", category="valves",
              tag_id=None, description="Gate Valve", confidence=0.9,
              x=100.0, y=100.0, detection=None, **kwargs):
        return EngineeringEntity(
            entity_id=entity_id,
            symbol_detection=detection if detection is not None else detection_factory(x=x, y=y),
            entity_type=entity_type,
            component_category=category,
            tag_id=tag_id,
            description=description,
            position=(x, y),
            confidence=confidence,
            **kwargs,
        )

    return _make
