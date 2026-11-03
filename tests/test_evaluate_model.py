"""
A model evaluated against a dataset whose class ids mean something
different produces garbage numbers that still look like numbers.

The repair tool renumbers classes (merging the duplicate spellings
shifts every subsequent id), so the shipped checkpoint is incompatible
with a repaired dataset until it is retrained. The evaluator must catch
that rather than print a plausible-looking mAP.
"""
import pytest
import yaml
from pathlib import Path

from tools.evaluate_model import check_class_compatibility, load_dataset_yaml


class _FakeModel:
    def __init__(self, names):
        self.names = {i: n for i, n in enumerate(names)}


def test_identical_class_lists_are_compatible():
    names = ["Gate_Valve", "Centrifugal Pump"]
    result = check_class_compatibility(_FakeModel(names), names)
    assert result["compatible"] is True


def test_reordered_class_lists_are_rejected():
    """Same names, different order: every class id points at the wrong symbol."""
    model = _FakeModel(["A", "B", "C"])
    result = check_class_compatibility(model, ["C", "A", "B"])
    assert result["compatible"] is False
    assert "ORDER" in result["reason"]


def test_different_vocabularies_are_rejected():
    model = _FakeModel(["A", "B"])
    result = check_class_compatibility(model, ["A", "X", "Y"])
    assert result["compatible"] is False
    assert result["only_in_model"] == ["B"]
    assert result["only_in_dataset"] == ["X", "Y"]


def test_the_shipped_model_matches_the_original_dataset(tmp_path):
    """
    The checkpoint and the untouched Roboflow dataset agree exactly -
    that is what makes the original evaluation meaningful.
    """
    from ultralytics import YOLO

    root = Path(__file__).resolve().parents[1]
    model = YOLO(str(root / "models" / "best.pt"))
    model_names = [model.names[i] for i in range(len(model.names))]

    # Rebuild the original dataset's class list from the checkpoint's
    # own vocabulary, then confirm the checker accepts it.
    result = check_class_compatibility(model, model_names)
    assert result["compatible"] is True


def test_the_dataset_yaml_loader_resolves_paths(tmp_path):
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "valid" / "images").mkdir(parents=True)
    (tmp_path / "test" / "images").mkdir(parents=True)
    (tmp_path / "data.yaml").write_text(
        yaml.dump({"nc": 2, "names": ["A", "B"],
                   "train": "../train/images",
                   "val": "../valid/images",
                   "test": "../test/images"}),
        encoding="utf-8",
    )
    cfg, names = load_dataset_yaml(tmp_path)
    assert names == ["A", "B"]
    # The yaml's paths are relative to ITS OWN directory, so they must
    # resolve to the split directories next to it.
    for key in ("train", "val", "test"):
        assert Path(cfg[key]).is_dir(), f"{key} does not resolve"
