"""
The dataset repair tool must be correct, not just present.

It merges duplicate class names, corrects misspellings, removes duplicate
image files, and re-splits so that no image appears in two splits. Each
of those is a change to the training data, so each needs a test.
"""
import hashlib
from pathlib import Path

import pytest
import yaml

from tools.repair_dataset import build_canonical_map, main


def _write_dataset(root: Path, splits, names):
    """
    splits: {split: [(filename, content_key, [label lines])]}

    `content_key` determines the image BYTES, so two entries with the
    same key are byte-identical files - which is what the dedup step
    looks for.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.dump({"nc": len(names), "names": names, "train": "../train/images",
                   "val": "../valid/images", "test": "../test/images"}),
        encoding="utf-8",
    )
    for split, items in splits.items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for fname, content, lines in items:
            (root / split / "images" / fname).write_bytes(
                content.encode() + b"-pixels"
            )
            (root / split / "labels" / (Path(fname).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )


NAMES = [
    "Back Pressure Regulator",
    "BackPressure Regulator",
    "bolier",
    "Boiler",
    "Gate_Valve",
    "Gate Valve",
]


def test_duplicate_class_names_are_merged():
    mapping, canonical = build_canonical_map(NAMES)
    assert mapping[0] == mapping[1], "the two spellings were not merged"
    assert canonical.count("Back Pressure Regulator") == 1
    assert len(canonical) == len(set(canonical)), "canonical names are not unique"


def test_misspellings_are_corrected():
    mapping, canonical = build_canonical_map(NAMES)
    assert mapping[2] == "Boiler"
    assert "bolier" not in canonical


def test_the_class_count_shrinks_by_the_number_of_merges():
    _, canonical = build_canonical_map(NAMES)
    # 6 names -> 4 canonical (2 duplicate pairs merged)
    assert len(canonical) == 4


def test_the_repair_removes_duplicate_images_and_leaks(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    # The SAME image content appears in train and test, and twice in train.
    _write_dataset(
        src,
        {
            "train": [
                ("a.jpg", "A", ["0 0.5 0.5 0.1 0.1"]),
                ("a_copy.jpg", "A", ["0 0.5 0.5 0.1 0.1"]),   # duplicate content
                ("b.jpg", "B", ["1 0.5 0.5 0.1 0.1"]),
            ],
            "valid": [("c.jpg", "C", ["2 0.5 0.5 0.1 0.1"])],
            "test": [("a_test.jpg", "A", ["0 0.5 0.5 0.1 0.1"])],  # leaks a.jpg
        },
        NAMES,
    )

    report = main(["--src", str(src), "--dst", str(dst),
                   "--report", str(tmp_path / "r.json")])

    assert report["classes_before"] == 6
    assert report["classes_after"] == 4
    assert report["dropped_duplicate_files"] == 2, (
        "the two duplicate files were not dropped"
    )

    # No image content may appear in two splits.
    hashes = {}
    for split in ("train", "valid", "test"):
        d = dst / split / "images"
        assert d.exists(), f"{split} directory missing"
        for p in d.iterdir():
            h = hashlib.md5(p.read_bytes()).hexdigest()
            hashes.setdefault(h, []).append(split)
    cross = [v for v in hashes.values() if len(set(v)) > 1]
    assert not cross, f"images still leak across splits: {cross}"


def test_the_repaired_labels_use_the_canonical_class_ids(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_dataset(
        src,
        {"train": [("a.jpg", "A", ["0 0.5 0.5 0.1 0.1", "1 0.4 0.4 0.1 0.1",
                                   "2 0.3 0.3 0.1 0.1", "3 0.2 0.2 0.1 0.1"])],
         "valid": [], "test": []},
        NAMES,
    )
    main(["--src", str(src), "--dst", str(dst)])

    cfg = yaml.safe_load((dst / "data.yaml").read_text(encoding="utf-8"))
    canonical = cfg["names"]
    # class 0 and 1 are the same symbol -> same new id
    # class 2 ('bolier') and 3 ('Boiler') -> same new id
    lines = (dst / "train" / "labels" / "a.txt").read_text(encoding="utf-8").split()
    ids = [int(lines[i]) for i in range(0, len(lines), 5)]
    assert ids[0] == ids[1], "the duplicate class ids were not merged"
    assert ids[2] == ids[3], "the misspelling did not merge into Boiler"
    assert max(ids) < len(canonical)


def test_the_repaired_data_yaml_has_absolute_resolvable_paths(tmp_path):
    """
    The shipped Roboflow data.yaml uses '../train/images', which does
    not resolve in a flat download. The repaired yaml must be usable
    as-is.
    """
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _write_dataset(src, {"train": [("a.jpg", "A", ["0 0.5 0.5 0.1 0.1"])],
                          "valid": [], "test": []}, NAMES)
    main(["--src", str(src), "--dst", str(dst)])

    cfg = yaml.safe_load((dst / "data.yaml").read_text(encoding="utf-8"))
    assert Path(cfg["train"]).is_dir(), "the train path does not resolve"
    assert Path(cfg["val"]).is_dir(), "the val path does not resolve"
    assert cfg["nc"] == len(cfg["names"])
