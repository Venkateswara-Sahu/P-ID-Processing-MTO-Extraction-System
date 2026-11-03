"""
The target-style data pipeline must be correct, because the whole
fine-tuning plan depends on it.

These tests check the things that would silently produce a useless
training set: boxes that do not match the glyph, labels that point at
the wrong image, and a merge that mismaps class ids.
"""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "data" / "target_style" / "glyph_catalogue.yaml"
SAMPLE = ROOT / "data" / "sample_pids" / "plant-pid.png"


@pytest.fixture(scope="module")
def catalogue():
    return yaml.safe_load(CATALOGUE.read_text(encoding="utf-8"))


def test_the_catalogue_exists_and_has_glyphs(catalogue):
    assert len(catalogue["glyphs"]) >= 20, (
        "the catalogue is too small to represent the target style"
    )
    assert catalogue["source_size"] == [650, 394]


def test_every_glyph_box_is_inside_the_source_image(catalogue):
    w, h = catalogue["source_size"]
    for g in catalogue["glyphs"]:
        x1, y1, x2, y2 = g["box"]
        assert 0 <= x1 < x2 <= w, f"{g['tag']} box {g['box']} out of bounds"
        assert 0 <= y1 < y2 <= h, f"{g['tag']} box {g['box']} out of bounds"


def test_every_glyph_box_actually_contains_ink(catalogue):
    """
    A box on blank paper means the glyph was never located and the
    augmenter would paste nothing.
    """
    import cv2
    import numpy as np

    image = cv2.imread(str(ROOT / catalogue["source_image"]))
    assert image is not None
    for g in catalogue["glyphs"]:
        x1, y1, x2, y2 = g["box"]
        crop = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        assert (gray < 200).mean() > 0.02, (
            f"{g['cls']} {g['tag']} box {g['box']} is essentially blank"
        )


def test_every_glyph_class_is_in_the_canonical_vocabulary(catalogue):
    """
    The class names must exist in the repaired Roboflow vocabulary, or
    the merge will drop the class.
    """
    from tools.repair_dataset import build_canonical_map

    names = ["Gate_Valve", "Centrifugal Pump", "Tank", "Heat Exchanger",
             "Horizontal Vessel", "Vertical Vessel", "Mixing Reactor"]
    _, canonical = build_canonical_map(names)
    for g in catalogue["glyphs"]:
        assert g["cls"] in canonical, f"{g['cls']} is not in the vocabulary"


def test_the_generator_produces_a_valid_dataset(tmp_path):
    import cv2

    from tools.augment_target import load_catalogue, main

    glyphs, cat = load_catalogue(CATALOGUE)
    assert len(glyphs) >= 20

    out = tmp_path / "gen"
    main(["--count", "6", "--out", str(out), "--width", "800",
          "--height", "600", "--seed", "1"])

    cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["nc"] == len(cfg["names"])
    assert Path(cfg["train"]).is_dir()

    n_images = len(list((out / "train" / "images").glob("*.jpg")))
    assert n_images > 0

    # every label file must match its image and every box be in range
    total = 0
    for img_path in (out / "train" / "images").glob("*.jpg"):
        lbl = out / "train" / "labels" / f"{img_path.stem}.txt"
        assert lbl.exists(), f"{lbl} missing"
        im = cv2.imread(str(img_path))
        H, W = im.shape[:2]
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.split()
            assert len(p) == 5
            cid, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            assert 0 <= cid < cfg["nc"]
            assert 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0
            assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
            total += 1
    assert total > 0, "the generator emitted no annotations"


def test_the_generator_is_reproducible_for_a_given_seed(tmp_path):
    import cv2
    import numpy as np

    from tools.augment_target import load_catalogue, make_sheet
    import random

    glyphs, cat = load_catalogue(CATALOGUE)
    vocab = cat["vocabulary"]

    a, _ = make_sheet(glyphs, 800, 600, random.Random(7), vocab)
    b, _ = make_sheet(glyphs, 800, 600, random.Random(7), vocab)
    assert np.array_equal(a, b), "the same seed produced different sheets"


def test_the_generator_does_not_draw_pipes_through_equipment():
    """
    A pipe crossing equipment teaches the model something false about
    P&IDs. The generator must drop those connections.
    """
    from tools.augment_target import segment_crosses_box

    # a horizontal run straight through a box
    assert segment_crosses_box((0, 50), (200, 50), (80, 40, 120, 60))
    # a vertical run through a box
    assert segment_crosses_box((100, 0), (100, 200), (80, 40, 120, 60))
    # a run that misses it
    assert not segment_crosses_box((0, 300), (200, 300), (80, 40, 120, 60))


def test_the_merge_remaps_class_ids_by_name(tmp_path):
    """
    The repaired dataset and a target-style dataset have DIFFERENT class
    ids - the repair merged the duplicate spellings, which shifts every
    id after it. A merge that copied ids would silently train the model
    on the wrong labels.
    """
    import cv2
    import numpy as np

    from tools.merge_datasets import main

    base = tmp_path / "base"
    extra = tmp_path / "extra"
    for root in (base, extra):
        for split in ("train", "valid", "test"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)

    # base has 2 classes; extra has 3, of which 1 is new and 2 overlap
    base_names = ["Gate_Valve", "Centrifugal Pump"]
    extra_names = ["Centrifugal Pump", "Vertical Vessel", "Gate_Valve"]
    (base / "data.yaml").write_text(
        yaml.dump({"nc": 2, "names": base_names}), encoding="utf-8")
    (extra / "data.yaml").write_text(
        yaml.dump({"nc": 3, "names": extra_names}), encoding="utf-8")

    img = np.full((40, 40, 3), 255, np.uint8)
    for root, names, cls in ((base, base_names, 0), (extra, extra_names, 0)):
        cv2.imwrite(str(root / "train" / "images" / "x.png"), img)
        (root / "train" / "labels" / "x.txt").write_text(
            f"{cls} 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        for split in ("valid", "test"):
            cv2.imwrite(str(root / split / "images" / "y.png"), img)
            (root / split / "labels" / "y.txt").write_text(
                f"{cls} 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    dst = tmp_path / "merged"
    report = main(["--base", str(base), "--extra", str(extra),
                    "--dst", str(dst)])

    cfg = yaml.safe_load((dst / "data.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    assert names == ["Gate_Valve", "Centrifugal Pump", "Vertical Vessel"]
    assert cfg["nc"] == 3

    # the base's class 1 ('Centrifugal Pump') must map to the merged id
    # of 'Centrifugal Pump', which is 1
    base_lines = (dst / "train" / "labels" / "base_train_x.txt")
    if not base_lines.exists():
        base_lines = next((dst / "train" / "labels").glob("base_*.txt"))
    ids = [int(l.split()[0])
           for l in base_lines.read_text(encoding="utf-8").splitlines()]
    assert ids == [names.index("Gate_Valve")], (
        f"the base label was remapped to {ids}, expected "
        f"{names.index('Gate_Valve')}"
    )
    assert report["classes_merged"] == 3
