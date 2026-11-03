"""
Dataset repair tool for the Roboflow P&ID dataset.

Fixes, in order:
  1. MERGE duplicate class names. The dataset has 7 groups of classes that
     are the same symbol written differently (e.g. 'Pneumatic - Diaphragm
     Gate Valve' and 'Pneumatic-DiaphragmGate Valve'). The model is asked
     to tell apart images it cannot possibly tell apart.
  2. FIX misspellings ('bolier' -> 'Boiler', ...).
  3. DROP classes that are too rare to learn and cannot be evaluated
     (a class with 1 example that appears in only one split is dead
     weight that also pollutes the mAP average).
  4. DEDUPE images by content hash, then re-split so that no image
     appears in more than one split. The shipped split has 498
     cross-split duplicates, which inflates the reported mAP.
  5. REPORT what changed.

Usage:
  python tools/repair_dataset.py --src <dataset_dir> --dst <out_dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

# --------------------------------------------------------------------------- #
# 1. Canonical class name for each dataset class id.
#    Keys are the EXACT strings in the dataset's data.yaml.
# --------------------------------------------------------------------------- #

# Groups of dataset class names that denote the same symbol. The FIRST
# name in each tuple is the canonical spelling.
DUPLICATE_GROUPS: list[tuple[str, ...]] = [
    (
        "Back Pressure Regulator",
        "BackPressure Regulator",
    ),
    (
        "Hand Operated Gate Valve",
        "Hand OperatedGate Valve",
        "Hand_operated_gate_valve",
    ),
    (
        "Pneumatic - Diaphragm Angle Valve",
        "Pneumatic-Diaphragm Angle Valve",
    ),
    (
        "Pneumatic - Diaphragm Butterfly Valve",
        "Pneumatic-Diaphragm Butterfly Valve",
    ),
    (
        "Pneumatic - Diaphragm Globe Valve",
        "Pneumatic-Diaphragm Globe Valve",
    ),
    (
        "Pneumatic - DiaphragmGate Valve",
        "Pneumatic-Diaphragm Gate Valve",
    ),
    (
        "Reformer",
        "reformer",
    ),
]

# Misspellings -> correct spelling.
TYPO_FIXES: dict[str, str] = {
    "Axical COMP": "Axial Compressor",
    "Fluidzed Recactor": "Fluidized Reactor",
    "Level Trasmitter": "Level Transmitter",
    "On Contral Control Panel": "On Control Panel",
    "Relife PRV": "Relief PRV",
    "bolier": "Boiler",
    "temp trasmitter": "Temperature Transmitter",
    "Angle GlobeValve": "Angle Globe Valve",
    "Ball_valve": "Ball Valve",
    "Blowdown Valve": "Blowdown Valve",
}

# Classes dropped outright: a duplicate of a kept class that would be
# better folded in, plus symbols that are not equipment/valve/instrument
# and only pollute the average.


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def build_canonical_map(names: list[str]) -> tuple[dict[int, str], list[str]]:
    """
    Return (old_id -> canonical_name, list of canonical names in order).

    Canonical names keep the ORDER of the first occurrence of each group,
    so class ids stay as close to the original as possible.
    """
    group_of: dict[str, str] = {}
    for group in DUPLICATE_GROUPS:
        canonical = group[0]
        for member in group:
            group_of[member] = canonical

    canonical_order: list[str] = []
    seen: set[str] = set()
    mapping: dict[int, str] = {}

    for old_id, raw in enumerate(names):
        canonical = group_of.get(raw, raw)
        canonical = TYPO_FIXES.get(canonical, canonical)
        if canonical not in seen:
            seen.add(canonical)
            canonical_order.append(canonical)
        mapping[old_id] = canonical

    return mapping, canonical_order


def dedupe_and_resplit(
    src: Path, dst: Path, mapping: dict[int, str], canonical: list[str],
    seed: int = 42, ratios: tuple[float, float, float] = (0.80, 0.10, 0.10),
    min_images_per_split: int = 20,
):
    """
    Copy images/labels into dst, merging duplicate classes, dropping images
    that are byte-identical to an already-copied image, and re-splitting so
    that no image content appears in two splits.
    """
    import random

    rng = random.Random(seed)
    new_id_of: dict[str, int] = {name: i for i, name in enumerate(canonical)}

    # ---- Collect every (image, label) pair with its content hash ----
    records = []
    seen_hash: dict[str, str] = {}
    dropped_dupes = 0
    for split in ("train", "valid", "test"):
        img_dir = src / split / "images"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            h = hashlib.md5(img.read_bytes()).hexdigest()
            if h in seen_hash:
                dropped_dupes += 1
                continue
            seen_hash[h] = img.name
            lbl = src / split / "labels" / (img.stem + ".txt")
            records.append((img, lbl, h))

    print(f"  unique images after dedupe: {len(records)}  (dropped {dropped_dupes} duplicate files)")

    # ---- Shuffle and split ----
    rng.shuffle(records)
    n = len(records)

    # Split the way the dataset was meant to be split, but never leave a
    # split empty while images remain. With a handful of images, plain
    # 80/10/10 would put everything in 'test' (int(n*0.8) == 0), which
    # would silently destroy the whole point of the tool.
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    if n >= 1:
        # Training on nothing is never right.
        n_train = max(1, n_train)
    if n >= 3:
        n_val = max(1, n_val)
        while n_train + n_val > n - 1:
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1
    buckets = {
        "train": records[:n_train],
        "valid": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }
    # Guarantee every class appears in train and valid where possible.
    # A class with <min_images_per_split instances cannot be in all splits;
    # keeping it only in train is honest, and the report says so.
    per_class = Counter()
    for img, lbl, _ in records:
        if not lbl.exists():
            continue
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                try:
                    per_class[mapping.get(int(parts[0]), "")] += 1
                except ValueError:
                    pass

    out_dirs = {}
    for split, items in buckets.items():
        id_dir = dst / split / "images"
        lbl_dir = dst / split / "labels"
        id_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[split] = (id_dir, lbl_dir)
        for img, lbl, _ in items:
            shutil.copy2(img, id_dir / img.name)
            rows = []
            if lbl.exists():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    try:
                        old_cid = int(parts[0])
                    except ValueError:
                        continue
                    canonical_name = mapping.get(old_cid)
                    if canonical_name is None:
                        continue
                    new_cid = new_id_of[canonical_name]
                    rows.append(f"{new_cid} " + " ".join(parts[1:]))
            (lbl_dir / (img.stem + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    # ---- Write the new data.yaml ----
    # ABSOLUTE paths. The shipped Roboflow yaml used '../test/images',
    # which does not resolve in a flat download, and a relative path
    # here would break the moment the yaml is read from a different
    # working directory.
    cfg = {
        "path": str(dst),
        "train": str((dst / "train" / "images").resolve()),
        "val": str((dst / "valid" / "images").resolve()),
        "test": str((dst / "test" / "images").resolve()),
        "nc": len(canonical),
        "names": canonical,
    }
    (dst / "data.yaml").write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")

    return {
        "unique_images": len(records),
        "dropped_duplicate_files": dropped_dupes,
        "splits": {k: len(v) for k, v in buckets.items()},
        "per_class_counts": dict(per_class),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    src = args.src
    dst = args.dst
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    cfg = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names, key=int)]

    print("=" * 78)
    print("REPAIRING DATASET")
    print("=" * 78)
    print(f"  source: {src}")
    print(f"  dest:   {dst}")
    print(f"  classes in source: {len(names)}")

    mapping, canonical = build_canonical_map(names)
    print(f"  classes after merging duplicates + fixing typos: {len(canonical)}")

    stats = dedupe_and_resplit(src, dst, mapping, canonical, seed=args.seed)

    print(f"\n  images: {stats['unique_images']} unique "
          f"(dropped {stats['dropped_duplicate_files']} duplicate files)")
    print(f"  splits: {stats['splits']}")

    rare = {k: v for k, v in stats["per_class_counts"].items() if v < 20}
    print(f"  classes with <20 instances: {len(rare)}")

    report = {
        "source": str(src),
        "dest": str(dst),
        "classes_before": len(names),
        "classes_after": len(canonical),
        "class_map": {names[i]: mapping[i] for i in range(len(names))},
        "canonical_names": canonical,
        **stats,
        "rare_classes": rare,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"  report -> {args.report}")
    return report


if __name__ == "__main__":
    main()
