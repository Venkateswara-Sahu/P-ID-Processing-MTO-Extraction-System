"""
Merge a target-style dataset into the repaired Roboflow dataset.

The two datasets use DIFFERENT class ids: repairing the Roboflow set
merges the duplicate spellings, which shifts every subsequent id, and a
target-style set has its own class list entirely. Merging therefore has
to remap ids by CLASS NAME, not by number.

Output is a single YOLO dataset with one merged class list, written in
the flat layout with absolute paths in data.yaml.

Usage:
    python tools/merge_datasets.py \
        --base dataset_repaired/ \
        --extra target_style/ \
        --dst dataset_merged/ \
        --extra-repeat 3
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

SPLIT_DIRS = {
    "train": ("train", "train"),
    "valid": ("valid", "valid"),
    "test": ("test", "test"),
}


def read_class_names(dataset_dir: Path) -> list[str]:
    src = dataset_dir / "data.yaml"
    if not src.exists():
        found = list(dataset_dir.rglob("data.yaml"))
        if not found:
            raise FileNotFoundError(f"no data.yaml under {dataset_dir}")
        src = found[0]
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names, key=int)]
    return names


def find_split(dataset_dir: Path, split: str) -> Path | None:
    """Return the images dir for a split, tolerating both layouts."""
    for candidate in (
        dataset_dir / split / "images",
        dataset_dir / "images" / split,
        dataset_dir / split,
    ):
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    return None


def load_split(dataset_dir: Path, split: str):
    """Return (records, class_names) for one split.

    records: list of (image_path, label_path, {class_name: count})
    """
    img_dir = find_split(dataset_dir, split)
    if img_dir is None:
        return [], []
    names = read_class_names(dataset_dir)
    label_dir = img_dir.parent / "labels"
    records = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = label_dir / (img.stem + ".txt")
        rows = []
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                if 0 <= cid < len(names):
                    rows.append((cid, parts[1:]))
        records.append((img, lbl, rows))
    return records, names


def remap_rows(rows, old_names, new_index):
    """Rewrite label rows so every class id points at the MERGED id."""
    out = []
    for cid, rest in rows:
        name = old_names[cid]
        new_cid = new_index.get(name)
        if new_cid is None:
            continue
        out.append(f"{new_cid} " + " ".join(rest))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path,
                    help="the repaired Roboflow dataset")
    ap.add_argument("--extra", type=Path, default=None,
                    help="a target-style dataset to merge in")
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--extra-repeat", type=int, default=1,
                    help="how many times to repeat the extra dataset "
                         "(oversampling a small target-style set)")
    ap.add_argument("--drop-classes", default="",
                    help="comma-separated class names to drop entirely "
                         "(e.g. the generic catch-alls)")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)

    base_names = read_class_names(args.base)
    extra_names = read_class_names(args.extra) if args.extra else []

    drop = {c.strip() for c in args.drop_classes.split(",") if c.strip()}

    # The merged class list: base classes first (keeping their order), then
    # any extra classes that are not already present.
    merged = [n for n in base_names if n not in drop]
    for n in extra_names:
        if n not in drop and n not in merged:
            merged.append(n)
    new_index = {n: i for i, n in enumerate(merged)}

    if args.dst.exists():
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True)

    stats = {}
    for split in ("train", "valid", "test"):
        base_records, _ = load_split(args.base, split)
        extra_records = []
        if args.extra:
            extra_records, _ = load_split(args.extra, split)

        out_img = args.dst / split / "images"
        out_lbl = args.dst / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        n_base = n_extra = 0
        counts = Counter()

        for img, lbl, rows in base_records:
            new_rows = remap_rows(rows, base_names, new_index)
            stem = f"base_{split}_{img.stem}"
            shutil.copy2(img, out_img / f"{stem}{img.suffix}")
            (out_lbl / f"{stem}.txt").write_text(
                "\n".join(new_rows) + ("\n" if new_rows else ""), encoding="utf-8")
            n_base += 1
            for r in new_rows:
                counts[merged[int(r.split()[0])]] += 1

        for rep in range(max(1, args.extra_repeat)):
            for img, lbl, rows in extra_records:
                new_rows = remap_rows(rows, extra_names, new_index)
                stem = f"target{rep}_split{split}_{img.stem}"
                shutil.copy2(img, out_img / f"{stem}{img.suffix}")
                (out_lbl / f"{stem}.txt").write_text(
                    "\n".join(new_rows) + ("\n" if new_rows else ""), encoding="utf-8")
                n_extra += 1
                for r in new_rows:
                    counts[merged[int(r.split()[0])]] += 1

        stats[split] = {
            "base_images": n_base,
            "target_images": n_extra,
            "total": n_base + n_extra,
            "instances": dict(counts),
        }
        print(f"  {split:6s} base={n_base:6d}  target={n_extra:6d}  "
              f"total={n_base + n_extra:6d}")

    cfg = {
        "path": str(args.dst),
        "train": str((args.dst / "train" / "images").resolve()),
        "val": str((args.dst / "valid" / "images").resolve()),
        "test": str((args.dst / "test" / "images").resolve()),
        "nc": len(merged),
        "names": merged,
    }
    (args.dst / "data.yaml").write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")

    print(f"\n  classes: {len(base_names)} base -> {len(merged)} merged"
          + (f"  (dropped {len(drop)}: {sorted(drop)})" if drop else ""))
    if extra_names:
        print(f"  target-style classes added: "
              f"{[n for n in extra_names if n in merged and n not in base_names]}")

    report = {
        "base": str(args.base),
        "extra": str(args.extra) if args.extra else None,
        "dst": str(args.dst),
        "classes_base": len(base_names),
        "classes_merged": len(merged),
        "dropped": sorted(drop),
        "extra_repeat": args.extra_repeat,
        "splits": stats,
        "class_names": merged,
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"  report -> {args.report}")
    return report


if __name__ == "__main__":
    main()
