"""
Generate target-style P&ID sheets with exact ground truth.

WHY THIS EXISTS
---------------
The shipped model scores mAP50 0.849 on the Roboflow dataset but
finds NONE of the vessels, columns, exchangers and reactors on a
P&ID drawn in a different style. It is not a data-volume problem:
`Centrifugal Pump` has 2,013 training instances and mAP50 0.935, and
the model still misses 6 of 8 pumps on the target drawing, because the
GLYPH is drawn differently.

Training on more of the same style cannot close that gap. Training on
the target style can - and because the target style is clean thin
black line art with no scan noise, it can be SYNTHESISED with exact
ground truth instead of hand-annotated.

This tool takes the verified glyphs in
`data/target_style/glyph_catalogue.yaml`, cuts them out of the sample
drawing, and composes them into new sheets with random pipe networks,
random positions, random scales and random degradation.

WHAT IT DOES AND DOES NOT PRODUCE
----------------------------------
This produces symbols that look like the target style. It does NOT
produce real process topology, real tag semantics, or the noise of a
scanned original. Treat it as a strong prior that a real annotated
drawing would then improve on - not a replacement for one.

Usage:
    python tools/augment_target.py --count 500 --out data/target_style/generated/
"""
from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import yaml

# Class order for the generated dataset. Must match the vocabulary the
# merged dataset uses.
DEFAULT_VOCAB = [
    "Centrifugal Pump",
    "Heat Exchanger",
    "Vertical Vessel",
    "Horizontal Vessel",
    "Mixing Reactor",
    "Gate_Valve",
    "Tank",
]

TAG_PREFIX = {
    "Centrifugal Pump": "P",
    "Heat Exchanger": "E",
    "Vertical Vessel": "T",
    "Horizontal Vessel": "E",
    "Mixing Reactor": "R",
    "Gate_Valve": "HV",
    "Tank": "TK",
}


def load_catalogue(path: Path):
    cat = yaml.safe_load(path.read_text(encoding="utf-8"))
    src = ROOT / cat["source_image"]
    image = cv2.imread(str(src))
    if image is None:
        raise ValueError(f"cannot read {src}")
    glyphs = []
    for g in cat["glyphs"]:
        x1, y1, x2, y2 = g["box"]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        glyphs.append({
            "cls": g["cls"],
            "raster": crop,
            "alpha": (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 200),
        })
    return glyphs, cat


def paste(canvas, raster, alpha, cx, cy, angle, scale, line_w=1):
    """Paste a glyph centred at (cx, cy), rotated and scaled."""
    h, w = raster.shape[:2]
    nh, nw = max(3, int(h * scale)), max(3, int(w * scale))
    r = cv2.resize(raster, (nw, nh), interpolation=cv2.INTER_CUBIC)
    a = cv2.resize(alpha.astype(np.uint8), (nw, nh),
                   interpolation=cv2.INTER_NEAREST).astype(bool)
    if angle:
        M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
        r = cv2.warpAffine(r, M, (nw, nh), flags=cv2.INTER_CUBIC,
                           borderValue=(255, 255, 255))
        a = cv2.warpAffine(a.astype(np.uint8), M, (nw, nh),
                           flags=cv2.INTER_NEAREST).astype(bool)
    # the glyph body is black ink on white
    ink = (cv2.cvtColor(r, cv2.COLOR_BGR2GRAY) < 200)
    y0, x0 = int(cy - nh / 2), int(cx - nw / 2)
    H, W = canvas.shape[:2]
    y1, x1 = max(0, y0), max(0, x0)
    y2, x2 = min(H, y0 + nh), min(W, x0 + nw)
    if y2 <= y1 or x2 <= x1:
        return None
    sub_ink = ink[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    region = canvas[y1:y2, x1:x2]
    region[sub_ink] = (30, 30, 30)
    return (x1, y1, x2, y2)


def draw_pipe(canvas, p0, p1, width=1):
    """
    Draw a pipe run the way a P&ID does: horizontal, then vertical,
    with a square elbow. Never a straight diagonal - a diagonal line
    running through a symbol teaches the model that pipes overlap
    equipment, which is not true of a real drawing.
    """
    mid = (p1[0], p0[1])
    cv2.line(canvas, p0, mid, (40, 40, 40), width, cv2.LINE_AA)
    cv2.line(canvas, mid, p1, (40, 40, 40), width, cv2.LINE_AA)


def draw_tag(canvas, text, x, y, scale):
    """Draw a tag label next to a symbol."""
    h = int(7 * scale)
    if h < 5:
        return
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                h / 22.0, (60, 60, 60), 1, cv2.LINE_AA)


def segment_crosses_box(p0, p1, box, pad=2) -> bool:
    """True if the axis-aligned pipe run p0->p1 passes through the box."""
    x1, y1, x2, y2 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    # The run is horizontal then vertical: (p0 -> mid -> p1)
    mid = (p1[0], p0[1])
    for a, b in ((p0, mid), (mid, p1)):
        if a[1] == b[1]:                      # horizontal segment
            if y1 <= a[1] <= y2:
                lo, hi = sorted((a[0], b[0]))
                if hi > x1 and lo < x2:
                    return True
        else:                                 # vertical segment
            if x1 <= a[0] <= x2:
                lo, hi = sorted((a[1], b[1]))
                if hi > y1 and lo < y2:
                    return True
    return False


def make_sheet(glyphs, width, height, rng, vocab):
    """
    Compose one synthetic sheet. Returns (image, [(cls, box)]).

    The order matters: the pipe network is drawn FIRST and the symbols
    are pasted on top, so the equipment occludes the pipe ends. That is
    what a real drawing looks like, and it stops the model learning
    that pipes run through equipment.
    """
    canvas = 255 * np.ones((height, width, 3), dtype=np.uint8)

    # Symbol density. A real P&ID is sparse; 60 symbols on one sheet
    # is not a P&ID, and the model would learn a cluttered prior.
    n = max(3, int(rng.gauss(0.0016, 0.0006) * width * height))
    n = max(2, min(n, 34))

    by_cls: dict[str, list] = {}
    for g in glyphs:
        by_cls.setdefault(g["cls"], []).append(g)

    planned = []
    counters: dict[str, int] = {}

    # Lay the symbols out on a loose grid so they do not overlap.
    cols = max(2, int(math.sqrt(n * width / height)))
    rows = max(2, int(n / cols) + 1)
    cell_w, cell_h = width / cols, height / rows

    for i in range(n):
        cls = rng.choice(vocab)
        pool = by_cls.get(cls)
        if not pool:
            continue
        g = rng.choice(pool)
        base_h = g["raster"].shape[0]
        # Scale so the symbol occupies a realistic fraction of the sheet.
        target = base_h * (width / 650.0)
        scale = max(0.6, target / base_h) * rng.uniform(0.85, 1.15)
        angle = rng.uniform(-4, 4)
        cx = int((i % cols + rng.uniform(0.2, 0.8)) * cell_w)
        cy = int((i // cols + rng.uniform(0.25, 0.75)) * cell_h)
        counters[cls] = counters.get(cls, 0) + 1
        tag = f"{TAG_PREFIX.get(cls, 'X')}-{1000 + counters[cls]}"
        planned.append((g, cls, tag, cx, cy, angle, scale))

    # 1. the pipe network, drawn first.
    #    Symbols are connected in reading order, and a connection that
    #    would run through an unrelated symbol is dropped. A pipe
    #    crossing equipment teaches the model something false.
    boxes = []
    for g, cls, tag, cx, cy, angle, scale in planned:
        h, w = g["raster"].shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        boxes.append((cx - nw // 2, cy - nh // 2, cx + nw // 2, cy + nh // 2))

    for i in range(len(planned) - 1):
        a, b = planned[i], planned[i + 1]
        p0 = (a[3], a[4])
        p1 = (b[3], b[4])
        if abs(p0[0] - p1[0]) < 3 and abs(p0[1] - p1[1]) < 3:
            continue
        if any(segment_crosses_box(p0, p1, bx)
               for j, bx in enumerate(boxes) if j not in (i, i + 1)):
            continue
        draw_pipe(canvas, p0, p1, width=rng.choice([1, 1, 2]))

    # 2. the symbols, pasted on top so they occlude the pipe ends
    annotations = []
    for (g, cls, tag, cx, cy, angle, scale), box in zip(planned, boxes):
        placed = paste(canvas, g["raster"], g["alpha"], cx, cy, angle, scale)
        if placed is None:
            continue
        annotations.append((cls, placed))
        # a tag beside the symbol, offset so it does not sit on top of it
        draw_tag(canvas, tag, placed[0] - 3, placed[1] - 4,
                 max(0.8, width / 650.0))

    return canvas, annotations


def degrade(canvas, rng):
    """Add the imperfections a real scan or export has."""
    out = canvas
    if rng.random() < 0.5:
        # slight blur
        k = rng.choice([3, 3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.4:
        # sensor / compression noise
        noise = np.random.normal(0, rng.uniform(1.5, 5.0), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        # JPEG artefacts
        q = rng.randint(55, 92)
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", type=Path,
                    default=ROOT / "data" / "target_style" / "glyph_catalogue.yaml")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "target_style" / "generated")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--height", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--preview", type=Path, default=None,
                    help="write a preview sheet here")
    args = ap.parse_args(argv)

    glyphs, cat = load_catalogue(args.catalogue)
    vocab = [c for c in cat.get("vocabulary", DEFAULT_VOCAB) if c in DEFAULT_VOCAB]
    print(f"catalogue: {len(glyphs)} glyphs over {len(vocab)} classes")
    for c in vocab:
        n = sum(1 for g in glyphs if g["cls"] == c)
        print(f"    {c:22s} {n} glyphs")

    if args.out.exists():
        shutil.rmtree(args.out)
    for split in ("train", "valid", "test"):
        (args.out / split / "images").mkdir(parents=True, exist_ok=True)
        (args.out / split / "labels").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    n_val = int(args.count * args.val_ratio)
    n_test = int(args.count * args.val_ratio)
    plan = (["train"] * (args.count - n_val - n_test)
            + ["valid"] * n_val + ["test"] * n_test)
    rng.shuffle(plan)

    id_of = {c: i for i, c in enumerate(vocab)}
    total_boxes = 0

    for i, split in enumerate(plan):
        img, anns = make_sheet(glyphs, args.width, args.height, rng, vocab)
        img = degrade(img, rng)
        name = f"gen_{i:05d}.jpg"
        cv2.imwrite(str(args.out / split / "images" / name), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        H, W = img.shape[:2]
        rows = []
        for cls, (x1, y1, x2, y2) in anns:
            cx = (x1 + x2) / 2 / W
            cy = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            rows.append(f"{id_of[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            total_boxes += 1
        (args.out / split / "labels" / f"gen_{i:05d}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    cfg = {
        "path": str(args.out),
        "train": str((args.out / "train" / "images").resolve()),
        "val": str((args.out / "valid" / "images").resolve()),
        "test": str((args.out / "test" / "images").resolve()),
        "nc": len(vocab),
        "names": vocab,
    }
    (args.out / "data.yaml").write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")

    print(f"\ngenerated {args.count} sheets "
          f"({args.width}x{args.height}), {total_boxes} annotations")
    print(f"  train {len(plan) - n_val - n_test}  valid {n_val}  test {n_test}")
    print(f"  -> {args.out}")

    if args.preview:
        img, _ = make_sheet(glyphs, args.width, args.height, rng, vocab)
        cv2.imwrite(str(args.preview), img)
        print(f"  preview -> {args.preview}")


if __name__ == "__main__":
    main()
