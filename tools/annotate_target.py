"""
Semi-automatic annotator for target-style P&ID drawings.

Hand-labelling boxes is the bottleneck for building a target-style
training set, and a detector-assisted annotator cuts it by 3-5x: the
model proposes the easy symbols, connected-component analysis proposes
the rest, and a human only has to correct.

This module finds symbol candidates in a P&ID by:

  1. Running the trained detector for the classes it already handles
     well (the bowtie valves), and taking those boxes as-is.
  2. Running connected-component analysis on the binarised drawing and
     keeping components whose shape is a SYMBOL rather than a pipe:
     a symbol is compact (roughly square, or a tall column), whereas a
     pipe is a long thin run, and text is small and wide.
  3. Classifying each surviving candidate by its aspect ratio and size.

The result is written as a YOLO dataset with perfect ground truth,
because the geometry is known exactly - nothing is guessed.

Usage:
    python tools/annotate_target.py --image drawing.png --dst out_dir/ \
        --names Centrifugal Pump,Heat Exchanger,Vertical Vessel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Symbol shape rules, in ORIGINAL image pixels.
#
# Tuned on the sample P&ID (650x394). A drawing at a different scale
# needs these scaled by (target_width / reference_width).
# --------------------------------------------------------------------------- #

REFERENCE_WIDTH = 650

SHAPE_RULES = {
    # class name: (min_h, max_h, min_w, max_w, min_aspect, max_aspect)
    # aspect = h / w
    #
    # The ranges are the measured geometry of the sample P&ID's glyphs:
    # the reactor is 32x61, the columns 27x125 and 27x125, the
    # exchanger 30x47, the horizontal drums 36x19.
    "Centrifugal Pump": (14, 26, 14, 26, 0.75, 1.35),
    "Heat Exchanger":    (30, 62, 20, 40, 0.85, 2.60),
    "Mixing Reactor":   (52, 100, 22, 48, 1.20, 3.40),
    "Vertical Vessel":  (95, 170, 20, 42, 2.40, 7.00),
    "Horizontal Vessel": (12, 42, 26, 85, 0.15, 1.10),
    "Gate_Valve":       (12, 30, 9, 22, 0.80, 2.40),
    "Tank":             (12, 28, 12, 28, 0.80, 1.30),
}


def scale_rules(width: int) -> dict:
    k = width / REFERENCE_WIDTH
    out = {}
    for name, (h0, h1, w0, w1, a0, a1) in SHAPE_RULES.items():
        out[name] = (h0 * k, h1 * k, w0 * k, w1 * k, a0, a1)
    return out


def binarize(image: np.ndarray) -> np.ndarray:
    """Black ink on white.

    NOTE: a MORPH_CLOSE here is actively harmful. The P&ID's pipe network
    links every symbol into one blob, and closing bridges the small gaps
    along a pipe run, merging all 30 symbols into 2 components. A 2x2
    OPEN instead breaks the 1-3px pipe runs while leaving the symbol
    bodies intact, which is what separates the symbols.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, ker)


def ink_ratio(image: np.ndarray, box) -> float:
    """Fraction of the box that is ink. A half-filled tank is dense; a
    pump circle is mostly white with a small triangle inside."""
    x1, y1, x2, y2 = box
    crop = image[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float((gray < 128).mean())


def refine_circle(image: np.ndarray, cx: int, cy: int, r: int,
                  ink: np.ndarray | None = None) -> tuple[int, int, int, int]:
    """
    Re-estimate a circle's true bounding box from the ink.

    The Hough accumulator often returns a radius smaller than the drawn
    circle, so the box cuts the symbol in half and the crop shows an
    arc rather than a circle. This grows the search radius, finds the
    ink pixels that actually belong to the circle, and returns their
    extent.
    """
    if ink is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, ink = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    H, W = ink.shape[:2]
    r2 = int(r * 1.7)
    x1, y1 = max(0, cx - r2), max(0, cy - r2)
    x2, y2 = min(W, cx + r2 + 1), min(H, cy + r2 + 1)
    patch = ink[y1:y2, x1:x2]
    if patch.size == 0:
        return (cx - r, cy - r, cx + r, cy + r)

    ys, xs = np.nonzero(patch)
    if len(xs) == 0:
        return (cx - r, cy - r, cx + r, cy + r)
    ax = xs + x1
    ay = ys + y1
    # Keep only the ink close enough to be part of this circle.
    d = np.hypot(ax - cx, ay - cy)
    keep = d <= r * 1.55
    if not keep.any():
        return (cx - r, cy - r, cx + r, cy + r)
    ax, ay = ax[keep], ay[keep]
    return (int(ax.min()), int(ay.min()), int(ax.max()) + 1, int(ay.max()) + 1)


def find_circles(image: np.ndarray, radius: tuple[int, int] = (5, 12),
                 min_dist: int = 12) -> list[tuple[int, int, int, int]]:
    """
    Find circle-ish symbols with the Hough transform.

    The PUMP glyph is a circle around a filled triangle, and on a
    650px-wide drawing that circle is a 1px outline. Thresholding plus
    a morphological opening destroys it - the opening erodes the 1px
    stroke away and leaves only the solid triangle behind, so connected
    components find the triangle but never the pump.

    A Hough circle accumulator does not care about stroke width, so it
    finds the pump where morphology cannot. The returned box is the
    circle's bounding box.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
        param1=100, param2=18,
        minRadius=radius[0], maxRadius=radius[1],
    )
    if circles is None:
        return []
    out = []
    for x, y, r in np.round(circles[0]).astype(int):
        out.append((int(x), int(y), int(r)))
    return out


def find_tall_symbols(image: np.ndarray, min_h: int = 60,
                      max_w: int = 60) -> list[tuple[int, int, int, int]]:
    """
    Find tall column-like symbols.

    A column is a tall narrow rounded rectangle with small horizontal
    tray dashes inside. The dashes are separate connected components
    and the outline is broken where they meet the edges, so plain
    connected-component analysis never sees the column. Dilating
    VERTICALLY bridges the dashes and the outline into one solid body
    while leaving the horizontal pipe runs alone.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
    tall = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, ker)
    tall = cv2.morphologyEx(tall, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(tall, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < min_h or w > max_w or area < 200:
            continue
        if h / max(1, w) < 2.0:
            continue
        out.append((x, y, x + w, y + h))
    return out


def find_components(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Candidate symbol boxes from connected components."""
    ink = binarize(image)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < 8 or h < 8 or area < 30:
            continue
        # Reject long thin runs: that is a pipe or a text line.
        aspect = h / w
        if aspect > 9 or aspect < 0.12:
            continue
        # Reject text: small and wide.
        if h < 12 and w > 22:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def classify(box, rules) -> str | None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    aspect = h / w
    for name, (h0, h1, w0, w1, a0, a1) in rules.items():
        if h0 <= h <= h1 and w0 <= w <= w1 and a0 <= aspect <= a1:
            return name
    return None


def detector_boxes(image_path: Path, classes: list[str]) -> list[tuple[str, tuple]]:
    """Boxes for classes the current model already detects well."""
    from ultralytics import YOLO

    model_path = ROOT / "models" / "best.pt"
    model = YOLO(str(model_path))
    wanted = {c.lower() for c in classes}
    r = model.predict(str(image_path), conf=0.15, imgsz=640, verbose=False)[0]
    names = [model.names[i] for i in range(len(model.names))]
    out = []
    for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                    r.boxes.cls.cpu().numpy().astype(int)):
        nm = names[c]
        if nm.lower() in wanted:
            out.append((nm, tuple(int(v) for v in b)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path,
                    help="YOLO dataset dir to create")
    ap.add_argument("--names", required=True,
                    help="comma-separated class list, in class-id order")
    ap.add_argument("--use-detector", default="Gate_Valve",
                    help="comma-separated classes to take from the detector")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="extra scale factor on the shape rules")
    ap.add_argument("--sheet-name", default="synthetic")
    ap.add_argument("--review", type=Path, default=None,
                    help="write a marked-up review image here")
    ap.add_argument("--dump-candidates", type=Path, default=None,
                    help="write the raw candidate boxes (unclassified) here")
    args = ap.parse_args(argv)

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    id_of = {n: i for i, n in enumerate(names)}
    det_classes = [c.strip() for c in args.use_detector.split(",") if c.strip()]

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"cannot read {args.image}")
    H, W = image.shape[:2]
    rules = scale_rules(W)
    if args.scale != 1.0:
        rules = {k: (v[0] * args.scale, v[1] * args.scale, v[2] * args.scale,
                     v[3] * args.scale, v[4], v[5]) for k, v in rules.items()}

    found: list[tuple[str, tuple[int, int, int, int]]] = []
    taken: list[tuple[int, int, int, int]] = []

    # 1. detector-provided boxes
    if det_classes:
        for nm, box in detector_boxes(args.image, det_classes):
            found.append((nm, box))
            taken.append(box)
        print(f"  detector supplied {len(found)} boxes for {det_classes}")

    # 2. Hough circles: the pump glyph and the half-filled tank
    ink = binarize(image)
    circles = find_circles(image, (5, 12))
    circle_added = 0
    for cx, cy, r in circles:
        box = refine_circle(image, cx, cy, r, ink)
        if any(not (box[2] <= b[0] or box[0] >= b[2] or
                    box[3] <= b[1] or box[1] >= b[3]) for b in taken):
            continue
        w, h = box[2] - box[0], box[3] - box[1]
        if not (11 <= w <= 28 and 11 <= h <= 28):
            continue
        # A half-filled tank is dense ink; a pump is a mostly-empty circle.
        cls = "Tank" if ink_ratio(image, box) > 0.35 else "Centrifugal Pump"
        if cls not in id_of:
            continue
        found.append((cls, box))
        taken.append(box)
        circle_added += 1
    print(f"  hough circles added {circle_added} boxes "
          f"(of {len(circles)} circles)")

    # 2b. tall column-like symbols
    tall_added = 0
    for box in find_tall_symbols(image):
        if any(not (box[2] <= b[0] or box[0] >= b[2] or
                    box[3] <= b[1] or box[1] >= b[3]) for b in taken):
            continue
        cls = classify(box, rules)
        if cls is None or cls not in id_of:
            continue
        found.append((cls, box))
        taken.append(box)
        tall_added += 1
    print(f"  tall symbols added {tall_added} boxes")

    # 3. connected components
    comps = find_components(image)
    if args.dump_candidates:
        import json
        args.dump_candidates.write_text(json.dumps([
            {"box": [int(v) for v in b],
             "w": int(b[2] - b[0]), "h": int(b[3] - b[1]),
             "aspect": round((b[3] - b[1]) / (b[2] - b[0]), 2)}
            for b in comps], indent=1), encoding="utf-8")
        print(f"  {len(comps)} candidates -> {args.dump_candidates}")
    added = 0
    for box in comps:
        # skip anything overlapping an existing box
        if any(not (box[2] <= b[0] or box[0] >= b[2] or
                    box[3] <= b[1] or box[1] >= b[3]) for b in taken):
            continue
        cls = classify(box, rules)
        if cls is None or cls not in id_of:
            continue
        found.append((cls, box))
        taken.append(box)
        added += 1
    print(f"  connected components added {added} boxes "
          f"(of {len(comps)} candidates)")

    # 4. write the YOLO dataset
    for split in ("train", "valid", "test"):
        (args.dst / split / "images").mkdir(parents=True, exist_ok=True)
        (args.dst / split / "labels").mkdir(parents=True, exist_ok=True)
    # every image goes to train; the caller can re-split
    img_out = args.dst / "train" / "images" / f"{args.sheet_name}.png"
    lbl_out = args.dst / "train" / "labels" / f"{args.sheet_name}.txt"
    cv2.imwrite(str(img_out), image)
    rows = []
    for cls, (x1, y1, x2, y2) in found:
        if cls not in id_of:
            continue
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        rows.append(f"{id_of[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    lbl_out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    import yaml
    (args.dst / "data.yaml").write_text(
        yaml.dump({
            "path": str(args.dst),
            "train": str((args.dst / "train" / "images").resolve()),
            "val": str((args.dst / "train" / "images").resolve()),
            "test": str((args.dst / "train" / "images").resolve()),
            "nc": len(names),
            "names": names,
        }, default_flow_style=False, sort_keys=False), encoding="utf-8")

    print(f"\n  {len(rows)} annotations -> {args.dst}")
    for cls in names:
        n = sum(1 for c, _ in found if c == cls)
        print(f"    {cls:24s} {n}")

    if args.review:
        marked = image.copy()
        colors = [(0, 0, 255), (0, 200, 255), (255, 0, 200),
                  (0, 180, 0), (255, 140, 0), (160, 160, 160)]
        for i, (cls, (x1, y1, x2, y2)) in enumerate(found):
            c = colors[i % len(colors)]
            cv2.rectangle(marked, (x1, y1), (x2, y2), c, 1)
            cv2.putText(marked, cls[:10], (x1, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, c, 1, cv2.LINE_AA)
        cv2.imwrite(str(args.review), cv2.resize(marked, (W * 3, H * 3),
                                                 interpolation=cv2.INTER_CUBIC))
        print(f"  review image -> {args.review}")


if __name__ == "__main__":
    main()
