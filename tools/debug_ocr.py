"""
Inspect the connected-component OCR on a single drawing.

Prints every text region the extractor returns, with its confidence and
centre, sorted by confidence. Useful when a tag comes out mangled or
missing: you see whether OCR read the glyph at all before the tag parser
or the entity mapper gets a chance to drop it.

Usage:
    python tools/debug_ocr.py
    python tools/debug_ocr.py data/sample_pids/plant-pid.png
    python tools/debug_ocr.py my_drawing.png --confidence 0.15 --no-upscale
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_IMAGE = ROOT / "data" / "sample_pids" / "plant-pid.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the OCR regions found in a P&ID drawing.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"drawing to inspect (default: {DEFAULT_IMAGE.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="minimum OCR confidence to report (default: 0.3)",
    )
    parser.add_argument(
        "--upscale-to",
        type=int,
        default=1920,
        help="upscale images narrower than this before OCR (default: 1920, 0 disables)",
    )
    parser.add_argument("--json", action="store_true", help="print regions as JSON")
    return parser.parse_args()


def load_image(path: Path, upscale_to: int) -> "cv2.Mat":
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"cannot read image: {path}")
    height, width = image.shape[:2]
    print(f"image: {path} ({width}x{height})")
    if upscale_to and width < upscale_to:
        scale = upscale_to / width
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        print(f"upscaled: {image.shape[1]}x{image.shape[0]}")
    return image


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    image = load_image(args.image, args.upscale_to)

    from src.ocr.text_extractor import TextExtractor

    extractor = TextExtractor(confidence_threshold=args.confidence)
    regions = extractor.extract(image)
    regions = sorted(regions, key=lambda region: -region.confidence)

    if args.json:
        import json

        print(json.dumps([region.to_dict() for region in regions], indent=2))
        return

    print(f"\ntext regions: {len(regions)}")
    for region in regions:
        x, y = region.center
        print(f"  [{region.confidence:.2f}] {region.text!r} @ ({x:.0f},{y:.0f})")


if __name__ == "__main__":
    main()
