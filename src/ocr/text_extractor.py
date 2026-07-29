"""
Text Extractor for P&ID drawings.

Strategy: Connected-Component Text Detection + Tesseract OCR.

After dual-binarization (which makes text clearly black on white),
we use OpenCV's connected component analysis to find blobs that match
the aspect ratio and size of text characters/words. These text-like
regions are then grouped into word-level crops and passed to Tesseract
(psm 7/8 — single line/word mode) which is highly accurate on clean crops.

This avoids the fundamental problem of all full-image OCR on P&IDs:
pipe lines, circles, and valve symbols produce noise that overwhelms
both EasyOCR's CRAFT detector and Tesseract's layout analysis.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


import cv2
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """A single text region detected by OCR."""

    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in original image coords
    center: Tuple[float, float]
    polygon: List[List[float]]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox],
            "center": [round(v, 1) for v in self.center],
        }


def _binarize(image: np.ndarray, scale: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upscale + binarize. Returns (binary_inv [black text on white], upscaled BGR).
    binary_inv: 255 where text/lines are, 0 for background.
    """
    h, w = image.shape[:2]
    up = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    gray = clahe.apply(gray)

    # Dual binarization — catches light-gray italic labels
    _, fixed = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

    block = 31
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block,
        C=5,
    )
    combined = cv2.bitwise_or(fixed, adaptive)

    # Noise removal
    kernel = np.ones((2, 2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    return combined, up  # combined: 255=ink, 0=background


def _find_text_candidates(
    binary_ink: np.ndarray,
    scale: float,
    min_h_px: int = 12,
    max_h_px: int = 130,
    min_w_px: int = 5,
    max_w_px: int = 700,
) -> List[Tuple[int, int, int, int]]:
    """
    Find connected components that have the geometry of text characters.
    Returns list of (x, y, w, h) in *scaled* image space.

    Text-like criteria (at scale=3.0):
    - Height: 8–120px (covers ~3–40px characters in original image)
    - Width: 4–600px (single char to full tag label)
    - Aspect: width/height typically 0.2–8 (not extreme horizontal/vertical lines)
    - Fill ratio: > 0.05 (not hollow outlines like circles/diamonds)
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_ink, connectivity=8
    )

    candidates = []
    for i in range(1, num_labels):  # skip background (0)
        x, y, w, h, area = (
            stats[i, cv2.CC_STAT_LEFT],
            stats[i, cv2.CC_STAT_TOP],
            stats[i, cv2.CC_STAT_WIDTH],
            stats[i, cv2.CC_STAT_HEIGHT],
            stats[i, cv2.CC_STAT_AREA],
        )

        # Size filter
        if h < min_h_px or h > max_h_px:
            continue
        if w < min_w_px or w > max_w_px:
            continue

        # Aspect ratio filter: exclude extreme horizontal pipe lines
        aspect = w / h
        if aspect > 12 or aspect < 0.15:
            continue

        # Fill ratio filter:
        # - Pipe lines (thin horizontal): very low fill < 0.05
        # - Pump impeller coils / hollow circles: medium fill 0.05-0.20
        # - Text characters: medium-high fill 0.15-0.85
        # - Solid filled dots/circles: very high fill > 0.90
        # We exclude very low AND very high fill to target text.
        fill = area / (w * h)
        if fill < 0.08 or fill > 0.90:
            continue

        # Minimum area: avoid sub-pixel noise
        if area < 30:
            continue

        candidates.append((x, y, w, h))

    return candidates


def _group_into_words(
    candidates: List[Tuple[int, int, int, int]],
    h_gap_ratio: float = 1.8,
    v_overlap_ratio: float = 0.4,
) -> List[Tuple[int, int, int, int]]:
    """
    Group individual character bounding boxes into word-level regions.
    h_gap_ratio=1.8 bridges the dash in tags like 'E-1001' without
    merging words that are clearly separated (label to label gap is
    typically 2-5x the character height).
    """
    if not candidates:
        return []

    # Sort left-to-right
    sorted_c = sorted(candidates, key=lambda c: c[0])

    groups: List[List[Tuple[int, int, int, int]]] = []
    current_group = [sorted_c[0]]

    for box in sorted_c[1:]:
        bx, by, bw, bh = box
        # Compare with rightmost box in current group
        prev = current_group[-1]
        px, py, pw, ph = prev

        prev_right = px + pw
        h_gap = bx - prev_right
        avg_h = (bh + ph) / 2

        # Vertical overlap
        top = max(by, py)
        bot = min(by + bh, py + ph)
        v_overlap = bot - top

        if h_gap <= h_gap_ratio * avg_h and v_overlap >= v_overlap_ratio * min(bh, ph):
            current_group.append(box)
        else:
            groups.append(current_group)
            current_group = [box]

    groups.append(current_group)

    # Merge each group into one bbox with padding
    merged = []
    for group in groups:
        xs = [c[0] for c in group]
        ys = [c[1] for c in group]
        x2s = [c[0] + c[2] for c in group]
        y2s = [c[1] + c[3] for c in group]
        pad = 4
        merged.append((
            max(0, min(xs) - pad),
            max(0, min(ys) - pad),
            max(x2s) - min(xs) + 2 * pad,
            max(y2s) - min(ys) + 2 * pad,
        ))

    return merged


# Known Tesseract noise patterns for P&ID drawings
_NOISE_PATTERN = re.compile(
    r'^[^a-zA-Z0-9]+$'           # all special chars
    r'|^[oe]{1,3}$'              # common pump-pattern misreads
    r'|^[ry]{1,2}$'              # common exchanger-coil misreads
    r'|^[&%@|><.]+$'             # pipe/arrow chars
    r'|^[\u0080-\uffff]+$',      # non-ASCII garbage
    re.IGNORECASE,
)


def _is_valid_text(text: str) -> bool:
    """
    Return True if the OCR result is likely a real engineering label.
    Filters single characters (unless digit), known noise patterns,
    and strings that are mostly special characters.
    """
    if not text or len(text.strip()) == 0:
        return False
    t = text.strip()
    # Allow single digits (stream numbers 1-17)
    if len(t) == 1 and t.isdigit():
        return True
    # Allow two-digit stream numbers
    if len(t) == 2 and t.isdigit():
        return True
    # Reject single non-digit chars
    if len(t) == 1:
        return False
    # Reject known noise
    if _NOISE_PATTERN.match(t):
        return False
    # Must have at least one letter or digit
    if not any(c.isalnum() for c in t):
        return False
    # Reject if >50% special/non-ascii chars
    alnum_count = sum(1 for c in t if c.isalnum() or c in '-_./() ')
    if alnum_count / len(t) < 0.5:
        return False
    return True


def _merge_fragments(regions: List['TextRegion']) -> List['TextRegion']:
    """
    Merge adjacent OCR fragments on the same baseline into compound tags.
    Handles 'E' + '-' + '1001' → 'E-1001' by merging regions that:
    - Are on the same vertical baseline (y centers within 0.6 * avg height)
    - Are horizontally close (gap < 2 * avg char width)
    Also removes remaining noise fragments after merging.
    """
    if not regions:
        return []

    # Sort left to right
    sorted_r = sorted(regions, key=lambda r: r.center[0])
    merged: List['TextRegion'] = []
    used = set()

    for i, r in enumerate(sorted_r):
        if i in used:
            continue
        group = [r]
        used.add(i)

        for j, s in enumerate(sorted_r[i + 1:], start=i + 1):
            if j in used:
                continue
            last = group[-1]
            h_avg = (abs(last.bbox[3] - last.bbox[1]) + abs(s.bbox[3] - s.bbox[1])) / 2
            # Horizontal gap
            gap = s.bbox[0] - last.bbox[2]
            # Vertical alignment
            v_diff = abs(s.center[1] - last.center[1])

            if gap < 2.5 * h_avg and v_diff < 0.7 * h_avg:
                group.append(s)
                used.add(j)
            else:
                break  # sorted, so no point continuing

        if len(group) == 1:
            merged.append(r)
        else:
            # Combine text and take average confidence
            combined_text = ''.join(g.text for g in group)
            avg_conf = sum(g.confidence for g in group) / len(group)
            x1 = min(g.bbox[0] for g in group)
            y1 = min(g.bbox[1] for g in group)
            x2 = max(g.bbox[2] for g in group)
            y2 = max(g.bbox[3] for g in group)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            import dataclasses
            merged_r = dataclasses.replace(
                group[0],
                text=combined_text,
                confidence=avg_conf,
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            )
            merged.append(merged_r)

    return merged

def _tesseract_crop(crop_bgr: np.ndarray) -> Tuple[str, float]:
    """Run Tesseract on a single clean crop. Returns (text, confidence)."""
    import pytesseract

    results = []
    for psm in [8, 7, 6]:  # word, single line, uniform block
        cfg = f"--oem 3 --psm {psm}"
        try:
            data = pytesseract.image_to_data(
                crop_bgr, config=cfg, output_type=pytesseract.Output.DICT
            )
            texts, confs = [], []
            for i, t in enumerate(data["text"]):
                t = t.strip()
                c = float(data["conf"][i])
                if t and c > 0:
                    texts.append(t)
                    confs.append(c)
            if texts:
                combined = " ".join(texts)
                avg_conf = sum(confs) / len(confs) / 100.0
                results.append((combined, avg_conf))
        except Exception:
            pass

    if not results:
        return "", 0.0

    # Return highest-confidence result
    return max(results, key=lambda x: x[1])


def _set_tesseract_path():
    """Set Tesseract binary path on Windows."""
    try:
        import pytesseract
        for path in [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


class TextExtractor:
    """
    OCR text extractor for P&ID drawings using connected-component detection.

    Binarizes the image, finds character-sized connected components,
    groups them into word-level crops, then runs Tesseract psm 7/8 on each.
    Falls back to EasyOCR full-image if Tesseract is unavailable.
    """

    def __init__(
        self,
        language: str = None,
        confidence_threshold: float = None,
        use_gpu: bool = False,
    ):
        self.language = language or settings.ocr_language
        self.confidence_threshold = confidence_threshold or settings.ocr_confidence_threshold
        self.use_gpu = use_gpu
        self._reader = None
        self._tesseract_ok = _set_tesseract_path()
        self._ocr_scale = 3.0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def extract(self, image: np.ndarray) -> List[TextRegion]:
        """Extract text from image using connected-component analysis + Tesseract."""
        return self._run(image)

    def extract_with_detections(
        self, image: np.ndarray, detections: list, **kwargs
    ) -> List[TextRegion]:
        """Same as extract() — detections are not needed with this strategy."""
        return self._run(image)

    def _run(self, image: np.ndarray) -> List[TextRegion]:
        scale = self._ocr_scale

        # Step 1: Binarize (upscale x3 + dual threshold)
        binary_ink, upscaled = _binarize(image, scale=scale)
        h_up, w_up = upscaled.shape[:2]

        # White-background, black-text image for Tesseract
        ocr_image = cv2.bitwise_not(binary_ink)
        ocr_image_bgr = cv2.cvtColor(ocr_image, cv2.COLOR_GRAY2BGR)

        # Step 2: CC analysis — WHERE is text?
        candidates = _find_text_candidates(binary_ink, scale=scale)
        logger.info(f"Connected components: {len(candidates)} text-like blobs")
        word_boxes = _group_into_words(candidates)
        logger.info(f"Grouped into {len(word_boxes)} word candidate regions")

        if not word_boxes:
            logger.warning("No text candidates found")
            return []

        # Step 3: Single OCR call on full image (fast)
        if self._tesseract_ok:
            regions = self._tesseract_full_image(ocr_image_bgr, word_boxes, scale, h_up, w_up)
        else:
            regions = self._easyocr_full_image(ocr_image_bgr, word_boxes, scale, h_up, w_up)

        logger.info(f"OCR raw pass: {len(regions)} valid regions before merge")

        # Merge adjacent fragments (e.g. 'FV' + '3040' -> 'FV3040')
        regions = _merge_fragments(regions)

        # Final validity check
        regions = [r for r in regions if _is_valid_text(r.text)]

        logger.info(f"OCR complete: {len(regions)} text regions (conf>={self.confidence_threshold})")
        if regions:
            sample = [f'"{r.text}" ({r.confidence:.2f})' for r in regions[:25]]
            logger.info(f"Detected: {', '.join(sample)}")

        return regions

    def _tesseract_full_image(
        self,
        ocr_image_bgr: np.ndarray,
        word_boxes: List[Tuple[int, int, int, int]],
        scale: float,
        h_up: int,
        w_up: int,
    ) -> List[TextRegion]:
        """
        Run a SINGLE Tesseract psm-11 call on the full binarized image.
        Filter results to only words that overlap CC-detected text regions.
        ~50x faster than per-crop approach (one process spawn vs 136).
        """
        import pytesseract

        config = "--oem 3 --psm 11"
        try:
            data = pytesseract.image_to_data(
                ocr_image_bgr, config=config, output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            logger.error(f"Tesseract failed: {e}")
            return []

        # Build overlap mask from CC word boxes
        word_mask = np.zeros((h_up, w_up), dtype=np.uint8)
        for (x, y, w, h) in word_boxes:
            word_mask[y:min(h_up, y + h), x:min(w_up, x + w)] = 255

        regions: List[TextRegion] = []
        seen: set = set()

        for i in range(len(data["text"])):
            text = str(data["text"][i]).strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i]) / 100.0
            except (ValueError, TypeError):
                continue
            if conf < 0:
                continue

            tx = int(data["left"][i])
            ty = int(data["top"][i])
            tw = int(data["width"][i])
            th = int(data["height"][i])
            if tw < 3 or th < 3:
                continue

            tx2 = min(w_up, tx + tw)
            ty2 = min(h_up, ty + th)
            if tx2 <= tx or ty2 <= ty:
                continue

            # Only keep words overlapping a CC text region
            overlap = word_mask[ty:ty2, tx:tx2]
            if overlap.size == 0 or overlap.mean() < 25:
                continue

            if not _is_valid_text(text) or conf < self.confidence_threshold:
                continue

            ox = tx / scale
            oy = ty / scale
            bbox = (ox, oy, (tx2) / scale, (ty2) / scale)
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            key = f"{text.lower()}_{int(center[0]/10)}_{int(center[1]/10)}"
            if key in seen:
                continue
            seen.add(key)

            regions.append(TextRegion(
                text=text, confidence=conf, bbox=bbox, center=center,
                polygon=[[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                         [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
            ))

        return regions

    def _easyocr_full_image(
        self,
        ocr_image_bgr: np.ndarray,
        word_boxes: List[Tuple[int, int, int, int]],
        scale: float,
        h_up: int,
        w_up: int,
    ) -> List[TextRegion]:
        """EasyOCR fallback: run on full image, filter by CC mask."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=self.use_gpu, verbose=False)

        word_mask = np.zeros((h_up, w_up), dtype=np.uint8)
        for (x, y, w, h) in word_boxes:
            word_mask[y:min(h_up, y + h), x:min(w_up, x + w)] = 255

        results = self._reader.readtext(ocr_image_bgr, paragraph=False,
                                        min_size=8, text_threshold=0.4)
        regions: List[TextRegion] = []
        seen: set = set()

        for bbox_pts, text, conf in results:
            text = text.strip()
            if not text or not _is_valid_text(text) or conf < self.confidence_threshold:
                continue
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            tx, ty = int(min(xs)), int(min(ys))
            tx2, ty2 = int(max(xs)), int(max(ys))
            overlap = word_mask[max(0, ty):min(h_up, ty2), max(0, tx):min(w_up, tx2)]
            if overlap.size == 0 or overlap.mean() < 25:
                continue
            ox, oy = tx / scale, ty / scale
            bbox = (ox, oy, tx2 / scale, ty2 / scale)
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            key = f"{text.lower()}_{int(center[0]/10)}_{int(center[1]/10)}"
            if key in seen:
                continue
            seen.add(key)
            regions.append(TextRegion(
                text=text, confidence=float(conf), bbox=bbox, center=center,
                polygon=[[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                         [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
            ))
        return regions

    def draw_text_regions(
        self,
        image: np.ndarray,
        text_regions: List[TextRegion],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 1,
        font_scale: float = 0.4,
    ) -> np.ndarray:
        """Draw text bounding boxes and labels on the image."""
        result = image.copy()
        for tr in text_regions:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            label = f"{tr.text} ({tr.confidence:.2f})"
            cv2.putText(result, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        return result
