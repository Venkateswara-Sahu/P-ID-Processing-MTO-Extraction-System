"""
Post-processor for merging tiled detections.

When large P&ID images are split into overlapping tiles for detection,
the same symbol may be detected in multiple tiles. This module merges
detections from tiles back into original coordinates and applies NMS
to eliminate duplicates.
"""

import logging
from typing import List, Optional, Tuple

from config.settings import settings
from src.detection.symbol_detector import Detection
from src.preprocessing.image_tiler import Tile, TilingResult

logger = logging.getLogger(__name__)


class DetectionPostProcessor:
    """Merges and deduplicates detections from overlapping tiles."""

    def __init__(
        self,
        iou_threshold: float = None,
        drop_edge_detections: bool = None,
        edge_tolerance_px: int = None,
    ):
        """
        Args:
            iou_threshold: IoU threshold for NMS deduplication. Defaults to
                the configured value so that changing it in .env actually
                affects the merge step.
            drop_edge_detections: Discard detections cut by an internal
                tile boundary. Defaults to the configured value.
            edge_tolerance_px: How close to the boundary a box must be to
                count as cut.
        """
        self.iou_threshold: float = (
            iou_threshold if iou_threshold is not None else settings.yolo_iou_threshold
        )
        self.drop_edge_detections: bool = (
            drop_edge_detections
            if drop_edge_detections is not None
            else settings.drop_edge_detections
        )
        self.edge_tolerance_px: int = (
            edge_tolerance_px
            if edge_tolerance_px is not None
            else settings.edge_tolerance_px
        )

    def _is_cut(
        self,
        bbox: Tuple[float, float, float, float],
        cut_left: float,
        cut_top: float,
        cut_right: float,
        cut_bottom: float,
    ) -> bool:
        """
        True when the box touches an internal tile boundary.

        The cut coordinates are in original-image space; a negative value
        means that edge is the image border, not a cut.
        """
        x1, y1, x2, y2 = bbox
        tol = self.edge_tolerance_px
        if cut_left >= 0 and x1 <= cut_left + tol:
            return True
        if cut_top >= 0 and y1 <= cut_top + tol:
            return True
        if cut_right >= 0 and x2 >= cut_right - tol:
            return True
        if cut_bottom >= 0 and y2 >= cut_bottom - tol:
            return True
        return False

    def merge_tiled_detections(
        self,
        tile_detections: List[List[Detection]],
        tiling_result: TilingResult,
    ) -> List[Detection]:
        """
        Merge detections from multiple tiles into original image coordinates
        and apply NMS to remove duplicates from overlapping regions.

        Args:
            tile_detections: List of detection lists, one per tile.
            tiling_result: The TilingResult containing tile metadata.

        Returns:
            Deduplicated list of detections in original image coordinates.
        """
        all_detections = []
        cut_count = 0

        for tile_dets, tile in zip(tile_detections, tiling_result.tiles):
            cut_left, cut_top, cut_right, cut_bottom = tile.cut_edges()
            for det in tile_dets:
                # Convert bbox from tile coords to original coords
                x1, y1, x2, y2 = det.bbox_xyxy
                orig_bbox = (
                    x1 + tile.x_offset,
                    y1 + tile.y_offset,
                    x2 + tile.x_offset,
                    y2 + tile.y_offset,
                )

                # Clip to original image boundaries. BOTH ends must be
                # clipped: clipping only the upper bound produced inverted
                # boxes (negative width/area) for detections that fell in a
                # tile's zero-padded region, and an inverted box has garbage
                # IoU so it could neither suppress nor be suppressed.
                clipped = (
                    max(0, min(orig_bbox[0], tiling_result.original_width)),
                    max(0, min(orig_bbox[1], tiling_result.original_height)),
                    max(0, min(orig_bbox[2], tiling_result.original_width)),
                    max(0, min(orig_bbox[3], tiling_result.original_height)),
                )
                cx1, cy1, cx2, cy2 = clipped

                # Drop degenerate boxes -- they are not real symbols.
                if cx2 <= cx1 or cy2 <= cy1:
                    logger.debug(
                        f"Dropping degenerate detection {clipped} "
                        f"(clipped from {orig_bbox})"
                    )
                    continue

                # Drop detections sliced by an INTERNAL tile boundary.
                # The detector saw only part of the symbol, so the box is
                # a partial shape: a false positive that also suppresses
                # the correct detection of the same symbol in the
                # neighbouring tile. Measured on a 3936x3936 sheet this
                # raised precision from 0.870 to 0.946.
                if self.drop_edge_detections and self._is_cut(
                    clipped, cut_left, cut_top, cut_right, cut_bottom
                ):
                    cut_count += 1
                    continue

                cx = (cx1 + cx2) / 2
                cy = (cy1 + cy2) / 2

                merged_det = Detection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=clipped,
                    center=(cx, cy),
                )
                all_detections.append(merged_det)

        if cut_count:
            logger.info(
                f"Dropped {cut_count} detections cut by a tile boundary"
            )

        logger.info(f"Total detections before NMS: {len(all_detections)}")

        # Class-AGNOSTIC pass first. The model has 82 near-synonymous valve
        # classes, so a symbol on a tile boundary can be classified
        # differently in each tile; class-aware NMS alone would keep both
        # and emit two MTO line items for one physical valve.
        deduped = self._class_agnostic_nms(all_detections)
        logger.info(
            f"After class-agnostic dedup: {len(deduped)} "
            f"(removed {len(all_detections) - len(deduped)})"
        )

        # Then the class-aware pass, to drop same-class duplicates.
        deduplicated = self._class_aware_nms(deduped)

        logger.info(f"Detections after NMS: {len(deduplicated)}")
        return deduplicated

    def _class_agnostic_nms(self, detections: List[Detection]) -> List[Detection]:
        """
        Greedy NMS that ignores the class label.

        Detections of the same physical symbol that the model labelled
        differently across tiles are collapsed into the most confident one.
        """
        if len(detections) <= 1:
            return list(detections)

        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []
        suppressed = set()

        for i, det_i in enumerate(dets):
            if i in suppressed:
                continue
            keep.append(det_i)
            for j in range(i + 1, len(dets)):
                if j in suppressed:
                    continue
                if self._compute_iou(det_i.bbox_xyxy, dets[j].bbox_xyxy) > self.iou_threshold:
                    suppressed.add(j)

        return keep

    def _class_aware_nms(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply Non-Maximum Suppression separately per class.

        Args:
            detections: All detections (may contain duplicates).

        Returns:
            Filtered detections after NMS.
        """
        if not detections:
            return []

        # Group by class
        class_groups = {}
        for det in detections:
            if det.class_id not in class_groups:
                class_groups[det.class_id] = []
            class_groups[det.class_id].append(det)

        # Apply NMS per class
        result = []
        for class_id, dets in class_groups.items():
            filtered = self._nms(dets)
            result.extend(filtered)

        # Sort by confidence (highest first)
        result.sort(key=lambda d: d.confidence, reverse=True)
        return result

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """
        Standard NMS implementation.

        Args:
            detections: Detections of the same class.

        Returns:
            Filtered detections.
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence (descending)
        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

        keep = []
        suppressed = set()

        for i, det_i in enumerate(dets):
            if i in suppressed:
                continue

            keep.append(det_i)

            for j in range(i + 1, len(dets)):
                if j in suppressed:
                    continue

                iou = self._compute_iou(det_i.bbox_xyxy, dets[j].bbox_xyxy)
                if iou > self.iou_threshold:
                    suppressed.add(j)

        return keep

    @staticmethod
    def _compute_iou(
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def get_detection_summary(detections: List[Detection]) -> dict:
        """Generate a summary of detections by class."""
        summary = {}
        for det in detections:
            name = det.class_name
            if name not in summary:
                summary[name] = {"count": 0, "avg_confidence": 0.0}
            summary[name]["count"] += 1
            summary[name]["avg_confidence"] += det.confidence

        for name in summary:
            count = summary[name]["count"]
            summary[name]["avg_confidence"] = round(
                summary[name]["avg_confidence"] / count, 4
            )

        return dict(sorted(summary.items(), key=lambda x: x[1]["count"], reverse=True))
