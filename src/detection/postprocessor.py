"""
Post-processor for merging tiled detections.

When large P&ID images are split into overlapping tiles for detection,
the same symbol may be detected in multiple tiles. This module merges
detections from tiles back into original coordinates and applies NMS
to eliminate duplicates.
"""

import logging
from typing import List, Tuple

import numpy as np

from src.detection.symbol_detector import Detection
from src.preprocessing.image_tiler import Tile, TilingResult

logger = logging.getLogger(__name__)


class DetectionPostProcessor:
    """Merges and deduplicates detections from overlapping tiles."""

    def __init__(self, iou_threshold: float = 0.45):
        """
        Args:
            iou_threshold: IoU threshold for NMS deduplication.
        """
        self.iou_threshold = iou_threshold

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

        for tile_dets, tile in zip(tile_detections, tiling_result.tiles):
            for det in tile_dets:
                # Convert bbox from tile coords to original coords
                x1, y1, x2, y2 = det.bbox_xyxy
                orig_bbox = (
                    x1 + tile.x_offset,
                    y1 + tile.y_offset,
                    x2 + tile.x_offset,
                    y2 + tile.y_offset,
                )

                # Clip to original image boundaries
                orig_bbox = (
                    max(0, orig_bbox[0]),
                    max(0, orig_bbox[1]),
                    min(tiling_result.original_width, orig_bbox[2]),
                    min(tiling_result.original_height, orig_bbox[3]),
                )

                cx = (orig_bbox[0] + orig_bbox[2]) / 2
                cy = (orig_bbox[1] + orig_bbox[3]) / 2

                merged_det = Detection(
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=orig_bbox,
                    center=(cx, cy),
                )
                all_detections.append(merged_det)

        logger.info(f"Total detections before NMS: {len(all_detections)}")

        # Apply class-aware NMS
        deduplicated = self._class_aware_nms(all_detections)

        logger.info(f"Detections after NMS: {len(deduplicated)}")
        return deduplicated

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
    def filter_by_area(
        detections: List[Detection],
        min_area: float = 100,
        max_area: float = 50000,
    ) -> List[Detection]:
        """
        Filter detections by bounding box area.
        Removes very small (noise) and very large (false positive) detections.
        """
        filtered = [d for d in detections if min_area <= d.area <= max_area]
        removed = len(detections) - len(filtered)
        if removed > 0:
            logger.info(f"Area filter removed {removed} detections")
        return filtered

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
