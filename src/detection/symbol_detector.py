"""
YOLOv8 Symbol Detector for P&ID drawings.

Wraps the Ultralytics YOLOv8 model for CPU inference on P&ID
symbol detection. Processes individual tiles and returns
structured detection results.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single symbol detection from YOLOv8."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple  # (x1, y1, x2, y2) in pixel coordinates
    center: tuple  # (cx, cy) center point

    @property
    def width(self) -> float:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox_xyxy],
            "center": [round(v, 1) for v in self.center],
            "width": round(self.width, 1),
            "height": round(self.height, 1),
        }


class SymbolDetector:
    """YOLOv8-based P&ID symbol detector for CPU inference."""

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        image_size: int = None,
    ):
        """
        Initialize the symbol detector.

        Args:
            model_path: Path to the trained YOLOv8 weights (.pt file).
            confidence_threshold: Minimum confidence for detections.
            iou_threshold: NMS IoU threshold.
            image_size: Input image size for the model.
        """
        self.model_path = model_path or settings.yolo_model_path
        self.confidence_threshold = confidence_threshold or settings.yolo_confidence_threshold
        self.iou_threshold = iou_threshold or settings.yolo_iou_threshold
        self.image_size = image_size or settings.yolo_image_size
        self.model = None

    def load_model(self):
        """Load the YOLOv8 model. Lazy-loaded on first detection call."""
        from ultralytics import YOLO

        model_path = Path(self.model_path)

        if model_path.exists():
            logger.info(f"Loading trained YOLOv8 model from: {model_path}")
            self.model = YOLO(str(model_path))
        else:
            # Fall back to pre-trained YOLOv8s for demo purposes
            logger.warning(
                f"Trained model not found at {model_path}. "
                "Loading pre-trained YOLOv8s (COCO). "
                "Train a custom model using the Colab notebook for P&ID symbols."
            )
            self.model = YOLO("yolov8s.pt")

        # Force CPU inference (no CUDA on Intel Iris Xe)
        self.model.to("cpu")
        logger.info(f"Model loaded successfully. Classes: {len(self.model.names)}")

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection on a single image or tile.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            List of Detection objects.
        """
        if self.model is None:
            self.load_model()

        # Run inference
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device="cpu",
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                # Extract bounding box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

                # Extract confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                detection = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center=((x1 + x2) / 2, (y1 + y2) / 2),
                )
                detections.append(detection)

        logger.debug(f"Detected {len(detections)} symbols")
        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """
        Run detection on a batch of images/tiles.

        Args:
            images: List of images as numpy arrays.

        Returns:
            List of detection lists, one per image.
        """
        if self.model is None:
            self.load_model()

        all_detections = []
        for idx, image in enumerate(images):
            dets = self.detect(image)
            all_detections.append(dets)
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(images)} tiles")

        return all_detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """
        Draw detection bounding boxes and labels on the image.

        Args:
            image: Input image (will be copied).
            detections: List of Detection objects.
            thickness: Box line thickness.
            font_scale: Label font scale.

        Returns:
            Image with drawn detections.
        """
        result = image.copy()

        # Color palette for different class categories
        colors = {
            "valve": (0, 255, 0),       # Green
            "instrument": (255, 165, 0), # Orange
            "equipment": (0, 0, 255),    # Red
            "piping": (255, 255, 0),     # Cyan
        }
        default_color = (128, 128, 128)  # Gray

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]

            # Pick color based on class category
            color = default_color
            for category, cat_color in colors.items():
                if category in det.class_name.lower():
                    color = cat_color
                    break

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(
                result,
                (x1, y1 - label_size[1] - 6),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                result,
                label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
            )

        return result

    def get_class_names(self) -> dict:
        """Return the model's class name mapping."""
        if self.model is None:
            self.load_model()
        return dict(self.model.names)
