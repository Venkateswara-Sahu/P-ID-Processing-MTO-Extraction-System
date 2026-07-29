"""
P&ID MTO Extraction Pipeline — Main Orchestrator.

Coordinates the full extraction pipeline:
1. Preprocessing (PDF → enhanced images → tiles)
2. Detection (YOLOv8 symbol detection on tiles)
3. OCR (PaddleOCR text extraction)
4. Graph (Entity mapping + relationship resolution)
5. Validation (Rule-based + LLM validation)
6. MTO Generation (Excel output)
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import networkx as nx
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result of the P&ID extraction pipeline."""

    # Extracted data
    entities: list = field(default_factory=list)
    graph: Optional[nx.DiGraph] = None
    graph_summary: dict = field(default_factory=dict)

    # Detection details
    detections: list = field(default_factory=list)
    text_regions: list = field(default_factory=list)

    # Validation
    rule_report: dict = field(default_factory=dict)
    llm_report: dict = field(default_factory=dict)
    confidence_summary: dict = field(default_factory=dict)

    # Output
    mto_path: Optional[str] = None
    annotated_image: Optional[np.ndarray] = None

    # Metadata
    processing_time: float = 0.0
    image_size: tuple = (0, 0)

    def to_summary(self) -> dict:
        return {
            "total_entities": len(self.entities),
            "total_detections": len(self.detections),
            "total_text_regions": len(self.text_regions),
            "graph": self.graph_summary,
            "confidence": self.confidence_summary,
            "mto_path": self.mto_path,
            "processing_time_seconds": round(self.processing_time, 2),
            "image_size": self.image_size,
        }


class PIDPipeline:
    """Orchestrates the full P&ID MTO extraction pipeline."""

    def __init__(self):
        """Initialize pipeline components (lazy-loaded)."""
        self._preprocessor = None
        self._tiler = None
        self._detector = None
        self._postprocessor = None
        self._ocr = None
        self._tag_parser = None
        self._entity_mapper = None
        self._graph_builder = None
        self._relationship_resolver = None
        self._rule_validator = None
        self._confidence_scorer = None
        self._llm_validator = None
        self._mto_generator = None

    def _init_components(self):
        """Initialize all pipeline components."""
        from src.preprocessing.enhancer import ImageEnhancer
        from src.preprocessing.image_tiler import ImageTiler
        from src.detection.symbol_detector import SymbolDetector
        from src.detection.postprocessor import DetectionPostProcessor
        from src.ocr.text_extractor import TextExtractor
        from src.ocr.tag_parser import TagParser
        from src.graph.entity_mapper import EntityMapper
        from src.graph.graph_builder import PIDGraphBuilder
        from src.graph.relationship_resolver import RelationshipResolver
        from src.validation.engineering_rules import EngineeringRuleValidator
        from src.validation.confidence_scorer import ConfidenceScorer
        from src.validation.llm_validator import LLMValidator
        from src.mto.mto_generator import MTOGenerator

        self._preprocessor = ImageEnhancer()
        self._tiler = ImageTiler()
        self._detector = SymbolDetector()
        self._postprocessor = DetectionPostProcessor()
        self._ocr = TextExtractor()
        self._tag_parser = TagParser()
        self._entity_mapper = EntityMapper()
        self._graph_builder = PIDGraphBuilder()
        self._relationship_resolver = RelationshipResolver()
        self._rule_validator = EngineeringRuleValidator()
        self._confidence_scorer = ConfidenceScorer()
        self._llm_validator = LLMValidator()
        self._mto_generator = MTOGenerator()

        logger.info("Pipeline components initialized")

    def process(
        self,
        input_path: str,
        project_name: str = "P&ID Extraction",
        run_llm_validation: bool = True,
        generate_mto: bool = True,
    ) -> PipelineResult:
        """
        Run the full extraction pipeline on a P&ID image or PDF.

        Args:
            input_path: Path to P&ID image (PNG, JPG) or PDF.
            project_name: Project name for MTO header.
            run_llm_validation: Whether to run LLM validation (requires API key).
            generate_mto: Whether to generate MTO Excel.

        Returns:
            PipelineResult with all extraction data.
        """
        start_time = time.time()
        self._init_components()

        result = PipelineResult()
        input_path = Path(input_path)

        logger.info(f"{'='*60}")
        logger.info(f"Processing: {input_path.name}")
        logger.info(f"{'='*60}")

        # ---- Step 1: Load & Preprocess ----
        logger.info("Step 1/6: Loading and preprocessing...")

        if input_path.suffix.lower() == ".pdf":
            from src.preprocessing.pdf_converter import PDFConverter
            converter = PDFConverter()
            images = converter.convert_to_numpy(str(input_path))
            image = images[0]  # Process first page
        else:
            image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"Failed to load image: {input_path}")

        result.image_size = (image.shape[1], image.shape[0])

        # Upscale small images before detection/OCR.
        # YOLO tiles at 640px — an image narrower than 1280px means symbols
        # are smaller than the model was trained on → poor detection.
        # EasyOCR CRAFT needs ≥ 20px character height.
        min_width = 1920
        h_orig, w_orig = image.shape[:2]
        if w_orig < min_width:
            scale_up = min_width / w_orig
            new_w = int(w_orig * scale_up)
            new_h = int(h_orig * scale_up)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.info(
                f"  Image upscaled {w_orig}×{h_orig} → {new_w}×{new_h} "
                f"(×{scale_up:.1f}) for better detection & OCR"
            )

        enhanced = self._preprocessor.enhance(image, apply_clahe=True, apply_denoise=True)

        # ---- Step 2: Symbol Detection ----
        logger.info("Step 2/6: Detecting symbols (YOLOv8)...")

        tiling_result = self._tiler.tile(enhanced)
        tile_images = [t.image for t in tiling_result.tiles]
        tile_detections = self._detector.detect_batch(tile_images)
        detections = self._postprocessor.merge_tiled_detections(
            tile_detections, tiling_result
        )
        result.detections = detections
        logger.info(f"  Found {len(detections)} symbols")

        # ---- Step 3: OCR Text Extraction ----
        logger.info("Step 3/6: Extracting text (detection-guided Tesseract OCR)...")

        # Detection-guided: for each YOLO bbox, crop a label search region
        # below/beside the symbol and run Tesseract on just that clean patch.
        # This avoids the pipe-line confusion that kills full-image OCR.
        # Also runs a full-image Tesseract pass (psm 11) for standalone labels.
        text_regions = self._ocr.extract_with_detections(image, detections)
        result.text_regions = text_regions
        if text_regions:
            detected_texts = [f'"{tr.text}" ({tr.confidence:.2f})' for tr in text_regions]
            logger.info(f"  Found {len(text_regions)} text regions: {', '.join(detected_texts[:25])}")
        else:
            logger.warning("  No text detected!")

        # ---- Step 4: Entity Mapping & Graph ----
        logger.info("Step 4/6: Building entity graph...")

        entities = self._entity_mapper.map_entities(detections, text_regions)
        graph = self._graph_builder.build(entities)
        graph = self._relationship_resolver.resolve(graph, entities)

        result.entities = entities
        result.graph = graph
        result.graph_summary = self._graph_builder.get_graph_summary()
        logger.info(f"  Graph: {result.graph_summary['total_nodes']} nodes, "
                     f"{result.graph_summary['total_edges']} edges")

        # ---- Step 5: Validation ----
        logger.info("Step 5/6: Validating extraction...")

        # Confidence scoring
        breakdowns = self._confidence_scorer.score_entities(entities)
        result.confidence_summary = self._confidence_scorer.get_summary_stats(breakdowns)

        # Rule-based validation
        rule_report = self._rule_validator.validate(entities, graph)
        result.rule_report = rule_report.to_dict()

        # LLM validation (optional)
        if run_llm_validation:
            entities_data = [e.to_dict() for e in entities]
            llm_result = self._llm_validator.validate(
                entities_data,
                result.graph_summary,
                result.rule_report.get("issues", []),
            )
            result.llm_report = llm_result

        # ---- Step 6: MTO Generation ----
        if generate_mto:
            logger.info("Step 6/6: Generating MTO spreadsheet...")
            result.mto_path = self._mto_generator.generate(
                entities=entities,
                graph=graph,
                project_name=project_name,
                sheet_ref=input_path.stem,
                validation_report=result.llm_report if run_llm_validation else None,
            )
            logger.info(f"  MTO saved: {result.mto_path}")

        # ---- Generate annotated image ----
        annotated = self._detector.draw_detections(image, detections)
        annotated = self._ocr.draw_text_regions(annotated, text_regions)
        result.annotated_image = annotated

        result.processing_time = time.time() - start_time

        logger.info(f"{'='*60}")
        logger.info(f"Pipeline complete in {result.processing_time:.1f}s")
        logger.info(f"{'='*60}")

        return result

    def process_image_array(
        self,
        image: np.ndarray,
        project_name: str = "P&ID Extraction",
        run_llm_validation: bool = True,
        generate_mto: bool = True,
    ) -> PipelineResult:
        """
        Run the pipeline on a numpy array image (used by the API).

        Args:
            image: BGR numpy array.
            project_name: Project name.
            run_llm_validation: Run LLM validation.
            generate_mto: Generate MTO Excel.

        Returns:
            PipelineResult.
        """
        start_time = time.time()
        self._init_components()

        result = PipelineResult()
        result.image_size = (image.shape[1], image.shape[0])

        # Upscale small images (same logic as process())
        min_width = 1920
        h_orig, w_orig = image.shape[:2]
        if w_orig < min_width:
            scale_up = min_width / w_orig
            new_w = int(w_orig * scale_up)
            new_h = int(h_orig * scale_up)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            logger.info(f"  Image upscaled {w_orig}×{h_orig} → {new_w}×{new_h}")

        # Preprocess
        enhanced = self._preprocessor.enhance(image)

        # Detect
        tiling_result = self._tiler.tile(enhanced)
        tile_images = [t.image for t in tiling_result.tiles]
        tile_detections = self._detector.detect_batch(tile_images)
        detections = self._postprocessor.merge_tiled_detections(
            tile_detections, tiling_result
        )
        result.detections = detections

        # OCR — detection-guided: crop label regions around each symbol
        text_regions = self._ocr.extract_with_detections(image, detections)
        result.text_regions = text_regions
        if text_regions:
            detected_texts = [f'"{tr.text}"' for tr in text_regions[:10]]
            logger.info(f"  OCR found {len(text_regions)} regions: {', '.join(detected_texts)}")
        else:
            logger.warning("  OCR: no text detected.")

        # Graph
        entities = self._entity_mapper.map_entities(detections, text_regions)
        graph = self._graph_builder.build(entities)
        graph = self._relationship_resolver.resolve(graph, entities)
        result.entities = entities
        result.graph = graph
        result.graph_summary = self._graph_builder.get_graph_summary()

        # Validation
        breakdowns = self._confidence_scorer.score_entities(entities)
        result.confidence_summary = self._confidence_scorer.get_summary_stats(breakdowns)
        rule_report = self._rule_validator.validate(entities, graph)
        result.rule_report = rule_report.to_dict()

        if run_llm_validation:
            entities_data = [e.to_dict() for e in entities]
            result.llm_report = self._llm_validator.validate(
                entities_data, result.graph_summary,
                result.rule_report.get("issues", []),
            )

        # MTO
        if generate_mto:
            result.mto_path = self._mto_generator.generate(
                entities, graph, project_name,
                validation_report=result.llm_report if run_llm_validation else None,
            )

        # Annotated image
        annotated = self._detector.draw_detections(image.copy(), detections)
        annotated = self._ocr.draw_text_regions(annotated, text_regions)
        result.annotated_image = annotated

        result.processing_time = time.time() - start_time
        return result
