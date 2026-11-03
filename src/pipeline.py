"""
P&ID MTO Extraction Pipeline — Main Orchestrator.

Coordinates the full extraction pipeline:
1. Preprocessing (PDF → enhanced images → tiles)
2. Detection (YOLOv8 symbol detection on tiles)
3. OCR (CC-guided Tesseract text extraction)
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
    scale_factor: float = 1.0
    per_page: list = field(default_factory=list)

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
            "scale_factor": round(self.scale_factor, 3),
            "pages": self.per_page,
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
        """
        Initialize all pipeline components.

        Idempotent: components that are already present are kept, so the
        22MB YOLOv8 model is loaded once per PIDPipeline rather than on
        every process() call.
        """
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

        if self._preprocessor is None:
            self._preprocessor = ImageEnhancer()
        if self._tiler is None:
            self._tiler = ImageTiler()
        if self._detector is None:
            self._detector = SymbolDetector()
        if self._postprocessor is None:
            self._postprocessor = DetectionPostProcessor()
        if self._ocr is None:
            self._ocr = TextExtractor()
        if self._tag_parser is None:
            self._tag_parser = TagParser()
        if self._entity_mapper is None:
            self._entity_mapper = EntityMapper()
        if self._graph_builder is None:
            self._graph_builder = PIDGraphBuilder()
        if self._relationship_resolver is None:
            self._relationship_resolver = RelationshipResolver()
        if self._rule_validator is None:
            self._rule_validator = EngineeringRuleValidator()
        if self._confidence_scorer is None:
            self._confidence_scorer = ConfidenceScorer()
        if self._llm_validator is None:
            self._llm_validator = LLMValidator()
        if self._mto_generator is None:
            self._mto_generator = MTOGenerator()

        logger.info("Pipeline components initialized")

    def process(
        self,
        input_path: str,
        project_name: str = "P&ID Extraction",
        run_llm_validation: bool = True,
        generate_mto: bool = True,
        max_pages: Optional[int] = None,
    ) -> PipelineResult:
        """
        Run the full extraction pipeline on a P&ID image or PDF.

        Args:
            input_path: Path to P&ID image (PNG, JPG) or PDF.
            project_name: Project name for MTO header.
            run_llm_validation: Whether to run LLM validation (requires API key).
            generate_mto: Whether to generate MTO Excel.
            max_pages: Maximum number of PDF pages to process. None means
                every page. P&ID packages are routinely multi-sheet and
                processing only the first page silently discarded the rest.

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

        pages: List[np.ndarray] = []
        if input_path.suffix.lower() == ".pdf":
            from src.preprocessing.pdf_converter import PDFConverter
            converter = PDFConverter()
            pages = converter.convert_to_numpy(str(input_path))
            if max_pages is not None:
                pages = pages[:max_pages]
            if not pages:
                raise ValueError(f"No renderable pages in PDF: {input_path}")
            logger.info(f"  PDF has {len(pages)} page(s) to process")
            image = pages[0]
        else:
            image = cv2.imread(str(input_path))
            if image is None:
                raise ValueError(f"Failed to load image: {input_path}")
            pages = [image]

        # The image is NOT upscaled for detection.
        #
        # The model was trained at imgsz=640 on images that were 640x640
        # (1:1) and 1312x1312 (0.49x). Upscaling a sheet before tiling
        # pushed the symbols far outside that distribution: measured on
        # the dataset's own val split, the old "upscale then tile at
        # 640" configuration scored F1 0.462, against 0.970 for
        # whole-image detection. Tiling at the model's native 1312
        # instead scored 0.958 on a 3936px sheet.
        #
        # OCR does not need the upscale: TextExtractor already inflates
        # the image internally for binarisation, and divides the
        # coordinates back out.
        h_orig, w_orig = image.shape[:2]
        scale_factor = 1.0

        # The reported size must describe the frame the coordinates live in.
        result.image_size = (w_orig, h_orig)
        result.scale_factor = scale_factor

        # ---- Steps 2-5 for every page ----
        all_detections: List = []
        all_text_regions: List = []
        all_entities: List = []
        all_graphs: List = []
        per_page: List[dict] = []

        for page_index, page_image in enumerate(pages):
            if len(pages) > 1:
                logger.info(f"  --- Page {page_index + 1} of {len(pages)} ---")

            page_result = self._process_page(page_image, page_index)

            all_detections.extend(page_result["detections"])
            all_text_regions.extend(page_result["text_regions"])
            all_entities.extend(page_result["entities"])
            if page_result["graph"] is not None:
                all_graphs.append(page_result["graph"])

            per_page.append({
                "page": page_index + 1,
                "detections": len(page_result["detections"]),
                "text_regions": len(page_result["text_regions"]),
                "entities": len(page_result["entities"]),
            })

        result.detections = all_detections
        result.text_regions = all_text_regions
        result.entities = all_entities
        result.per_page = per_page

        # Each page's mapper restarts its counter, so entity ids collide
        # across pages. Renumber them so every entity in the result has a
        # unique id.
        for index, entity in enumerate(all_entities, start=1):
            entity.entity_id = f"E{index:04d}"

        # ---- Build one graph over every page's entities ----
        graph = self._graph_builder.build(all_entities)
        graph = self._relationship_resolver.resolve(graph, all_entities)
        breakdowns = self._confidence_scorer.score_entities(all_entities, graph=graph)
        result.confidence_summary = self._confidence_scorer.get_summary_stats(breakdowns)

        result.graph = graph
        result.graph_summary = self._graph_builder.get_graph_summary()
        logger.info(f"  Graph: {result.graph_summary['total_nodes']} nodes, "
                     f"{result.graph_summary['total_edges']} edges")

        # ---- Step 5: Validation ----
        logger.info("Step 5/6: Validating extraction...")

        rule_report = self._rule_validator.validate(all_entities, graph)
        result.rule_report = rule_report.to_dict()

        # LLM validation (optional)
        if run_llm_validation:
            entities_data = [e.to_dict() for e in all_entities]
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
                entities=all_entities,
                graph=graph,
                project_name=project_name,
                sheet_ref=input_path.stem,
                validation_report=result.llm_report if run_llm_validation else None,
                rule_report=result.rule_report,
            )
            logger.info(f"  MTO saved: {result.mto_path}")

        # ---- Generate annotated image (first page) ----
        annotated = self._detector.draw_detections(image, all_detections)
        annotated = self._ocr.draw_text_regions(annotated, all_text_regions)
        result.annotated_image = annotated

        result.processing_time = time.time() - start_time

        logger.info(f"{'='*60}")
        logger.info(f"Pipeline complete in {result.processing_time:.1f}s")
        logger.info(f"{'='*60}")

        return result

    def _process_page(self, image: np.ndarray, page_index: int = 0) -> dict:
        """
        Run detection, OCR, entity mapping and graph building for one page.

        Returns a dict with the page's detections, text regions, entities
        and graph.
        """
        enhanced = self._preprocessor.enhance(image)

        # ---- Symbol Detection ----
        logger.info("Step 2/6: Detecting symbols (YOLOv8)...")

        tiling_result = self._tiler.tile(enhanced)
        tile_images = [t.image for t in tiling_result.tiles]
        tile_detections = self._detector.detect_batch(tile_images)
        detections = self._postprocessor.merge_tiled_detections(
            tile_detections, tiling_result
        )
        logger.info(f"  Found {len(detections)} symbols")

        # ---- OCR Text Extraction ----
        logger.info("Step 3/6: Extracting text (CC-guided Tesseract OCR)...")

        # The CC-guided extractor finds text-like blobs first, then reads
        # only those regions, which avoids the pipe-line confusion that
        # kills full-image OCR.
        text_regions = self._ocr.extract_with_detections(image, detections)
        if text_regions:
            detected_texts = [f'"{tr.text}" ({tr.confidence:.2f})' for tr in text_regions]
            logger.info(f"  Found {len(text_regions)} text regions: {', '.join(detected_texts[:25])}")
        else:
            logger.warning("  No text detected!")

        # ---- Entity Mapping & Graph ----
        logger.info("Step 4/6: Building entity graph...")

        entities = self._entity_mapper.map_entities(detections, text_regions)
        graph = self._graph_builder.build(entities)
        graph = self._relationship_resolver.resolve(graph, entities)

        return {
            "detections": detections,
            "text_regions": text_regions,
            "entities": entities,
            "graph": graph,
        }

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

        # NOT upscaled: see the note in process(). The model was trained
        # at a fixed scale and the OCR stage inflates the image itself.
        h_orig, w_orig = image.shape[:2]
        scale_factor = 1.0

        # Report the size of the frame the coordinates actually live in.
        result.image_size = (w_orig, h_orig)
        result.scale_factor = scale_factor

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

        # Score with the graph in hand so node confidences match the entities.
        breakdowns = self._confidence_scorer.score_entities(entities, graph=graph)
        result.confidence_summary = self._confidence_scorer.get_summary_stats(breakdowns)
        result.entities = entities
        result.graph = graph
        result.graph_summary = self._graph_builder.get_graph_summary()

        # Validation
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
                sheet_ref="P&ID",
                validation_report=result.llm_report if run_llm_validation else None,
                rule_report=result.rule_report,
            )

        # Annotated image
        annotated = self._detector.draw_detections(image.copy(), detections)
        annotated = self._ocr.draw_text_regions(annotated, text_regions)
        result.annotated_image = annotated

        result.processing_time = time.time() - start_time
        return result
