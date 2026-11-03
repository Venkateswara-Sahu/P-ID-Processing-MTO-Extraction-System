"""
Centralized configuration for the P&ID MTO Extraction Pipeline.
Uses pydantic-settings for type-safe environment variable management.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def _resolve_against_root(value: str) -> str:
    """
    Resolve a possibly-relative path against the project root.

    A relative path in .env silently overrides an absolute default and becomes
    CWD-dependent: launched from anywhere other than the project root, the
    trained weights are "not found" and the code falls back to a different
    (COCO) model that has no P&ID classes.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # ---- Groq LLM ----
    groq_api_key: str = Field(default="", description="Groq API key for LLM inference")
    # NOTE: 'llama-3.3-70b-versatile' was retired by Groq and returns HTTP 404,
    # which made LLM validation silently never run. Verified working models:
    # 'openai/gpt-oss-120b', 'openai/gpt-oss-20b'.
    groq_model: str = Field(default="openai/gpt-oss-120b", description="Groq model name")
    groq_temperature: float = Field(default=0.1, description="LLM temperature for validation")
    groq_timeout: float = Field(default=60.0, description="Groq request timeout in seconds")
    groq_max_retries: int = Field(default=3, description="Groq retry attempts on rate limit")

    # ---- YOLOv8 Detection ----
    yolo_model_path: str = Field(
        default=str(PROJECT_ROOT / "models" / "best.pt"),
        description="Path to trained YOLOv8 weights (relative paths resolve against the project root)",
    )
    yolo_confidence_threshold: float = Field(default=0.5, description="Minimum detection confidence")
    yolo_iou_threshold: float = Field(default=0.45, description="NMS IoU threshold")
    yolo_image_size: int = Field(default=640, description="YOLOv8 input image size")

    # ---- Image Tiling ----
    # The model was trained at imgsz=640 on images that were 640x640 (1:1)
    # and 1312x1312 (downscaled 0.49x). Measured on the dataset's own val
    # split, tiling at 640 dropped F1 from 0.970 to 0.773, and the
    # pipeline's upscale-then-tile-640 combination scored 0.462.
    # A 1312 tile at imgsz=640 reproduces the training scale exactly, and
    # on a 3936x3936 sheet it scored F1 0.958.
    tile_size: int = Field(
        default=1312,
        description="Tile dimension in pixels (matches the model's native training resolution)",
    )
    tile_overlap: int = Field(
        default=128,
        description="Overlap between adjacent tiles in pixels",
    )
    drop_edge_detections: bool = Field(
        default=True,
        description=(
            "Discard detections whose box touches an internal tile boundary. "
            "A symbol cut by the boundary is detected as a partial shape, "
            "which is a false positive: on a 3936px sheet this raised "
            "precision from 0.870 to 0.946."
        ),
    )
    edge_tolerance_px: int = Field(
        default=6,
        description="How close to a tile boundary a box must be to count as cut",
    )

    # ---- Preprocessing ----
    pdf_dpi: int = Field(default=300, description="DPI for PDF to image conversion")
    clahe_clip_limit: float = Field(default=2.0, description="CLAHE clip limit for contrast enhancement")
    clahe_grid_size: int = Field(default=8, description="CLAHE grid size")
    enable_denoise: bool = Field(
        default=False,
        description=(
            "Apply non-local means denoising. Off by default: it costs ~20 min "
            "on an A1 sheet at 300 DPI and 1px line art survives CLAHE alone."
        ),
    )

    # ---- OCR ----
    ocr_language: str = Field(default="en", description="OCR language")
    ocr_max_dimension: int = Field(
        default=6000,
        description=(
            "Maximum pixel dimension of the image the OCR stage works on. "
            "The extractor upscales by 3x for binarisation, so an A1 sheet "
            "at 300 DPI would otherwise be inflated to ~21000px and blow "
            "up memory. The internal upscale factor is reduced instead."
        ),
    )
    ocr_confidence_threshold: float = Field(
        default=0.2,
        description=(
            "Minimum OCR confidence. Low on purpose: Tesseract's per-word "
            "confidence is unreliable on binarised line art, and the tag "
            "parser rejects noise better than a confidence gate does."
        ),
    )

    # ---- MTO Output ----
    output_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "output"),
        description="Directory for MTO output files",
    )

    # ---- FastAPI ----
    app_host: str = Field(default="0.0.0.0", description="API host")
    app_port: int = Field(default=8000, description="API port")
    max_upload_mb: int = Field(default=25, description="Maximum upload size in megabytes")
    job_retention_seconds: int = Field(
        default=3600, description="How long a job and its uploads are retained"
    )
    api_key: str = Field(
        default="", description="Optional API key required on every request"
    )
    cors_origins: str = Field(
        default="", description="Comma-separated allowed CORS origins (empty = same-origin only)"
    )

    # ---- Paths ----
    sample_pids_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "sample_pids"),
        description="Directory for sample P&ID files",
    )

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("yolo_model_path", "output_dir", "sample_pids_dir", mode="after")
    @classmethod
    def _resolve_paths(cls, value: str) -> str:
        return _resolve_against_root(value)


# Singleton settings instance
settings = Settings()


# --------------------------------------------------------------------------- #
# Engineering symbol categories                                           #
# --------------------------------------------------------------------------- #
#
# NOTE: this was previously a dead table of dotted names ("valve.gate")
# that matched none of the model's 229 real class names ("Gate Valve").
# The live mapping now lives in src/graph/class_map.py and is derived from
# the model's own vocabulary.

# Instrument tag prefixes (ISA standard)
INSTRUMENT_TAG_MAP = {
    "F": "Flow", "P": "Pressure", "T": "Temperature", "L": "Level",
    "A": "Analyzer", "C": "Controller", "D": "Density", "E": "Voltage",
    "H": "Hand", "I": "Current", "J": "Power", "K": "Time",
    "M": "Moisture", "N": "User Choice", "O": "User Choice",
    "Q": "Quantity", "R": "Radiation", "S": "Speed", "U": "Multivariable",
    "V": "Vibration", "W": "Weight", "X": "Unclassified", "Y": "Event",
    "Z": "Position",
}

# Instrument function suffixes
INSTRUMENT_FUNCTION_MAP = {
    "T": "Transmitter", "I": "Indicator", "C": "Controller",
    "V": "Valve", "E": "Element", "S": "Switch", "A": "Alarm",
    "R": "Recorder", "G": "Glass/Gauge", "H": "High", "L": "Low",
    "Y": "Relay/Compute", "Z": "Driver/Actuator",
}
