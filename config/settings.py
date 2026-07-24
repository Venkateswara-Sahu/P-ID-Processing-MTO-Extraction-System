"""
Centralized configuration for the P&ID MTO Extraction Pipeline.
Uses pydantic-settings for type-safe environment variable management.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # ---- Groq LLM ----
    groq_api_key: str = Field(default="", description="Groq API key for LLM inference")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Groq model name")
    groq_temperature: float = Field(default=0.1, description="LLM temperature for validation")

    # ---- YOLOv8 Detection ----
    yolo_model_path: str = Field(
        default=str(PROJECT_ROOT / "models" / "best.pt"),
        description="Path to trained YOLOv8 weights",
    )
    yolo_confidence_threshold: float = Field(default=0.5, description="Minimum detection confidence")
    yolo_iou_threshold: float = Field(default=0.45, description="NMS IoU threshold")
    yolo_image_size: int = Field(default=640, description="YOLOv8 input image size")

    # ---- Image Tiling ----
    tile_size: int = Field(default=640, description="Tile dimension in pixels")
    tile_overlap: int = Field(default=64, description="Overlap between adjacent tiles in pixels")

    # ---- Preprocessing ----
    pdf_dpi: int = Field(default=300, description="DPI for PDF to image conversion")
    clahe_clip_limit: float = Field(default=2.0, description="CLAHE clip limit for contrast enhancement")
    clahe_grid_size: int = Field(default=8, description="CLAHE grid size")

    # ---- PaddleOCR ----
    ocr_language: str = Field(default="en", description="OCR language")
    ocr_confidence_threshold: float = Field(default=0.6, description="Minimum OCR confidence")

    # ---- MTO Output ----
    output_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "output"),
        description="Directory for MTO output files",
    )

    # ---- FastAPI ----
    app_host: str = Field(default="0.0.0.0", description="API host")
    app_port: int = Field(default=8000, description="API port")

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


# Singleton settings instance
settings = Settings()


# Engineering symbol categories for MTO grouping
SYMBOL_CATEGORIES = {
    "valves": [
        "valve.gate", "valve.globe", "valve.ball", "valve.check",
        "valve.butterfly", "valve.plug", "valve.needle", "valve.relief",
        "valve.control", "valve.solenoid", "valve.diaphragm",
    ],
    "instruments": [
        "instrument.flow", "instrument.pressure", "instrument.temperature",
        "instrument.level", "instrument.analyzer", "instrument.controller",
        "instrument.transmitter", "instrument.indicator",
    ],
    "equipment": [
        "equipment.pump", "equipment.compressor", "equipment.heat_exchanger",
        "equipment.vessel", "equipment.tank", "equipment.reactor",
        "equipment.column", "equipment.filter", "equipment.mixer",
    ],
    "piping": [
        "piping.reducer", "piping.tee", "piping.elbow",
        "piping.flange", "piping.cap", "piping.strainer",
        "piping.spectacle_blind", "piping.expansion_joint",
    ],
}

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
