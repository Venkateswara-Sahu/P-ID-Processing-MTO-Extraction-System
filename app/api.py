"""
FastAPI Backend for P&ID MTO Extraction Pipeline.

Provides REST API endpoints for uploading P&IDs, running extraction,
and downloading MTO results.
"""

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.pipeline import PIDPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- App Setup ----

app = FastAPI(
    title="P&ID MTO Extraction API",
    description="AI-powered Material Take-Off extraction from P&ID engineering drawings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (for demo purposes)
jobs: Dict[str, dict] = {}

# Pipeline instance (lazy-loaded)
pipeline: Optional[PIDPipeline] = None


def get_pipeline() -> PIDPipeline:
    global pipeline
    if pipeline is None:
        pipeline = PIDPipeline()
    return pipeline


# ---- Endpoints ----


@app.get("/")
async def root():
    """API health check."""
    return {
        "service": "P&ID MTO Extraction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "POST /upload — Upload a P&ID image or PDF",
            "POST /process/{job_id} — Run extraction pipeline",
            "GET /results/{job_id} — Get extraction results",
            "GET /mto/{job_id}/download — Download MTO Excel",
            "GET /graph/{job_id} — Get graph visualization data",
        ],
    }


@app.post("/upload")
async def upload_pid(file: UploadFile = File(...)):
    """Upload a P&ID image or PDF for processing."""
    # Validate file type
    allowed_types = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf", ".bmp"}
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_types}",
        )

    # Generate job ID and save file
    job_id = str(uuid.uuid4())[:8]
    upload_dir = Path(settings.output_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{job_id}{suffix}"
    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    # Store job
    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "file_path": str(file_path),
        "status": "uploaded",
        "result": None,
    }

    logger.info(f"Uploaded: {file.filename} → Job {job_id}")

    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "uploaded",
        "message": f"File uploaded. Call POST /process/{job_id} to start extraction.",
    }


@app.post("/process/{job_id}")
async def process_pid(
    job_id: str,
    project_name: str = "P&ID Extraction",
    run_llm: bool = True,
):
    """Run the extraction pipeline on an uploaded P&ID."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if job["status"] == "processing":
        return {"job_id": job_id, "status": "processing", "message": "Already processing"}

    job["status"] = "processing"

    try:
        pipe = get_pipeline()
        result = pipe.process(
            input_path=job["file_path"],
            project_name=project_name,
            run_llm_validation=run_llm,
            generate_mto=True,
        )

        # Save annotated image
        if result.annotated_image is not None:
            annotated_path = str(
                Path(settings.output_dir) / f"{job_id}_annotated.png"
            )
            cv2.imwrite(annotated_path, result.annotated_image)
            job["annotated_path"] = annotated_path

        # Store results
        job["status"] = "completed"
        job["result"] = {
            "summary": result.to_summary(),
            "entities": [e.to_dict() for e in result.entities],
            "graph": result.graph_summary,
            "rule_validation": result.rule_report,
            "llm_validation": result.llm_report,
            "confidence": result.confidence_summary,
            "mto_path": result.mto_path,
        }

        # Store graph visualization data
        if result.graph is not None:
            from src.graph.graph_builder import PIDGraphBuilder
            builder = PIDGraphBuilder()
            builder.graph = result.graph
            job["graph_vis"] = builder.export_for_visualization()
            job["graph_data"] = builder.to_serializable()

        return {
            "job_id": job_id,
            "status": "completed",
            "summary": result.to_summary(),
        }

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.error(f"Processing failed for job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get the extraction results for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        return {"job_id": job_id, "status": job["status"]}

    return {"job_id": job_id, "status": "completed", **job["result"]}


@app.get("/mto/{job_id}/download")
async def download_mto(job_id: str):
    """Download the generated MTO Excel file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if job["status"] != "completed" or not job.get("result"):
        raise HTTPException(status_code=400, detail="Job not completed yet")

    mto_path = job["result"].get("mto_path")
    if not mto_path or not Path(mto_path).exists():
        raise HTTPException(status_code=404, detail="MTO file not found")

    return FileResponse(
        path=mto_path,
        filename=Path(mto_path).name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/graph/{job_id}")
async def get_graph(job_id: str):
    """Get graph visualization data for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if "graph_vis" not in job:
        raise HTTPException(status_code=400, detail="Graph data not available")

    return job["graph_vis"]


@app.get("/annotated/{job_id}")
async def get_annotated_image(job_id: str):
    """Get the annotated P&ID image with detection overlays."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    annotated_path = job.get("annotated_path")

    if not annotated_path or not Path(annotated_path).exists():
        raise HTTPException(status_code=404, detail="Annotated image not available")

    return FileResponse(path=annotated_path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
    )
