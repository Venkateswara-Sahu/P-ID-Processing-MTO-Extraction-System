"""
FastAPI Backend for P&ID MTO Extraction Pipeline.

Provides REST API endpoints for uploading P&IDs, running extraction,
and downloading MTO results.

Security posture
----------------
- Uploads are capped by size, checked by magic bytes, and stored under a
  server-generated name (never the client filename).
- An optional API key gates every endpoint.
- CORS defaults to same-origin only; an allowlist is opt-in.
- Jobs and their uploads expire after a configurable retention window.
"""

import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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
    version="1.1.0",
)

# CORS: same-origin only unless an allowlist is configured. The previous
# default of '*' with no authentication let any web page drive this API
# from a user's browser, spending the operator's Groq credits.
_allowed_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)

# In-memory job storage. Entries expire after job_retention_seconds.
jobs: Dict[str, dict] = {}

# Pipeline instance (lazy-loaded)
pipeline: Optional[PIDPipeline] = None

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf", ".bmp"}

# Magic-byte signatures for the allowed types. Extension checks alone let
# a .exe through with a .png name.
_MAGIC_BYTES = {
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".bmp": [b"BM"],
    ".tiff": [b"II*\x00", b"MM\x00*"],
    ".tif": [b"II*\x00", b"MM\x00*"],
    ".pdf": [b"%PDF-"],
}


def get_pipeline() -> PIDPipeline:
    global pipeline
    if pipeline is None:
        pipeline = PIDPipeline()
    return pipeline


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Reject requests without the configured API key (if one is set)."""
    if not settings.api_key:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _sanitize_filename(name: Optional[str]) -> str:
    """
    Reduce a client-supplied filename to a safe display name.

    The name is only echoed back to the client and written to the log, so
    it must not carry path separators or newlines.
    """
    if not name:
        return "upload"
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" .")
    return base[:120] or "upload"


def _cleanup_expired_jobs() -> None:
    """Remove jobs (and their uploaded files) past the retention window."""
    cutoff = time.time() - settings.job_retention_seconds
    expired = [jid for jid, j in jobs.items() if j.get("created_at", 0) < cutoff]
    for jid in expired:
        job = jobs.pop(jid, None)
        if not job:
            continue
        for key in ("file_path", "annotated_path"):
            path = job.get(key)
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass


def _looks_like_allowed_type(content: bytes, suffix: str) -> bool:
    """Check the file's magic bytes against the allowed types."""
    signatures = _MAGIC_BYTES.get(suffix)
    if not signatures:
        return False
    return any(content.startswith(sig) for sig in signatures)


# ---- Endpoints ----


@app.get("/")
async def root():
    """API health check."""
    return {
        "service": "P&ID MTO Extraction API",
        "version": "1.1.0",
        "status": "running",
        "endpoints": [
            "POST /upload — Upload a P&ID image or PDF",
            "POST /process/{job_id} — Run extraction pipeline",
            "GET /results/{job_id} — Get extraction results",
            "GET /mto/{job_id}/download — Download MTO Excel",
            "GET /graph/{job_id} — Get graph visualization data",
        ],
    }


@app.post("/upload", dependencies=[Depends(require_api_key)])
async def upload_pid(file: UploadFile = File(...)):
    """Upload a P&ID image or PDF for processing."""
    _cleanup_expired_jobs()

    safe_name = _sanitize_filename(file.filename)
    suffix = Path(safe_name).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {suffix or '(none)'}. "
                f"Allowed: {sorted(ALLOWED_SUFFIXES)}"
            ),
        )

    # Read at most the limit plus one byte, so an oversized upload is
    # rejected without pulling the whole thing into memory.
    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_mb} MB limit",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    # Validate the content, not just the extension.
    if not _looks_like_allowed_type(content, suffix):
        raise HTTPException(
            status_code=400,
            detail=(
                "File content does not match its extension. "
                f"Expected {suffix} data."
            ),
        )

    # Generate job ID and save the file under a SERVER-generated name.
    job_id = str(uuid.uuid4())[:8]
    upload_dir = Path(settings.output_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / f"{job_id}{suffix}"
    file_path.write_bytes(content)

    jobs[job_id] = {
        "id": job_id,
        "filename": safe_name,
        "file_path": str(file_path),
        "status": "uploaded",
        "result": None,
        "created_at": time.time(),
    }

    logger.info(f"Uploaded: {safe_name} ({len(content)} bytes) -> Job {job_id}")

    return {
        "job_id": job_id,
        "filename": safe_name,
        "status": "uploaded",
        "message": f"File uploaded. Call POST /process/{job_id} to start extraction.",
    }


@app.post("/process/{job_id}", dependencies=[Depends(require_api_key)])
def process_pid(
    job_id: str,
    project_name: str = "P&ID Extraction",
    run_llm: bool = True,
):
    """
    Run the extraction pipeline on an uploaded P&ID.

    Deliberately a sync `def`: FastAPI dispatches sync endpoints to a
    threadpool, so this 30-60 second job no longer blocks the event loop
    and freeze every other request while it runs.
    """
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


@app.get("/results/{job_id}", dependencies=[Depends(require_api_key)])
async def get_results(job_id: str):
    """Get the extraction results for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        return {"job_id": job_id, "status": job["status"]}

    return {"job_id": job_id, "status": "completed", **job["result"]}


@app.get("/mto/{job_id}/download", dependencies=[Depends(require_api_key)])
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


@app.get("/graph/{job_id}", dependencies=[Depends(require_api_key)])
async def get_graph(job_id: str):
    """Get graph visualization data for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if "graph_vis" not in job:
        raise HTTPException(status_code=400, detail="Graph data not available")

    return job["graph_vis"]


@app.get("/annotated/{job_id}", dependencies=[Depends(require_api_key)])
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

    # reload=True is a development setting: it watches the filesystem and
    # restarts the worker on any change. It is not the production config.
    uvicorn.run(
        "app.api:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
