"""
API tests.

Covers upload validation, the size limit, content sniffing, filename
sanitisation, auth, CORS, and job expiry.
"""
import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api as api


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture(autouse=True)
def clean_jobs():
    api.jobs.clear()
    yield
    api.jobs.clear()


def _png_bytes(size=64):
    """A minimal valid PNG header plus padding."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * size


# --------------------------------------------------------------------------- #
# Upload validation
# --------------------------------------------------------------------------- #

def test_valid_png_is_accepted(client):
    r = client.post("/upload", files={"file": ("plan.png", io.BytesIO(_png_bytes()), "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "uploaded"


def test_executable_extension_is_rejected(client):
    r = client.post("/upload", files={"file": ("bad.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]


def test_content_must_match_the_extension(client):
    """
    An .exe with a .png name was accepted and saved verbatim. The content
    must be checked, not just the extension.
    """
    r = client.post(
        "/upload",
        files={"file": ("evil.png", io.BytesIO(b"MZ\x90\x00 windows exe"), "image/png")},
    )
    assert r.status_code == 400
    assert "does not match" in r.json()["detail"]


def test_oversized_upload_is_rejected(client, monkeypatch):
    """An upload must not be able to exhaust memory or disk."""
    monkeypatch.setattr(api.settings, "max_upload_mb", 1)
    payload = _png_bytes(2 * 1024 * 1024)
    r = client.post("/upload", files={"file": ("big.png", io.BytesIO(payload), "image/png")})
    assert r.status_code == 413
    assert "limit" in r.json()["detail"]


def test_empty_upload_is_rejected(client):
    r = client.post("/upload", files={"file": ("empty.png", io.BytesIO(b""), "image/png")})
    assert r.status_code == 400


def test_missing_file_is_rejected(client):
    r = client.post("/upload")
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# Path traversal
# --------------------------------------------------------------------------- #

def test_path_traversal_in_the_filename_is_neutralised(client):
    """
    The on-disk path is derived from a server-generated UUID, never the
    client filename.
    """
    r = client.post(
        "/upload",
        files={"file": ("../../etc/passwd.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert r.status_code == 200
    job = api.jobs[r.json()["job_id"]]
    stored = Path(job["file_path"]).resolve()
    upload_root = (Path(api.settings.output_dir) / "uploads").resolve()
    assert upload_root in stored.parents, (
        f"the file was stored outside the uploads dir: {stored}"
    )


def test_filename_is_sanitised_for_display(client):
    r = client.post(
        "/upload",
        files={"file": ("../../evil\nname.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert r.status_code == 200
    name = r.json()["filename"]
    assert "/" not in name and "\\" not in name
    assert "\n" not in name


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #

def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setattr(api.settings, "api_key", "secret-key")
    api.jobs.clear()

    r = client.post("/upload", files={"file": ("plan.png", io.BytesIO(_png_bytes()), "image/png")})
    assert r.status_code == 401

    r = client.post(
        "/upload",
        headers={"X-API-Key": "secret-key"},
        files={"file": ("plan.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert r.status_code == 200


def test_no_api_key_means_no_auth_required(client, monkeypatch):
    monkeypatch.setattr(api.settings, "api_key", "")
    r = client.post("/upload", files={"file": ("plan.png", io.BytesIO(_png_bytes()), "image/png")})
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
# CORS
# --------------------------------------------------------------------------- #

def test_cors_is_not_wide_open_by_default(client):
    """allow_origins=['*'] with no auth let any site drive this API."""
    r = client.options(
        "/upload",
        headers={
            "Origin": "http://evil.test",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.headers.get("access-control-allow-origin") != "*"


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #

def test_unknown_job_is_404(client):
    assert client.get("/results/doesnotexist").status_code == 404


def test_processing_an_unknown_job_is_404(client):
    assert client.post("/process/doesnotexist").status_code == 404


def test_expired_jobs_are_cleaned_up(client, monkeypatch):
    """Jobs and their uploads must not accumulate forever."""
    import time

    r = client.post("/upload", files={"file": ("plan.png", io.BytesIO(_png_bytes()), "image/png")})
    job_id = r.json()["job_id"]
    path = Path(api.jobs[job_id]["file_path"])
    assert path.exists()

    # Backdate the job past the retention window.
    api.jobs[job_id]["created_at"] = time.time() - 10_000
    monkeypatch.setattr(api.settings, "job_retention_seconds", 1)

    client.post("/upload", files={"file": ("plan2.png", io.BytesIO(_png_bytes()), "image/png")})

    assert job_id not in api.jobs, "the expired job was not removed"
    assert not path.exists(), "the expired job's upload was not deleted"


# --------------------------------------------------------------------------- #
# The process endpoint must not block the event loop
# --------------------------------------------------------------------------- #

def test_process_endpoint_is_a_sync_def():
    """
    An `async def` endpoint running a 30-60 second blocking job freezes
    every other request for its full duration.
    """
    import inspect

    assert not inspect.iscoroutinefunction(api.process_pid), (
        "/process must be a sync def so FastAPI runs it in a threadpool"
    )
