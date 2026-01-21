import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend_app.redis_client import redis_client
from backend_app.schemas.jobs import GenerateRequest, GenerateResponse
from backend_app.services.job_store import job_store

router = APIRouter(prefix="/api", tags=["jobs"])

# Define OUTPUT_DIR as a module-level constant
OUTPUT_DIR = Path("output").resolve()


@router.post("/generate", response_model=GenerateResponse)
async def generate_job(request: GenerateRequest):
    job_id = str(uuid.uuid4())

    job_store.create_job(job_id, request.dict())
    job_store.enqueue(job_id)

    return GenerateResponse(
        job_id=job_id,
        status="queued",
        message=f"Job submitted. Poll /api/status/{job_id}",
    )


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    status = job_store.get_status(job_id)

    if not status:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {"job_id": job_id, "status": status}

    if status == "completed":
        response["parameters"] = job_store.get_parameters(job_id)
        response["metrics"] = job_store.get_metrics(job_id)
        response["download_url"] = f"/api/download/{job_id}"

    if status == "failed":
        response["error"] = job_store.get_error(job_id)

    return response


@router.get("/download/{job_id}")
async def download_results(job_id: str):
    from fastapi.responses import FileResponse

    output_file = job_store.get_output_file(job_id)

    if not output_file:
        raise HTTPException(status_code=404, detail="File not ready")

    # Validate that the output_file path is within the expected OUTPUT_DIR
    # to prevent path traversal attacks
    try:
        output_file_path = Path(output_file).resolve()
        # Check if the output_file_path is a child of OUTPUT_DIR
        output_file_path.relative_to(OUTPUT_DIR)
    except (ValueError, OSError):
        # ValueError: not relative to OUTPUT_DIR
        # OSError: malformed path
        raise HTTPException(status_code=403, detail="Access denied")

    if not output_file_path.exists():
        raise HTTPException(status_code=404, detail="File not ready")

    return FileResponse(
        path=str(output_file_path),
        media_type="text/csv",
        filename=f"synthetic_garch_{job_id}.csv",
    )
