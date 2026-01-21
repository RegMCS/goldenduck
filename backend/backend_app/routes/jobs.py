import json
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException

from backend_app.redis_client import redis_client
from backend_app.schemas.jobs import GenerateRequest, GenerateResponse
from backend_app.services.job_store import job_store

router = APIRouter(prefix="/api", tags=["jobs"])


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
    import os

    output_file = job_store.get_output_file(job_id)

    if not output_file or not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="File not ready")

    return FileResponse(
        output_file,
        media_type="text/csv",
        filename=f"synthetic_garch_{job_id}.csv",
    )
