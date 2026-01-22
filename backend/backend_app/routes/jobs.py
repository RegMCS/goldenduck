import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend_app.schemas.jobs import GenerateRequest, GenerateResponse
from backend_app.models.enums import JobStatus
from backend_app.services.job_store import job_store

router = APIRouter(prefix="/api", tags=["jobs"])

# Define OUTPUT_DIR as a module-level constant
OUTPUT_DIR = Path("output").resolve()


@router.post("/generate/user/{user_id}", response_model=GenerateResponse)
async def generate_job(user_id: str, request: GenerateRequest):
    job_id = str(uuid.uuid4())

    job_store.create_job(
        job_id=job_id,
        user_id=user_id,
        parameters=request.dict(),
        status=JobStatus.queued,
    )
    job_store.enqueue(job_id)

    return GenerateResponse(
        job_id=job_id,
        status=JobStatus.queued,
        message=f"Job submitted. Poll /api/status/user/{user_id}/{job_id}",
    )


@router.get("/status/user/{user_id}/{job_id}")
async def get_job_status(user_id: str, job_id: str):
    job = job_store.get_job(job_id)

    if not job or job["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Job not found")

    status = JobStatus(job["status"])

    response = {
        "job_id": job_id,
        "status": status,
    }

    if status == JobStatus.completed:
        response.update(
            {
                "parameters": job_store.get_parameters(job_id),
                "metrics": job_store.get_metrics(job_id),
                "download_url": f"/api/download/user/{user_id}/{job_id}",
            }
        )

    if status == JobStatus.failed:
        response["error"] = job_store.get_error(job_id)

    return response


@router.get("/download/user/{user_id}/{job_id}")
async def download_results(user_id: str, job_id: str):
    job = job_store.get_job(job_id)

    if not job or job["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Job not found")

    output_file = job_store.get_output_file(job_id)

    if not output_file:
        raise HTTPException(status_code=404, detail="File not ready")

    # Prevent path traversal
    try:
        output_file_path = Path(output_file).resolve()
        output_file_path.relative_to(OUTPUT_DIR)
    except (ValueError, OSError):
        raise HTTPException(status_code=403, detail="Access denied")

    if not output_file_path.exists():
        raise HTTPException(status_code=404, detail="File not ready")

    return FileResponse(
        path=output_file_path,
        media_type="text/csv",
        filename=f"synthetic_garch_{job_id}.csv",
    )
