from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path

from backend_app.db.session import get_db
from backend_app.models.user import User
from fastapi.responses import FileResponse, RedirectResponse
import boto3
import os

from backend_app.services.job_service import create_job
from job_scheduler.models.enums import JobStatus
from job_scheduler.services.job_store import job_store
from job_scheduler.schemas.jobs import GenerateRequest, GenerateResponse

router = APIRouter(prefix="/api", tags=["jobs"])


@router.post("/generate/user/{user_id}", response_model=GenerateResponse)
async def generate_job(
    user_id: str,
    request: GenerateRequest,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        user = User(id=user_id)
        db.add(user)
        db.commit()

    # Create job in DB
    job = create_job(
        db=db,
        user_id=user_id,
        job_type=request.job_type,
    )

    job_store.create_job(
        job_id=str(job.id),
        user_id=user_id,
        parameters=request.model_dump(),
        status=JobStatus.queued,
    )

    # Enqueue job
    job_store.enqueue(str(job.id))

    return GenerateResponse(
        job_id=str(job.id),
        status=JobStatus.queued,
        message=f"Job submitted. Poll /api/status/user/{user_id}/{job.id}",
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

    # Handle S3 URLs
    if not output_file.startswith("s3://"):
        raise HTTPException(
            status_code=500, detail="Invalid S3 URL format stored in DB"
        )

    # Expected format: s3://bucket/key
    parts = output_file.replace("s3://", "").split("/", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=500, detail="Invalid S3 URL format stored in DB"
        )

    bucket_name, key = parts

    # Use env vars for creds
    s3_client = boto3.client("s3")
    try:
        url = s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket_name, "Key": key}, ExpiresIn=3600
        )
        return RedirectResponse(url=url)
    except Exception as e:
        # logger.error/print would be better but keeping it simple
        raise HTTPException(
            status_code=500, detail=f"Failed to generate download URL: {str(e)}"
        )
