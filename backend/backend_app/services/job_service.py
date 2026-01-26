from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from backend_app.models.ai_model_job import AIModelJob
from job_scheduler.models.enums import JobStatus, JobType


def create_job(
    db: Session, user_id: str, job_type: JobType, commit: bool = True
) -> AIModelJob:
    job = AIModelJob(
        status=JobStatus.queued,
        job_type=job_type,
        requestor=user_id,
    )
    db.add(job)
    db.flush()  # Flush to get the job ID without committing
    if commit:
        db.commit()
    db.refresh(job)
    return job


def update_job_status(
    db: Session,
    job_id,
    status: JobStatus,
    s3_url: str | None = None,
):
    values = {"status": status}
    if status == JobStatus.completed:
        values["s3_url"] = s3_url
        values["completed_at"] = func.now()

    db.query(AIModelJob).filter_by(id=job_id).update(values)
    db.commit()
