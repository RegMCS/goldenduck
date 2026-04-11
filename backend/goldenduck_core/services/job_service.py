from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from goldenduck_core.models.ai_model_job import AIModelJob
from goldenduck_core.models.enums import JobStatus, JobType


def create_job(db: Session, user_id: str, job_type: JobType) -> AIModelJob:
    job = AIModelJob(
        status=JobStatus.queued,
        job_type=job_type,
        requestor=user_id,
    )
    db.add(job)
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
    if status in (JobStatus.completed, JobStatus.failed):
        values["completed_at"] = func.now()
    if status == JobStatus.completed:
        values["s3_url"] = s3_url

    db.query(AIModelJob).filter_by(id=job_id).update(values)
    db.commit()
