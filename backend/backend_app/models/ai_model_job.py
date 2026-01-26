import uuid
from sqlalchemy import Column, DateTime, Text, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from backend_app.db.base import Base
from job_scheduler.models.enums import JobStatus, JobType


class AIModelJob(Base):
    __tablename__ = "ai_model_jobs"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    status = Column(Enum(JobStatus, name="job_status"), nullable=False)

    job_type = Column(Enum(JobType, name="job_type"))

    requestor = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False,
    )

    requested_at = Column(
        DateTime,
        nullable=False,
        default=func.now(),
    )

    s3_url = Column(Text)
    completed_at = Column(DateTime)
