import uuid

from sqlalchemy import Column, Text, Boolean, BigInteger, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from job_scheduler.db.base import Base


class UploadedModel(Base):
    __tablename__ = "uploaded_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(Text, unique=True, nullable=False)
    s3_key = Column(Text, nullable=False)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    file_size_bytes = Column(BigInteger)
    is_active = Column(Boolean, nullable=False, default=False)
