import uuid
from sqlalchemy import Column, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import DateTime
from sqlalchemy.sql import func

from goldenduck_core.db.base import Base


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(Text, nullable=False, default="queued")
    triggered_by = Column(UUID(as_uuid=True), nullable=False)

    started_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at = Column(DateTime(timezone=True))

    # JSON blob: testing_mode, n_assets, n_scenarios, run_evaluation
    config = Column(JSONB, nullable=False, default=dict)

    # Coarse step progress
    step = Column(Text)
    step_index = Column(Integer)

    # Saved model filename e.g. rf_delta_20260402_143000.pkl
    model_name = Column(Text)

    # Is this run's model the currently active inference model?
    is_active = Column(Boolean, nullable=False, default=False)

    # Full evaluation_report.json contents stored inline
    evaluation_report = Column(JSONB)

    error = Column(Text)
