from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from job_scheduler.models.enums import JobStatus, JobType


class GenerateRequest(BaseModel):
    job_type: JobType = JobType.garch  # default
    ticker: str
    horizon: int

    # GARCH-FX user knobs
    desired_volatility: float = Field(1.0, ge=0.5, le=2.0)
    desired_trend: float = Field(0.0, ge=-1.0, le=1.0)
    desired_fat_tails: float = Field(1.0, ge=0.8, le=1.5)
    desired_momentum: float = Field(0.5, ge=0.2, le=1.0)


class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
