from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from job_scheduler.models.enums import JobStatus, JobType


class GenerateRequest(BaseModel):
    job_type: JobType = JobType.garch  # default
    ticker: str
    p: int
    q: int
    horizon: int
    num_scenarios: int
    volatility_multiplier: float = Field(1.0, gt=0.0, lt=10.0)


class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
