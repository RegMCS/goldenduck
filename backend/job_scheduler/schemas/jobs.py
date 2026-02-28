from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any
from job_scheduler.models.enums import JobStatus, JobType


class GenerateRequest(BaseModel):
    job_type: JobType = JobType.garch  # default
    ticker: Optional[str] = None  # Optional - provide either ticker or csv_data
    csv_data: Optional[str] = None  # CSV content as string (OHLCV format)
    horizon: int

    # GARCH-FX user knobs
    desired_volatility: float = Field(1.0, ge=0.5, le=2.0)
    desired_trend: float = Field(0.0, ge=-1.0, le=1.0)
    desired_fat_tails: float = Field(1.0, ge=0.5, le=2.0)
    desired_momentum: float = Field(0.5, ge=0.0, le=1.0)
    
    @model_validator(mode='after')
    def check_data_source(self):
        # Ensure either ticker or csv_data is provided (but not both or neither)
        if not self.ticker and not self.csv_data:
            raise ValueError('Either ticker or csv_data must be provided')
        if self.ticker and self.csv_data:
            raise ValueError('Provide either ticker or csv_data, not both')
        return self


class GenerateResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
