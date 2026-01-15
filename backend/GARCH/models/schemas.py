# models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional


class GenerateRequest(BaseModel):
    """Request model for synthetic data generation"""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    num_scenarios: int = Field(1000, ge=1, le=10000, description="Number of scenarios")
    horizon: int = Field(252, ge=1, le=1000, description="Days per scenario")
    volatility_multiplier: float = Field(
        1.0, ge=0.1, le=5.0, description="Volatility scaling"
    )
    p: int = Field(1, ge=1, le=5, description="GARCH p order")
    q: int = Field(1, ge=1, le=5, description="ARCH q order")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "num_scenarios": 1000,
                "horizon": 252,
                "volatility_multiplier": 1.2,
                "p": 1,
                "q": 1,
            }
        }


class GARCHParameters(BaseModel):
    """GARCH model parameters"""

    omega: float
    alpha: float
    beta: float
    converged: bool
    aic: float
    bic: float


class ValidationMetrics(BaseModel):
    """Validation metrics for synthetic data"""

    ks_statistic: float
    ks_pvalue: float
    kurtosis_historical: float
    kurtosis_synthetic: float
    acf_lag1_historical: float
    acf_lag1_synthetic: float


class GenerateResponse(BaseModel):
    """Response model for generation request"""

    job_id: str
    status: str
    message: str
    parameters: Optional[GARCHParameters] = None
    validation_metrics: Optional[ValidationMetrics] = None
    download_url: Optional[str] = None
