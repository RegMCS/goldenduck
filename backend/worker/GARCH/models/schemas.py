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
    error: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response model for generation request"""

    job_id: str
    status: str
    message: str
    parameters: Optional[GARCHParameters] = None
    validation_metrics: Optional[ValidationMetrics] = None
    download_url: Optional[str] = None


class GenerateFXRequest(BaseModel):
    """Request schema for GARCH-FX generation"""

    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, SPY)")
    num_scenarios: int = Field(
        1000, ge=1, le=10000, description="Number of independent scenarios to generate"
    )
    horizon: int = Field(
        252, ge=1, le=1000, description="Forecast horizon in trading days"
    )

    # GARCH parameters
    p: int = Field(1, ge=1, le=5, description="GARCH p parameter (lag order)")
    q: int = Field(1, ge=1, le=5, description="GARCH q parameter (lag order)")

    # GARCH-FX specific parameters
    theta: float = Field(
        0.005,
        ge=0.0001,
        le=0.1,
        description="Stochasticity parameter (0.001-0.005 balanced, >0.01 for stress)",
    )

    # Scenario selection
    scenario_type: Optional[str] = Field(
        None,
        description="Predefined scenario: sudden_crisis, gradual_escalation, crisis_waves, flash_crash, prolonged_stress",
    )

    delta_sequence: Optional[List[float]] = Field(
        None, description="Custom delta multiplier sequence (overrides scenario_type)"
    )

    # Regime switching
    regime_switching: bool = Field(
        False, description="Enable stochastic Markov chain regime switching"
    )

    regime_states: Optional[List[List[float]]] = Field(
        None, description="Custom transition probability matrix for regime switching"
    )

    regimes: Optional[List[float]] = Field(
        None, description="Delta multipliers for each regime state"
    )

    seed_start: int = Field(42, ge=0, description="Starting seed for reproducibility")

    class Config:
        schema_extra = {
            "example": {
                "ticker": "AAPL",
                "num_scenarios": 500,
                "horizon": 252,
                "p": 1,
                "q": 1,
                "theta": 0.01,
                "scenario_type": "sudden_crisis",
                "regime_switching": False,
                "seed_start": 42,
            }
        }


class GenerateFXResponse(BaseModel):
    """Response schema for GARCH-FX generation"""

    job_id: str
    status: str
    message: str
    scenario_type: Optional[str] = None
    theta: float


class ScenarioInfo(BaseModel):
    """Information about a predefined scenario"""

    name: str
    description: str


class ScenarioListResponse(BaseModel):
    """List of available scenarios"""

    scenarios: List[ScenarioInfo]
