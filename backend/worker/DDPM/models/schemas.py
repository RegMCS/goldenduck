"""
Pydantic schemas for DDPM requests and responses
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class GenerateDDPMRequest(BaseModel):
    """Request to generate DDPM paths (Stage 1 or Stage 2)"""

    stage: int = Field(1, ge=1, le=2, description="Stage 1 (unconditional) or Stage 2 (conditional)")
    num_paths: int = Field(2000, ge=100, le=10000, description="Number of paths to generate")
    guidance_scale: Optional[float] = Field(1.0, ge=1.0, le=10.0, description="CFG scale (only for Stage 2)")

    # Stage 2 conditioning parameters
    realised_vol: Optional[float] = Field(None, ge=0.01, le=1.0, description="Realised volatility (annualised)")
    drift: Optional[float] = Field(None, ge=-0.5, le=0.5, description="Drift/trend (annualised)")
    tail_index: Optional[float] = Field(None, ge=0.001, le=0.1, description="Tail index (99th percentile)")
    momentum: Optional[float] = Field(None, ge=-1.0, le=1.0, description="Momentum (autocorrelation)")


class DDPMPath(BaseModel):
    """Single generated path"""

    path_id: int
    returns: List[List[float]] = Field(..., description="Log returns (num_assets, seq_len)")
    regime: Optional[str] = Field(None, description="Detected regime: calm, highvol, crisis")


class ValidationMetrics(BaseModel):
    """Validation metrics for generated paths"""

    ks_statistic: float = Field(..., description="Kolmogorov-Smirnov test statistic")
    ks_pvalue: float = Field(..., description="KS test p-value (>0.05 is good)")
    kurtosis_historical: float = Field(..., description="Historical excess kurtosis")
    kurtosis_synthetic: float = Field(..., description="Synthetic excess kurtosis")
    skewness_historical: float = Field(..., description="Historical skewness")
    skewness_synthetic: float = Field(..., description="Synthetic skewness")
    acf_lag1_historical: float = Field(..., description="Historical ACF lag-1")
    acf_lag1_synthetic: float = Field(..., description="Synthetic ACF lag-1")
    acf_vol_historical: float = Field(..., description="Historical squared returns ACF")
    acf_vol_synthetic: float = Field(..., description="Synthetic squared returns ACF")


class RegimeHitMetrics(BaseModel):
    """Regime hit rate metrics"""

    calm_hit_rate: float = Field(..., description="% segments in calm regime")
    highvol_hit_rate: float = Field(..., description="% segments in high-vol regime")
    crisis_hit_rate: float = Field(..., description="% segments in crisis regime")
    target_calm: float = Field(..., description="Target % for calm regime")
    target_highvol: float = Field(..., description="Target % for high-vol regime")
    target_crisis: float = Field(..., description="Target % for crisis regime")


class GenerateDDPMResponse(BaseModel):
    """Response from DDPM generation"""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: queued, generating, validating, completed, failed")
    message: str = Field(..., description="Status message")
    num_paths_generated: Optional[int] = Field(None, description="Number of paths successfully generated")
    validation_metrics: Optional[ValidationMetrics] = Field(None, description="Validation metrics for generated data")
    regime_metrics: Optional[RegimeHitMetrics] = Field(None, description="Regime hit rate metrics (Stage 2 only)")
    download_url: Optional[str] = Field(None, description="URL to download generated CSV")
    error: Optional[str] = Field(None, description="Error message if job failed")


class DDPMParameters(BaseModel):
    """Saved DDPM model parameters and metadata"""

    stage: int = Field(description="Stage 1 or 2")
    num_assets: int = Field(7, description="Number of assets")
    window_len: int = Field(1260, description="Sequence length")
    t_steps: int = Field(200, description="Number of diffusion steps")
    base_channels: int = Field(64, description="Base channel count")
    time_emb_dim: int = Field(128, description="Time embedding dimension")
    depth: int = Field(4, description="Network depth")
    conditioning_dim: Optional[int] = Field(4, description="Conditioning dimension (Stage 2 only)")
    trained_epochs: int = Field(description="Number of epochs trained")
    best_loss: float = Field(description="Best validation loss achieved")
