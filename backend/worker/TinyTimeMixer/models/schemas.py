from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional


# -----------------------------
# Shared specs
# -----------------------------

TaskType = Literal["forecast", "generate_paths"]
TargetType = Literal["price", "return", "log_return"]
ScalingType = Literal["none", "standard", "minmax", "revin"]
OutputFormat = Literal["csv", "parquet"]
ModelFamily = Literal["ttm_r1", "ttm_r2", "ttm_research", "custom"]


class FeatureSpec(BaseModel):
    """
    Defines what goes into the model and how it is preprocessed.
    This should be saved alongside the trained model artifacts.
    """

    # What columns/features are provided to the model input.
    # Examples: ["open", "high", "low", "close", "volume"]
    features: List[str] = Field(..., min_length=1)

    # What the model is trained to predict.
    # For most forecasting: target=close or target=log_return
    target: str = Field(..., description="Target column name, e.g., 'close' or 'log_return'")

    target_type: TargetType = Field(
        "log_return",
        description="Whether target is raw price, simple return, or log return",
    )

    scaling: ScalingType = Field(
        "revin",
        description="Normalization strategy applied during training/inference",
    )

    # If you do log-return modeling, you may want to reconstruct price from last observed price
    reconstruct_price: bool = Field(
        True,
        description="If target_type is return/log_return, reconstruct price level outputs",
    )


class ModelSpec(BaseModel):
    """
    Identifies which trained model artifact to load.
    """

    family: ModelFamily = Field("ttm_r2")
    name: str = Field(..., description="Model name, e.g., 'ttm_r2_spy_ohlcv'")
    version: str = Field("latest", description="Artifact version tag, e.g., 'v1' or 'latest'")

    device: Literal["cpu", "cuda"] = Field("cpu")
    dtype: Literal["float32", "float16", "bfloat16"] = Field("float32")


class WindowSpec(BaseModel):
    """
    Defines the temporal shape.
    """

    context_len: int = Field(..., ge=16, le=4096)
    pred_len: int = Field(..., ge=1, le=1024)


# -----------------------------
# Inference / Generation
# -----------------------------

class TinyMixerGenerateRequest(BaseModel):
    """
    Request for forecasting or generating synthetic paths using a trained TinyTimeMixer/TTM model.
    """

    ticker: str = Field(..., description="Ticker symbol (e.g., AAPL)")

    task: TaskType = Field(
        "forecast",
        description="forecast = deterministic forecast; generate_paths = stochastic paths from residual bootstrapping",
    )

    # How many future steps, and how much past context
    window: WindowSpec

    # How many synthetic scenarios (for generate_paths)
    num_scenarios: int = Field(
        100, ge=1, le=10000, description="Number of scenarios/paths for generate_paths"
    )

    # Used when task=generate_paths
    noise_method: Literal["gaussian", "bootstrap_residuals"] = Field(
        "bootstrap_residuals",
        description="How to inject randomness for multiple scenario generation",
    )
    noise_scale: float = Field(
        1.0, ge=0.0, le=10.0, description="Scale multiplier on injected noise"
    )
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")

    # Model selection
    model: ModelSpec

    # Feature + preprocessing contract
    feature_spec: FeatureSpec

    output_format: OutputFormat = Field("csv")

    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "AAPL",
                "task": "generate_paths",
                "window": {"context_len": 512, "pred_len": 96},
                "num_scenarios": 100,
                "noise_method": "bootstrap_residuals",
                "noise_scale": 1.0,
                "seed": 42,
                "model": {"family": "ttm_r2", "name": "ttm_r2_aapl_ohlcv", "version": "latest", "device": "cpu", "dtype": "float32"},
                "feature_spec": {
                    "features": ["open", "high", "low", "close", "volume"],
                    "target": "close",
                    "target_type": "log_return",
                    "scaling": "revin",
                    "reconstruct_price": True
                },
                "output_format": "csv"
            }
        }


class ForecastPoint(BaseModel):
    """
    Optional: structured response for direct JSON outputs (if not using download-only).
    """
    t: int
    values: Dict[str, float]


class TinyMixerGenerateResponse(BaseModel):
    """
    Generate Response
    """

    job_id: str
    status: str
    message: str

    ticker: str
    task: TaskType

    # For traceability
    model: ModelSpec
    feature_spec: FeatureSpec
    window: WindowSpec

    # Optional metrics/debug info
    metrics: Optional[Dict[str, float]] = None

    download_url: Optional[str] = None


# -----------------------------
# Training (optional but recommended)
# -----------------------------

class TinyMixerTrainRequest(BaseModel):
    """
    A training job spec. 
    """

    tickers: List[str] = Field(..., min_length=1)
    window: WindowSpec

    feature_spec: FeatureSpec

    # Architecture
    family: ModelFamily = Field("ttm_r2")
    d_model: int = Field(256, ge=16, le=2048)
    n_layers: int = Field(8, ge=1, le=64)
    patch_len: int = Field(16, ge=1, le=256)

    token_mlp_ratio: float = Field(2.0, ge=0.5, le=8.0)
    channel_mlp_ratio: float = Field(2.0, ge=0.5, le=8.0)
    dropout: float = Field(0.1, ge=0.0, le=0.8)

    # Optimization
    epochs: int = Field(20, ge=1, le=500)
    batch_size: int = Field(64, ge=1, le=4096)
    lr: float = Field(1e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(0.0, ge=0.0, le=1.0)

    seed: int = Field(42, ge=0)

    # Artifact naming
    artifact_name: str = Field(..., description="e.g., 'ttm_r2_aapl_ohlcv_v1'")
    artifact_version: str = Field("v1")


class TinyMixerTrainResponse(BaseModel):
    job_id: str
    status: str
    message: str

    artifact_name: str
    artifact_version: str

    metrics: Optional[Dict[str, float]] = None
    artifact_path: Optional[str] = None
