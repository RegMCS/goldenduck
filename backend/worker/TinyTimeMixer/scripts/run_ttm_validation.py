"""Generate TTM validation CSVs from backend/AAPL.csv.

Default behavior:
- Input: backend/AAPL.csv
- Output folder: backend/ttm_validation
- Output file: baseline.csv
- Baseline controls: volatility=1.0, fat_tails=1.0, momentum=0.5, trend=0.0

Edit the CONFIG block below to change controls and output filename.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.worker.TinyTimeMixer.services.ttm_controlled_dataset import (
    CONTROL_NAMES,
    ControlRanges,
    ControlValues,
    StandardScaler,
    TTMControlledConfig,
    apply_inference_noise,
    reconstruct_ohlcv_from_features,
    scale_controls,
)

try:
    from tsfm_public.toolkit.get_model import get_model
except Exception as e:
    raise ImportError("Failed to import get_model from tsfm_public.") from e

try:
    from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING
except Exception:
    DEFAULT_FREQUENCY_MAPPING = {
        "min": 1,
        "2min": 2,
        "5min": 3,
        "10min": 4,
        "15min": 5,
        "30min": 6,
        "h": 7,
        "H": 7,
        "d": 8,
        "D": 8,
        "W": 9,
    }


# ==============================
# Editable config
# ==============================
PROJECT_ROOT = ROOT / "backend"
INPUT_CSV_PATH = PROJECT_ROOT / "AAPL.csv"
OUTPUT_DIR = PROJECT_ROOT / "ttm_validation"
OUTPUT_FILENAME = "baseline.csv"

MODEL_DIR = PROJECT_ROOT / "worker" / "TinyTimeMixer" / "outputs" / "ttm_controlled"
MODEL_CONFIG_PATH = MODEL_DIR / "ttm_controlled_config.json"
MODEL_WEIGHTS_PATH = MODEL_DIR / "ttm_controlled_weights.pt"
MODEL_SCALER_PATH = MODEL_DIR / "ttm_controlled_scaler.pt"

HORIZON = 120
ROLL_STEP = 1
SEED = 41
APPLY_NOISE = True

CONTROLS = ControlValues(
    volatility_mult=1.0,
    trend=0.0,
    fat_tails=1.0,
    momentum=0.5,
    horizon=float(HORIZON),
)


def extract_predictions(outputs) -> torch.Tensor:
    if hasattr(outputs, "predictions"):
        y_hat = outputs.predictions
    elif hasattr(outputs, "prediction_outputs"):
        y_hat = outputs.prediction_outputs
    elif hasattr(outputs, "logits"):
        y_hat = outputs.logits
    elif (
        isinstance(outputs, (tuple, list))
        and len(outputs) > 0
        and torch.is_tensor(outputs[0])
    ):
        y_hat = outputs[0]
    elif torch.is_tensor(outputs):
        y_hat = outputs
    else:
        raise RuntimeError(
            f"Could not extract predictions from outputs type: {type(outputs)}"
        )
    if not torch.is_tensor(y_hat):
        raise RuntimeError("Extracted predictions is not a tensor.")
    return y_hat


def maybe_fix_pred_shape(y_hat: torch.Tensor, num_channels: int) -> torch.Tensor:
    if y_hat.ndim != 3:
        raise RuntimeError(f"Expected 3D predictions, got {tuple(y_hat.shape)}")
    _, d1, d2 = y_hat.shape
    if d1 == num_channels and d2 != num_channels:
        return y_hat.transpose(1, 2).contiguous()
    return y_hat


def _load_input_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "date"})
        else:
            raise ValueError("Input CSV missing date column (date/Date/Datetime).")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=required)
    return df


def _build_feature_frame(
    raw: pd.DataFrame,
    *,
    target_features: list[str],
    exogenous_features: list[str],
    detrend_returns: bool,
    detrend_window: int,
    detrend_mode: str,
    realized_vol_window: int,
) -> pd.DataFrame:
    df = raw.copy()

    close = df["Close"].astype(float).clip(lower=1e-12)
    high = df["High"].astype(float).clip(lower=1e-12)
    low = df["Low"].astype(float).clip(lower=1e-12)
    volume = df["Volume"].astype(float).clip(lower=0.0)

    df["log_price"] = np.log(close)
    df["log_return"] = df["log_price"].diff()
    df["log_range"] = np.log(high / low)
    df["log_volume"] = np.log1p(volume)

    if detrend_returns:
        if detrend_mode == "rolling":
            roll = df["log_return"].rolling(detrend_window, min_periods=1).mean()
            df["log_return"] = df["log_return"] - roll
        elif detrend_mode == "mean":
            df["log_return"] = df["log_return"] - df["log_return"].mean()
        else:
            raise ValueError(
                f"Invalid detrend_mode={detrend_mode}. Use 'rolling' or 'mean'."
            )

    returns_proxy = df["log_return"]
    df["realized_vol"] = (
        returns_proxy.rolling(realized_vol_window, min_periods=2).std().shift(1)
    )

    needed = ["date", "Close", *target_features, *exogenous_features]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Feature '{col}' not available from input data.")

    feats = df[needed].dropna().reset_index(drop=True)
    return feats


def _identity_scaler(num_channels: int) -> StandardScaler:
    return StandardScaler(
        mean=np.zeros((num_channels,), dtype=np.float32),
        std=np.ones((num_channels,), dtype=np.float32),
        eps=1e-6,
    )


def _load_scalers(
    scaler_path: Path,
    *,
    num_target_features: int,
    num_exogenous_features: int,
) -> tuple[StandardScaler, StandardScaler]:
    scaler_state = torch.load(scaler_path, weights_only=False)
    if (
        isinstance(scaler_state, dict)
        and "targets" in scaler_state
        and "exog" in scaler_state
    ):
        target_scaler = StandardScaler.from_state_dict(scaler_state["targets"])
        exog_scaler = StandardScaler.from_state_dict(scaler_state["exog"])
        return target_scaler, exog_scaler

    target_scaler = StandardScaler.from_state_dict(scaler_state)
    exog_scaler = _identity_scaler(num_exogenous_features)

    if len(target_scaler.mean) != num_target_features:
        raise ValueError(
            "Target scaler dimension mismatch: "
            f"expected {num_target_features}, got {len(target_scaler.mean)}"
        )
    return target_scaler, exog_scaler


def _derive_returns_from_targets(
    targets_raw: np.ndarray, target_features: list[str]
) -> np.ndarray:
    if "log_return" in target_features:
        idx = target_features.index("log_return")
        return targets_raw[:, idx]
    if "log_price" in target_features:
        idx = target_features.index("log_price")
        lp = targets_raw[:, idx]
        return np.diff(lp, prepend=lp[0])
    return targets_raw[:, 0]


def _build_exogenous_context(
    targets_raw: np.ndarray,
    *,
    target_features: list[str],
    exogenous_features: list[str],
    realized_vol_window: int,
) -> np.ndarray:
    if not exogenous_features:
        return np.zeros((targets_raw.shape[0], 0), dtype=np.float32)

    returns = _derive_returns_from_targets(targets_raw, target_features)
    realized_vol = (
        pd.Series(returns)
        .rolling(realized_vol_window, min_periods=2)
        .std()
        .shift(1)
        .bfill()
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    cols: list[np.ndarray] = []
    for name in exogenous_features:
        if name == "realized_vol":
            cols.append(realized_vol)
        else:
            cols.append(np.zeros_like(realized_vol, dtype=np.float32))

    return np.column_stack(cols).astype(np.float32)


@torch.no_grad()
def _rollout_forecast(
    model: torch.nn.Module,
    *,
    past_targets_scaled: np.ndarray,
    past_targets_raw: np.ndarray,
    cfg: TTMControlledConfig,
    target_scaler: StandardScaler,
    exog_scaler: StandardScaler,
    controls: ControlValues,
    horizon: int,
    roll_step: int,
    target_features: list[str],
    exogenous_features: list[str],
    num_input_channels: int,
    device: str,
    freq_token_value: Optional[int],
) -> np.ndarray:
    ctrl_scaled = scale_controls(controls, cfg.control_ranges)
    current_scaled = past_targets_scaled.copy()
    current_raw = past_targets_raw.copy()
    remaining = int(horizon)
    outputs_scaled: list[np.ndarray] = []

    while remaining > 0:
        step = min(int(roll_step), remaining)
        exog_raw = _build_exogenous_context(
            current_raw,
            target_features=target_features,
            exogenous_features=exogenous_features,
            realized_vol_window=int(cfg.realized_vol_window),
        )
        exog_scaled = exog_scaler.transform(exog_raw)
        ctrl_block = np.repeat(ctrl_scaled[None, :], cfg.context_length, axis=0)
        past_values = np.concatenate([current_scaled, exog_scaled, ctrl_block], axis=1)

        x = torch.tensor(past_values, dtype=torch.float32, device=device).unsqueeze(0)
        if freq_token_value is not None:
            freq_token = torch.full(
                (x.shape[0],), int(freq_token_value), device=device, dtype=torch.long
            )
            out = model(past_values=x, freq_token=freq_token)
        else:
            out = model(past_values=x)

        y_hat = extract_predictions(out)
        y_hat = maybe_fix_pred_shape(y_hat, num_channels=num_input_channels)
        if y_hat.shape[1] != cfg.prediction_length:
            y_hat = y_hat[:, : cfg.prediction_length, :]

        y_hat = y_hat[:, :step, : len(target_features)]
        pred_scaled = y_hat.squeeze(0).detach().cpu().numpy()
        outputs_scaled.append(pred_scaled)

        pred_raw = target_scaler.inverse_transform(pred_scaled)
        current_scaled = np.concatenate([current_scaled, pred_scaled], axis=0)[
            -cfg.context_length :
        ]
        current_raw = np.concatenate([current_raw, pred_raw], axis=0)[-cfg.context_length :]
        remaining -= step

    return target_scaler.inverse_transform(np.concatenate(outputs_scaled, axis=0))


def main() -> None:
    if not MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Model config not found: {MODEL_CONFIG_PATH}")
    if not MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS_PATH}")
    if not MODEL_SCALER_PATH.exists():
        raise FileNotFoundError(f"Model scaler not found: {MODEL_SCALER_PATH}")

    raw = _load_input_csv(INPUT_CSV_PATH)
    config = json.loads(MODEL_CONFIG_PATH.read_text())

    target_features = list(config.get("target_features", ["log_return"]))
    exogenous_features = list(config.get("exogenous_features", ["realized_vol"]))

    cfg = TTMControlledConfig(
        context_length=int(config["context_length"]),
        prediction_length=int(config["prediction_length"]),
    )
    cfg.control_ranges = ControlRanges(**config["control_ranges"])
    cfg.detrend_returns = bool(config.get("detrend_returns", cfg.detrend_returns))
    cfg.detrend_window = int(config.get("detrend_window", cfg.detrend_window))
    cfg.detrend_mode = str(config.get("detrend_mode", cfg.detrend_mode))
    cfg.realized_vol_window = int(
        config.get("realized_vol_window", cfg.realized_vol_window)
    )
    cfg.infer_noise_enabled = bool(APPLY_NOISE and config.get("infer_noise_enabled", True))
    cfg.infer_return_noise_scale = float(
        config.get("infer_return_noise_scale", cfg.infer_return_noise_scale)
    )
    cfg.infer_range_noise_scale = float(
        config.get("infer_range_noise_scale", cfg.infer_range_noise_scale)
    )
    cfg.infer_volume_noise_scale = float(
        config.get("infer_volume_noise_scale", cfg.infer_volume_noise_scale)
    )
    cfg.noise_df_min = float(config.get("noise_df_min", cfg.noise_df_min))
    cfg.noise_df_max = float(config.get("noise_df_max", cfg.noise_df_max))

    target_scaler, exog_scaler = _load_scalers(
        MODEL_SCALER_PATH,
        num_target_features=len(target_features),
        num_exogenous_features=len(exogenous_features),
    )

    feats = _build_feature_frame(
        raw,
        target_features=target_features,
        exogenous_features=exogenous_features,
        detrend_returns=cfg.detrend_returns,
        detrend_window=cfg.detrend_window,
        detrend_mode=cfg.detrend_mode,
        realized_vol_window=cfg.realized_vol_window,
    )
    if len(feats) < cfg.context_length:
        raise ValueError(
            f"Not enough rows for context_length={cfg.context_length}. Have {len(feats)}."
        )
    feats = feats.iloc[-cfg.context_length :].copy()

    past_targets_raw = feats[target_features].to_numpy(dtype=np.float32)
    past_targets_scaled = target_scaler.transform(past_targets_raw)
    returns_proxy = _derive_returns_from_targets(past_targets_raw, target_features)
    past_sigma = float(np.std(returns_proxy) + 1e-8)
    last_close = float(feats["Close"].iloc[-1])
    last_date = pd.to_datetime(feats["date"].iloc[-1])

    num_target = len(target_features)
    num_exog = len(exogenous_features)
    num_ctrl = len(CONTROL_NAMES)
    num_input_channels = num_target + num_exog + num_ctrl
    prediction_channel_indices = list(range(num_target))
    exogenous_channel_indices = list(range(num_target, num_input_channels))

    model = get_model(
        config["model_id"],
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        freq=config.get("freq", "D"),
        freq_prefix_tuning=True,
        num_input_channels=num_input_channels,
        prediction_channel_indices=prediction_channel_indices,
        exogenous_channel_indices=exogenous_channel_indices,
        decoder_mode="mix_channel",
        scaling=None,
    )
    state = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    use_freq_token = bool(getattr(model.config, "resolution_prefix_tuning", False))
    freq_str = config.get("freq", "D")
    freq_token_value = (
        DEFAULT_FREQUENCY_MAPPING.get(freq_str, None) if use_freq_token else None
    )
    if use_freq_token and freq_token_value is None:
        raise ValueError(f"Frequency token not found for freq={freq_str}.")

    controls = ControlValues(
        volatility_mult=float(CONTROLS.volatility_mult),
        trend=float(CONTROLS.trend),
        fat_tails=float(CONTROLS.fat_tails),
        momentum=float(CONTROLS.momentum),
        horizon=float(HORIZON),
    )

    pred_features = _rollout_forecast(
        model,
        past_targets_scaled=past_targets_scaled,
        past_targets_raw=past_targets_raw,
        cfg=cfg,
        target_scaler=target_scaler,
        exog_scaler=exog_scaler,
        controls=controls,
        horizon=int(HORIZON),
        roll_step=int(ROLL_STEP),
        target_features=target_features,
        exogenous_features=exogenous_features,
        num_input_channels=num_input_channels,
        device=device,
        freq_token_value=freq_token_value,
    )

    if APPLY_NOISE:
        pred_features = apply_inference_noise(
            pred_features,
            sigma=past_sigma,
            controls=controls,
            cfg=cfg,
            rng=np.random.default_rng(SEED),
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    ohlcv = reconstruct_ohlcv_from_features(last_close, pred_features, last_date)
    ohlcv.to_csv(output_path, index=False)

    print("TTM validation run complete")
    print("Input CSV:", INPUT_CSV_PATH)
    print("Output CSV:", output_path)
    print("Rows generated:", len(ohlcv))
    print(
        "Controls:",
        {
            "volatility_mult": controls.volatility_mult,
            "trend": controls.trend,
            "fat_tails": controls.fat_tails,
            "momentum": controls.momentum,
            "horizon": controls.horizon,
        },
    )


if __name__ == "__main__":
    main()
