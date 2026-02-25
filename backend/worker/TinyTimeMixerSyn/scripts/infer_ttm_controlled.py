"""
Inference script for controlled TTM (daily) with rolling horizon support.

Outputs CSV with OHLCV predictions.

Run:
  python infer_ttm_controlled.py --ticker AAPL --horizon 120 --output-csv out.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.worker.TinyTimeMixer.services.ttm_controlled_dataset import (
    TARGET_FEATURES,
    EXOG_FEATURES,
    ControlRanges,
    ControlValues,
    StandardScaler,
    TTMControlledConfig,
    build_base_features,
    download_daily_ohlcv,
    apply_inference_noise,
    reconstruct_ohlcv_from_features,
    scale_controls,
)

# optional: map frequency string to token id for prefix tuning
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


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def load_input_ohlcv(
    ticker: str,
    start: str,
    end: str,
    input_csv: Optional[Path],
    scenario_id: Optional[int],
) -> pd.DataFrame:
    if input_csv is None:
        return download_daily_ohlcv(ticker, start=start, end=end)

    df = pd.read_csv(input_csv)
    if "date" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "date"})
        else:
            raise ValueError("Input CSV missing a date column (date/Date/Datetime).")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns {missing}.")

    df["date"] = pd.to_datetime(df["date"])
    if "scenario_id" in df.columns and scenario_id is not None:
        df = df[df["scenario_id"] == scenario_id].reset_index(drop=True)
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=required)
    return df


def build_context(
    df: pd.DataFrame,
    cfg: TTMControlledConfig,
    target_scaler: StandardScaler,
    exog_scaler: StandardScaler,
    controls: ControlValues,
) -> Tuple[np.ndarray, np.ndarray, float, pd.Timestamp, float]:
    feats = build_base_features(df, cfg)
    if len(feats) < cfg.context_length:
        raise ValueError(
            f"Not enough rows for context_length={cfg.context_length}. "
            f"Have {len(feats)} rows after feature construction."
        )
    feats = feats.iloc[-cfg.context_length :].copy()

    past_raw = feats[TARGET_FEATURES].values.astype(np.float32)
    past_exog = feats[EXOG_FEATURES].values.astype(np.float32)
    past_sigma = float(np.std(past_raw[:, 0]) + 1e-8)
    past_scaled = target_scaler.transform(past_raw)
    past_exog_scaled = exog_scaler.transform(past_exog)

    ctrl_scaled = scale_controls(controls, cfg.control_ranges)
    past_ctrl = np.repeat(ctrl_scaled[None, :], cfg.context_length, axis=0)
    past_values = np.concatenate([past_scaled, past_exog_scaled, past_ctrl], axis=1)

    last_close = float(feats["Close"].iloc[-1])
    last_date = pd.to_datetime(feats["date"].iloc[-1])
    return past_values, past_raw, last_close, last_date, past_sigma


@torch.no_grad()
def rollout_forecast(
    model: torch.nn.Module,
    past_scaled: np.ndarray,
    cfg: TTMControlledConfig,
    target_scaler: StandardScaler,
    exog_scaler: StandardScaler,
    controls: ControlValues,
    *,
    horizon: int,
    roll_step: int,
    device: str,
    freq_token_value: Optional[int] = None,
    past_raw: Optional[np.ndarray] = None,
) -> np.ndarray:
    ctrl_scaled = scale_controls(controls, cfg.control_ranges)
    remaining = int(horizon)

    current = past_scaled.copy()
    if past_raw is None:
        raise ValueError("past_raw is required for exogenous feature computation.")
    current_raw = past_raw.copy()
    outputs_scaled = []

    while remaining > 0:
        step = min(roll_step, remaining)
        past_ctrl = np.repeat(ctrl_scaled[None, :], cfg.context_length, axis=0)
        returns_series = pd.Series(current_raw[:, 0])
        vol_window = int(cfg.realized_vol_window)
        realized_vol = returns_series.rolling(vol_window, min_periods=2).std().shift(1)
        realized_vol = realized_vol.bfill().fillna(0.0).values.astype(np.float32)
        past_exog = realized_vol.reshape(-1, 1)
        past_exog_scaled = exog_scaler.transform(past_exog)

        past_values = np.concatenate([current, past_exog_scaled, past_ctrl], axis=1)

        x = torch.tensor(past_values, dtype=torch.float32, device=device).unsqueeze(0)
        if freq_token_value is not None:
            freq_token = torch.full(
                (x.shape[0],), int(freq_token_value), device=device, dtype=torch.long
            )
            out = model(past_values=x, freq_token=freq_token)
        else:
            out = model(past_values=x)
        y_hat = extract_predictions(out)
        y_hat = maybe_fix_pred_shape(y_hat, num_channels=past_values.shape[-1])
        if y_hat.shape[1] != cfg.prediction_length:
            y_hat = y_hat[:, : cfg.prediction_length, :]

        y_hat = y_hat[:, :step, : cfg.num_target_features]
        pred_scaled = y_hat.squeeze(0).detach().cpu().numpy()
        outputs_scaled.append(pred_scaled)

        pred_raw = target_scaler.inverse_transform(pred_scaled)
        current = np.concatenate([current, pred_scaled], axis=0)[-cfg.context_length :]
        current_raw = np.concatenate([current_raw, pred_raw], axis=0)[
            -cfg.context_length :
        ]
        remaining -= step

    pred_scaled_all = np.concatenate(outputs_scaled, axis=0)
    pred_raw = target_scaler.inverse_transform(pred_scaled_all)
    return pred_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled TTM inference (daily).")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., AAPL).")
    parser.add_argument(
        "--start", default=None, help="Start date (YYYY-MM-DD). Optional."
    )
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Optional.")
    parser.add_argument(
        "--lookback-days", type=int, default=365, help="Lookback days if start not set."
    )
    parser.add_argument(
        "--horizon", type=int, default=60, help="Forecast horizon (days)."
    )
    parser.add_argument(
        "--roll-step",
        type=int,
        default=1,
        help="Rollout step size (days). Use 1 for day-by-day generation.",
    )
    parser.add_argument("--volatility-mult", type=float, default=1.0)
    parser.add_argument("--trend", type=float, default=0.0)
    parser.add_argument("--fat-tails", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to finetuned weights."
    )
    parser.add_argument("--scaler", type=str, default=None, help="Path to scaler.")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON.")
    parser.add_argument(
        "--output-csv", type=str, required=True, help="CSV output path."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Optional OHLCV CSV for inference (overrides yfinance).",
    )
    parser.add_argument(
        "--scenario-id",
        type=int,
        default=None,
        help="Scenario ID filter if input CSV contains multiple scenarios.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Defaults follow training outputs
    base_dir = Path(__file__).resolve().parents[1]
    default_dir = base_dir / "model" / "ttm_controlled"
    weights_path = (
        Path(args.weights)
        if args.weights
        else (default_dir / "ttm_controlled_weights.pt")
    )
    scaler_path = (
        Path(args.scaler) if args.scaler else (default_dir / "ttm_controlled_scaler.pt")
    )
    config_path = (
        Path(args.config)
        if args.config
        else (default_dir / "ttm_controlled_config.json")
    )

    config = load_config(config_path)
    cfg = TTMControlledConfig(
        context_length=int(config["context_length"]),
        prediction_length=int(config["prediction_length"]),
    )
    cfg.control_ranges = ControlRanges(**config["control_ranges"])
    cfg.detrend_returns = False
    cfg.detrend_window = int(config.get("detrend_window", cfg.detrend_window))
    cfg.detrend_mode = str(config.get("detrend_mode", cfg.detrend_mode))
    cfg.realized_vol_window = int(
        config.get("realized_vol_window", cfg.realized_vol_window)
    )

    scaler_state = torch.load(scaler_path, weights_only=False)
    if (
        isinstance(scaler_state, dict)
        and "targets" in scaler_state
        and "exog" in scaler_state
    ):
        target_scaler = StandardScaler.from_state_dict(scaler_state["targets"])
        exog_scaler = StandardScaler.from_state_dict(scaler_state["exog"])
    else:
        target_scaler = StandardScaler.from_state_dict(scaler_state)
        exog_scaler = StandardScaler(
            mean=np.array([0.0], dtype=np.float32),
            std=np.array([1.0], dtype=np.float32),
            eps=1e-6,
        )

    controls = ControlValues(
        volatility_mult=float(args.volatility_mult),
        trend=float(args.trend),
        fat_tails=float(args.fat_tails),
        momentum=float(args.momentum),
        horizon=float(args.horizon),
    )

    end = args.end
    if end is None:
        end = pd.Timestamp.utcnow().date().isoformat()
    if args.start:
        start = args.start
    else:
        start = (
            (pd.Timestamp(end) - pd.Timedelta(days=args.lookback_days))
            .date()
            .isoformat()
        )

    raw = load_input_ohlcv(
        args.ticker,
        start=start,
        end=end,
        input_csv=(Path(args.input_csv) if args.input_csv else None),
        scenario_id=args.scenario_id,
    )
    past_values, past_raw, last_close, last_date, past_sigma = build_context(
        raw, cfg, target_scaler, exog_scaler, controls
    )

    try:
        from tsfm_public.toolkit.get_model import get_model
    except Exception as e:
        raise ImportError("Failed to import get_model from tsfm_public.") from e

    model = get_model(
        config["model_id"],
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        freq=config.get("freq", "D"),
        freq_prefix_tuning=True,
        num_input_channels=cfg.num_input_channels,
        prediction_channel_indices=cfg.prediction_channel_indices,
        exogenous_channel_indices=cfg.exogenous_channel_indices,
        decoder_mode="mix_channel",
        scaling=None,
    )

    state = torch.load(weights_path, map_location="cpu")
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

    pred_features = rollout_forecast(
        model,
        past_values[:, : cfg.num_target_features],
        cfg,
        target_scaler,
        exog_scaler,
        controls,
        horizon=int(args.horizon),
        roll_step=int(args.roll_step),
        device=device,
        freq_token_value=freq_token_value,
        past_raw=past_raw,
    )

    pred_features = apply_inference_noise(
        pred_features,
        sigma=past_sigma,
        controls=controls,
        cfg=cfg,
    )

    ohlcv = reconstruct_ohlcv_from_features(last_close, pred_features, last_date)
    ohlcv.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
