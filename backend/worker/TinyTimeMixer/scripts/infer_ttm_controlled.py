"""
Inference script for controlled TTM (daily) with rolling horizon support.

Outputs CSV with OHLCV predictions.

Run:
  python infer_ttm_controlled.py --ticker AAPL --horizon 120 --output-csv out.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
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


def build_context(
    df: pd.DataFrame,
    cfg: TTMControlledConfig,
    scaler: StandardScaler,
    controls: ControlValues,
) -> Tuple[np.ndarray, float, pd.Timestamp, float]:
    feats = build_base_features(df, cfg)
    if len(feats) < cfg.context_length:
        raise ValueError(
            f"Not enough rows for context_length={cfg.context_length}. "
            f"Have {len(feats)} rows after feature construction."
        )
    feats = feats.iloc[-cfg.context_length :].copy()

    past_raw = feats[TARGET_FEATURES].values.astype(np.float32)
    past_sigma = float(np.std(past_raw[:, 0]) + 1e-8)
    past_scaled = scaler.transform(past_raw)

    ctrl_scaled = scale_controls(controls, cfg.control_ranges)
    past_ctrl = np.repeat(ctrl_scaled[None, :], cfg.context_length, axis=0)
    past_values = np.concatenate([past_scaled, past_ctrl], axis=1)

    last_close = float(feats["Close"].iloc[-1])
    last_date = pd.to_datetime(feats["date"].iloc[-1])
    return past_values, last_close, last_date, past_sigma


@torch.no_grad()
def rollout_forecast(
    model: torch.nn.Module,
    past_scaled: np.ndarray,
    cfg: TTMControlledConfig,
    scaler: StandardScaler,
    controls: ControlValues,
    *,
    horizon: int,
    roll_step: int,
    device: str,
    freq_token_value: Optional[int] = None,
) -> np.ndarray:
    ctrl_scaled = scale_controls(controls, cfg.control_ranges)
    remaining = int(horizon)

    current = past_scaled.copy()
    outputs_scaled = []

    while remaining > 0:
        step = min(roll_step, remaining)
        past_ctrl = np.repeat(ctrl_scaled[None, :], cfg.context_length, axis=0)
        past_values = np.concatenate([current, past_ctrl], axis=1)

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

        current = np.concatenate([current, pred_scaled], axis=0)[-cfg.context_length :]
        remaining -= step

    pred_scaled_all = np.concatenate(outputs_scaled, axis=0)
    pred_raw = scaler.inverse_transform(pred_scaled_all)
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
    parser.add_argument("--mean-reversion", type=float, default=0.0)
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to finetuned weights."
    )
    parser.add_argument("--scaler", type=str, default=None, help="Path to scaler.")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON.")
    parser.add_argument(
        "--output-csv", type=str, required=True, help="CSV output path."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Defaults follow training outputs
    default_dir = Path(__file__).resolve().parents[1] / "outputs" / "ttm_controlled"
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
    cfg.detrend_returns = bool(config.get("detrend_returns", cfg.detrend_returns))
    cfg.detrend_window = int(config.get("detrend_window", cfg.detrend_window))
    cfg.detrend_mode = str(config.get("detrend_mode", cfg.detrend_mode))

    scaler_state = torch.load(scaler_path, weights_only=False)
    scaler = StandardScaler.from_state_dict(scaler_state)

    controls = ControlValues(
        volatility_mult=float(args.volatility_mult),
        trend=float(args.trend),
        fat_tails=float(args.fat_tails),
        momentum=float(args.momentum),
        mean_reversion=float(args.mean_reversion),
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

    raw = download_daily_ohlcv(args.ticker, start=start, end=end)
    past_values, last_close, last_date, past_sigma = build_context(raw, cfg, scaler, controls)

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
        scaler,
        controls,
        horizon=int(args.horizon),
        roll_step=int(args.roll_step),
        device=device,
        freq_token_value=freq_token_value,
    )

    pred_features = apply_inference_noise(
        pred_features,
        sigma=past_sigma,
        controls=controls,
        cfg=cfg,
    )

    if abs(controls.trend) < 1e-8:
        pred_features[:, 0] = pred_features[:, 0] - float(pred_features[:, 0].mean())

    ohlcv = reconstruct_ohlcv_from_features(last_close, pred_features, last_date)
    ohlcv.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
