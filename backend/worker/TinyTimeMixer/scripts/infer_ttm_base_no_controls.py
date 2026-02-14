"""
Inference script for base TTM (daily) without control channels.

Outputs CSV with OHLCV predictions.

Run:
  python infer_ttm_base_no_controls.py --ticker AAPL --horizon 120 --output-csv out.csv
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

from backend.worker.TinyTimeMixer.services.ttm_base_dataset import (
    StandardScaler,
    TTMBaseConfig,
)
from backend.worker.TinyTimeMixer.services.ttm_controlled_dataset import (
    build_base_features,
    download_daily_ohlcv,
    reconstruct_ohlcv_from_features,
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
    start: Optional[str],
    end: Optional[str],
    lookback_days: int,
    input_csv: Optional[Path],
) -> pd.DataFrame:
    if input_csv is None:
        if start is None or end is None:
            end_ts = pd.Timestamp.utcnow().normalize()
            start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
            start = start_ts.strftime("%Y-%m-%d")
            end = end_ts.strftime("%Y-%m-%d")
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
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=required)
    return df


def build_context(
    df: pd.DataFrame,
    cfg: TTMBaseConfig,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, float, pd.Timestamp, float]:
    feats = build_base_features(df, cfg)
    if len(feats) < cfg.context_length:
        raise ValueError(
            f"Not enough rows for context_length={cfg.context_length}. "
            f"Have {len(feats)} rows after feature construction."
        )
    feats = feats.iloc[-cfg.context_length :].copy()

    past_raw = feats[["log_return", "log_range", "log_volume"]].values.astype(
        np.float32
    )
    past_sigma = float(np.std(past_raw[:, 0]) + 1e-8)
    past_scaled = scaler.transform(past_raw)

    last_close = float(feats["Close"].iloc[-1])
    last_date = pd.to_datetime(feats["date"].iloc[-1])
    return past_scaled, last_close, last_date, past_sigma


@torch.no_grad()
def rollout_forecast(
    model: torch.nn.Module,
    past_scaled: np.ndarray,
    cfg: TTMBaseConfig,
    scaler: StandardScaler,
    *,
    horizon: int,
    roll_step: int,
    device: str,
    freq_token_value: Optional[int] = None,
) -> np.ndarray:
    remaining = int(horizon)
    current = past_scaled.copy()
    outputs_scaled = []

    while remaining > 0:
        step = min(roll_step, remaining)
        x = torch.tensor(current, dtype=torch.float32, device=device).unsqueeze(0)
        if freq_token_value is not None:
            freq_token = torch.full(
                (x.shape[0],), int(freq_token_value), device=device, dtype=torch.long
            )
            out = model(past_values=x, freq_token=freq_token)
        else:
            out = model(past_values=x)
        y_hat = extract_predictions(out)
        y_hat = maybe_fix_pred_shape(y_hat, num_channels=current.shape[-1])
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
    parser = argparse.ArgumentParser(description="Base TTM inference (daily).")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., AAPL).")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD). Optional.")
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
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to output CSV (synthetic OHLCV).",
    )
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to finetuned weights."
    )
    parser.add_argument("--scaler", type=str, default=None, help="Path to scaler.")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON.")
    parser.add_argument(
        "--anchor-blend",
        type=float,
        default=0.25,
        help="Anchor blend factor (0 no anchor, 1 full anchor).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    default_dir = Path(__file__).resolve().parents[1] / "outputs" / "ttm_base_version_no_controls"
    config_path = Path(args.config) if args.config else (default_dir / "ttm_base_config.json")
    weights_path = Path(args.weights) if args.weights else (default_dir / "ttm_base_weights.pt")
    scaler_path = Path(args.scaler) if args.scaler else (default_dir / "ttm_base_scaler.pt")

    config = load_config(config_path)
    cfg = TTMBaseConfig(
        context_length=int(config["context_length"]),
        prediction_length=int(config["prediction_length"]),
    )
    cfg.detrend_returns = bool(config.get("detrend_returns", cfg.detrend_returns))
    cfg.detrend_window = int(config.get("detrend_window", cfg.detrend_window))
    cfg.detrend_mode = str(config.get("detrend_mode", cfg.detrend_mode))

    scaler_state = torch.load(scaler_path, weights_only=False)
    scaler = StandardScaler.from_state_dict(scaler_state)

    raw = load_input_ohlcv(
        args.ticker,
        start=args.start,
        end=args.end,
        lookback_days=args.lookback_days,
        input_csv=(Path(args.input_csv) if args.input_csv else None),
    )

    past_values, last_close, last_date, _past_sigma = build_context(raw, cfg, scaler)

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

    pred_features = rollout_forecast(
        model,
        past_values,
        cfg,
        scaler,
        horizon=int(args.horizon),
        roll_step=int(args.roll_step),
        device=device,
        freq_token_value=freq_token_value,
    )

    synth_df = reconstruct_ohlcv_from_features(last_close, pred_features, last_date)

    if not synth_df.empty and args.anchor_blend > 0:
        anchor_close = float(raw["Close"].iloc[-1])
        first_synth_close = float(synth_df["Close"].iloc[0])
        if abs(first_synth_close) > 1e-12:
            raw_scale = anchor_close / first_synth_close
            scale = 1.0 + args.anchor_blend * (raw_scale - 1.0)
            price_cols = ["Open", "High", "Low", "Close"]
            synth_df[price_cols] = synth_df[price_cols] * scale

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(out_path, index=False)
    print(f"Saved synthetic CSV: {out_path}")


if __name__ == "__main__":
    main()
