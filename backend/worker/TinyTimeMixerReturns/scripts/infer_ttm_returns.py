"""
Inference script for TTM returns-only model.

Outputs CSV with predicted log-returns and reconstructed close prices.

Run:
  python infer_ttm_returns.py --ticker AAPL --horizon 120 --output-csv out.csv
"""

from __future__ import annotations

import argparse
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
    StandardScaler,
    download_daily_ohlcv,
)

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


def build_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out = out.dropna().reset_index(drop=True)
    return out[["date", "Close", "log_return"]]


@torch.no_grad()
def rollout_forecast(
    model: torch.nn.Module,
    past_scaled: np.ndarray,
    *,
    horizon: int,
    pred_len: int,
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
        if y_hat.shape[1] != pred_len:
            y_hat = y_hat[:, :pred_len, :]

        y_hat = y_hat[:, :step, :]
        pred_scaled = y_hat.squeeze(0).detach().cpu().numpy()
        outputs_scaled.append(pred_scaled)

        current = np.concatenate([current, pred_scaled], axis=0)[-current.shape[0] :]
        remaining -= step

    return np.concatenate(outputs_scaled, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTM returns inference.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., AAPL).")
    parser.add_argument(
        "--start", default=None, help="Start date (YYYY-MM-DD). Optional."
    )
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Optional.")
    parser.add_argument(
        "--lookback-days", type=int, default=365 * 2, help="Lookback days if start not set."
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
    parser.add_argument("--weights", type=str, default=None, help="Path to weights.")
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

    base_dir = Path(__file__).resolve().parents[1]
    default_dir = base_dir / "model" / "ttm_returns"
    weights_path = (
        Path(args.weights) if args.weights else (default_dir / "ttm_returns_weights.pt")
    )
    scaler_path = (
        Path(args.scaler) if args.scaler else (default_dir / "ttm_returns_scaler.pt")
    )
    config_path = (
        Path(args.config) if args.config else (default_dir / "ttm_returns_config.json")
    )

    config = load_config(config_path)
    context_length = int(config["context_length"])
    pred_len = int(config["prediction_length"])

    scaler_state = torch.load(scaler_path, weights_only=False)
    scaler = StandardScaler.from_state_dict(scaler_state)

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
    feats = build_log_returns(raw)
    if len(feats) < context_length:
        raise ValueError(
            f"Not enough rows for context_length={context_length}. "
            f"Have {len(feats)} rows after feature construction."
        )

    past = feats["log_return"].values.astype(np.float32).reshape(-1, 1)
    past = past[-context_length:]
    past_scaled = scaler.transform(past)

    last_close = float(feats["Close"].iloc[-1])
    last_date = pd.to_datetime(feats["date"].iloc[-1])

    try:
        from tsfm_public.toolkit.get_model import get_model
    except Exception as e:
        raise ImportError("Failed to import get_model from tsfm_public.") from e

    model = get_model(
        config["model_id"],
        context_length=context_length,
        prediction_length=pred_len,
        freq=config.get("freq", "D"),
        freq_prefix_tuning=True,
        num_input_channels=1,
        prediction_channel_indices=[0],
        exogenous_channel_indices=[],
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

    pred_scaled = rollout_forecast(
        model,
        past_scaled,
        horizon=int(args.horizon),
        pred_len=pred_len,
        roll_step=int(args.roll_step),
        device=device,
        freq_token_value=freq_token_value,
    )

    pred_returns = scaler.inverse_transform(pred_scaled).squeeze(1)

    dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=len(pred_returns))
    close = [last_close]
    for r in pred_returns:
        close.append(close[-1] * float(np.exp(r)))
    close = close[1:]

    out_df = pd.DataFrame(
        {
            "Date": dates,
            "LogReturn": pred_returns,
            "Close": close,
        }
    )
    out_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
