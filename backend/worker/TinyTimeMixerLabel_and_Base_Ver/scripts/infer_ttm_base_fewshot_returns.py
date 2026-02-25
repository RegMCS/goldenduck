"""
Few-shot inference with base TTM (no controls) for returns only.

This downloads the IBM Granite TTM model, optionally adapts it to recent data,
and runs inference on log_return.

Run:
  python infer_ttm_base_fewshot_returns.py --ticker AAPL --horizon 60

Optional few-shot adaptation:
  python infer_ttm_base_fewshot_returns.py --ticker AAPL --horizon 60 --fewshot-steps 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.worker.TinyTimeMixer.services.ttm_controlled_dataset import (
    download_daily_ohlcv,
)

try:
    from tsfm_public.toolkit.time_series_preprocessor import (
        DEFAULT_FREQUENCY_MAPPING,
        TimeSeriesPreprocessor,
    )
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
    TimeSeriesPreprocessor = None


MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
FREQ = "D"

DEFAULT_TICKER = "RDDT"
DEFAULT_OUTPUT_CSV_TEMPLATE = (
    "backend/worker/TinyTimeMixerNew/outputs/ttm_label/{ticker}_fewshot_returns.csv"
)

# Few-shot config defaults
FEWSHOT_UNFREEZE_KEYWORDS = [
    "head",
    "decoder",
    "prediction",
    "pred",
    "proj",
    "output",
    "lm_head",
]


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


def freeze_for_fewshot(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in FEWSHOT_UNFREEZE_KEYWORDS):
            p.requires_grad = True


class WindowDataset(Dataset):
    def __init__(
        self,
        series: np.ndarray,
        *,
        context_length: int,
        prediction_length: int,
        max_windows: Optional[int] = None,
    ) -> None:
        self.series = series
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.max_windows = max_windows

        n = len(series)
        max_start = n - (context_length + prediction_length)
        if max_start <= 0:
            raise ValueError("Not enough data to build few-shot windows.")
        indices = list(range(max_start + 1))
        if max_windows is not None and len(indices) > max_windows:
            indices = indices[-max_windows:]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        mid = start + self.context_length
        end = mid + self.prediction_length
        past = self.series[start:mid].copy()
        future = self.series[mid:end].copy()
        return (
            torch.tensor(past, dtype=torch.float32),
            torch.tensor(future, dtype=torch.float32),
        )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out = out.dropna().reset_index(drop=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot base TTM returns inference."
    )
    parser.add_argument("--ticker", default=None, help="Ticker symbol (e.g., AAPL).")
    parser.add_argument(
        "--start", default=None, help="Start date (YYYY-MM-DD). Optional."
    )
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Optional.")
    parser.add_argument(
        "--lookback-days", type=int, default=365, help="Lookback days if start not set."
    )
    parser.add_argument(
        "--context-length", type=int, default=180, help="Past window length."
    )
    parser.add_argument(
        "--prediction-length", type=int, default=60, help="Model prediction length."
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
        default=None,
        help="Path to output CSV (predicted returns).",
    )
    parser.add_argument(
        "--price-chart",
        type=str,
        default=None,
        help="Optional output path for a price chart PNG.",
    )
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument(
        "--fewshot-steps",
        type=int,
        default=0,
        help="If >0, perform few-shot adaptation steps before inference.",
    )
    parser.add_argument(
        "--fewshot-lr",
        type=float,
        default=1e-4,
        help="Few-shot learning rate.",
    )
    parser.add_argument(
        "--fewshot-batch-size",
        type=int,
        default=32,
        help="Few-shot batch size.",
    )
    parser.add_argument(
        "--fewshot-max-windows",
        type=int,
        default=512,
        help="Max windows for few-shot adaptation (most recent windows).",
    )
    return parser.parse_args()


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


@torch.no_grad()
def rollout_forecast(
    model: torch.nn.Module,
    past_scaled: np.ndarray,
    *,
    context_length: int,
    prediction_length: int,
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
        if y_hat.shape[1] != prediction_length:
            y_hat = y_hat[:, :prediction_length, :]

        y_hat = y_hat[:, :step, :]
        pred_scaled = y_hat.squeeze(0).detach().cpu().numpy()
        outputs_scaled.append(pred_scaled)

        current = np.concatenate([current, pred_scaled], axis=0)[-context_length:]
        remaining -= step

    return np.concatenate(outputs_scaled, axis=0)


def main() -> None:
    args = parse_args()
    if not args.ticker:
        args.ticker = DEFAULT_TICKER
    if not args.output_csv:
        args.output_csv = DEFAULT_OUTPUT_CSV_TEMPLATE.format(
            ticker=str(args.ticker).lower()
        )

    raw = load_input_ohlcv(
        args.ticker,
        start=args.start,
        end=args.end,
        lookback_days=args.lookback_days,
        input_csv=(Path(args.input_csv) if args.input_csv else None),
    )

    feats = build_features(raw)
    if len(feats) < args.context_length:
        raise ValueError(
            f"Not enough rows for context_length={args.context_length}. "
            f"Have {len(feats)} rows after feature construction."
        )

    if TimeSeriesPreprocessor is None:
        raise ImportError(
            "TimeSeriesPreprocessor is not available. Install tsfm_public to proceed."
        )

    tsp = TimeSeriesPreprocessor(
        id_columns=[],
        timestamp_column="date",
        target_columns=["log_return"],
        context_length=int(args.context_length),
        prediction_length=int(args.prediction_length),
        scaling=True,
        scaler_type="standard",
        scaling_id_columns=[],
        freq=FREQ,
    )
    tsp.train(feats)
    scaled_df = tsp.preprocess(feats)
    scaler = next(iter(tsp.target_scaler_dict.values()))

    context_feats = feats.iloc[-args.context_length :].copy()
    past_scaled = scaled_df.iloc[-args.context_length :][["log_return"]].values.astype(
        np.float32
    )

    last_date = pd.to_datetime(context_feats["date"].iloc[-1])
    dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=args.horizon)

    try:
        from tsfm_public.toolkit.get_model import get_model
    except Exception as e:
        raise ImportError("Failed to import get_model from tsfm_public.") from e

    model = get_model(
        MODEL_ID,
        context_length=int(args.context_length),
        prediction_length=int(args.prediction_length),
        freq=FREQ,
        freq_prefix_tuning=True,
        num_input_channels=1,
        prediction_channel_indices=[0],
        exogenous_channel_indices=[],
        decoder_mode="mix_channel",
        scaling=None,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    use_freq_token = bool(getattr(model.config, "resolution_prefix_tuning", False))
    freq_token_value = (
        DEFAULT_FREQUENCY_MAPPING.get(FREQ, None) if use_freq_token else None
    )

    if args.fewshot_steps and args.fewshot_steps > 0:
        freeze_for_fewshot(model)
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters for few-shot adaptation.")

        optimizer = torch.optim.AdamW(trainable, lr=float(args.fewshot_lr))
        loss_fn = torch.nn.MSELoss()

        series_scaled = scaled_df[["log_return"]].values.astype(np.float32)
        ds = WindowDataset(
            series_scaled,
            context_length=int(args.context_length),
            prediction_length=int(args.prediction_length),
            max_windows=int(args.fewshot_max_windows),
        )
        loader = DataLoader(ds, batch_size=int(args.fewshot_batch_size), shuffle=True)

        step = 0
        while step < int(args.fewshot_steps):
            for past, future in loader:
                past = past.to(device)
                future = future.to(device)

                freq_token = None
                if freq_token_value is not None:
                    freq_token = torch.full(
                        (past.shape[0],),
                        int(freq_token_value),
                        device=device,
                        dtype=torch.long,
                    )
                if freq_token is not None:
                    out = model(past_values=past, freq_token=freq_token)
                else:
                    out = model(past_values=past)
                y_hat = extract_predictions(out)
                y_hat = maybe_fix_pred_shape(y_hat, num_channels=past.shape[-1])
                if y_hat.shape[1] != int(args.prediction_length):
                    y_hat = y_hat[:, : int(args.prediction_length), :]

                loss = loss_fn(y_hat, future)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                step += 1
                if step >= int(args.fewshot_steps):
                    break

        model.eval()

    pred_scaled = rollout_forecast(
        model,
        past_scaled,
        context_length=int(args.context_length),
        prediction_length=int(args.prediction_length),
        horizon=int(args.horizon),
        roll_step=int(args.roll_step),
        device=device,
        freq_token_value=freq_token_value,
    )

    pred_raw = scaler.inverse_transform(pred_scaled)
    pred_log_return = pred_raw[:, 0]
    pred_return = np.expm1(pred_log_return)

    last_close = float(raw["Close"].iloc[-1])
    close_series = last_close * np.exp(np.cumsum(pred_log_return))

    out_df = pd.DataFrame(
        {
            "Date": dates,
            "pred_log_return": pred_log_return,
            "pred_return": pred_return,
            "pred_close": close_series,
        }
    )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predicted returns CSV: {out_path}")

    chart_path = (
        Path(args.price_chart)
        if args.price_chart
        else (out_path.parent / f"{str(args.ticker).lower()}_fewshot_price.png")
    )
    hist_df = raw.tail(int(args.context_length)).copy()
    hist_df = hist_df.rename(columns={"date": "Date"})

    plt.figure(figsize=(10, 4))
    plt.plot(hist_df["Date"], hist_df["Close"], label="Historical Close", linewidth=1.2)
    plt.plot(
        out_df["Date"], out_df["pred_close"], label="Predicted Close", linewidth=1.5
    )
    plt.title("Zero-shot TTM Predicted Price (Close) with History")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved price chart: {chart_path}")


if __name__ == "__main__":
    main()
