"""
Test script for Bitcoin price forecasting with Tiny Time Mixer (TTM).

This is a runnable, no-CLI version of the BTC notebook pipeline:
  - loads BTC OHLCV data (CSV or yfinance)
  - optional resample
  - builds train/val/test splits
  - runs zero-shot (and optional fine-tune)
  - saves forecast vs actual + chart
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import Trainer, TrainingArguments, set_seed

from tsfm_public import TimeSeriesPreprocessor, TinyTimeMixerForPrediction, get_datasets
from tsfm_public.toolkit.time_series_forecasting_pipeline import (
    TimeSeriesForecastingPipeline,
)

# ----------------------------
# User-configurable section
# ----------------------------
INPUT_CSV = None  # e.g., "C:/path/to/ohlcv.csv"
TICKER = "AAPL"  # Used only if INPUT_CSV is None
INPUT_START = "2014-01-01"  # e.g., "2020-01-01" (only used if INPUT_CSV is None)
INPUT_END = "2024-12-31"  # e.g., "2024-12-31" (only used if INPUT_CSV is None)

TIMESTAMP_COL = "Timestamp"
TARGET_COL = "log_price"
OBSERVABLE_COLS = []
TICKER_COL = "ticker"

RESAMPLE = "1d"  # use "none" to disable resampling
LOOKBACK_DAYS = 365 * 3  # used if INPUT_CSV is None

CONTEXT_LENGTH = 512
FORECAST_LENGTH = 96
PREDICTION_FILTER_LENGTH = 24

FINETUNE = False
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SEED = 42
FEWSHOT_FRACTION = None  # e.g., 0.05 for 5% of training data
FEWSHOT_LOCATION = "last"  # "first" | "last" | "uniform"

MODEL_PATH = None  # set to local model dir if available

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs" / "ttm_bitcoin_test"
CHART_DIR = BASE_DIR / "outputs" / "charts"
OUTPUT_CSV = OUTPUT_DIR / "bitcoin_forecast_vs_actual.csv"
CHART_PATH = CHART_DIR / "bitcoin_pred_vs_actual.png"


def _maybe_convert_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        max_val = series.dropna().max()
        if max_val > 1e11:
            return pd.to_datetime(series, unit="ms", errors="coerce")
        return pd.to_datetime(series, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def load_input_csv(path: Path, timestamp_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": timestamp_col})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": timestamp_col})
        else:
            raise ValueError(f"Missing timestamp column: {timestamp_col}")
    df[timestamp_col] = _maybe_convert_timestamp(df[timestamp_col])
    df = df.dropna(subset=[timestamp_col])
    return df


def download_btc_data(
    ticker: str,
    *,
    lookback_days: int,
    interval: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    import yfinance as yf

    if end is None:
        end = pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")
    if start is None:
        start = (
            pd.Timestamp(end) - pd.Timedelta(days=int(lookback_days))
        ).strftime("%Y-%m-%d")
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    # yfinance may return MultiIndex columns (field, ticker)
    if isinstance(df.columns, pd.MultiIndex):
        level_values = df.columns.get_level_values(-1)
        if ticker in level_values:
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": TIMESTAMP_COL})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": TIMESTAMP_COL})
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    return df


def add_log_price(df: pd.DataFrame, *, close_col: str = "Close") -> pd.DataFrame:
    out = df.copy()
    if close_col not in out.columns:
        raise ValueError(f"Missing close column: {close_col}")
    out["log_price"] = np.log(out[close_col])
    return out


def load_data() -> pd.DataFrame:
    if INPUT_CSV:
        df = load_input_csv(Path(INPUT_CSV), TIMESTAMP_COL)
    else:
        if RESAMPLE and RESAMPLE.lower() not in ("none", ""):
            interval = RESAMPLE
        else:
            interval = "1d"
        df = download_btc_data(
            TICKER,
            lookback_days=LOOKBACK_DAYS,
            interval=interval,
            start=INPUT_START,
            end=INPUT_END,
        )

    if RESAMPLE and RESAMPLE.lower() not in ("none", ""):
        df = (
            df.resample(RESAMPLE, on=TIMESTAMP_COL)
            .mean()
            .dropna()
            .reset_index()
        )
    return add_log_price(df, close_col="Close")


def build_split_config(n: int, context_length: int) -> Dict[str, Tuple[int, int]]:
    train_end = round(n * 0.8)
    eval_start = train_end - context_length
    eval_end = round(n * 0.9)
    test_start = eval_end - context_length
    test_end = n
    return {
        "train": (0, train_end),
        "valid": (eval_start, eval_end),
        "test": (test_start, test_end),
    }


def compare_forecast(
    forecast: pd.DataFrame,
    date_col: str,
    prediction_col: str,
    actual_col: str,
    horizon_out: int,
) -> pd.DataFrame:
    rows = []
    for _, row in forecast.iterrows():
        try:
            pred_vals = row[prediction_col]
            actual_vals = row[actual_col]
            if len(pred_vals) <= horizon_out or len(actual_vals) <= horizon_out:
                continue
            rows.append(
                {
                    date_col: row[date_col],
                    "pred": pred_vals[horizon_out],
                    "actual": actual_vals[horizon_out],
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def plot_predictions(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    if "pred_price_from_return" in df.columns:
        pred_col = "pred_price_from_return"
        actual_col = "actual_price_from_return"
        title = "Pred vs Actual (Price from % Return)"
    elif "pred_price" in df.columns:
        pred_col = "pred_price"
        actual_col = "actual_price"
        title = "Pred vs Actual (Price)"
    else:
        pred_col = "pred"
        actual_col = "actual"
        title = "Pred vs Actual"
    ax.plot(df[TIMESTAMP_COL], df[pred_col], label=pred_col)
    ax.plot(df[TIMESTAMP_COL], df[actual_col], label=actual_col)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    n_rows = len(df)
    context_length = CONTEXT_LENGTH
    forecast_length = FORECAST_LENGTH
    min_required = context_length + forecast_length + 1
    if n_rows < min_required:
        # Auto-shrink to fit the available data.
        context_length = min(context_length, max(30, n_rows // 2))
        forecast_length = min(forecast_length, max(12, n_rows - context_length - 1))
        min_required = context_length + forecast_length + 1
        if n_rows < min_required:
            raise ValueError(
                "Not enough rows for the configured context/forecast lengths. "
                f"Have {n_rows} rows, need at least {min_required}. "
                "Reduce CONTEXT_LENGTH/FORECAST_LENGTH or provide more data."
            )
        print(
            f"[auto] Adjusted context_length={context_length}, "
            f"forecast_length={forecast_length} (rows={n_rows})"
        )

    if TICKER_COL not in df.columns:
        df[TICKER_COL] = TICKER

    tsp = TimeSeriesPreprocessor(
        id_columns=[TICKER_COL],
        scaling_id_columns=[TICKER_COL],
        timestamp_column=TIMESTAMP_COL,
        target_columns=[TARGET_COL],
        observable_columns=OBSERVABLE_COLS,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    split_config = build_split_config(len(df), context_length)
    if FEWSHOT_FRACTION is not None and not (0.0 < FEWSHOT_FRACTION <= 1.0):
        raise ValueError("FEWSHOT_FRACTION must be in (0, 1].")
    train_dataset, valid_dataset, _ = get_datasets(
        tsp,
        df,
        split_config,
        fewshot_fraction=FEWSHOT_FRACTION,
        fewshot_location=FEWSHOT_LOCATION,
    )

    model_id = MODEL_PATH if MODEL_PATH else "ibm/TTM"
    model = TinyTimeMixerForPrediction.from_pretrained(
        model_id, revision="main", prediction_filter_length=PREDICTION_FILTER_LENGTH
    )

    if FINETUNE:
        train_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / "train_output"),
            overwrite_output_dir=True,
            learning_rate=LEARNING_RATE,
            num_train_epochs=EPOCHS,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            save_strategy="epoch",
            save_total_limit=1,
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )
        trainer.train()

    pipeline = TimeSeriesForecastingPipeline(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        timestamp_column=TIMESTAMP_COL,
        id_columns=[TICKER_COL],
        target_columns=[TARGET_COL],
        observable_columns=OBSERVABLE_COLS,
        freq=RESAMPLE if RESAMPLE.lower() != "none" else "1h",
    )

    test_start, test_end = split_config["test"]
    test_data = tsp.preprocess(df[test_start:test_end])
    forecasts = pipeline(test_data)

    pred_col = f"{TARGET_COL}_prediction"
    forecast_predictions = compare_forecast(
        forecasts, TIMESTAMP_COL, pred_col, TARGET_COL, 12
    )
    forecast_out = forecast_predictions.dropna(subset=["actual", "pred"])

    if forecast_out.empty:
        print("No forecast rows available for evaluation.")
        return

    rmse = np.sqrt(np.mean((forecast_out["actual"] - forecast_out["pred"]) ** 2))
    print(f"RMSE (12 out): {rmse:.6f}")

    if TARGET_COL == "log_price":
        forecast_out = forecast_out.copy()
        forecast_out["pred_price"] = np.exp(forecast_out["pred"].astype(float))
        forecast_out["actual_price"] = np.exp(forecast_out["actual"].astype(float))
        rmse_price = np.sqrt(
            np.mean((forecast_out["actual_price"] - forecast_out["pred_price"]) ** 2)
        )
        print(f"RMSE price (12 out): {rmse_price:.6f}")

        # Convert to percentage returns from the start price at each timestep,
        # and apply to that start price to reconstruct forecasted price.
        # Use the start price from the input OHLCV used for inference.
        start_lookup = df[[TIMESTAMP_COL, "Close", "log_price"]].copy()
        start_lookup[TIMESTAMP_COL] = pd.to_datetime(start_lookup[TIMESTAMP_COL])
        start_lookup = start_lookup.sort_values(TIMESTAMP_COL)
        forecast_out[TIMESTAMP_COL] = pd.to_datetime(forecast_out[TIMESTAMP_COL])
        forecast_out = forecast_out.sort_values(TIMESTAMP_COL)

        # Align to the nearest prior timestamp if exact match is missing.
        forecast_out = pd.merge_asof(
            forecast_out,
            start_lookup,
            on=TIMESTAMP_COL,
            direction="backward",
        )
        forecast_out = forecast_out.rename(
            columns={"Close": "start_close", "log_price": "start_log_price"}
        )
        if forecast_out["start_log_price"].isna().any():
            print(
                "[warn] Missing start_log_price for some timestamps; return conversion may be partial."
            )

        forecast_out["pred_return_pct"] = (
            np.exp(forecast_out["pred"] - forecast_out["start_log_price"]) - 1.0
        ) * 100.0
        forecast_out["actual_return_pct"] = (
            np.exp(forecast_out["actual"] - forecast_out["start_log_price"]) - 1.0
        ) * 100.0

        forecast_out["pred_price_from_return"] = forecast_out["start_close"] * (
            1.0 + forecast_out["pred_return_pct"] / 100.0
        )
        forecast_out["actual_price_from_return"] = forecast_out["start_close"] * (
            1.0 + forecast_out["actual_return_pct"] / 100.0
        )

    forecast_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

    plot_predictions(forecast_out, CHART_PATH)
    print(f"Saved chart: {CHART_PATH}")


if __name__ == "__main__":
    main()
