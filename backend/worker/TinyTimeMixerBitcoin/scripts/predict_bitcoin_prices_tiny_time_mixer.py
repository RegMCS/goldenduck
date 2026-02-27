"""
Runnable script derived from Kaggle notebook:
https://www.kaggle.com/code/spencer1129/predict-bitcoin-prices-tiny-time-mixer/notebook
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import Trainer, TrainingArguments, set_seed

from tsfm_public import TimeSeriesPreprocessor, TinyTimeMixerForPrediction, get_datasets
from tsfm_public.toolkit.time_series_forecasting_pipeline import TimeSeriesForecastingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Bitcoin prices with TTM.")
    parser.add_argument("--csv", required=True, help="Path to BTC OHLCV CSV.")
    parser.add_argument("--timestamp-col", default="Timestamp")
    parser.add_argument("--target-col", default="log_price")
    parser.add_argument("--observable-cols", default="")
    parser.add_argument("--ticker-col", default="ticker")
    parser.add_argument("--ticker", default="BTC-USD", help="Ticker label to use if CSV has no ticker column.")
    parser.add_argument("--resample", default="1h", help="Resample frequency (e.g., 1h). Use 'none' to disable.")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--forecast-length", type=int, default=96)
    parser.add_argument("--prediction-filter-length", type=int, default=24)
    parser.add_argument("--finetune", action="store_true", help="Fine-tune the model.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--fewshot-fraction",
        type=float,
        default=None,
        help="Fraction of training data to keep for few-shot (0 < f <= 1).",
    )
    parser.add_argument(
        "--fewshot-location",
        default="last",
        choices=["first", "last", "uniform"],
        help="How to select few-shot data from the training split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=None, help="Local model path (optional).")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--horizon-out", type=int, default=12, help="Horizon for RMSE comparison.")
    return parser.parse_args()


def add_log_price(df: pd.DataFrame, *, close_col: str = "Close") -> pd.DataFrame:
    out = df.copy()
    if close_col not in out.columns:
        raise ValueError(f"Missing close column: {close_col}")
    out["log_price"] = np.log(out[close_col])
    return out


def load_and_prepare(df: pd.DataFrame, timestamp_col: str, resample: str) -> pd.DataFrame:
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    if resample and resample.lower() != "none":
        df = (
            df.resample(resample, on=timestamp_col)
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
    hours_out: int,
) -> pd.DataFrame:
    rows = []
    for _, row in forecast.iterrows():
        try:
            pred_vals = row[prediction_col]
            actual_vals = row[actual_col]
            if len(pred_vals) <= hours_out or len(actual_vals) <= hours_out:
                continue
            rows.append(
                {
                    "Timestamp": row[date_col],
                    "pred": pred_vals[hours_out],
                    "actual": actual_vals[hours_out],
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def plot_predictions(forecast_out: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(forecast_out["Timestamp"], forecast_out["pred"], label="pred")
    ax.plot(forecast_out["Timestamp"], forecast_out["actual"], label="actual")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[1] / "outputs" / "ttm_bitcoin"
    charts_dir = output_dir.parent / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = load_and_prepare(df, args.timestamp_col, args.resample)

    timestamp_col = args.timestamp_col
    target_cols = [args.target_col]
    observable_cols = [c.strip() for c in args.observable_cols.split(",") if c.strip()]
    ticker_col = args.ticker_col
    if ticker_col not in df.columns:
        df[ticker_col] = args.ticker

    tsp = TimeSeriesPreprocessor(
        id_columns=[ticker_col],
        timestamp_column=timestamp_col,
        target_columns=target_cols,
        observable_columns=observable_cols,
        scaling_id_columns=[ticker_col],
        context_length=args.context_length,
        prediction_length=args.forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    split_config = build_split_config(len(df), args.context_length)
    if args.fewshot_fraction is not None and not (0.0 < args.fewshot_fraction <= 1.0):
        raise ValueError("--fewshot-fraction must be in (0, 1].")
    train_dataset, valid_dataset, test_dataset = get_datasets(
        tsp,
        df,
        split_config,
        fewshot_fraction=args.fewshot_fraction,
        fewshot_location=args.fewshot_location,
    )

    model_id = args.model_path if args.model_path else "ibm/TTM"
    model = TinyTimeMixerForPrediction.from_pretrained(
        model_id, revision="main", prediction_filter_length=args.prediction_filter_length
    )

    if args.finetune:
        train_args = TrainingArguments(
            output_dir=str(output_dir / "train_output"),
            overwrite_output_dir=True,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            do_eval=True,
            eval_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
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
        timestamp_column=timestamp_col,
        id_columns=[ticker_col],
        target_columns=target_cols,
        observable_columns=observable_cols,
        freq=args.resample if args.resample.lower() != "none" else "1h",
    )

    test_start, test_end = split_config["test"]
    test_data = tsp.preprocess(df[test_start:test_end])
    forecasts = pipeline(test_data)

    pred_col = f"{args.target_col}_prediction"
    forecast_predictions = compare_forecast(
        forecasts, timestamp_col, pred_col, args.target_col, args.horizon_out
    )
    forecast_out = forecast_predictions.dropna(subset=["actual", "pred"])

    if forecast_out.empty:
        print("No forecast rows available for evaluation.")
        return

    rmse = np.sqrt(np.mean((forecast_out["actual"] - forecast_out["pred"]) ** 2))
    print(f"RMSE ({args.horizon_out} out): {rmse:.6f}")

    plot_predictions(
        forecast_out,
        charts_dir / "bitcoin_pred_vs_actual.png",
        title=f"Pred vs Actual (RMSE {rmse:.6f})",
    )

    forecast_out.to_csv(output_dir / "bitcoin_forecast_vs_actual.csv", index=False)
    print(f"Saved forecast CSV: {output_dir / 'bitcoin_forecast_vs_actual.csv'}")


if __name__ == "__main__":
    main()
