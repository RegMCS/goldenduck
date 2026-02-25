"""
Generate GARCH-FX synthetic OHLCV data for TTM training (local, no API).

This uses the ML delta predictor (rf_delta.pkl) and heuristic theta, then
generates synthetic scenarios via GARCH-FX and writes CSV outputs.

Example:
  python generate_garch_training_data_ttm.py --tickers AAPL,MSFT --num-scenarios 200
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.worker.GARCH.services.garch_service import GARCHService


def _load_predict_parameters() -> callable:
    ml_training_path = ROOT / "backend" / "ml_training"
    predict_path = ml_training_path / "scripts" / "06_predict_parameters.py"
    spec = importlib.util.spec_from_file_location(
        "predict_parameters_module", predict_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load predictor module from {predict_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict_parameters"):
        raise RuntimeError("predict_parameters function not found in predictor module.")
    return module.predict_parameters


def _parse_tickers(raw: str) -> List[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _download_data(
    ticker: str,
    *,
    start: Optional[str],
    end: Optional[str],
    lookback_days: int,
) -> pd.DataFrame:
    if start is None or end is None:
        end_ts = pd.Timestamp.utcnow().normalize()
        start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
        start = start_ts.strftime("%Y-%m-%d")
        end = end_ts.strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, progress=False, threads=True)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        level_values = df.columns.get_level_values(-1)
        if ticker in level_values:
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "date"})
    else:
        df = df.rename(columns={df.columns[0]: "date"})
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker} missing columns {missing}")
    return df


def _add_dates(df: pd.DataFrame, last_date: pd.Timestamp) -> pd.DataFrame:
    start = pd.Timestamp(last_date) + pd.offsets.BDay(1)
    dates = pd.bdate_range(start=start, periods=len(df))
    out = df.copy()
    out.insert(0, "Date", dates)
    return out


def _combine_scenarios(
    scenarios: List[pd.DataFrame],
    *,
    last_date: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for i, scenario_df in enumerate(scenarios, start=1):
        df = _add_dates(scenario_df, last_date)
        df["scenario_id"] = i
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def _write_outputs(
    *,
    out_dir: Path,
    ticker: str,
    combined: pd.DataFrame,
    metadata: Dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{ticker.lower()}_garch_fx_training.csv"
    combined.to_csv(out_csv, index=False)

    meta_path = out_dir / f"{ticker.lower()}_garch_fx_training_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GARCH-FX synthetic OHLCV data for TTM training."
    )
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers.")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument("--lookback-days", type=int, default=730)
    parser.add_argument("--num-scenarios", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=252)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing rf_delta.pkl (for delta prediction).",
    )
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--desired-volatility", type=float, default=1.0)
    parser.add_argument("--desired-trend", type=float, default=0.0)
    parser.add_argument("--desired-fat-tails", type=float, default=1.0)
    parser.add_argument("--desired-momentum", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = _parse_tickers(args.tickers)
    if not tickers:
        raise ValueError("No tickers provided.")

    predict_parameters = _load_predict_parameters()

    default_out_dir = (
        Path(__file__).resolve().parents[1] / "outputs" / "ttm_garch_training"
    )
    out_dir = Path(args.output_dir) if args.output_dir else default_out_dir

    user_knobs = {
        "desired_volatility": float(args.desired_volatility),
        "desired_trend": float(args.desired_trend),
        "desired_fat_tails": float(args.desired_fat_tails),
        "desired_momentum": float(args.desired_momentum),
    }

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        data = _download_data(
            ticker,
            start=args.start,
            end=args.end,
            lookback_days=args.lookback_days,
        )
        last_date = pd.to_datetime(data["date"].iloc[-1])

        returns = np.log(data["Close"].values[1:] / data["Close"].values[:-1])
        pred_params = predict_parameters(
            historical_returns=returns,
            user_knobs=user_knobs,
            model_dir=args.model_dir,
        )

        garch = GARCHService()
        fit_params = garch.fit_with_retry(data, p=args.p, q=args.q)

        delta_sequence = np.full(args.horizon, float(pred_params["delta"]))
        scenarios = garch.generate_scenarios_fx(
            num_scenarios=args.num_scenarios,
            horizon=args.horizon,
            theta=float(pred_params["theta"]),
            delta_sequence=delta_sequence,
            user_knobs=user_knobs,
        )

        combined = _combine_scenarios(scenarios, last_date=last_date)
        out_path = _write_outputs(
            out_dir=out_dir,
            ticker=ticker,
            combined=combined,
            metadata={
                "ticker": ticker,
                "num_scenarios": int(args.num_scenarios),
                "horizon": int(args.horizon),
                "p": int(args.p),
                "q": int(args.q),
                "delta_predicted": float(pred_params["delta"]),
                "theta_predicted": float(pred_params["theta"]),
                "delta_confidence": pred_params.get("delta_confidence"),
                "fit_params": fit_params,
                "user_knobs": user_knobs,
            },
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
