"""
Download OHLCV data from yfinance and save to CSV.

Example:
  python backend/GARCH/scripts/download_ohlcv_yf.py --ticker AAPL --start 2022-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


DEFAULT_TICKER = "AAPL"
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2023-12-31"
DEFAULT_PERIOD = None  # if start/end are None, period is used
DEFAULT_INTERVAL = "1d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OHLCV from yfinance.")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument("--output", default=None, help="Output CSV path.")
    return parser.parse_args()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    out = out.reset_index()
    if "Date" in out.columns:
        out = out.rename(columns={"Date": "date"})
    elif "Datetime" in out.columns:
        out = out.rename(columns={"Datetime": "date"})
    else:
        out = out.rename(columns={out.columns[0]: "date"})

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()

    if args.start or args.end:
        df = yf.download(
            args.ticker,
            start=args.start,
            end=args.end,
            interval=args.interval,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    else:
        df = yf.download(
            args.ticker,
            period=args.period or "2y",
            interval=args.interval,
            auto_adjust=False,
            progress=False,
            threads=True,
        )

    if df is None or df.empty:
        raise SystemExit(f"No data returned for {args.ticker}")

    out = _normalize(df)

    if args.output:
        out_path = Path(args.output)
    else:
        suffix = (
            f"{args.start}_to_{args.end}"
            if (args.start or args.end)
            else f"{args.period or '2y'}"
        )
        out_path = Path("backend/GARCH/output") / f"{args.ticker.lower()}_{suffix}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
