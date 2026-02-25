"""
Visualize a CSV file with common OHLCV columns.

Examples:
  python visualize_csv.py --file path/to/data.csv
  python visualize_csv.py --file data.csv --columns Close,Volume --scenario-id 3
  python visualize_csv.py --file data.csv --columns Open,High,Low,Close --overlay
  python visualize_csv.py --file data.csv --columns Close --output close.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt


def _parse_columns(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [c.strip() for c in raw.split(",") if c.strip()]


def _pick_date_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("Date", "date", "Datetime", "datetime"):
        if candidate in df.columns:
            return candidate
    return None


def _pick_default_columns(df: pd.DataFrame) -> List[str]:
    if "Close" in df.columns:
        return ["Close"]
    exclude = {"scenario_id"}
    date_col = _pick_date_column(df)
    if date_col:
        exclude.add(date_col)
    numeric_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_cols[:4] if numeric_cols else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CSV time series data.")
    parser.add_argument("--file", required=True, help="Path to CSV file.")
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated columns to plot (default: Close or numeric columns).",
    )
    parser.add_argument(
        "--scenario-id",
        type=int,
        default=None,
        help="Filter by scenario_id (if column exists).",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay multiple columns on a single axis.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save plot to file instead of showing (e.g., plot.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.file).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty.")

    if args.scenario_id is not None and "scenario_id" in df.columns:
        df = df[df["scenario_id"] == args.scenario_id]
        if df.empty:
            raise ValueError(f"No rows found for scenario_id={args.scenario_id}")

    date_col = _pick_date_column(df)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
        x = df[date_col]
        xlabel = date_col
    else:
        x = df.index
        xlabel = "index"

    columns = _parse_columns(args.columns) or _pick_default_columns(df)
    if not columns:
        raise ValueError("No columns to plot. Provide --columns explicitly.")

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    if args.overlay or len(columns) == 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in columns:
            ax.plot(x, df[col], label=col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(len(columns), 1, figsize=(12, 3 * len(columns)), sharex=True)
        if len(columns) == 1:
            axes = [axes]
        for ax, col in zip(axes, columns):
            ax.plot(x, df[col], label=col)
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel(xlabel)

    fig.tight_layout()

    if args.output:
        out_path = Path(args.output).expanduser()
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
