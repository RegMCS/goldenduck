"""
Plot MS-GARCH return diagnostics for a fixed historical window.

Outputs:
  - returns_timeseries.png
  - sample_returns_path.png
  - returns_distribution.png

Example:
  python backend/GARCH/scripts/plot_msgarch_returns.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys
import json

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `pip install matplotlib`."
    ) from exc

# Add GARCH package root for local imports
ROOT = Path(__file__).resolve().parents[1]  # .../backend/GARCH
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from services.msgarch_service import MSGARCHService

# ----------------------------
# Defaults (edit these)
# ----------------------------
DEFAULT_TICKER = "AAPL"
DEFAULT_CSV = "backend/GARCH/output/aapl_2022-01-01_to_2023-12-31.csv"
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2023-12-31"
DEFAULT_PERIOD = None  # if start/end are None, period is used

DEFAULT_P = 1
DEFAULT_Q = 1
DEFAULT_DIST = "normal"
DEFAULT_N_ITER = 5
DEFAULT_INIT_SPLIT_QUANTILE = 0.5
DEFAULT_MIN_POINTS_PER_REGIME = 30

DEFAULT_NUM_SCENARIOS = 200
DEFAULT_VOL_MULT = 1.0
DEFAULT_SEED = 3
DEFAULT_OUT_DIR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MS-GARCH returns plotting utility.")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Optional CSV with Close data.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--period", default=DEFAULT_PERIOD)

    parser.add_argument("--p", type=int, default=DEFAULT_P)
    parser.add_argument("--q", type=int, default=DEFAULT_Q)
    parser.add_argument("--dist", default=DEFAULT_DIST, help="normal|t|skewt")
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument(
        "--init-split-quantile",
        type=float,
        default=DEFAULT_INIT_SPLIT_QUANTILE,
    )
    parser.add_argument(
        "--min-points-per-regime",
        type=int,
        default=DEFAULT_MIN_POINTS_PER_REGIME,
    )

    parser.add_argument("--num-scenarios", type=int, default=DEFAULT_NUM_SCENARIOS)
    parser.add_argument("--vol-mult", type=float, default=DEFAULT_VOL_MULT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    def _norm(col: str) -> str:
        return col.strip().lower().replace(" ", "").replace("_", "")

    col_map = {_norm(c): c for c in out.columns}

    if "date" in col_map:
        out = out.rename(columns={col_map["date"]: "date"})
    elif "datetime" in col_map:
        out = out.rename(columns={col_map["datetime"]: "date"})
    else:
        out = out.reset_index().rename(columns={out.columns[0]: "date"})

    rename = {}
    for key, target in {"close": "Close", "adjclose": "Close"}.items():
        if key in col_map:
            rename[col_map[key]] = target
            break
    if rename:
        out = out.rename(columns=rename)

    if "Close" not in out.columns:
        raise ValueError(
            f"Missing Close column. Available columns: {list(out.columns)}"
        )

    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    if out["date"].isna().any():
        # Fallback for mixed date formats
        out["date"] = pd.to_datetime(
            out["date"], errors="coerce", format="mixed", dayfirst=True
        )
    if out["date"].isna().all():
        raise ValueError("Could not parse any dates from the 'date' column.")
    if out["date"].isna().any():
        out = out.dropna(subset=["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out = out.dropna(subset=["date", "Close"])
    return out


def _download_data(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    period: Optional[str],
) -> pd.DataFrame:
    if start or end:
        df = yf.download(ticker, start=start, end=end, progress=False, threads=True)
    else:
        df = yf.download(ticker, period=period or "2y", progress=False, threads=True)

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    return _normalize_ohlcv(df)


def _plot_returns_series(dates: np.ndarray, returns: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, returns, lw=1.0, color="tab:blue")
    ax.set_title("Historical Log Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sample_returns(
    hist_dates: np.ndarray,
    hist_returns: np.ndarray,
    sample_returns: np.ndarray,
    out_path: Path,
) -> None:
    sample_dates = pd.date_range(hist_dates[-1], periods=len(sample_returns) + 1)[1:]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist_dates, hist_returns, lw=1.0, label="Historical", color="tab:blue")
    ax.plot(sample_dates, sample_returns, lw=1.0, label="Sample Path #1", color="tab:orange")
    ax.set_title("Historical Returns + First Sample Path")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_returns_distribution(
    hist_returns: np.ndarray,
    synth_returns: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(hist_returns, bins=60, alpha=0.6, density=True, label="Historical", color="tab:blue")
    ax.hist(
        synth_returns,
        bins=60,
        alpha=0.6,
        density=True,
        label="Synthetic (MS-GARCH)",
        color="tab:orange",
    )
    ax.set_title("Return Distribution")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_price_series(dates: np.ndarray, prices: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, prices, lw=1.5, color="tab:green")
    ax.set_title("Actual Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sample_price_path(
    hist_dates: np.ndarray,
    hist_prices: np.ndarray,
    sample_prices: np.ndarray,
    out_path: Path,
) -> None:
    if sample_prices is None or len(sample_prices) == 0:
        return
    sample_dates = pd.date_range(hist_dates[-1], periods=len(sample_prices) + 1)[1:]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist_dates, hist_prices, lw=1.5, label="Historical Close", color="tab:green")
    ax.plot(sample_dates, sample_prices, lw=1.5, label="Sample Price Path #1", color="tab:orange")
    ax.set_title("Historical Close + Synthetic Price Path")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _plot_regime_background(
    dates: np.ndarray,
    values: np.ndarray,
    regime_probs: np.ndarray,
    out_path: Path,
    *,
    title: str,
    y_label: str,
) -> None:
    if regime_probs is None or len(regime_probs) == 0:
        return
    n = len(regime_probs)
    dates = dates[-n:]
    values = values[-n:]
    regimes = (regime_probs[:, 1] >= 0.5).astype(int)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, values, color="tab:blue", lw=1.2)

    # Color background by most likely regime
    last_idx = 0
    last_regime = regimes[0]
    for i in range(1, n):
        if regimes[i] != last_regime:
            ax.axvspan(
                dates[last_idx],
                dates[i],
                color="tab:orange" if last_regime == 1 else "tab:green",
                alpha=0.15,
                lw=0,
            )
            last_idx = i
            last_regime = regimes[i]
    ax.axvspan(
        dates[last_idx],
        dates[-1],
        color="tab:orange" if last_regime == 1 else "tab:green",
        alpha=0.15,
        lw=0,
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _plot_transition_matrix(P: np.ndarray, out_path: Path) -> None:
    if P is None:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(P, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low", "High"])
    ax.set_yticklabels(["Low", "High"])
    ax.set_title("Transition Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.csv:
        raw = pd.read_csv(args.csv)
        hist = _normalize_ohlcv(raw)
    else:
        hist = _download_data(args.ticker, args.start, args.end, args.period)
    close = hist["Close"].astype(float).values
    hist_returns = np.log(close[1:] / close[:-1])
    hist_dates = hist["date"].values[1:]
    price_dates = hist["date"].values

    horizon = len(hist_returns)
    if horizon < 2:
        raise RuntimeError("Not enough data to compute returns.")

    service = MSGARCHService()
    fit_summary = service.fit_model(
        historical_data=hist,
        p=args.p,
        q=args.q,
        dist=args.dist,
        n_iter=args.n_iter,
        init_split_quantile=args.init_split_quantile,
        min_points_per_regime=args.min_points_per_regime,
    )

    scenarios = service.generate_scenarios(
        num_scenarios=args.num_scenarios,
        horizon=horizon,
        volatility_multiplier=args.vol_mult,
        random_seed=args.seed,
    )

    # Returns from scenarios
    synth_returns_all = []
    first_path_returns = None
    first_path_prices = None
    for idx, s in enumerate(scenarios):
        c = s["Close"].astype(float).values
        r = np.log(c[1:] / c[:-1])
        if idx == 0:
            first_path_returns = r
            first_path_prices = c
        synth_returns_all.append(r)
    synth_returns = np.concatenate(synth_returns_all)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = ROOT / "output" / "msgarch_returns" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_returns_series(hist_dates, hist_returns, out_dir / "returns_timeseries.png")
    _plot_price_series(price_dates, close, out_dir / "price_timeseries.png")
    _plot_sample_price_path(
        price_dates,
        close,
        first_path_prices,
        out_dir / "sample_price_path.png",
    )
    _plot_regime_background(
        hist["date"].values[1:],
        hist_returns,
        service.filtered_probs,
        out_dir / "returns_regime_background.png",
        title="Returns with Regime Background",
        y_label="Log return",
    )
    _plot_regime_background(
        hist["date"].values,
        close,
        service.filtered_probs,
        out_dir / "price_regime_background.png",
        title="Price with Regime Background",
        y_label="Price",
    )
    _plot_transition_matrix(service.P, out_dir / "transition_matrix.png")
    _plot_sample_returns(
        hist_dates,
        hist_returns,
        first_path_returns,
        out_dir / "sample_returns_path.png",
    )
    _plot_returns_distribution(
        hist_returns,
        synth_returns,
        out_dir / "returns_distribution.png",
    )

    summary_path = out_dir / "fit_summary.json"
    summary_path.write_text(json.dumps(fit_summary, indent=2))

    print(f"Saved plots to: {out_dir}")
    print(f"Fit summary: {summary_path}")


if __name__ == "__main__":
    main()
