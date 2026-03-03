"""
Plot MS-GARCH scenarios and diagnostics.

Example:
  python backend/GARCH/scripts/plot_msgarch_scenarios.py --ticker AAPL --period 2y --num-scenarios 200 --horizon 252
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
DEFAULT_START = None  # "YYYY-MM-DD"
DEFAULT_END = None  # "YYYY-MM-DD"
DEFAULT_PERIOD = "2y"

DEFAULT_P = 1
DEFAULT_Q = 1
DEFAULT_DIST = "normal"
DEFAULT_N_ITER = 5
DEFAULT_INIT_SPLIT_QUANTILE_LOW = 0.1
DEFAULT_INIT_SPLIT_QUANTILE_HIGH = 0.9
DEFAULT_MIN_POINTS_PER_REGIME = 100

DEFAULT_NUM_SCENARIOS = 200
DEFAULT_HORIZON = 504
DEFAULT_VOL_MULT = 1.0
DEFAULT_SEED = 43
DEFAULT_MIN_RUN_LENGTH = 100
DEFAULT_SWITCH_SCALE = 1.0

DEFAULT_SAMPLE_PATHS = 20
DEFAULT_HIST_WINDOW = 0  # 0 = full history
DEFAULT_OUT_DIR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MS-GARCH plotting utility.")
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help="Ticker symbol (e.g., AAPL).",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Optional CSV with OHLCV data.",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help="End date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="yfinance period (e.g., 2y).",
    )

    parser.add_argument("--p", type=int, default=DEFAULT_P)
    parser.add_argument("--q", type=int, default=DEFAULT_Q)
    parser.add_argument("--dist", default=DEFAULT_DIST, help="normal|t|skewt")
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument(
        "--init-split-quantile-low",
        type=float,
        default=DEFAULT_INIT_SPLIT_QUANTILE_LOW,
    )
    parser.add_argument(
        "--init-split-quantile-high",
        type=float,
        default=DEFAULT_INIT_SPLIT_QUANTILE_HIGH,
    )
    parser.add_argument(
        "--min-points-per-regime",
        type=int,
        default=DEFAULT_MIN_POINTS_PER_REGIME,
    )

    parser.add_argument("--num-scenarios", type=int, default=DEFAULT_NUM_SCENARIOS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--vol-mult", type=float, default=DEFAULT_VOL_MULT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--min-run-length",
        type=int,
        default=DEFAULT_MIN_RUN_LENGTH,
        help="Minimum consecutive days in a regime (0 disables).",
    )
    parser.add_argument(
        "--switch-scale",
        type=float,
        default=DEFAULT_SWITCH_SCALE,
        help="Scale off-diagonal transition probs (e.g., 0.5 = fewer switches).",
    )

    parser.add_argument(
        "--sample-paths",
        type=int,
        default=DEFAULT_SAMPLE_PATHS,
        help="Number of scenario paths to overlay in the price plot.",
    )
    parser.add_argument(
        "--hist-window",
        type=int,
        default=DEFAULT_HIST_WINDOW,
        help="Number of most recent historical days to plot.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory for plots (default: backend/GARCH/output/msgarch_plots/<stamp>).",
    )
    return parser.parse_args()


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        # yfinance can return (field, ticker)
        out.columns = out.columns.get_level_values(0)

    if "Date" in out.columns:
        out = out.rename(columns={"Date": "date"})
    elif "Datetime" in out.columns:
        out = out.rename(columns={"Datetime": "date"})
    elif "date" not in out.columns:
        # assume index is date-like
        out = out.reset_index().rename(columns={out.columns[0]: "date"})

    # Prefer Adj Close if present
    if "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"].astype(float)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    if out["date"].isna().any():
        out["date"] = pd.to_datetime(
            out["date"], errors="coerce", format="mixed", dayfirst=True
        )
    if out["date"].isna().all():
        raise ValueError("Could not parse any dates from the 'date' column.")
    if out["date"].isna().any():
        out = out.dropna(subset=["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out = out.dropna(subset=required)
    return out


def _load_historical_data(
    ticker: Optional[str],
    csv_path: Optional[str],
    start: Optional[str],
    end: Optional[str],
    period: str,
) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
        return _normalize_ohlcv(df)

    if not ticker:
        raise ValueError("Provide --ticker or --csv.")

    if start or end:
        df = yf.download(ticker, start=start, end=end, progress=False, threads=True)
    else:
        df = yf.download(ticker, period=period, progress=False, threads=True)

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    return _normalize_ohlcv(df)


def _plot_price_paths(
    hist: pd.DataFrame,
    close_matrix: np.ndarray,
    out_path: Path,
    sample_paths: int,
    hist_window: int,
) -> None:
    hist_slice = hist if hist_window <= 0 else hist.tail(hist_window)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist_slice["date"], hist_slice["Close"], label="Historical Close", lw=2)

    if close_matrix.size > 0:
        n_paths = min(sample_paths, close_matrix.shape[0])
        for i in range(n_paths):
            ax.plot(
                pd.date_range(
                    hist_slice["date"].iloc[-1], periods=close_matrix.shape[1] + 1
                )[1:],
                close_matrix[i],
                alpha=0.4,
                lw=1,
                color="tab:orange",
            )

    ax.set_title("Historical Close + Sample MS-GARCH Paths")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_first_sample_path(
    hist: pd.DataFrame,
    close_matrix: np.ndarray,
    out_path: Path,
    hist_window: int,
) -> None:
    if close_matrix.size == 0:
        return
    hist_slice = hist if hist_window <= 0 else hist.tail(hist_window)
    first_path = close_matrix[0]
    future_dates = pd.date_range(
        hist_slice["date"].iloc[-1], periods=len(first_path) + 1
    )[1:]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist_slice["date"], hist_slice["Close"], label="Historical Close", lw=2)
    ax.plot(future_dates, first_path, label="Sample Path #1", lw=2, color="tab:orange")
    ax.set_title("Historical Close + First Sample Path")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_synthetic_only(
    close_matrix: np.ndarray,
    out_path: Path,
    sample_path_index: int,
    regimes: Optional[np.ndarray] = None,
) -> None:
    if close_matrix.size == 0:
        return
    idx = max(0, min(int(sample_path_index), close_matrix.shape[0] - 1))
    x = np.arange(close_matrix.shape[1])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        x,
        close_matrix[idx],
        alpha=0.8,
        lw=2,
        color="tab:orange",
        label=None,
    )
    if regimes is not None and len(regimes) == len(x):
        # Color background by regime (0=green, 1=orange)
        last_i = 0
        last_reg = int(regimes[0])
        for i in range(1, len(regimes)):
            r = int(regimes[i])
            if r != last_reg:
                ax.axvspan(
                    last_i,
                    i,
                    color="tab:orange" if last_reg == 1 else "tab:green",
                    alpha=0.15,
                    lw=0,
                )
                last_i = i
                last_reg = r
        ax.axvspan(
            last_i,
            len(regimes) - 1,
            color="tab:orange" if last_reg == 1 else "tab:green",
            alpha=0.15,
            lw=0,
        )
        ax.set_title(f"Synthetic Price Path Only (path #{idx + 1}) + Regimes")
    else:
        ax.set_title(f"Synthetic Price Path Only (path #{idx + 1})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    if regimes is not None and len(regimes) == len(x):
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(color="tab:green", alpha=0.15, label="Regime 0"),
                Patch(color="tab:orange", alpha=0.15, label="Regime 1"),
            ],
            loc="best",
        )
    else:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def _plot_fan_chart(
    hist: pd.DataFrame,
    close_matrix: np.ndarray,
    out_path: Path,
) -> None:
    if close_matrix.size == 0:
        return

    dates = pd.date_range(hist["date"].iloc[-1], periods=close_matrix.shape[1] + 1)[1:]
    q10 = np.quantile(close_matrix, 0.10, axis=0)
    q50 = np.quantile(close_matrix, 0.50, axis=0)
    q90 = np.quantile(close_matrix, 0.90, axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist["date"].tail(200), hist["Close"].tail(200), label="Historical Close")
    ax.plot(dates, q50, label="Median Forecast", color="tab:blue")
    ax.fill_between(dates, q10, q90, color="tab:blue", alpha=0.2, label="10-90%")
    ax.set_title("MS-GARCH Fan Chart")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_return_hist(
    hist_returns: np.ndarray,
    synth_returns: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        hist_returns,
        bins=60,
        alpha=0.6,
        label="Historical",
        density=True,
        color="tab:blue",
    )
    ax.hist(
        synth_returns,
        bins=60,
        alpha=0.6,
        label="Synthetic (MS-GARCH)",
        density=True,
        color="tab:orange",
    )
    ax.set_title("Log Return Distribution")
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_regime_probs(
    hist: pd.DataFrame,
    filtered_probs: np.ndarray,
    out_path: Path,
) -> None:
    if filtered_probs is None or len(filtered_probs) == 0:
        return

    dates = hist["date"].iloc[1:].values[-len(filtered_probs) :]
    prob_hi = filtered_probs[:, 1]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(dates, prob_hi, color="tab:red", label="Regime 1 Prob")
    ax1.set_ylabel("Regime 1 Probability", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_ylim(0.0, 1.0)

    ax2 = ax1.twinx()
    ax2.plot(dates, hist["Close"].iloc[1:].values[-len(filtered_probs) :], color="tab:blue", alpha=0.4)
    ax2.set_ylabel("Close", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title("Filtered Regime Probability vs Close")
    ax1.set_xlabel("Date")
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_transition_matrix(P: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(P, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{P[i, j]:.5f}", ha="center", va="center", color="black")
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

    hist = _load_historical_data(
        ticker=args.ticker,
        csv_path=args.csv,
        start=args.start,
        end=args.end,
        period=args.period,
    )

    service = MSGARCHService()
    fit_summary = service.fit_model(
        historical_data=hist,
        p=args.p,
        q=args.q,
        dist=args.dist,
        n_iter=args.n_iter,
        init_split_quantile_low=args.init_split_quantile_low,
        init_split_quantile_high=args.init_split_quantile_high,
        min_points_per_regime=args.min_points_per_regime,
    )

    scenarios = service.generate_scenarios(
        num_scenarios=args.num_scenarios,
        horizon=args.horizon,
        volatility_multiplier=args.vol_mult,
        random_seed=args.seed,
        include_regime=True,
        min_run_length=args.min_run_length,
        switch_scale=args.switch_scale,
    )

    close_matrix = np.stack([s["Close"].values for s in scenarios], axis=0)

    hist_close = hist["Close"].astype(float).values
    hist_returns = np.log(hist_close[1:] / hist_close[:-1])

    synth_returns = []
    for s in scenarios:
        c = s["Close"].astype(float).values
        if len(c) >= 2:
            synth_returns.append(np.log(c[1:] / c[:-1]))
    synth_returns = np.concatenate(synth_returns) if synth_returns else np.array([])

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = ROOT / "output" / "msgarch_plots" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plots
    _plot_price_paths(
        hist, close_matrix, out_dir / "price_paths.png", args.sample_paths, args.hist_window
    )
    _plot_first_sample_path(
        hist, close_matrix, out_dir / "first_sample_path.png", args.hist_window
    )
    first_regimes = None
    if scenarios and "Regime" in scenarios[0].columns:
        first_regimes = scenarios[0]["Regime"].to_numpy()
    _plot_synthetic_only(
        close_matrix, out_dir / "synthetic_only_path_1.png", 0, regimes=first_regimes
    )
    _plot_fan_chart(hist, close_matrix, out_dir / "fan_chart.png")
    if synth_returns.size > 0:
        _plot_return_hist(hist_returns, synth_returns, out_dir / "return_hist.png")
    _plot_regime_probs(hist, service.filtered_probs, out_dir / "regime_probs.png")
    if service.P is not None:
        _plot_transition_matrix(service.P, out_dir / "transition_matrix.png")

    # Save fit summary
    summary_path = out_dir / "fit_summary.json"
    summary_path.write_text(json.dumps(fit_summary, indent=2))

    print(f"Saved plots to: {out_dir}")
    print(f"Fit summary: {summary_path}")


if __name__ == "__main__":
    main()
