"""Local runner for GC-GARCH inference.

Edit CSV_PATH, HORIZON, and KNOBS below.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis as kurtosis_fn, skew as skew_fn

from services.gc_garch_service import (
    GCGarchGenerator,
    GCGarchKnobs,
    normalize_knobs,
    apply_trend,
)

# ==============================
# Editable inputs (change these)
# ==============================
CSV_PATH = r"C:\Users\guiqu\OneDrive\Documents\GitHub\goldenduck\backend\AAPL_10.csv"
HORIZON = 500
KNOBS = GCGarchKnobs(
    volatility=1,
    trend=0.0,
    fat_tails=1,
    momentum=0.5,
)
SEED = 41
SEED_START = 1
SEED_END = 100
SEED_FOR_OUTPUT = 41
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_PATH = str(RESULTS_DIR / "gc_garch_output.csv")  # set to "" to skip
PLOT_PATH = str(RESULTS_DIR / "gc_garch_characteristics.png")
PLOT_OVERLAY_PATH = str(RESULTS_DIR / "gc_garch_overlay.png")
PLOT_RETURNS_OVERLAY_PATH = str(RESULTS_DIR / "gc_garch_returns_overlay.png")
PLOT_QUALITY_PATH = str(RESULTS_DIR / "gc_garch_quality_metrics.png")
DRY_RUN = False  # set True to skip generation


REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _validate_ohlcv(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _compute_baseline_mu(close: np.ndarray) -> float:
    returns = np.log(close[1:] / close[:-1])
    return float(np.mean(returns))


def _lag1_autocorr(series: np.ndarray) -> float:
    if series.size < 2:
        return 0.0
    x = series[:-1]
    y = series[1:]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def _compute_characteristics(close: np.ndarray) -> Dict[str, float]:
    returns = np.log(close[1:] / close[:-1])
    if returns.size == 0:
        return {"volatility": 0.0, "trend": 0.0, "fat_tails": 0.0, "momentum": 0.0}

    volatility = float(np.std(returns, ddof=1))
    trend = float(np.mean(returns))
    fat_tails = float(kurtosis_fn(returns, fisher=True, bias=False))

    # Momentum proxy: lag-1 autocorrelation of squared returns (volatility persistence)
    momentum = _lag1_autocorr(returns**2)

    return {
        "volatility": volatility,
        "trend": trend,
        "fat_tails": fat_tails,
        "momentum": momentum,
    }


def _plot_characteristics(
    actual: Dict[str, float],
    synthetic: Dict[str, float],
    output_path: str,
) -> None:
    labels = ["volatility", "trend", "momentum"]
    actual_vals = [actual[k] for k in labels]
    synthetic_vals = [synthetic[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, actual_vals, width, label="input (AAPL_100)")
    ax.bar(x + width / 2, synthetic_vals, width, label="synthetic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("GC-GARCH Characteristics Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_overlay(
    actual_close: np.ndarray,
    synthetic_close: np.ndarray,
    output_path: str,
) -> None:
    n = min(len(actual_close), len(synthetic_close))
    if n == 0:
        return

    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, actual_close[:n], label="input (AAPL_100)", linewidth=1.2)
    ax.plot(x, synthetic_close[:n], label="synthetic", linewidth=1.2)
    ax.set_title("GC-GARCH Close: Input vs Synthetic (Overlay)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Close")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_returns_overlay(
    actual_close: np.ndarray,
    synthetic_close: np.ndarray,
    output_path: str,
) -> None:
    if len(actual_close) < 2 or len(synthetic_close) < 2:
        return

    actual_returns = np.log(actual_close[1:] / actual_close[:-1])
    synthetic_returns = np.log(synthetic_close[1:] / synthetic_close[:-1])

    n = min(len(actual_returns), len(synthetic_returns))
    if n == 0:
        return

    a = actual_returns[:n]
    s = synthetic_returns[:n]

    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.hist(a, bins=60, density=True, alpha=0.55, label="input returns (AAPL_100)")
    ax.hist(s, bins=60, density=True, alpha=0.55, label="synthetic returns")
    ax.set_title("GC-GARCH Returns Distribution: Input vs Synthetic")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _compute_target_kurtosis(
    desired_fat_tails: float, desired_momentum: float
) -> float:
    baseline_kurtosis = 2.0
    max_kurtosis = 8.0

    if desired_fat_tails <= 1.0:
        kurtosis_from_fat_tails = baseline_kurtosis * desired_fat_tails
    else:
        excess = desired_fat_tails - 1.0
        kurtosis_from_fat_tails = baseline_kurtosis + (
            max_kurtosis - baseline_kurtosis
        ) * min(excess / 2.0, 1.0)

    momentum_boost = 0.0
    if desired_momentum > 0.7:
        momentum_boost = (desired_momentum - 0.7) / 0.3 * 2.0

    target_kurtosis = kurtosis_from_fat_tails + momentum_boost
    return min(target_kurtosis, max_kurtosis)


def _compute_quality_metrics(
    historical_close: np.ndarray,
    synthetic_close: np.ndarray,
    knobs: GCGarchKnobs,
) -> Dict[str, float]:
    hist_returns = np.log(historical_close[1:] / historical_close[:-1])
    synth_returns = np.log(synthetic_close[1:] / synthetic_close[:-1])

    hist_returns = hist_returns[np.isfinite(hist_returns)]
    synth_returns = synth_returns[np.isfinite(synth_returns)]

    if hist_returns.size == 0 or synth_returns.size == 0:
        return {"overall_match": 0.0}

    hist_vol = float(np.std(hist_returns, ddof=1))
    synth_vol = float(np.std(synth_returns, ddof=1))
    hist_kurt = float(kurtosis_fn(hist_returns, fisher=True, bias=False))
    synth_kurt = float(kurtosis_fn(synth_returns, fisher=True, bias=False))
    hist_skew = float(skew_fn(hist_returns, bias=False))
    synth_skew = float(skew_fn(synth_returns, bias=False))

    hist_acf = _lag1_autocorr(hist_returns)
    synth_acf = _lag1_autocorr(synth_returns)

    # Compare synthetic statistics directly against historical statistics
    volatility_match = 1.0 - abs(synth_vol - hist_vol) / (abs(hist_vol) + 1e-6)
    kurtosis_match = 1.0 - min(
        abs(synth_kurt - hist_kurt) / (abs(hist_kurt) + 0.5),
        1.0,
    )
    skewness_match = 1.0 - abs(synth_skew - hist_skew) / (
        abs(hist_skew) + 0.2
    )
    acf_match = 1.0 - abs(synth_acf - hist_acf) / (abs(hist_acf) + 0.1)

    # Clamp to [0, 1] for readability
    volatility_match = float(np.clip(volatility_match, 0.0, 1.0))
    kurtosis_match = float(np.clip(kurtosis_match, 0.0, 1.0))
    skewness_match = float(np.clip(skewness_match, 0.0, 1.0))
    acf_match = float(np.clip(acf_match, 0.0, 1.0))

    overall_match = (
        volatility_match + kurtosis_match + skewness_match + acf_match
    ) / 4.0

    return {
        "volatility_synthetic": synth_vol,
        "volatility_historical": hist_vol,
        "volatility_match": volatility_match,
        "kurtosis_synthetic": synth_kurt,
        "kurtosis_historical": hist_kurt,
        "kurtosis_match": kurtosis_match,
        "skewness_synthetic": synth_skew,
        "skewness_historical": hist_skew,
        "skewness_match": skewness_match,
        "acf_synthetic": synth_acf,
        "acf_historical": hist_acf,
        "acf_match": acf_match,
        "overall_match": overall_match,
    }


def _plot_quality_metrics(quality: Dict[str, float], output_path: str) -> None:
    metric_keys = [
        "volatility_match",
        "kurtosis_match",
        "skewness_match",
        "acf_match",
        "overall_match",
    ]
    values = [quality.get(k, 0.0) for k in metric_keys]

    x = np.arange(len(metric_keys))
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(x, values, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("GC-GARCH Quality Metrics (Match Scores)")
    ax.set_ylabel("Match Score (0–1)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for i, v in enumerate(values):
        ax.text(i, min(1.02, v + 0.03), f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _average_quality_metrics(qualities: list[Dict[str, float]]) -> Dict[str, float]:
    if not qualities:
        return {}
    keys = list(qualities[0].keys())
    avg: Dict[str, float] = {}
    for k in keys:
        vals = [q.get(k) for q in qualities if isinstance(q.get(k), (int, float))]
        if vals:
            avg[k] = float(np.mean(vals))
        else:
            avg[k] = qualities[0].get(k)
    return avg


def _summarize_inputs(df: pd.DataFrame, knobs: GCGarchKnobs, horizon: int) -> Dict:
    close = df["close"].to_numpy(dtype=float)
    mu_base = _compute_baseline_mu(close)
    mu_adj = apply_trend(mu_base, knobs.trend)
    return {
        "rows": int(len(df)),
        "mu_base": mu_base,
        "mu_adjusted": mu_adj,
        "volatility_scale": knobs.volatility,
        "horizon": horizon,
    }


def main() -> None:
    path = Path(CSV_PATH)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    df = _normalize_columns(df)
    _validate_ohlcv(df)

    knobs = normalize_knobs(KNOBS)

    summary = _summarize_inputs(df, knobs, HORIZON)

    print("GC-GARCH Inference")
    print("CSV:", path)
    print("Knobs:", asdict(knobs))
    print("Summary:", summary)

    if DRY_RUN:
        print("DRY_RUN=True: generation skipped.")
        return

    generator = GCGarchGenerator()
    generator.fit(df)
    scenario = generator.generate(horizon=HORIZON, knobs=knobs, seed=SEED_FOR_OUTPUT)

    print("Generated rows:", len(scenario))
    print("Head:\n", scenario.head())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH:
        scenario.to_csv(OUTPUT_PATH, index=False)
        print("Saved to:", OUTPUT_PATH)

    # Prepare series for plotting/metrics
    actual_close = df["close"].to_numpy(dtype=float)
    synthetic_close = scenario["Close"].to_numpy(dtype=float)

    # Plot overlay of close prices
    _plot_overlay(actual_close, synthetic_close, PLOT_OVERLAY_PATH)
    print("Saved overlay plot to:", PLOT_OVERLAY_PATH)

    _plot_returns_overlay(actual_close, synthetic_close, PLOT_RETURNS_OVERLAY_PATH)
    print("Saved returns overlay plot to:", PLOT_RETURNS_OVERLAY_PATH)

    # Compute and display average quality metrics across multiple seeds
    qualities = []
    for seed in range(SEED_START, SEED_END + 1):
        scenario_seed = generator.generate(horizon=HORIZON, knobs=knobs, seed=seed)
        synth_close_seed = scenario_seed["Close"].to_numpy(dtype=float)
        qualities.append(_compute_quality_metrics(actual_close, synth_close_seed, knobs))

    avg_quality = _average_quality_metrics(qualities)
    print(f"Average quality metrics (seeds {SEED_START}-{SEED_END}):", avg_quality)
    _plot_quality_metrics(avg_quality, PLOT_QUALITY_PATH)
    print("Saved quality metrics plot to:", PLOT_QUALITY_PATH)


if __name__ == "__main__":
    main()
