import json
import sys
from pathlib import Path

import pandas as pd

# Ensure project backend is on sys.path when running this script directly
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR))

from worker.GARCH.services.validation_service_v2 import ValidationServiceV2


# ============================================
# CONFIG: Set paths and knobs here
# ============================================
ORIGINAL_OHLCV_PATH = Path("backend/AAPL.csv")
SYNTHETIC_OHLCV_PATH = Path("backend/gc_garch_validation/baseline.csv")

# Knobs used to generate the synthetic data
USER_KNOBS = {
    "volatility": 1,  # 0.5 - 2.0
    "fat_tails": 1,  # 0.5 - 2.0
    "momentum": 0.5,  # 0.0 - 1.0
    "trend": 0.0,  # -1.0 - 1.0
}


def _load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if "scenario_id" in df.columns:
        df = df[df["scenario_id"] == 1].copy()
    return df


def _fmt_pct(value: float) -> str:
    if value != value:  # NaN
        return "nan"
    return f"{value:.2f}%"


def _fmt_num(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.6f}"


def _print_section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def _print_metric_row(name: str, orig, target, synth, match) -> None:
    print(
        f"{name:24} | orig {orig:>12} | target {target:>12} | synth {synth:>12} | match {match:>8}"
    )


def _pretty_print(results: dict) -> None:
    _print_section("Volatility")
    vol = results["volatility"]
    _print_metric_row(
        "Std Dev (log returns)",
        _fmt_num(vol["original"]),
        _fmt_num(vol["target"]),
        _fmt_num(vol["synthetic"]),
        _fmt_pct(vol["match_pct"]),
    )

    _print_section("Fat Tails")
    ft = results["fat_tails"]
    k = ft["kurtosis"]
    _print_metric_row(
        "Kurtosis (normalized)",
        _fmt_num(k["original_normalized"]),
        _fmt_num(k["target_normalized"]),
        _fmt_num(k["synthetic_normalized"]),
        _fmt_pct(k["match_pct"]),
    )
    print(f"\nFat Tails Match: {_fmt_pct(ft['match_pct'])}")

    _print_section("Momentum")
    mom = results["momentum"]["hurst"]
    _print_metric_row(
        "Hurst exponent",
        _fmt_num(mom["original"]),
        _fmt_num(mom["target"]),
        _fmt_num(mom["synthetic"]),
        _fmt_pct(mom["match_pct"]),
    )

    _print_section("Trend")
    tr = results["trend"]
    mr = tr["mean_return"]
    _print_metric_row(
        "Mean log return",
        _fmt_num(mr["original"]),
        _fmt_num(mr["target"]),
        _fmt_num(mr["synthetic"]),
        _fmt_pct(mr["match_pct"]),
    )
    print(f"\nTrend Match: {_fmt_pct(tr['match_pct'])}")

    _print_section("Overall")
    print(f"Overall Match: {_fmt_pct(results['overall_match_pct'])}")


def main() -> None:
    original_ohlcv = _load_ohlcv(ORIGINAL_OHLCV_PATH)
    synthetic_ohlcv = _load_ohlcv(SYNTHETIC_OHLCV_PATH)

    validator = ValidationServiceV2(decay_k=3.0)
    results = validator.validate(
        original_ohlcv=original_ohlcv,
        synthetic_ohlcv=synthetic_ohlcv,
        user_knobs=USER_KNOBS,
    )

    _pretty_print(results)
    print("\nRaw JSON\n--------")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
