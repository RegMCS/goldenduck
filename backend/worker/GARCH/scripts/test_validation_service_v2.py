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
ORIGINAL_OHLCV_PATH = Path("backend/AAPL_real.csv")
SYNTHETIC_OHLCV_PATH = Path("backend/OHLCV_output default.csv")

# Knobs used to generate the synthetic data
USER_KNOBS = {
    "volatility": 1.0,  # 0.5 - 2.0
    "fat_tails": 1.0,   # 0.5 - 2.0
    "momentum": 0.5,    # 0.0 - 1.0
    "trend": 0.0,       # -1.0 - 1.0
}


def _load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


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
        "Kurtosis",
        _fmt_num(k["original"]),
        _fmt_num(k["target"]),
        _fmt_num(k["synthetic"]),
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
    ms = tr["ma20_slope_avg"]
    ur = tr["ma20_up_ratio"]
    se = tr["start_end_return"]
    _print_metric_row(
        "MA20 slope avg",
        _fmt_num(ms["original"]),
        _fmt_num(ms["target"]),
        _fmt_num(ms["synthetic"]),
        _fmt_pct(ms["match_pct"]),
    )
    _print_metric_row(
        "MA20 uptrend ratio",
        _fmt_num(ur["original"]),
        _fmt_num(ur["target"]),
        _fmt_num(ur["synthetic"]),
        _fmt_pct(ur["match_pct"]),
    )
    _print_metric_row(
        "Start-End return",
        _fmt_num(se["original"]),
        _fmt_num(se["target"]),
        _fmt_num(se["synthetic"]),
        _fmt_pct(se["match_pct"]),
    )
    print(f"\nTrend Match: {_fmt_pct(tr['match_pct'])}")

    _print_section("Overall")
    print(f"Overall Match: {_fmt_pct(results['overall_match_pct'])}")


def main() -> None:
    original_ohlcv = _load_ohlcv(ORIGINAL_OHLCV_PATH)
    synthetic_ohlcv = _load_ohlcv(SYNTHETIC_OHLCV_PATH)

    validator = ValidationServiceV2()
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
