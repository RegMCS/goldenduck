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
SCENARIO_ID = 1  # Used only if synthetic CSV has scenario_id/path_id

RUNS = [
    {
        "name": "high_vol",
        "synthetic_path": Path("backend/gc_garch_validation/high_momentum.csv"),
        "user_knobs": {
            "volatility": 1,
            "fat_tails": 1,
            "momentum": 1,
            "trend": 0,
        },
    },
    {
        "name": "low_vol",
        "synthetic_path": Path("backend/gc_garch_validation/low_momentum.csv"),
        "user_knobs": {
            "volatility": 1,
            "fat_tails": 1,
            "momentum": 0,
            "trend": 0,
        },
    },
]


def _load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if "scenario_id" in df.columns:
        df = df[df["scenario_id"] == SCENARIO_ID].copy()
    elif "path_id" in df.columns:
        df = df[df["path_id"] == SCENARIO_ID].copy()
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


def _mean(values: list[float]) -> float:
    finite_vals = [v for v in values if v == v]
    if not finite_vals:
        return float("nan")
    return sum(finite_vals) / len(finite_vals)


def main() -> None:
    original_ohlcv = _load_ohlcv(ORIGINAL_OHLCV_PATH)
    validator = ValidationServiceV2(decay_k=3.0)

    run_scores = []
    all_results = {}

    for i, run in enumerate(RUNS, start=1):
        run_name = run["name"]
        synthetic_path = run["synthetic_path"]
        user_knobs = run["user_knobs"]

        synthetic_ohlcv = _load_ohlcv(synthetic_path)
        results = validator.validate(
            original_ohlcv=original_ohlcv,
            synthetic_ohlcv=synthetic_ohlcv,
            user_knobs=user_knobs,
        )

        run_scores.append(float(results.get("overall_match_pct", float("nan"))))
        all_results[run_name] = results

        print(f"\nRun {i}: AAPL.csv vs {synthetic_path.name}")
        _pretty_print(results)
        print("\nRaw JSON\n--------")
        print(json.dumps(results, indent=2))

    average_score = _mean(run_scores)
    _print_section("Average")
    print(f"Average Overall Match: {_fmt_pct(average_score)}")

    print("\nAverage JSON\n------------")
    print(
        json.dumps(
            {
                "runs": [r["name"] for r in RUNS],
                "overall_match_pct_per_run": run_scores,
                "average_overall_match_pct": average_score,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
