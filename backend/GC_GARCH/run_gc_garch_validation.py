"""Generate GC-GARCH validation CSVs from AAPL.csv.

Default behavior:
- Input: backend/AAPL.csv
- Output folder: backend/gc_garch_validation
- Output file: baseline.csv
- Baseline knobs: volatility=1.0, fat_tails=1.0, momentum=0.5, trend=0.0

Edit the CONFIG block below to create new validation outputs.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from services.gc_garch_service import GCGarchGenerator, GCGarchKnobs, normalize_knobs


# ==============================
# Editable config
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV_PATH = PROJECT_ROOT / "AAPL.csv"
OUTPUT_DIR = PROJECT_ROOT / "gc_garch_validation"
OUTPUT_FILENAME = "baseline.csv"
SEED = 41
HORIZON = None  # None -> use the same number of rows as the input CSV

# baseline.csv knobs (requested defaults)
KNOBS = GCGarchKnobs(
    volatility=1.0,
    fat_tails=1.0,
    momentum=0.5,
    trend=0.0,
)


REQUIRED_COLUMNS = {"close"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(c).strip().lower() for c in normalized.columns]
    return normalized


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")


def main() -> None:
    if not INPUT_CSV_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV_PATH}")

    df = pd.read_csv(INPUT_CSV_PATH)
    df = _normalize_columns(df)
    _validate_input(df)

    knobs = normalize_knobs(KNOBS)
    horizon = int(HORIZON) if HORIZON is not None else int(len(df))
    if horizon <= 0:
        raise ValueError("HORIZON must be > 0")

    generator = GCGarchGenerator()
    generator.fit(df)
    scenario = generator.generate(horizon=horizon, knobs=knobs, seed=SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME
    scenario.to_csv(output_path, index=False)

    print("GC-GARCH validation run complete")
    print("Input CSV:", INPUT_CSV_PATH)
    print("Output CSV:", output_path)
    print("Rows generated:", len(scenario))
    print("Knobs:", asdict(knobs))


if __name__ == "__main__":
    main()
