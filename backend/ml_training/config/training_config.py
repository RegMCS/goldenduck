"""
Configuration for Random Forest training
SIMPLIFIED FOR TESTING
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# MLFLOW CONFIGURATION (File-based for testing)
# ============================================================

# Simple file-based tracking (no database needed)
MLFLOW_TRACKING_URI = "./mlruns"  # Just use local filesystem
MLFLOW_EXPERIMENT_NAME = "garch-fx-parameter-prediction"

# NOTE: When ready for production, switch to PostgreSQL:
# MLFLOW_TRACKING_URI = f"postgresql://{user}:{pass}@{host}:{port}/mlflow_tracking"


# ============================================================
# DATA CONFIGURATION (TESTING: 50 assets instead of 500)
# ============================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data/raw"
PROCESSED_DATA_DIR = BASE_DIR / "data/processed"
MODEL_SAVE_DIR = BASE_DIR / "models/saved_models"
EVALUATION_DIR = BASE_DIR / "models/evaluation"

TESTING_MODE = False  # Set to False for production
# Testing: Use only 50 assets (instead of 500)
if TESTING_MODE:
    N_ASSETS = 100  # ← Changed from 50 to 100
    N_SCENARIOS_PER_ASSET = 20  # ← Changed from 10 to 20
    TOTAL_SAMPLES = N_ASSETS * N_SCENARIOS_PER_ASSET  # 2,000 samples
else:
    N_ASSETS = 500
    N_SCENARIOS_PER_ASSET = 50
    TOTAL_SAMPLES = 25000
TRAIN_SPLIT = 0.70  # 35 assets for training
VAL_SPLIT = 0.15  # 8 assets for validation
TEST_SPLIT = 0.15  # 7 assets for testing

# Total samples: 50 assets × 10 scenarios = 500 samples (much faster!)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

CATCH22_FEATURES = 22
FINANCIAL_FEATURES = 6
USER_KNOB_FEATURES = 4
TOTAL_FEATURES = CATCH22_FEATURES + FINANCIAL_FEATURES + USER_KNOB_FEATURES  # 32


# ============================================================
# RANDOM FOREST HYPERPARAMETERS (Simpler for testing)
# ============================================================

RF_PARAMS = {
    "n_estimators": 50,  # ← REDUCED from 100 (faster)
    "max_depth": 10,  # ← REDUCED from 15 (faster)
    "min_samples_split": 5,  # ← REDUCED from 10
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# GRID SEARCH (REDUCED for faster testing)
# ============================================================

# Testing: Use coarser grid (faster)
DELTA_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # 6 values (was 10)
THETA_GRID = [1e-5, 1e-4, 1e-3, 1e-2]  # 4 values (was 10)

# Total combinations: 6 × 4 = 24 (instead of 500)
# Per asset: 24 combinations × 10 scenarios = 240 simulations
# Total: 50 assets × 240 = 12,000 simulations (~30 minutes instead of 12 hours)


# ============================================================
# TESTING MODE
# ============================================================


if TESTING_MODE:
    print("⚠️  TESTING MODE ACTIVE")
    print(f"   Using {N_ASSETS} assets (not 500)")
    print(f"   Using {N_SCENARIOS_PER_ASSET} scenarios per asset (not 50)")
    print(f"   Total samples: {N_ASSETS * N_SCENARIOS_PER_ASSET}")
    print(
        f"   Grid search: {len(DELTA_GRID)} × {len(THETA_GRID)} = {len(DELTA_GRID) * len(THETA_GRID)} combinations"
    )
    print(f"   Estimated time: ~30-60 minutes (not 12 hours)")
