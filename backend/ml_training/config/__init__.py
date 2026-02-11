"""
Configuration module for ML training pipeline
Exports all configuration variables for easy import
"""

from .training_config import (
    # MLflow configuration
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    # Data paths
    BASE_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_SAVE_DIR,
    # Data generation parameters
    N_ASSETS,
    N_SCENARIOS_PER_ASSET,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    # Feature dimensions
    CATCH22_FEATURES,
    FINANCIAL_FEATURES,
    USER_KNOB_FEATURES,
    TOTAL_FEATURES,
    # Random Forest hyperparameters
    RF_PARAMS,
    # Grid search parameters
    DELTA_GRID,
    THETA_GRID,
    # Testing mode flag
    TESTING_MODE,
)

__all__ = [
    # MLflow
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    # Paths
    "BASE_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODEL_SAVE_DIR",
    # Data parameters
    "N_ASSETS",
    "N_SCENARIOS_PER_ASSET",
    "TRAIN_SPLIT",
    "VAL_SPLIT",
    "TEST_SPLIT",
    # Features
    "CATCH22_FEATURES",
    "FINANCIAL_FEATURES",
    "USER_KNOB_FEATURES",
    "TOTAL_FEATURES",
    # Model hyperparameters
    "RF_PARAMS",
    # Grid search
    "DELTA_GRID",
    "THETA_GRID",
    # Testing mode
    "TESTING_MODE",
]

# Print configuration on import (helpful for debugging)
if TESTING_MODE:
    print("=" * 60)
    print("⚠️  ML TRAINING CONFIGURATION (TESTING MODE)")
    print("=" * 60)
    print(f"Assets: {N_ASSETS} (testing)")
    print(f"Scenarios per asset: {N_SCENARIOS_PER_ASSET}")
    print(f"Total samples: {N_ASSETS * N_SCENARIOS_PER_ASSET}")
    print(f"Feature dimensions: {TOTAL_FEATURES}")
    print(f"MLflow: {MLFLOW_TRACKING_URI}")
    print("=" * 60)
