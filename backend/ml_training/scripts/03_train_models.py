"""
Train Random Forest models with MLflow tracking
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import sys
from pathlib import Path
import numpy as np
import joblib

SCRIPT_DIR = Path(__file__).parent
ML_TRAINING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ML_TRAINING_DIR))
from config.training_config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    PROCESSED_DATA_DIR,
    MODEL_SAVE_DIR,
)
from models.rf_predictor import train_random_forest


def main():
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load training data
    with open(f"{PROCESSED_DATA_DIR}/train_samples.pkl", "rb") as f:
        train_samples = pickle.load(f)

    with open(f"{PROCESSED_DATA_DIR}/val_samples.pkl", "rb") as f:
        val_samples = pickle.load(f)

    EPS = 1e-8  # to avoid log(0)

    X_train = np.array([s["X"] for s in train_samples])
    y_delta_train = np.array([s["y_delta"] for s in train_samples])
    # y_theta_train = np.array([s["y_theta"] for s in train_samples])
    # y_theta_train_log = np.log(y_theta_train)  # Direct log (all theta values > 0)

    X_val = np.array([s["X"] for s in val_samples])
    y_delta_val = np.array([s["y_delta"] for s in val_samples])
    # y_theta_val = np.array([s["y_theta"] for s in val_samples])
    # y_theta_val_log = np.log(y_theta_val)  # Direct log (all theta values > 0)

    # Train models
    rf_delta = train_random_forest(X_train, y_delta_train, X_val, y_delta_val, "delta")
    # rf_theta = train_random_forest(
    #     X_train, y_theta_train_log, X_val, y_theta_val_log, "theta"
    # )

    # Save models
    joblib.dump(rf_delta, f"{MODEL_SAVE_DIR}/rf_delta.pkl")
    # joblib.dump(rf_theta, f"{MODEL_SAVE_DIR}/rf_theta.pkl")

    print("✓ Models trained and saved!")


if __name__ == "__main__":
    main()
