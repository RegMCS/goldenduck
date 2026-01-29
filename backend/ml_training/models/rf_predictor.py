"""
Random Forest predictor for parameter optimization
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def train_random_forest(X_train, y_train, X_val, y_val, model_name="model"):
    """
    Train Random Forest with MLflow tracking

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        X_val: Validation features
        y_val: Validation targets
        model_name: Name for logging ('delta' or 'theta')

    Returns:
        Trained RandomForestRegressor
    """

    # Hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }

    # Train
    with mlflow.start_run(run_name=f"rf_{model_name}"):
        mlflow.log_params(params)

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Evaluate
        y_train_pred = rf.predict(X_train)
        y_val_pred = rf.predict(X_val)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.sklearn.log_model(rf, f"rf_{model_name}")

        print(f"✓ {model_name}: Train RMSE={train_rmse:.6f}, Val RMSE={val_rmse:.6f}")

        return rf
