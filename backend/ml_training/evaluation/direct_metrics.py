"""
Direct evaluation metrics for parameter prediction
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)


def evaluate_direct_metrics(
    model, X_test, y_test, param_name="delta", is_log_theta=False
):
    """
    Evaluate model on direct parameter prediction metrics

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets (original scale)
        param_name: 'delta' or 'theta'
        is_log_theta: If True, model predicts log(theta), so inverse transform predictions

    Returns:
        dict: RMSE, MAE, R², MAPE (all computed on original scale)
    """

    # Predictions
    y_pred_log = model.predict(X_test)

    # Inverse transform if theta model (log-scale predictions)
    if is_log_theta:
        y_pred = np.exp(y_pred_log)  # Inverse of ln(theta)
        y_pred = np.clip(y_pred, 1e-6, 0.1)  # Clip to realistic theta range [1e-6, 0.1]
    else:
        y_pred = y_pred_log

    # Calculate metrics (always on original scale)
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test, y_pred) * 100,
    }

    # Print
    print(f"\n{param_name.upper()} Prediction Metrics:")
    print(f"  RMSE:  {metrics['rmse']:.6f}")
    print(f"  MAE:   {metrics['mae']:.6f}")
    print(f"  R²:    {metrics['r2']:.4f}")
    print(f"  MAPE:  {metrics['mape']:.2f}%")

    if is_log_theta:
        print(f"  Pred range: [{np.min(y_pred):.6f}, {np.max(y_pred):.6f}]")
        print(f"  True range: [{np.min(y_test):.6f}, {np.max(y_test):.6f}]")

    # Additional statistics
    errors = y_pred - y_test
    print(f"\nError Distribution:")
    print(f"  Mean error:   {np.mean(errors):.6f}")
    print(f"  Std error:    {np.std(errors):.6f}")
    print(f"  Max error:    {np.max(np.abs(errors)):.6f}")
    print(f"  95th %ile:    {np.percentile(np.abs(errors), 95):.6f}")

    return metrics
