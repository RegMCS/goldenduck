"""
End-to-end evaluation: Generate synthetic data and evaluate quality
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append("../..")
from config.training_config import RAW_DATA_DIR
from parameter_optimization.grid_search import garch_fx_simulate, score_synthetic_data


def evaluate_end_to_end(rf_delta, rf_theta, test_samples, save_results=True):
    """
    End-to-end evaluation:
    1. Use AI to predict parameters
    2. Generate synthetic data
    3. Score quality

    Returns:
        float: Average quality score (0-1)
    """

    quality_scores = []
    results_list = []

    for sample in tqdm(test_samples, desc="End-to-end evaluation"):
        # Load historical data
        asset = sample["asset"]
        df = pd.read_csv(f"{RAW_DATA_DIR}/{asset}.csv")
        historical_returns = df["returns"].values

        # Extract inputs
        X = sample["X"]
        user_knobs = {
            "desired_volatility": X[28],
            "desired_trend": X[29],
            "desired_fat_tails": X[30],
            "desired_momentum": X[31],
            "desired_mean_reversion": X[32],
        }

        # AI predicts parameters
        delta_pred = rf_delta.predict(X.reshape(1, -1))[0]

        # Predict log-theta
        theta_log_pred = rf_theta.predict(X.reshape(1, -1))[0]

        # Convert back to theta (inverse of ln)
        theta_pred = float(np.exp(theta_log_pred))

        # Clip to sensible range
        theta_pred = float(np.clip(theta_pred, 1e-6, 0.1))

        # Generate synthetic data
        try:
            synthetic_returns = garch_fx_simulate(
                historical_returns=historical_returns,
                delta=delta_pred,
                theta=theta_pred,
                horizon=252,
                num_paths=100,
            )

            # Score quality
            quality = score_synthetic_data(
                synthetic_returns=synthetic_returns,
                historical_returns=historical_returns,
                user_knobs=user_knobs,
            )

            quality_scores.append(quality)

            results_list.append(
                {
                    "asset": asset,
                    "delta_pred": delta_pred,
                    "theta_pred": theta_pred,
                    "quality_score": quality,
                    "user_vol": user_knobs["desired_volatility"],
                    "user_tails": user_knobs["desired_fat_tails"],
                }
            )

        except Exception as e:
            print(f"⚠️  Failed for {asset}: {str(e)}")
            continue

    # Calculate statistics
    avg_quality = np.mean(quality_scores)
    std_quality = np.std(quality_scores)
    min_quality = np.min(quality_scores)
    max_quality = np.max(quality_scores)

    print(f"\nEnd-to-End Quality Scores:")
    print(f"  Average:  {avg_quality:.3f}")
    print(f"  Std Dev:  {std_quality:.3f}")
    print(f"  Min:      {min_quality:.3f}")
    print(f"  Max:      {max_quality:.3f}")
    print(f"  Median:   {np.median(quality_scores):.3f}")

    # Save results
    if save_results:
        from config.training_config import PROCESSED_DATA_DIR

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f"{PROCESSED_DATA_DIR}/end_to_end_results.csv", index=False)
        print(f"\n✓ Results saved to {PROCESSED_DATA_DIR}/end_to_end_results.csv")

    return avg_quality
