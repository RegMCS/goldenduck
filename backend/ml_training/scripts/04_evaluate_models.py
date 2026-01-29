"""
Evaluate trained Random Forest models on test set
Includes both direct metrics and end-to-end evaluation
"""

import sys
import pickle
import numpy as np
import pandas as pd
import joblib
import mlflow
import json
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
ML_TRAINING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ML_TRAINING_DIR))
from config.training_config import *
from evaluation.direct_metrics import evaluate_direct_metrics
from evaluation.end_to_end_eval import evaluate_end_to_end


def convert_to_json_serializable(obj):
    """
    Recursively convert numpy/pandas types to JSON-serializable Python types
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def main():
    """
    Complete evaluation pipeline on test set
    """

    print("=" * 80)
    print("EVALUATING MODELS ON TEST SET")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────
    # 1. Load models
    # ──────────────────────────────────────────────────────────
    print("\n📦 Loading trained models...")
    rf_delta = joblib.load(f"{MODEL_SAVE_DIR}/rf_delta.pkl")
    rf_theta = joblib.load(f"{MODEL_SAVE_DIR}/rf_theta.pkl")
    print("✓ Models loaded")

    # ──────────────────────────────────────────────────────────
    # 2. Load test data
    # ──────────────────────────────────────────────────────────
    print("\n📊 Loading test samples...")

    # Load train/val/test samples
    with open(f"{PROCESSED_DATA_DIR}/train_samples.pkl", "rb") as f:
        train_samples = pickle.load(f)

    with open(f"{PROCESSED_DATA_DIR}/val_samples.pkl", "rb") as f:
        val_samples = pickle.load(f)

    with open(f"{PROCESSED_DATA_DIR}/test_samples.pkl", "rb") as f:
        test_samples = pickle.load(f)

    X_test = np.array([s["X"] for s in test_samples])
    y_delta_test = np.array([s["y_delta"] for s in test_samples])
    y_theta_test = np.array([s["y_theta"] for s in test_samples])

    print(f"✓ Loaded {len(test_samples)} test samples")
    print(f"  Test assets: {len(set([s['asset'] for s in test_samples]))} unique")

    # ──────────────────────────────────────────────────────────
    # 3. EVALUATION LEVEL 1: Direct Metrics
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LEVEL 1: DIRECT PARAMETER PREDICTION METRICS")
    print("=" * 80)

    delta_metrics = evaluate_direct_metrics(
        rf_delta, X_test, y_delta_test, param_name="delta"
    )

    theta_metrics = evaluate_direct_metrics(
        rf_theta, X_test, y_theta_test, param_name="theta", is_log_theta=True
    )

    # Extract individual metrics
    delta_rmse = delta_metrics["rmse"]
    delta_mae = delta_metrics["mae"]
    delta_r2 = delta_metrics["r2"]

    theta_rmse = theta_metrics["rmse"]
    theta_mae = theta_metrics["mae"]
    theta_r2 = theta_metrics["r2"]

    # ──────────────────────────────────────────────────────────
    # 4. EVALUATION LEVEL 2: End-to-End Quality
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LEVEL 2: END-TO-END SYNTHETIC DATA QUALITY")
    print("=" * 80)
    print("(This will take ~10 minutes...)\n")

    end_to_end_scores = evaluate_end_to_end(
        rf_delta=rf_delta,
        rf_theta=rf_theta,
        test_samples=test_samples[:50],  # Test on 50 samples
        save_results=True,
    )

    # Extract end-to-end score
    if isinstance(end_to_end_scores, dict):
        end_to_end_score = end_to_end_scores.get("mean_score", 0)
    else:
        end_to_end_score = float(end_to_end_scores)

    # ──────────────────────────────────────────────────────────
    # 5. EVALUATION LEVEL 3: Compare to Baselines
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LEVEL 3: COMPARISON TO BASELINES")
    print("=" * 80)

    baseline_scores = evaluate_baselines(
        rf_delta=rf_delta, rf_theta=rf_theta, test_samples=test_samples[:50]
    )

    # ──────────────────────────────────────────────────────────
    # 6. Log to MLflow
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LOGGING RESULTS TO MLFLOW")
    print("=" * 80)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="FINAL_TEST_EVALUATION"):
        # Log direct metrics
        mlflow.log_metrics(
            {
                "test_delta_rmse": float(delta_rmse),
                "test_delta_mae": float(delta_mae),
                "test_delta_r2": float(delta_r2),
                "test_theta_rmse": float(theta_rmse),
                "test_theta_mae": float(theta_mae),
                "test_theta_r2": float(theta_r2),
            }
        )

        # Log end-to-end score
        mlflow.log_metric("test_end_to_end_quality", float(end_to_end_score))

        # Log baseline comparisons
        mlflow.log_metrics(
            {
                "test_ai_score": float(baseline_scores["ai_score"]),
                "test_baseline_fixed": float(baseline_scores["baseline_fixed"]),
                "test_baseline_heuristic": float(baseline_scores["baseline_heuristic"]),
                "test_p_value_vs_fixed": float(baseline_scores["p_value_vs_fixed"]),
                "test_p_value_vs_heuristic": float(
                    baseline_scores["p_value_vs_heuristic"]
                ),
            }
        )

        # Log test set info
        mlflow.log_params(
            {
                "test_samples": len(test_samples),
                "test_assets": len(set([s["asset"] for s in test_samples])),
                "evaluation_date": pd.Timestamp.now().isoformat(),
            }
        )

    print("✓ Results logged to MLflow")

    # ──────────────────────────────────────────────────────────
    # 7. Final Summary
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL TEST SET EVALUATION SUMMARY")
    print("=" * 80)

    print("\n📊 Direct Metrics:")
    print(f"  Delta - RMSE: {delta_rmse:.4f}, R²: {delta_r2:.3f}")
    print(f"  Theta - RMSE: {theta_rmse:.6f}, R²: {theta_r2:.3f}")

    print(f"\n🎯 End-to-End Quality: {end_to_end_score:.3f}")

    print("\n📈 vs Baselines:")
    print(f"  AI (RF):            {baseline_scores['ai_score']:.3f}")
    print(f"  Fixed parameters:   {baseline_scores['baseline_fixed']:.3f}")
    print(f"  Simple heuristic:   {baseline_scores['baseline_heuristic']:.3f}")
    print(
        f"  Improvement:        +{(baseline_scores['ai_score'] - baseline_scores['baseline_fixed']) * 100:.1f}%"
    )

    # Pass/Fail
    print("\n" + "=" * 80)
    print("PASS/FAIL CRITERIA")
    print("=" * 80)

    checks = {
        "Delta R² > 0.7": delta_r2 > 0.7,
        "Theta R² > 0.6": theta_r2 > 0.6,
        "End-to-End > 0.7": end_to_end_score > 0.7,
        "Better than fixed": baseline_scores["ai_score"]
        > baseline_scores["baseline_fixed"],
        "Better than heuristic": baseline_scores["ai_score"]
        > baseline_scores["baseline_heuristic"],
    }

    for criterion, passed in checks.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {criterion:30s} {status}")

    all_passed = all(checks.values())
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED - Model ready for production!")
    else:
        print("\n⚠️  SOME CHECKS FAILED - Model needs improvement")

    # ──────────────────────────────────────────────────────────
    # 8. Save Evaluation Report
    # ──────────────────────────────────────────────────────────
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset_size": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "direct_metrics": {"delta": delta_metrics, "theta": theta_metrics},
        "end_to_end_metrics": (
            {"mean_score": end_to_end_score}
            if isinstance(end_to_end_score, (int, float))
            else end_to_end_scores
        ),
        "baseline_comparison": baseline_scores,
        "quality_checks": checks,
        "overall_pass": all_passed,
    }

    # Convert to JSON-serializable types
    report = convert_to_json_serializable(report)

    # Save report
    Path(EVALUATION_DIR).mkdir(parents=True, exist_ok=True)
    report_path = f"{EVALUATION_DIR}/evaluation_report.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Full report saved to {report_path}")


def evaluate_baselines(rf_delta, rf_theta, test_samples):
    """
    Compare AI approach to baseline methods
    """
    from evaluation.end_to_end_eval import score_synthetic_data
    from parameter_optimization.grid_search import garch_fx_simulate

    ai_scores = []
    baseline_fixed_scores = []
    baseline_heuristic_scores = []

    for sample in tqdm(test_samples, desc="Comparing baselines"):
        try:
            # Load historical data
            asset = sample["asset"]
            df = pd.read_csv(f"{RAW_DATA_DIR}/{asset}.csv")
            historical_returns = df["returns"].values

            # Extract user knobs
            X = sample["X"]
            user_knobs = {
                "desired_volatility": float(X[28]),
                "desired_trend": float(X[29]),
                "desired_fat_tails": float(X[30]),
                "desired_momentum": float(X[31]),
                "desired_mean_reversion": float(X[32]),
            }

            # AI approach
            delta_ai = float(rf_delta.predict(X.reshape(1, -1))[0])
            # theta_ai = float(rf_theta.predict(X.reshape(1, -1))[0])

            # Predict log-theta
            theta_log_ai = rf_theta.predict(X.reshape(1, -1))[0]

            # Convert back to theta (inverse of ln)
            theta_ai = float(np.exp(theta_log_ai))

            # Clip to sensible range
            theta_ai = float(np.clip(theta_ai, 1e-6, 0.1))

            # Clip predictions
            delta_ai = np.clip(delta_ai, 0.001, 100)
            # theta_ai = np.clip(theta_ai, 0.001, 10)

            synthetic_ai = garch_fx_simulate(
                historical_returns, delta_ai, theta_ai, horizon=252, num_paths=10
            )
            ai_scores.append(
                score_synthetic_data(synthetic_ai, historical_returns, user_knobs)
            )

            # Baseline 1: Fixed parameters
            synthetic_fixed = garch_fx_simulate(
                historical_returns, delta=1.0, theta=0.001, horizon=252, num_paths=10
            )
            baseline_fixed_scores.append(
                score_synthetic_data(synthetic_fixed, historical_returns, user_knobs)
            )

            # Baseline 2: Simple heuristic
            delta_heur = float(user_knobs["desired_volatility"])
            theta_heur = 0.001 * float(user_knobs["desired_fat_tails"])
            synthetic_heur = garch_fx_simulate(
                historical_returns, delta_heur, theta_heur, horizon=252, num_paths=10
            )
            baseline_heuristic_scores.append(
                score_synthetic_data(synthetic_heur, historical_returns, user_knobs)
            )

        except Exception as e:
            print(f"\n⚠️  Error processing sample: {str(e)}")
            continue

    # Statistical test
    from scipy import stats

    t_stat_fixed, p_value_fixed = stats.ttest_rel(ai_scores, baseline_fixed_scores)
    t_stat_heur, p_value_heur = stats.ttest_rel(ai_scores, baseline_heuristic_scores)

    return {
        "ai_score": float(np.mean(ai_scores)),
        "baseline_fixed": float(np.mean(baseline_fixed_scores)),
        "baseline_heuristic": float(np.mean(baseline_heuristic_scores)),
        "p_value_vs_fixed": float(p_value_fixed),
        "p_value_vs_heuristic": float(p_value_heur),
        "significant_vs_fixed": bool(p_value_fixed < 0.05),
        "significant_vs_heuristic": bool(p_value_heur < 0.05),
    }


if __name__ == "__main__":
    main()
