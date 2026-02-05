"""
Production inference: Load trained models and predict GARCH-FX parameters
For a given asset and user knobs, predict optimal (delta, theta)
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent
ML_TRAINING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ML_TRAINING_DIR))

from config.training_config import MODEL_SAVE_DIR
from feature_extraction.catch22_extractor import extract_all_features
from parameter_optimization.heuristic_theta import compute_theta_hybrid

logger = logging.getLogger(__name__)


class ParameterPredictor:
    """
    Production-ready parameter predictor for GARCH-FX
    Loads trained Random Forest models and makes predictions
    """

    def __init__(self, model_dir=None):
        """
        Initialize the predictor by loading trained models

        Parameters:
        -----------
        model_dir : str, optional
            Directory containing saved models
            Default: MODEL_SAVE_DIR from config
        """
        if model_dir is None:
            model_dir = MODEL_SAVE_DIR

        self.model_dir = Path(model_dir)
        self.rf_delta = None
        self.rf_theta = None

        self._load_models()

    def _load_models(self):
        """Load trained Delta Random Forest model from disk"""
        try:
            delta_path = self.model_dir / "rf_delta.pkl"

            if not delta_path.exists():
                raise FileNotFoundError(f"Delta model not found: {delta_path}")

            self.rf_delta = joblib.load(delta_path)

            logger.info(f"✓ Delta model loaded from {self.model_dir}")
            logger.info(f"   Theta: Using heuristic approach (more reliable)")

        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise

    def predict(
        self,
        historical_returns: np.ndarray,
        user_knobs: dict,
    ) -> dict:
        """
        Predict optimal GARCH-FX parameters for an asset

        Parameters:
        -----------
        historical_returns : np.ndarray
            Historical returns (e.g., from last 2 years of data)
        user_knobs : dict
            User's financial objectives:
            {
                'desired_volatility': float (0.5-2.0),
                'desired_trend': float (-1.0 to 1.0),
                'desired_fat_tails': float (0.8-1.5),
                'desired_momentum': float (0.2-0.8)
            }

        Returns:
        --------
        dict : {
            'delta': float,
            'theta': float,
            'delta_confidence': str ('high', 'medium', 'low'),
            'theta_note': str (explanation of heuristic approach)
        }
        """
        try:
            # Validate inputs
            if len(historical_returns) < 100:
                raise ValueError(
                    f"Need at least 100 returns, got {len(historical_returns)}"
                )

            # Extract features for delta prediction
            asset_features = extract_all_features(historical_returns)
            user_knob_features = np.array(list(user_knobs.values()))
            X = np.concatenate([asset_features, user_knob_features])

            # Predict delta (ML model)
            delta_pred = float(self.rf_delta.predict(X.reshape(1, -1))[0])
            delta_pred = np.clip(delta_pred, 0.1, 10.0)  # Safety bounds

            # Compute theta (heuristic - more reliable than ML for small values)
            theta_pred = self._compute_theta_heuristic(historical_returns, user_knobs)

            # Estimate delta confidence
            delta_confidence = self._estimate_confidence(
                historical_returns, user_knobs, param="delta"
            )

            return {
                "delta": delta_pred,
                "theta": theta_pred,
                "delta_confidence": delta_confidence,
                "theta_note": "Computed via heuristic (inverse relationship with volatility)",
            }

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise

    def _compute_theta_heuristic(
        self, historical_returns: np.ndarray, user_knobs: dict
    ) -> float:
        """
        Compute theta using the heuristic function
        Based on fat_tails and momentum knobs for better stability
        """
        theta = compute_theta_hybrid(user_knobs, historical_returns)
        return theta

    def _estimate_confidence(
        self, historical_returns: np.ndarray, user_knobs: dict, param: str = "delta"
    ) -> str:
        """
        Estimate prediction confidence

        High confidence: Normal returns, well-behaved knobs
        Medium confidence: Slightly unusual returns
        Low confidence: Extreme returns or unusual knobs
        """
        vol = np.std(historical_returns)
        skew = np.abs(np.mean(historical_returns) / (vol + 1e-8))

        # Check if returns are well-behaved
        returns_ok = 0.005 < vol < 0.1 and skew < 2.0

        # Check if knobs are reasonable
        vol_knob = user_knobs.get("desired_volatility", 1.0)
        knobs_ok = 0.5 <= vol_knob <= 2.0

        if returns_ok and knobs_ok:
            confidence = "high"
        elif returns_ok or knobs_ok:
            confidence = "medium"
        else:
            confidence = "low"

        return confidence


def predict_parameters(
    historical_returns: np.ndarray,
    user_knobs: dict,
    model_dir: str = None,
) -> dict:
    """
    Standalone function for quick predictions

    Predicts:
    - delta: Using trained Random Forest model
    - theta: Using heuristic approach (more reliable)

    Usage:
    ------
    params = predict_parameters(
        historical_returns=asset_returns,
        user_knobs={
            'desired_volatility': 1.5,
            'desired_trend': 0.0,
            'desired_fat_tails': 1.2,
            'desired_momentum': 0.5
        }
    )
    print(f"Delta: {params['delta']:.3f}, Theta: {params['theta']:.6f}")
    """
    predictor = ParameterPredictor(model_dir=model_dir)
    return predictor.predict(
        historical_returns=historical_returns, user_knobs=user_knobs
    )


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta

    print("=" * 80)
    print("GARCH-FX PARAMETER PREDICTOR - PRODUCTION INFERENCE")
    print("=" * 80)

    # Download example data
    print("\n1. Downloading AAPL data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years

    try:
        data = yf.download("AAPL", start=start_date, end=end_date, progress=False)
        returns = data["Close"].pct_change().dropna().values
        print(f"   ✓ Downloaded {len(returns)} days")
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        print("   Using synthetic data for demo...")
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, 500)

    # Define user knobs
    user_knobs = {
        "desired_volatility": 1.5,  # 1.5x normal volatility
        "desired_trend": 0.0,  # No directional bias
        "desired_fat_tails": 1.2,  # Slightly heavier tails
        "desired_momentum": 1.0,  # Normal momentum
        # 'desired_mean_reversion': 0.7  # Mild mean reversion
    }

    print("\n2. User Knobs:")
    for knob, value in user_knobs.items():
        print(f"   {knob}: {value}")

    print("\n3. Predicting parameters...")
    try:
        # Predict using ML (delta) + Heuristic (theta)
        params = predict_parameters(historical_returns=returns, user_knobs=user_knobs)

        print(f"\n   Predictions:")
        print(f"     Delta (ML):           {params['delta']:.4f}")
        print(f"     Delta Confidence:     {params['delta_confidence']}")
        print(f"     Theta (Heuristic):    {params['theta']:.6f}")
        print(f"     Theta Note:           {params['theta_note']}")

        print("\n✓ Prediction successful!")
        print("\nThese parameters can now be used with GARCHFXEngine:")
        print(f"  engine.forecast(horizon=252, theta={params['theta']:.6f})")
        print(f"  delta_sequence=[{params['delta']:.2f}] * 252")

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nModels not found. Train them first:")
        print("  python3 03_train_models.py")
    except Exception as e:
        print(f"\n❌ Prediction failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
