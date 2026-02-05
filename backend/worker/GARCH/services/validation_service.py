# services/validation_service.py
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf
import logging

logger = logging.getLogger(__name__)


class ValidationService:
    def __init__(self, historical_data: pd.DataFrame):
        # Extract returns and force to 1D numpy array
        returns = historical_data["Close"].pct_change().dropna()
        self.historical_returns = np.asarray(returns).flatten()

        # Compute statistics with safe scalar extraction
        self.historical_volatility = self._to_scalar(np.std(self.historical_returns))
        self.historical_kurtosis = self._to_scalar(
            stats.kurtosis(self.historical_returns)
        )
        self.historical_skewness = self._to_scalar(stats.skew(self.historical_returns))

    def _to_scalar(self, value):
        """
        Safely convert numpy array/scalar to Python float
        """
        # If it's already a Python float/int
        if isinstance(value, (float, int)):
            return float(value)

        # If it's a numpy scalar or 0-d array
        if hasattr(value, "item"):
            return float(value.item())

        # If it's a 1-element array
        if hasattr(value, "__len__") and len(value) == 1:
            return float(value[0])

        # Fallback: try to convert directly
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            logger.error(f"Could not convert {type(value)} to scalar: {value}")
            raise

    def validate(self, synthetic_returns: np.ndarray) -> dict:
        """
        Validate synthetic data against historical data
        """
        # Ensure both are numpy arrays
        hist_returns = np.array(self.historical_returns).flatten()
        synth_returns = np.array(synthetic_returns).flatten()

        # Remove any NaN/inf values
        hist_returns = hist_returns[np.isfinite(hist_returns)]
        synth_returns = synth_returns[np.isfinite(synth_returns)]

        logger.info(
            f"Validating: hist={len(hist_returns)} samples, synth={len(synth_returns)} samples"
        )

        # KS test
        ks_stat, ks_pval = stats.ks_2samp(hist_returns, synth_returns)

        # Kurtosis (fat tails)
        hist_kurt = stats.kurtosis(hist_returns)
        synth_kurt = stats.kurtosis(synth_returns)

        # Skewness (asymmetry)
        hist_skew = stats.skew(hist_returns)
        synth_skew = stats.skew(synth_returns)

        # Autocorrelation
        hist_acf_returns = acf(hist_returns, nlags=10, fft=False)[1]
        synth_acf_returns = acf(synth_returns, nlags=10, fft=False)[1]

        hist_acf_vol = acf(hist_returns**2, nlags=10, fft=False)[1]
        synth_acf_vol = acf(synth_returns**2, nlags=10, fft=False)[1]

        return {
            "ks_statistic": self._to_scalar(ks_stat),
            "ks_pvalue": self._to_scalar(ks_pval),
            "kurtosis_historical": self._to_scalar(hist_kurt),
            "kurtosis_synthetic": self._to_scalar(synth_kurt),
            "skewness_historical": self._to_scalar(hist_skew),
            "skewness_synthetic": self._to_scalar(synth_skew),
            "acf_returns_historical": self._to_scalar(hist_acf_returns),
            "acf_returns_synthetic": self._to_scalar(synth_acf_returns),
            "acf_volatility_historical": self._to_scalar(hist_acf_vol),
            "acf_volatility_synthetic": self._to_scalar(synth_acf_vol),
        }

    def validate_against_desired(
        self, synthetic_returns: np.ndarray, user_knobs: dict
    ) -> dict:
        """
        Validate synthetic data against desired characteristics from user knobs.
        """
        # Clean synthetic returns
        synth_returns = np.array(synthetic_returns).flatten()
        synth_returns = synth_returns[np.isfinite(synth_returns)]

        # Calculate synthetic statistics
        synth_volatility = self._to_scalar(np.std(synth_returns))
        synth_kurtosis = self._to_scalar(stats.kurtosis(synth_returns))
        synth_skewness = self._to_scalar(stats.skew(synth_returns))
        synth_acf = self._to_scalar(acf(synth_returns, nlags=10, fft=False)[1])

        # Extract user knobs
        desired_volatility = float(user_knobs.get("desired_volatility", 1.0))
        desired_fat_tails = float(user_knobs.get("desired_fat_tails", 1.0))
        desired_trend = float(user_knobs.get("desired_trend", 0.0))
        desired_momentum = float(user_knobs.get("desired_momentum", 0.5))

        # ============================================================
        # Compute desired characteristics
        # ============================================================

        # Volatility
        desired_volatility_value = self.historical_volatility * desired_volatility

        # Kurtosis (with momentum adjustment)
        base_kurtosis = self.historical_kurtosis * desired_fat_tails
        momentum_boost = max(0.0, (desired_momentum - 0.7) * 2.0)
        desired_kurtosis_value = base_kurtosis + momentum_boost

        # Skewness (AR(1) + trend contributions)
        phi = -0.1 + 0.4 * desired_momentum

        if abs(phi) > 0.05:
            k = 0.5 + (desired_volatility - 1.0) * 0.2
            ar_skewness = k * phi
        else:
            ar_skewness = 0.0

        if abs(desired_trend) > 0.7:
            trend_skewness = desired_trend * 0.2
        else:
            trend_skewness = 0.0

        desired_skewness_value = ar_skewness + trend_skewness

        # ACF (with GARCH dampening)
        dampening = 0.9
        desired_acf_value = phi * dampening

        # ============================================================
        # Calculate matches
        # ============================================================

        volatility_match = 1.0 - abs(synth_volatility - desired_volatility_value) / (
            desired_volatility_value + 1e-6
        )
        kurtosis_match = 1.0 - min(
            abs(synth_kurtosis - desired_kurtosis_value)
            / (abs(desired_kurtosis_value) + 0.5),
            1.0,
        )
        skewness_match = 1.0 - abs(synth_skewness - desired_skewness_value) / (
            abs(desired_skewness_value) + 0.2
        )
        acf_match = 1.0 - abs(synth_acf - desired_acf_value) / (
            abs(desired_acf_value) + 0.1
        )

        # ============================================================
        # Logging
        # ============================================================

        logger.info(
            f"Validation vs desired:"
            f"\n  Volatility: synth={synth_volatility:.4f} vs desired={desired_volatility_value:.4f} (match={volatility_match:.2%})"
            f"\n  Kurtosis: synth={synth_kurtosis:.2f} vs desired={desired_kurtosis_value:.2f} (match={kurtosis_match:.2%})"
            f"\n  Skewness: synth={synth_skewness:.4f} vs desired={desired_skewness_value:.4f} (match={skewness_match:.2%})"
            f"\n  ACF: synth={synth_acf:.4f} vs desired={desired_acf_value:.4f} (match={acf_match:.2%})"
        )

        return {
            "volatility_synthetic": synth_volatility,
            "volatility_desired": desired_volatility_value,
            "volatility_match": volatility_match,
            "kurtosis_synthetic": synth_kurtosis,
            "kurtosis_desired": desired_kurtosis_value,
            "kurtosis_match": kurtosis_match,
            "skewness_synthetic": synth_skewness,
            "skewness_desired": desired_skewness_value,
            "skewness_match": skewness_match,
            "acf_synthetic": synth_acf,
            "acf_desired": desired_acf_value,
            "acf_match": acf_match,
            "overall_match": (
                volatility_match + kurtosis_match + skewness_match + acf_match
            )
            / 4.0,
        }
