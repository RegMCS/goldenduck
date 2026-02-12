# services/validation_service.py
import numpy as np
import pandas as pd
from scipy import stats
import logging

try:
    from statsmodels.tsa.stattools import acf as sm_acf
except Exception:  # pragma: no cover - optional dependency
    sm_acf = None

logger = logging.getLogger(__name__)


class ValidationService:
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_returns = historical_data["Close"].pct_change().dropna().values

    def validate(self, synthetic_returns: np.ndarray) -> dict:
        """
        Validate synthetic data against historical data
        """
        # Ensure both are numpy arrays
        hist_returns = np.array(self.historical_returns)
        synth_returns = np.array(synthetic_returns)

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

        # Autocorrelation (volatility clustering)
        hist_acf = self._acf_lag1(hist_returns**2)
        synth_acf = self._acf_lag1(synth_returns**2)

        return {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
            "kurtosis_historical": float(hist_kurt),
            "kurtosis_synthetic": float(synth_kurt),
            "skewness_historical": float(hist_skew),
            "skewness_synthetic": float(synth_skew),
            "acf_lag1_historical": float(hist_acf),
            "acf_lag1_synthetic": float(synth_acf),
        }

    @staticmethod
    def _acf_lag1(x: np.ndarray) -> float:
        x = np.asarray(x)
        if len(x) < 2:
            return float("nan")
        if sm_acf is not None:
            return float(sm_acf(x, nlags=10, fft=False)[1])
        x0 = x[:-1]
        x1 = x[1:]
        if np.std(x0) < 1e-12 or np.std(x1) < 1e-12:
            return 0.0
        return float(np.corrcoef(x0, x1)[0, 1])
