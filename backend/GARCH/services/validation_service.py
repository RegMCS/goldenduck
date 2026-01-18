# services/validation_service.py
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf
import logging

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
        hist_acf = acf(hist_returns**2, nlags=10, fft=False)[1]
        synth_acf = acf(synth_returns**2, nlags=10, fft=False)[1]

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
