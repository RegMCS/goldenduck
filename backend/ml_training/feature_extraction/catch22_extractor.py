"""
Layer 1: Feature extraction using catch22
"""

import pycatch22
import numpy as np
from scipy import stats


def _normalize_returns(returns):
    """Normalize input to a 1D float array and remove non-finite values."""
    arr = np.asarray(returns, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 10:
        raise ValueError("Not enough valid return values for feature extraction")
    return arr


def extract_catch22_features(returns):
    """Extract 22 catch22 features from time series"""
    returns = _normalize_returns(returns)
    c22 = pycatch22.catch22_all(returns.tolist())
    return np.array(c22["values"])


def extract_financial_features(returns):
    """Extract 6 financial domain features"""
    returns = _normalize_returns(returns)
    features = [
        np.std(returns) * np.sqrt(252),  # realized vol
        stats.kurtosis(returns),  # tail thickness
        stats.skew(returns),  # asymmetry
        np.corrcoef(returns[:-1] ** 2, returns[1:] ** 2)[0, 1],  # ARCH effect
        np.percentile(returns, 5),  # VaR 95%
        (
            np.maximum.accumulate(np.cumsum(returns)) - np.cumsum(returns)
        ).max(),  # max drawdown
    ]
    return np.array(features)


def extract_all_features(returns):
    """Extract all 28 asset features"""
    catch22 = extract_catch22_features(returns)
    financial = extract_financial_features(returns)
    return np.concatenate([catch22, financial])
