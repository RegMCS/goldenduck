"""
Scoring functions for evaluating synthetic data quality
"""

import numpy as np
from scipy import stats


def score_synthetic_data(synthetic_returns, historical_returns, user_knobs):
    """
    Main scoring function - imported by grid_search.py
    """
    from .grid_search import score_synthetic_data as _score

    return _score(synthetic_returns, historical_returns, user_knobs)


def detailed_score_breakdown(synthetic_returns, historical_returns, user_knobs):
    """
    Return detailed breakdown of each scoring component
    """

    synthetic_flat = synthetic_returns.flatten()

    # Extract all characteristics
    scores = {}

    # Volatility
    synthetic_vol = np.std(synthetic_flat) * np.sqrt(252)
    historical_vol = np.std(historical_returns) * np.sqrt(252)
    target_vol = historical_vol * user_knobs["desired_volatility"]
    vol_error = abs(synthetic_vol - target_vol) / target_vol
    scores["vol_score"] = max(0, 1 - vol_error)
    scores["vol_target"] = target_vol
    scores["vol_actual"] = synthetic_vol

    # Kurtosis
    synthetic_kurtosis = stats.kurtosis(synthetic_flat)
    historical_kurtosis = stats.kurtosis(historical_returns)
    target_kurtosis = historical_kurtosis * user_knobs["desired_fat_tails"]
    kurtosis_error = abs(synthetic_kurtosis - target_kurtosis) / max(
        abs(target_kurtosis), 3
    )
    scores["kurtosis_score"] = max(0, 1 - kurtosis_error)
    scores["kurtosis_target"] = target_kurtosis
    scores["kurtosis_actual"] = synthetic_kurtosis

    # Distribution similarity
    ks_stat = stats.ks_2samp(historical_returns, synthetic_flat).statistic
    scores["distribution_score"] = max(0, 1 - ks_stat)
    scores["ks_statistic"] = ks_stat

    # Overall
    scores["total_score"] = (
        0.35 * scores["vol_score"]
        + 0.25 * scores["kurtosis_score"]
        + 0.40 * scores["distribution_score"]
    )

    return scores
