"""
Scoring functions for evaluating synthetic data quality
Comprehensively covers all 4 user knobs: volatility, trend, fat_tails, momentum
"""

import numpy as np
from scipy import stats


def score_synthetic_data(synthetic_returns, historical_returns, user_knobs):
    """
    Score how well synthetic data matches desired characteristics
    Returns score between 0 (bad) and 1 (perfect)

    Covers all 4 user knobs:
    - desired_volatility: via volatility match (30%)
    - desired_trend: via mean return match (20%)
    - desired_fat_tails: via kurtosis match (25%)
    - desired_momentum: via autocorrelation match (25%)
    """

    synthetic_flat = synthetic_returns.flatten()

    # ============================================================
    # Validation: Remove NaN/Inf
    # ============================================================
    synthetic_flat = synthetic_flat[np.isfinite(synthetic_flat)]

    if len(synthetic_flat) < 100:
        return 0.0

    # ============================================================
    # Extract synthetic characteristics
    # ============================================================
    synthetic_vol = np.std(synthetic_flat) * np.sqrt(252)
    synthetic_mean = np.mean(synthetic_flat) * 252  # Annualized mean return
    synthetic_kurtosis = stats.kurtosis(synthetic_flat)
    synthetic_skew = stats.skew(synthetic_flat)

    # Autocorrelation (momentum)
    if len(synthetic_flat) > 1:
        try:
            synthetic_autocorr = np.corrcoef(synthetic_flat[:-1], synthetic_flat[1:])[
                0, 1
            ]
            if not np.isfinite(synthetic_autocorr):
                synthetic_autocorr = 0
        except:
            synthetic_autocorr = 0
    else:
        synthetic_autocorr = 0

    # ============================================================
    # Extract historical characteristics
    # ============================================================
    historical_vol = np.std(historical_returns) * np.sqrt(252)
    historical_mean = np.mean(historical_returns) * 252
    historical_kurtosis = stats.kurtosis(historical_returns)
    historical_autocorr = (
        np.corrcoef(historical_returns[:-1], historical_returns[1:])[0, 1]
        if len(historical_returns) > 1
        else 0
    )

    # ============================================================
    # Calculate target characteristics (historical × knobs)
    # ============================================================

    # 1. Target volatility
    target_vol = historical_vol * user_knobs["desired_volatility"]

    # 2. Target mean return (trend)
    # Map trend knob [-1, +1] to mean return adjustment
    # trend = -1 → -15% annual return (strong bear)
    # trend = 0 → historical mean (neutral)
    # trend = +1 → +15% annual return (strong bull)
    trend_knob = user_knobs["desired_trend"]
    if trend_knob < 0:
        # Bearish: scale down to negative
        target_mean = historical_mean + (trend_knob * 0.15)  # -1 → -15%
    elif trend_knob > 0:
        # Bullish: scale up to positive
        target_mean = historical_mean + (trend_knob * 0.15)  # +1 → +15%
    else:
        # Neutral: keep historical
        target_mean = historical_mean

    # 3. Target kurtosis (fat tails)
    # Additive adjustment (kurtosis doesn't scale linearly)
    target_kurtosis = historical_kurtosis + 3 * (user_knobs["desired_fat_tails"] - 1.0)

    # 4. Target autocorrelation (momentum)
    # Map momentum [0, 1] to autocorr [-0.1, 0.3]
    # momentum = 0 → autocorr = -0.1 (mean reverting)
    # momentum = 0.5 → autocorr = 0.1 (neutral)
    # momentum = 1.0 → autocorr = 0.3 (strong trending)
    target_autocorr = -0.1 + 0.4 * user_knobs["desired_momentum"]

    # ============================================================
    # Score each characteristic
    # ============================================================

    # 1. Volatility score (30%)
    vol_error = abs(synthetic_vol - target_vol) / max(target_vol, 0.01)
    vol_score = max(0, 1 - vol_error)

    # 2. Trend score (20%)
    # More lenient: within ±5% is good
    mean_error = abs(synthetic_mean - target_mean)
    mean_score = max(0, 1 - mean_error / 0.05)  # ±5% tolerance

    # 3. Kurtosis score (25%)
    kurtosis_error = abs(synthetic_kurtosis - target_kurtosis) / max(
        abs(target_kurtosis), 3
    )
    kurtosis_score = max(0, 1 - kurtosis_error)

    # 4. Momentum score (25%)
    momentum_error = abs(synthetic_autocorr - target_autocorr)
    momentum_score = max(0, 1 - 2 * momentum_error)  # ±0.5 tolerance

    # ============================================================
    # Weighted total score
    # ============================================================
    total_score = (
        0.30 * vol_score
        + 0.20 * mean_score
        + 0.25 * kurtosis_score
        + 0.25 * momentum_score
    )

    return float(np.clip(total_score, 0, 1))


def detailed_score_breakdown(synthetic_returns, historical_returns, user_knobs):
    """
    Return detailed breakdown of each scoring component
    Useful for debugging and understanding what's wrong
    """

    synthetic_flat = synthetic_returns.flatten()
    synthetic_flat = synthetic_flat[np.isfinite(synthetic_flat)]

    if len(synthetic_flat) < 100:
        return {"error": "Insufficient valid data"}

    # Extract characteristics
    synthetic_vol = np.std(synthetic_flat) * np.sqrt(252)
    synthetic_mean = np.mean(synthetic_flat) * 252
    synthetic_kurtosis = stats.kurtosis(synthetic_flat)

    try:
        synthetic_autocorr = np.corrcoef(synthetic_flat[:-1], synthetic_flat[1:])[0, 1]
        if not np.isfinite(synthetic_autocorr):
            synthetic_autocorr = 0
    except:
        synthetic_autocorr = 0

    historical_vol = np.std(historical_returns) * np.sqrt(252)
    historical_mean = np.mean(historical_returns) * 252
    historical_kurtosis = stats.kurtosis(historical_returns)

    # Calculate targets
    target_vol = historical_vol * user_knobs["desired_volatility"]

    trend_knob = user_knobs["desired_trend"]
    target_mean = historical_mean + (trend_knob * 0.15)

    target_kurtosis = historical_kurtosis + 3 * (user_knobs["desired_fat_tails"] - 1.0)
    target_autocorr = -0.1 + 0.4 * user_knobs["desired_momentum"]

    # Calculate scores
    vol_error = abs(synthetic_vol - target_vol) / max(target_vol, 0.01)
    vol_score = max(0, 1 - vol_error)

    mean_error = abs(synthetic_mean - target_mean)
    mean_score = max(0, 1 - mean_error / 0.05)

    kurtosis_error = abs(synthetic_kurtosis - target_kurtosis) / max(
        abs(target_kurtosis), 3
    )
    kurtosis_score = max(0, 1 - kurtosis_error)

    momentum_error = abs(synthetic_autocorr - target_autocorr)
    momentum_score = max(0, 1 - 2 * momentum_error)

    total_score = (
        0.30 * vol_score
        + 0.20 * mean_score
        + 0.25 * kurtosis_score
        + 0.25 * momentum_score
    )

    return {
        # Volatility
        "vol_target": float(target_vol),
        "vol_actual": float(synthetic_vol),
        "vol_error": float(vol_error),
        "vol_score": float(vol_score),
        # Trend (mean return)
        "mean_target": float(target_mean),
        "mean_actual": float(synthetic_mean),
        "mean_error": float(mean_error),
        "mean_score": float(mean_score),
        # Fat tails (kurtosis)
        "kurtosis_target": float(target_kurtosis),
        "kurtosis_actual": float(synthetic_kurtosis),
        "kurtosis_error": float(kurtosis_error),
        "kurtosis_score": float(kurtosis_score),
        # Momentum (autocorrelation)
        "autocorr_target": float(target_autocorr),
        "autocorr_actual": float(synthetic_autocorr),
        "momentum_error": float(momentum_error),
        "momentum_score": float(momentum_score),
        # Overall
        "total_score": float(total_score),
        # User knobs (for reference)
        "user_knobs": user_knobs,
    }


def compare_to_baseline(synthetic_returns, historical_returns, user_knobs):
    """
    Compare synthetic data to historical baseline
    Shows if synthetic is 'better' or 'worse' than just using historical
    """

    # Score synthetic data
    synthetic_score = score_synthetic_data(
        synthetic_returns, historical_returns, user_knobs
    )

    # Score historical data (as if it were synthetic)
    # This shows how well historical matches the desired knobs
    historical_as_synthetic = historical_returns.reshape(1, -1)
    baseline_score = score_synthetic_data(
        historical_as_synthetic, historical_returns, user_knobs
    )

    return {
        "synthetic_score": float(synthetic_score),
        "baseline_score": float(baseline_score),
        "improvement": float(synthetic_score - baseline_score),
        "better_than_baseline": bool(synthetic_score > baseline_score),
    }
