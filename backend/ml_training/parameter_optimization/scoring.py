"""
Scoring functions for evaluating synthetic data quality
Comprehensively covers all 4 user knobs: volatility, trend, fat_tails, momentum
"""

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf as sm_acf


def _compute_target_kurtosis(user_knobs):
    """
    Model-aware kurtosis target — mirrors compute_target_kurtosis in
    validation_service.py exactly so training and inference optimise
    against the same objective.

    fat_tails = 0.5  → ~2.0  (near-normal)
    fat_tails = 1.0  → ~4.5  (model baseline with t(6)+GARCH)
    fat_tails = 2.0  → ~10.0 (near model ceiling)
    """
    desired_fat_tails = float(user_knobs.get("desired_fat_tails", 1.0))
    desired_momentum = float(user_knobs.get("desired_momentum", 0.5))

    baseline_kurtosis = 4.5
    max_kurtosis = 10.0

    if desired_fat_tails <= 1.0:
        kurtosis_from_fat_tails = 2.0 + (baseline_kurtosis - 2.0) * (
            desired_fat_tails / 1.0
        )
    else:
        excess = desired_fat_tails - 1.0
        kurtosis_from_fat_tails = baseline_kurtosis + (
            max_kurtosis - baseline_kurtosis
        ) * min(excess, 1.0)

    momentum_boost = 0.0
    if desired_momentum > 0.7:
        momentum_boost = (desired_momentum - 0.7) / 0.3 * 2.0

    return float(min(kurtosis_from_fat_tails + momentum_boost, max_kurtosis))


def _compute_target_daily_mean(user_knobs):
    """
    Match validation_service.py trend target exactly (daily units).
    Absolute bull/bear semantics:
      desired_trend=-1 -> -15% annual, 0 -> 0%, +1 -> +15% annual.
    """
    desired_trend = float(user_knobs.get("desired_trend", 0.0))
    return desired_trend * 0.15 / 252


def _compute_mean_sign_score(synthetic_mean, desired_trend, synthetic_std, sample_size):
    """
    Trend score focused on direction (bull/bear), not exact mean magnitude.
    Mirrors ValidationService._compute_mean_sign_match().
    """
    neutral_trend_eps = 0.1
    fixed_floor_band = 0.05 / 252  # ±5% annualised treated as neutral
    if sample_size > 1 and synthetic_std > 0:
        se_mean = synthetic_std / np.sqrt(sample_size)
        noise_band = 2.0 * se_mean
    else:
        noise_band = fixed_floor_band
    neutral_mean_band = max(fixed_floor_band, noise_band)

    if abs(desired_trend) < neutral_trend_eps:
        return float(np.exp(-abs(synthetic_mean) / (neutral_mean_band + 1e-12)))

    desired_sign = 1.0 if desired_trend > 0 else -1.0
    return 1.0 if (synthetic_mean * desired_sign) > 0 else 0.0


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
    synthetic_std = np.std(synthetic_flat)
    synthetic_vol = synthetic_std * np.sqrt(252)
    synthetic_mean = np.mean(synthetic_flat)  # Daily mean return
    synthetic_kurtosis = stats.kurtosis(synthetic_flat)
    synthetic_skew = stats.skew(synthetic_flat)

    # Autocorrelation (momentum) — use statsmodels ACF to match inference
    try:
        synthetic_autocorr = float(sm_acf(synthetic_flat, nlags=10, fft=False)[1])
        if not np.isfinite(synthetic_autocorr):
            synthetic_autocorr = 0.0
    except Exception:
        synthetic_autocorr = 0.0

    # ============================================================
    # Extract historical characteristics
    # ============================================================
    historical_vol = np.std(historical_returns) * np.sqrt(252)
    historical_mean = np.mean(historical_returns)
    historical_kurtosis = stats.kurtosis(historical_returns)
    historical_autocorr = (
        float(sm_acf(historical_returns, nlags=10, fft=False)[1])
        if len(historical_returns) > 1
        else 0.0
    )

    # ============================================================
    # Calculate target characteristics (historical × knobs)
    # ============================================================

    # 1. Target volatility
    target_vol = historical_vol * user_knobs["desired_volatility"]

    # 2. Target mean return (trend) - daily units, aligned with inference
    target_mean = _compute_target_daily_mean(user_knobs)

    # 3. Target kurtosis — model-aware formula matching validation_service.py
    target_kurtosis = _compute_target_kurtosis(user_knobs)

    # 4. Target autocorrelation (momentum) with GARCH dampening
    # phi maps [0,1] → [-0.1, 0.3]; multiplied by 0.9 to match
    # the dampening applied at inference in validation_service.py
    phi = -0.1 + 0.4 * user_knobs["desired_momentum"]
    target_autocorr = phi * 0.9

    # ============================================================
    # Score each characteristic
    # ============================================================

    # 1. Volatility score (30%)
    vol_error = abs(synthetic_vol - target_vol) / max(target_vol, 0.01)
    vol_score = max(0, 1 - vol_error)

    # 2. Trend score (20%)
    # Same normalization as inference validation service
    mean_error = abs(synthetic_mean - target_mean)
    mean_score = _compute_mean_sign_score(
        synthetic_mean,
        float(user_knobs.get("desired_trend", 0.0)),
        synthetic_std,
        len(synthetic_flat),
    )

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
    synthetic_std = np.std(synthetic_flat)
    synthetic_vol = synthetic_std * np.sqrt(252)
    synthetic_mean = np.mean(synthetic_flat)
    synthetic_kurtosis = stats.kurtosis(synthetic_flat)

    try:
        synthetic_autocorr = float(sm_acf(synthetic_flat, nlags=10, fft=False)[1])
        if not np.isfinite(synthetic_autocorr):
            synthetic_autocorr = 0.0
    except Exception:
        synthetic_autocorr = 0.0

    historical_vol = np.std(historical_returns) * np.sqrt(252)
    historical_mean = np.mean(historical_returns)
    historical_kurtosis = stats.kurtosis(historical_returns)

    # Calculate targets
    target_vol = historical_vol * user_knobs["desired_volatility"]

    target_mean = _compute_target_daily_mean(user_knobs)

    target_kurtosis = _compute_target_kurtosis(user_knobs)
    phi = -0.1 + 0.4 * user_knobs["desired_momentum"]
    target_autocorr = phi * 0.9

    # Calculate scores
    vol_error = abs(synthetic_vol - target_vol) / max(target_vol, 0.01)
    vol_score = max(0, 1 - vol_error)

    mean_error = abs(synthetic_mean - target_mean)
    mean_score = _compute_mean_sign_score(
        synthetic_mean,
        float(user_knobs.get("desired_trend", 0.0)),
        synthetic_std,
        len(synthetic_flat),
    )

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
