"""
Find optimal (delta, theta) parameters via grid search
This generates the ground truth labels for training
"""

import numpy as np
from scipy import stats
import sys
from pathlib import Path
from arch import arch_model

SCRIPT_DIR = Path(__file__).parent
ML_TRAINING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ML_TRAINING_DIR))

# Import from worker (single source of truth)
WORKER_DIR = ML_TRAINING_DIR.parent / "worker"
sys.path.insert(0, str(WORKER_DIR))

from config.training_config import DELTA_GRID, THETA_GRID
from GARCH.services.garchfx_engine import GARCHFXEngine


def find_optimal_parameters(historical_returns, user_knobs, method="grid_search"):
    """
    Find (delta, theta) that produces synthetic data matching:
    1. Historical asset characteristics
    2. User's desired modifications (knobs)

    Returns: (optimal_delta, optimal_theta)
    """

    # Fit GARCH(1,1) to historical data
    from arch import arch_model

    model = arch_model(historical_returns * 100, vol="Garch", p=1, q=1, rescale=True)
    try:
        fitted = model.fit(disp="off", show_warning=False)
        omega = fitted.params["omega"]
        alpha = fitted.params["alpha[1]"]
        beta = fitted.params["beta[1]"]
    except:
        # If GARCH fitting fails, use default values
        omega = 0.01
        alpha = 0.1
        beta = 0.85

    # Grid search
    best_score = -np.inf
    best_params = (1.0, 0.001)  # Default

    for delta in DELTA_GRID:
        for theta in THETA_GRID:
            # Generate synthetic data
            try:
                synthetic_returns = garch_fx_simulate(
                    historical_returns=historical_returns,
                    delta=delta,
                    theta=theta,
                    omega=omega,
                    alpha=alpha,
                    beta=beta,
                    horizon=252,
                    num_paths=10,  # Use 10 paths for speed
                )

                # Score quality
                score = score_synthetic_data(
                    synthetic_returns=synthetic_returns,
                    historical_returns=historical_returns,
                    user_knobs=user_knobs,
                )

                if score > best_score:
                    best_score = score
                    best_params = (delta, theta)

            except Exception as e:
                # Skip if simulation fails
                continue

    return best_params


def garch_fx_simulate(
    historical_returns,
    delta,
    theta,
    omega=None,
    alpha=None,
    beta=None,
    horizon=252,
    num_paths=10,
    distribution: str = "normal",
):
    """
    Wrapper that uses GARCHFXEngine to simulate returns paths.
    - delta: used as a constant multiplicative stress on omega
    - theta: passed as the stochasticity parameter to the engine
    """

    # 1) Fit GARCH(1,1) if params not provided
    if omega is None or alpha is None or beta is None:
        am = arch_model(historical_returns * 100, vol="Garch", p=1, q=1, rescale=True)
        res = am.fit(disp="off", show_warning=False)
        omega = res.params["omega"]
        alpha = res.params["alpha[1]"]
        beta = res.params["beta[1]"]
        last_vol = res.conditional_volatility[-1] / 100.0
    else:
        # If you already computed them in find_optimal_parameters, also pass last_vol in
        # For now, approximate last_vol from historical data:
        last_vol = np.std(historical_returns)

    # 2) Apply delta as a level modifier on omega
    params = {
        "alpha": float(alpha),
        "beta": float(beta),
        "omega": float(omega * delta),  # delta modulates base variance level
    }

    # 3) Create engine
    engine = GARCHFXEngine(
        volatility=float(last_vol),
        params=params,
        scale_factor=100.0,
    )

    # 4) Generate volatility paths and returns
    paths = []
    for _ in range(num_paths):
        vol_path = engine.forecast(horizon=horizon, theta=float(theta))
        ret_path = engine.generate_returns_from_volatility(
            vol_path, distribution=distribution
        )
        paths.append(ret_path)

    return np.vstack(paths)


def score_synthetic_data(synthetic_returns, historical_returns, user_knobs):
    """
    Score how well synthetic data matches desired characteristics
    Returns score between 0 (bad) and 1 (perfect)
    """

    synthetic_flat = synthetic_returns.flatten()

    # ============================================================
    # FIX: Add validation for synthetic data
    # ============================================================
    # Remove any NaN or Inf values
    synthetic_flat = synthetic_flat[np.isfinite(synthetic_flat)]

    if len(synthetic_flat) < 100:  # Need minimum data points
        return 0.0

    # Extract characteristics
    synthetic_vol = np.std(synthetic_flat) * np.sqrt(252)
    synthetic_kurtosis = stats.kurtosis(synthetic_flat)
    synthetic_skew = stats.skew(synthetic_flat)

    # Autocorrelation
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

    historical_vol = np.std(historical_returns) * np.sqrt(252)
    historical_kurtosis = stats.kurtosis(historical_returns)

    # Target characteristics (historical × user knobs)
    target_vol = historical_vol * user_knobs["desired_volatility"]
    target_kurtosis = historical_kurtosis * user_knobs["desired_fat_tails"]
    target_momentum = user_knobs["desired_momentum"]

    # ============================================================
    # FIX: Add bounds checking for scores
    # ============================================================
    # Score each characteristic
    # 1. Volatility match
    vol_error = abs(synthetic_vol - target_vol) / max(target_vol, 0.01)
    vol_score = max(0, 1 - vol_error)

    # 2. Kurtosis match
    kurtosis_error = abs(synthetic_kurtosis - target_kurtosis) / max(
        abs(target_kurtosis), 3
    )
    kurtosis_score = max(0, 1 - kurtosis_error)

    # 3. Momentum match (via autocorrelation)
    target_autocorr = target_momentum - 0.5
    momentum_error = abs(synthetic_autocorr - target_autocorr)
    momentum_score = max(0, 1 - momentum_error)

    # 4. Distribution shape (KS test)
    try:
        ks_stat = stats.ks_2samp(historical_returns, synthetic_flat).statistic
        distribution_score = max(0, 1 - ks_stat)
    except:
        distribution_score = 0.5  # Neutral score if test fails

    # Weighted average
    total_score = (
        0.35 * vol_score
        + 0.25 * kurtosis_score
        + 0.20 * momentum_score
        + 0.20 * distribution_score
    )

    return float(np.clip(total_score, 0, 1))
