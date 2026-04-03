"""
Conditioning vector construction for Stage 2 DDPM.
Computes 4D market regime vectors: [realised_vol, drift, tail_index, momentum]
"""

import numpy as np
import pandas as pd
from typing import Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ConditioningVector:
    """
    Market regime representation for Stage 2 DDPM conditioning.
    4D vector: [realised_vol, drift, tail_index, momentum]
    """

    def __init__(self, realised_vol: float, drift: float, tail_index: float, momentum: float):
        """
        Args:
            realised_vol: Annualised realised volatility [0.01, 1.0]
            drift: Annualised drift/expected return [-0.5, 0.5]
            tail_index: Tail index (99th percentile of log returns) [0.001, 0.1]
            momentum: Autocorrelation at lag-1 [-1.0, 1.0]
        """
        self.realised_vol = float(realised_vol)
        self.drift = float(drift)
        self.tail_index = float(tail_index)
        self.momentum = float(momentum)

    def to_array(self) -> np.ndarray:
        """Return as numpy array [4,]"""
        return np.array(
            [self.realised_vol, self.drift, self.tail_index, self.momentum],
            dtype=np.float32,
        )

    def __repr__(self):
        return (
            f"ConditioningVector(RV={self.realised_vol:.3f}, "
            f"drift={self.drift:.3f}, tail={self.tail_index:.4f}, momentum={self.momentum:.3f})"
        )


def compute_conditioning_from_window(
    window: np.ndarray, annualization_factor: float = np.sqrt(252)
) -> ConditioningVector:
    """
    Compute conditioning vector from a single rolling window.

    Args:
        window: Array of shape (num_assets, window_len) with log returns
        annualization_factor: √252 for daily data

    Returns:
        ConditioningVector with [RV, drift, tail, momentum]
    """
    # Flatten to per-asset returns
    flat_returns = window.flatten()

    # 1. Realised Volatility (annualised)
    realised_vol = float(flat_returns.std() * annualization_factor)

    # 2. Drift (annualised) — mean return
    drift = float(flat_returns.mean() * annualization_factor)

    # 3. Tail Index — 99th percentile of absolute returns
    tail_index = float(np.percentile(np.abs(flat_returns), 99))

    # 4. Momentum — ACF lag-1 of flat returns
    acf = compute_acf(flat_returns, nlags=1)
    momentum = float(acf[1])

    return ConditioningVector(
        realised_vol=realised_vol,
        drift=drift,
        tail_index=tail_index,
        momentum=momentum,
    )


def compute_acf(x: np.ndarray, nlags: int = 1) -> np.ndarray:
    """
    Compute autocorrelation function.

    Args:
        x: Time series array
        nlags: Maximum lag

    Returns:
        ACF values [0:nlags+1]
    """
    x = x - x.mean()
    c0 = np.dot(x, x) / len(x)
    acf_vals = [1.0]
    for k in range(1, nlags + 1):
        ck = np.dot(x[:-k], x[k:]) / len(x)
        acf_vals.append(ck / c0)
    return np.array(acf_vals)


def compute_conditioning_batch(
    windows: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute conditioning vectors for a batch of windows.

    Args:
        windows: Array of shape (num_windows, num_assets, window_len)

    Returns:
        Tuple of:
          - conditioning_batch: Array of shape (num_windows, 4)
          - stress_scores: Array of shape (num_windows,) for regime detection
    """
    num_windows = windows.shape[0]
    conditioning_batch = np.zeros((num_windows, 4), dtype=np.float32)

    for i in range(num_windows):
        c_vec = compute_conditioning_from_window(windows[i])
        conditioning_batch[i] = c_vec.to_array()

    # Stress scoring for regime detection
    stress_scores = _compute_stress_scores(windows)

    logger.info(
        f"Computed conditioning batch: shape={conditioning_batch.shape}, "
        f"RV range=[{conditioning_batch[:, 0].min():.3f}, {conditioning_batch[:, 0].max():.3f}]"
    )
    return conditioning_batch, stress_scores


def _compute_stress_scores(windows: np.ndarray) -> np.ndarray:
    """
    Simple stress scoring: ratio of samples > 1.5 std to total samples.

    Args:
        windows: Array of shape (num_windows, num_assets, window_len)

    Returns:
        Stress scores array of shape (num_windows,)
    """
    flat = windows.flatten()
    threshold = 1.5 * flat.std()
    stress_flags = (windows > threshold).astype(float)
    stress_scores = stress_flags.mean(axis=(1, 2))  # Average across assets and time
    return stress_scores


def detect_regime(
    conditioning_vector: ConditioningVector,
    vol_threshold_low: float = 0.25,
    vol_threshold_high: float = 0.50,
) -> str:
    """
    Simple regime classifier based on realised volatility.

    Args:
        conditioning_vector: ConditioningVector
        vol_threshold_low: Below this → calm
        vol_threshold_high: Above this → crisis; in-between → highvol

    Returns:
        Regime string: "calm", "highvol", or "crisis"
    """
    rv = conditioning_vector.realised_vol
    if rv < vol_threshold_low:
        return "calm"
    elif rv < vol_threshold_high:
        return "highvol"
    else:
        return "crisis"
