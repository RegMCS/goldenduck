"""GC-based intrabar range sampling.

Uses a Cornish-Fisher style adjustment to inject skewness and kurtosis.
This is an approximation that behaves well for moderate knob ranges.
"""
from typing import Tuple

import numpy as np


def sample_gc_z(
    rng: np.random.Generator,
    n: int,
    skew: float,
    excess_kurtosis: float,
) -> np.ndarray:
    """Sample GC-adjusted innovations using a Cornish-Fisher expansion."""
    u = rng.standard_normal(n)

    s = float(skew)
    k = float(excess_kurtosis)

    # Cornish-Fisher expansion (approximate) for skewness and excess kurtosis
    z = (
        u
        + (s / 6.0) * (u**2 - 1.0)
        + (k / 24.0) * (u**3 - 3.0 * u)
        - (s**2 / 36.0) * (2.0 * u**3 - 5.0 * u)
    )

    # Guard against extreme tails from the approximation
    return np.clip(z, -10.0, 10.0)


def sample_gc_high_low_range(
    sigma: float,
    skew: float,
    excess_kurtosis: float,
    tail_scale: float = 1.0,
    range_scale: float = 1.5,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    """Return (high_mult, low_mult) where H = max(O,C)*(1+high_mult)
    and L = min(O,C)*(1-low_mult).

    Skew is preserved by using the positive tail for highs and the
    negative tail for lows.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = float(excess_kurtosis) * float(tail_scale)

    z_high = sample_gc_z(rng, 1, skew, k)[0]
    z_low = sample_gc_z(rng, 1, skew, k)[0]

    high_mult = max(0.0, z_high) * sigma * range_scale
    low_mult = max(0.0, -z_low) * sigma * range_scale

    return high_mult, low_mult
