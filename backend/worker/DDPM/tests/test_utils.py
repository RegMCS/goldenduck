import pytest
import numpy as np
from worker.DDPM.utils import (compute_conditioning, normalise_windows,
                                normalise_cond_vector, REGIME_PRESETS)


def test_compute_conditioning_shape():
    window = np.random.randn(7, 1260).astype(np.float32)
    c = compute_conditioning(window)
    assert c.shape == (4,), f"Expected (4,) got {c.shape}"


def test_normalise_windows_unit_std():
    windows = np.random.randn(10, 7, 252).astype(np.float32)
    norm, scales = normalise_windows(windows)
    stds = norm.std(axis=-1)
    assert np.allclose(stds, 1.0, atol=0.05), \
        f"Normalised windows std not ~1: mean={stds.mean():.4f}"


def test_normalise_cond_vector_bounds():
    cond_norm_params = {
        "min": np.array([0.05, -0.5, 0.001, -0.3], dtype=np.float32),
        "max": np.array([0.8,   0.5,  0.1,   0.3], dtype=np.float32),
    }
    for regime, preset in REGIME_PRESETS.items():
        normed = normalise_cond_vector(preset, cond_norm_params)
        assert normed.min() >= 0.0 and normed.max() <= 1.0, \
            f"Regime {regime}: normalised vector out of [0,1]: {normed}"


def test_regime_presets_ordering():
    assert (REGIME_PRESETS["crisis"]["realised_vol"]
            > REGIME_PRESETS["highvol"]["realised_vol"]
            > REGIME_PRESETS["calm"]["realised_vol"])