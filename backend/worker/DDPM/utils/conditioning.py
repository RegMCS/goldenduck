
import numpy as np
from typing import Dict, Tuple


def compute_conditioning(window: np.ndarray) -> np.ndarray:
    """
    Compute 4 conditioning scalars from a (n_assets, seq_len) window.
    Returns: [realised_vol, drift, tail_index, momentum]
    """
    n_assets = window.shape[0]
    rv    = np.mean([np.std(window[i]) * np.sqrt(252) for i in range(n_assets)])
    drift = np.mean([np.mean(window[i]) * 252         for i in range(n_assets)])
    tail  = np.mean([np.percentile(np.abs(window[i]), 99) for i in range(n_assets)])
    moms  = []
    for i in range(n_assets):
        w = window[i]
        if np.std(w[1:]) > 0 and np.std(w[:-1]) > 0:
            moms.append(np.corrcoef(w[:-1], w[1:])[0, 1])
        else:
            moms.append(0.0)
    mom = np.mean(moms)
    return np.array([rv, drift, tail, mom], dtype=np.float32)


def build_conditioning_matrix(windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Build and normalise conditioning matrix from all training windows.
    windows: (N, n_assets, seq_len)
    Returns: C_raw (N,4), C_norm (N,4), cond_norm_params dict
    """
    C_raw   = np.array([compute_conditioning(windows[i]) for i in range(len(windows))])
    C_min   = C_raw.min(axis=0)
    C_max   = C_raw.max(axis=0)
    C_range = C_max - C_min
    C_range[C_range == 0] = 1.0
    C_norm  = (C_raw - C_min) / C_range
    cond_norm_params = {"min": C_min, "max": C_max}
    return C_raw, C_norm, cond_norm_params


def normalise_cond_vector(cond_vector_raw: Dict, cond_norm_params: Dict) -> np.ndarray:
    """Normalise a single user-provided conditioning vector to [0,1]."""
    arr = np.array([
        cond_vector_raw["realised_vol"],
        cond_vector_raw["drift"],
        cond_vector_raw["tail_index"],
        cond_vector_raw["momentum"],
    ], dtype=np.float32)
    normed = (arr - cond_norm_params["min"]) / (
        cond_norm_params["max"] - cond_norm_params["min"] + 1e-8)
    return np.clip(normed, 0, 1)


def normalise_windows(windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalise each window by per-window per-asset std.
    Returns: windows_norm (N, n_assets, seq_len), window_scales (N, n_assets)
    """
    window_scales = windows.std(axis=-1, keepdims=True) + 1e-8
    windows_norm  = windows / window_scales
    return windows_norm, windows.std(axis=-1)


REGIME_PRESETS = {
    "calm"   : {"realised_vol": 0.10, "drift":  0.05, "tail_index": 0.008, "momentum": -0.05},
    "highvol": {"realised_vol": 0.30, "drift": -0.05, "tail_index": 0.025, "momentum": -0.10},
    "crisis" : {"realised_vol": 0.55, "drift": -0.25, "tail_index": 0.055, "momentum": -0.15},
}