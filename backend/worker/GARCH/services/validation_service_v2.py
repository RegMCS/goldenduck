# services/validation_service_v2.py
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ValidationServiceV2:
    """
    Validation Service V2

    Compares original OHLCV vs synthetic OHLCV using:
    - Volatility: std dev of log returns
    - Fat tails: kurtosis (matched on normalized kurtosis)
    - Momentum: Hurst exponent of cumulative return process
    - Trend: mean log return

    Matching uses hybrid linear decay (relative + absolute floor).
    """

    def __init__(
        self,
        *,
        # Shared hybrid-decay knobs:
        # err = |synth-target| / (abs_floor + rel_weight*|target|)
        # match = max(0, 1 - decay_k*err)
        decay_rel_weight: float = 1.0,
        decay_k: float = 1.0,
        # Per-metric absolute floors (protect very small targets from over-penalization)
        volatility_abs_floor: float = 0.10,
        fat_tail_abs_floor: float = 0.50,
        momentum_abs_floor: float = 0.10,
        trend_abs_floor: float = 0.01,
    ) -> None:
        self.decay_rel_weight = float(decay_rel_weight)
        self.decay_k = float(decay_k)
        self.volatility_abs_floor = float(volatility_abs_floor)
        self.fat_tail_abs_floor = float(fat_tail_abs_floor)
        self.momentum_abs_floor = float(momentum_abs_floor)
        self.trend_abs_floor = float(trend_abs_floor)

    def validate(
        self,
        original_ohlcv: pd.DataFrame,
        synthetic_ohlcv: pd.DataFrame,
        user_knobs: Dict | None = None,
    ) -> Dict:
        """
        Validate synthetic OHLCV against original OHLCV.
        Returns raw metrics and percent matches for each knob.
        """
        orig_close = self._extract_close(original_ohlcv, label="original")
        synth_close = self._extract_close(synthetic_ohlcv, label="synthetic")

        knobs = user_knobs or {}
        volatility_knob = float(knobs.get("volatility", 1.0))
        fat_tails_knob = float(knobs.get("fat_tails", 1.0))
        momentum_knob = float(knobs.get("momentum", 0.5))
        trend_knob = float(knobs.get("trend", 0.0))

        orig_returns = self._percent_returns(orig_close)
        synth_returns = self._percent_returns(synth_close)
        orig_log_returns = self._log_returns(orig_close)
        synth_log_returns = self._log_returns(synth_close)

        # =========================
        # Volatility
        # =========================
        vol_orig = self._to_scalar(np.std(orig_log_returns, ddof=1))
        vol_synth = self._to_scalar(np.std(synth_log_returns, ddof=1))
        vol_target = vol_orig * volatility_knob
        vol_match = self._match_hybrid(
            vol_target,
            vol_synth,
            abs_floor=self.volatility_abs_floor,
        )

        # =========================
        # Fat tails
        # =========================
        kurt_orig = self._to_scalar(stats.kurtosis(orig_returns))
        kurt_synth = self._to_scalar(stats.kurtosis(synth_returns))
        # Requested mapping:
        # target_kurt = hist_kurt * (1 + 0.15 * (fat_tails - 1))
        kurt_target = kurt_orig * (1.0 + 0.15 * (fat_tails_knob - 1.0))
        kurt_orig_norm = self._normalize_kurtosis(kurt_orig)
        kurt_target_norm = self._normalize_kurtosis(kurt_target)
        kurt_synth_norm = self._normalize_kurtosis(kurt_synth)
        kurt_match = self._match_hybrid(
            kurt_target_norm,
            kurt_synth_norm,
            abs_floor=self.fat_tail_abs_floor,
        )

        fat_tail_match = kurt_match

        # =========================
        # Momentum (Hurst exponent)
        # =========================
        # Use log returns and estimate H on cumulative demeaned increments
        # so random-walk-like behavior is centered around H ~= 0.5.
        hurst_orig = self._hurst_exponent(orig_log_returns)
        hurst_synth = self._hurst_exponent(synth_log_returns)
        # Keep knob semantics centered at 0.5:
        # knob=0.5 -> target = original
        # knob<0.5 -> move target toward 0.5 (more anti-persistent/random)
        # knob>0.5 -> amplify distance from 0.5 (more persistent)
        momentum_scale = momentum_knob / 0.5
        hurst_target = 0.5 + (hurst_orig - 0.5) * momentum_scale
        hurst_target = float(np.clip(hurst_target, 0.0, 1.0))
        momentum_match = self._match_hybrid(
            hurst_target,
            hurst_synth,
            abs_floor=self.momentum_abs_floor,
        )

        # =========================
        # Trend (mean returns)
        # =========================
        trend_orig = self._to_scalar(np.mean(orig_log_returns))
        trend_synth = self._to_scalar(np.mean(synth_log_returns))
        # Keep existing trend knob semantics:
        # - trend = 0.0 => multiplier = 1.0 (neutral)
        # - trend = +1.0 => multiplier = 2.0 (stronger same direction)
        # - trend = -1.0 => multiplier = -1.0 (full reversal)
        if trend_knob < 0:
            trend_multiplier = 1.0 + 2.0 * trend_knob
        else:
            trend_multiplier = 1.0 + trend_knob
        trend_target = trend_orig * trend_multiplier
        trend_match = self._match_hybrid(
            trend_target,
            trend_synth,
            abs_floor=self.trend_abs_floor,
        )

        # =========================
        # Overall
        # =========================
        overall_match = (vol_match + fat_tail_match + momentum_match + trend_match) / 4.0

        return {
            "volatility": {
                "original": vol_orig,
                "target": vol_target,
                "synthetic": vol_synth,
                "decay_abs_floor": self.volatility_abs_floor,
                "decay_rel_weight": self.decay_rel_weight,
                "decay_k": self.decay_k,
                "match_pct": vol_match * 100.0,
            },
            "fat_tails": {
                "kurtosis": {
                    "original": kurt_orig,
                    "target": kurt_target,
                    "synthetic": kurt_synth,
                    "original_normalized": kurt_orig_norm,
                    "target_normalized": kurt_target_norm,
                    "synthetic_normalized": kurt_synth_norm,
                    "decay_abs_floor": self.fat_tail_abs_floor,
                    "decay_rel_weight": self.decay_rel_weight,
                    "decay_k": self.decay_k,
                    "match_pct": kurt_match * 100.0,
                },
                "match_pct": fat_tail_match * 100.0,
            },
            "momentum": {
                "hurst": {
                    "original": hurst_orig,
                    "target": hurst_target,
                    "synthetic": hurst_synth,
                    "decay_abs_floor": self.momentum_abs_floor,
                    "decay_rel_weight": self.decay_rel_weight,
                    "decay_k": self.decay_k,
                    "match_pct": momentum_match * 100.0,
                },
                "match_pct": momentum_match * 100.0,
            },
            "trend": {
                "mean_return": {
                    "original": trend_orig,
                    "target": trend_target,
                    "synthetic": trend_synth,
                    "decay_abs_floor": self.trend_abs_floor,
                    "decay_rel_weight": self.decay_rel_weight,
                    "decay_k": self.decay_k,
                    "match_pct": trend_match * 100.0,
                },
                "match_pct": trend_match * 100.0,
            },
            "overall_match_pct": overall_match * 100.0,
        }

    def _extract_close(self, ohlcv: pd.DataFrame, label: str) -> pd.Series:
        if "Close" not in ohlcv.columns:
            raise ValueError(f"{label} OHLCV missing 'Close' column")
        return ohlcv["Close"].astype(float).dropna()

    def _percent_returns(self, close: pd.Series) -> np.ndarray:
        returns = close.pct_change().dropna().values
        returns = np.asarray(returns, dtype=float)
        return returns[np.isfinite(returns)]

    def _log_returns(self, close: pd.Series) -> np.ndarray:
        x = close.astype(float).values
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return np.asarray([], dtype=float)
        valid = (x[1:] > 0.0) & (x[:-1] > 0.0)
        if not np.any(valid):
            return np.asarray([], dtype=float)
        returns = np.log(x[1:][valid] / x[:-1][valid])
        returns = np.asarray(returns, dtype=float)
        return returns[np.isfinite(returns)]

    def _hurst_exponent(self, returns: np.ndarray) -> float:
        """
        Estimate Hurst exponent with variance-time scaling on cumulative returns:
        Let y_t = cumsum(r_t - mean(r)).
        Then std(y[t+lag] - y[t]) ~ lag^H.
        For uncorrelated returns (random walk), H is typically near 0.5.
        """
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        n = r.size
        if n < 20:
            return float("nan")

        y = np.cumsum(r - np.mean(r))
        max_lag = min(100, n // 2)
        if max_lag < 3:
            return float("nan")

        lags = np.arange(2, max_lag + 1, dtype=int)
        used_lags: list[float] = []
        tau: list[float] = []

        for lag in lags:
            diffs = y[lag:] - y[:-lag]
            s = float(np.std(diffs))
            if np.isfinite(s) and s > 0.0:
                used_lags.append(float(lag))
                tau.append(s)

        if len(used_lags) < 2:
            return float("nan")

        slope, _ = np.polyfit(np.log(used_lags), np.log(tau), 1)
        return float(np.clip(slope, 0.0, 1.0))

    def _ma20_trend_metrics(self, close: pd.Series) -> tuple[float, float]:
        ma20 = close.rolling(20).mean()
        diffs = ma20.diff().dropna()
        if diffs.empty:
            return float("nan"), float("nan")
        slope_avg = self._to_scalar(diffs.mean())
        up_ratio = self._to_scalar(np.mean(diffs > 0))
        return slope_avg, up_ratio

    def _start_end_return(self, close: pd.Series) -> float:
        if close.size < 2:
            return float("nan")
        first = float(close.iloc[0])
        last = float(close.iloc[-1])
        if first == 0:
            return float("nan")
        return (last - first) / first

    def _match_hybrid(self, target: float, synth: float, abs_floor: float) -> float:
        """
        Hybrid linear decay:
        err = |synth-target| / (abs_floor + rel_weight*|target|)
        match = max(0, 1 - decay_k*err)
        """
        if not np.isfinite(target) or not np.isfinite(synth):
            return 0.0
        denom = abs(abs_floor) + self.decay_rel_weight * abs(target)
        if denom <= 0:
            return 0.0
        err = abs(synth - target) / denom
        return float(max(0.0, 1.0 - (self.decay_k * err)))

    def _normalize_kurtosis(self, value: float) -> float:
        """
        Signed log normalization to compress heavy-tailed ranges while preserving sign.
        """
        if not np.isfinite(value):
            return float("nan")
        return float(np.sign(value) * np.log1p(abs(value)))

    def _to_scalar(self, value) -> float:
        if isinstance(value, (float, int)):
            return float(value)
        if hasattr(value, "item"):
            return float(value.item())
        if hasattr(value, "__len__") and len(value) == 1:
            return float(value[0])
        return float(value)
