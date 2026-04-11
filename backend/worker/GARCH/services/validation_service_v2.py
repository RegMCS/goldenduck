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
    - Fat tails: kurtosis only
    - Momentum: Hurst exponent of percent returns
    - Trend: MA(20) slope avg, MA(20) uptrend ratio, start-end return

    Matching uses linear decay without a tolerance band.
    """

    def __init__(
        self,
        *,
        volatility_tol_rel: float = 0.05,
        fat_tail_tol_rel: float = 0.10,
        momentum_tol_abs: float = 0.05,
        trend_start_end_tol_abs: float = 0.01,
        trend_ma_slope_tol_pct: float = 0.01,
        trend_up_ratio_tol_abs: float = 0.10,
    ) -> None:
        self.volatility_tol_rel = float(volatility_tol_rel)
        self.fat_tail_tol_rel = float(fat_tail_tol_rel)
        self.momentum_tol_abs = float(momentum_tol_abs)
        self.trend_start_end_tol_abs = float(trend_start_end_tol_abs)
        self.trend_ma_slope_tol_pct = float(trend_ma_slope_tol_pct)
        self.trend_up_ratio_tol_abs = float(trend_up_ratio_tol_abs)

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
        vol_match = self._match_relative(
            vol_target, vol_synth, self.volatility_tol_rel
        )

        # =========================
        # Fat tails
        # =========================
        kurt_orig = self._to_scalar(stats.kurtosis(orig_returns))
        kurt_synth = self._to_scalar(stats.kurtosis(synth_returns))
        # Requested mapping:
        # target_kurt = hist_kurt * (1 + 0.15 * (fat_tails - 1))
        kurt_target = kurt_orig * (1.0 + 0.15 * (fat_tails_knob - 1.0))
        kurt_match = self._match_relative(
            kurt_target, kurt_synth, self.fat_tail_tol_rel
        )

        fat_tail_match = kurt_match

        # =========================
        # Momentum (Hurst exponent)
        # =========================
        hurst_orig = self._hurst_exponent(orig_returns)
        hurst_synth = self._hurst_exponent(synth_returns)
        # Keep knob semantics centered at 0.5:
        # knob=0.5 -> target = original
        # knob<0.5 -> move target toward 0.5 (more anti-persistent/random)
        # knob>0.5 -> amplify distance from 0.5 (more persistent)
        momentum_scale = momentum_knob / 0.5
        hurst_target = 0.5 + (hurst_orig - 0.5) * momentum_scale
        hurst_target = float(np.clip(hurst_target, 0.0, 1.0))
        momentum_match = self._match_absolute(
            hurst_target, hurst_synth, self.momentum_tol_abs
        )

        # =========================
        # Trend
        # =========================
        ma_slope_orig, up_ratio_orig = self._ma20_trend_metrics(orig_close)
        ma_slope_synth, up_ratio_synth = self._ma20_trend_metrics(synth_close)

        start_end_orig = self._start_end_return(orig_close)
        start_end_synth = self._start_end_return(synth_close)

        # MA slope tolerance is % of average original price level
        avg_price_orig = self._to_scalar(np.mean(orig_close))
        ma_slope_tol_abs = abs(avg_price_orig) * self.trend_ma_slope_tol_pct
        # Trend mapping:
        # - trend = 0.0 => multiplier = 1.0 (neutral)
        # - trend = +1.0 => multiplier = 2.0 (stronger same direction)
        # - trend = -1.0 => multiplier = -1.0 (full reversal)
        if trend_knob < 0:
            trend_multiplier = 1.0 + 2.0 * trend_knob
        else:
            trend_multiplier = 1.0 + trend_knob
        ma_slope_target = ma_slope_orig * trend_multiplier
        start_end_target = start_end_orig * trend_multiplier

        if trend_knob >= 0:
            up_ratio_target = up_ratio_orig + trend_knob * (1.0 - up_ratio_orig)
        else:
            up_ratio_target = up_ratio_orig + trend_knob * (up_ratio_orig)
        up_ratio_target = float(np.clip(up_ratio_target, 0.0, 1.0))

        ma_slope_match = self._match_absolute(
            ma_slope_target, ma_slope_synth, ma_slope_tol_abs
        )
        up_ratio_match = self._match_absolute(
            up_ratio_target, up_ratio_synth, self.trend_up_ratio_tol_abs
        )
        start_end_match = self._match_absolute(
            start_end_target, start_end_synth, self.trend_start_end_tol_abs
        )

        trend_match = (ma_slope_match + up_ratio_match + start_end_match) / 3.0

        # =========================
        # Overall
        # =========================
        overall_match = (vol_match + fat_tail_match + momentum_match + trend_match) / 4.0

        return {
            "volatility": {
                "original": vol_orig,
                "target": vol_target,
                "synthetic": vol_synth,
                "tolerance_rel": self.volatility_tol_rel,
                "match_pct": vol_match * 100.0,
            },
            "fat_tails": {
                "kurtosis": {
                    "original": kurt_orig,
                    "target": kurt_target,
                    "synthetic": kurt_synth,
                    "tolerance_rel": self.fat_tail_tol_rel,
                    "match_pct": kurt_match * 100.0,
                },
                "match_pct": fat_tail_match * 100.0,
            },
            "momentum": {
                "hurst": {
                    "original": hurst_orig,
                    "target": hurst_target,
                    "synthetic": hurst_synth,
                    "tolerance_abs": self.momentum_tol_abs,
                    "match_pct": momentum_match * 100.0,
                },
                "match_pct": momentum_match * 100.0,
            },
            "trend": {
                "ma20_slope_avg": {
                    "original": ma_slope_orig,
                    "target": ma_slope_target,
                    "synthetic": ma_slope_synth,
                    "tolerance_abs": ma_slope_tol_abs,
                    "match_pct": ma_slope_match * 100.0,
                },
                "ma20_up_ratio": {
                    "original": up_ratio_orig,
                    "target": up_ratio_target,
                    "synthetic": up_ratio_synth,
                    "tolerance_abs": self.trend_up_ratio_tol_abs,
                    "match_pct": up_ratio_match * 100.0,
                },
                "start_end_return": {
                    "original": start_end_orig,
                    "target": start_end_target,
                    "synthetic": start_end_synth,
                    "tolerance_abs": self.trend_start_end_tol_abs,
                    "match_pct": start_end_match * 100.0,
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
        Estimate Hurst exponent from returns using the variance-time scaling method:
        std(x[t+lag] - x[t]) ~ lag^H
        """
        x = np.asarray(returns, dtype=float)
        x = x[np.isfinite(x)]
        n = x.size
        if n < 20:
            return float("nan")

        max_lag = min(100, n // 2)
        if max_lag < 3:
            return float("nan")

        lags = np.arange(2, max_lag + 1, dtype=int)
        used_lags: list[float] = []
        tau: list[float] = []

        for lag in lags:
            diffs = x[lag:] - x[:-lag]
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

    def _match_relative(self, orig: float, synth: float, tol_rel: float) -> float:
        if not np.isfinite(orig) or not np.isfinite(synth):
            return 0.0
        eps = 1e-12
        delta_rel = abs(synth - orig) / (abs(orig) + eps)
        # No tolerance band: decay starts immediately from delta=0.
        return self._decay_match(delta_rel, scale=tol_rel)

    def _match_absolute(self, orig: float, synth: float, tol_abs: float) -> float:
        if not np.isfinite(orig) or not np.isfinite(synth):
            return 0.0
        delta = abs(synth - orig)
        # No tolerance band: decay starts immediately from delta=0.
        return self._decay_match(delta, scale=tol_abs)

    def _decay_match(self, delta: float, scale: float) -> float:
        if scale <= 0:
            return 0.0
        return float(max(0.0, 1.0 - (delta / scale)))

    def _to_scalar(self, value) -> float:
        if isinstance(value, (float, int)):
            return float(value)
        if hasattr(value, "item"):
            return float(value.item())
        if hasattr(value, "__len__") and len(value) == 1:
            return float(value[0])
        return float(value)
