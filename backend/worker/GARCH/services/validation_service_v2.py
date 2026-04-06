# services/validation_service_v2.py
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

logger = logging.getLogger(__name__)


class ValidationServiceV2:
    """
    Validation Service V2

    Compares original OHLCV vs synthetic OHLCV using:
    - Volatility: std dev of percent returns
    - Fat tails: kurtosis + Q95-Q5 tail spread (50/50)
    - Momentum: lag-1 autocorrelation of percent returns
    - Trend: MA(20) slope avg, MA(20) uptrend ratio, start-end return

    Matching uses linear decay with a tolerance band.
    """

    def __init__(
        self,
        *,
        volatility_tol_rel: float = 0.05,
        fat_tail_tol_rel: float = 0.10,
        tail_spread_tol_rel: float = 0.10,
        momentum_tol_abs: float = 0.05,
        trend_start_end_tol_abs: float = 0.01,
        trend_ma_slope_tol_pct: float = 0.01,
        trend_up_ratio_tol_abs: float = 0.10,
    ) -> None:
        self.volatility_tol_rel = float(volatility_tol_rel)
        self.fat_tail_tol_rel = float(fat_tail_tol_rel)
        self.tail_spread_tol_rel = float(tail_spread_tol_rel)
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

        # =========================
        # Volatility
        # =========================
        vol_orig = self._to_scalar(np.std(orig_returns, ddof=1))
        vol_synth = self._to_scalar(np.std(synth_returns, ddof=1))
        vol_target = vol_orig * volatility_knob
        vol_match = self._match_relative(
            vol_target, vol_synth, self.volatility_tol_rel
        )

        # =========================
        # Fat tails
        # =========================
        kurt_orig = self._to_scalar(stats.kurtosis(orig_returns))
        kurt_synth = self._to_scalar(stats.kurtosis(synth_returns))
        kurt_target = kurt_orig * fat_tails_knob
        kurt_match = self._match_relative(
            kurt_target, kurt_synth, self.fat_tail_tol_rel
        )

        q95_orig = self._to_scalar(np.percentile(orig_returns, 95))
        q5_orig = self._to_scalar(np.percentile(orig_returns, 5))
        q95_synth = self._to_scalar(np.percentile(synth_returns, 95))
        q5_synth = self._to_scalar(np.percentile(synth_returns, 5))
        tail_spread_orig = q95_orig - q5_orig
        tail_spread_synth = q95_synth - q5_synth
        tail_spread_target = tail_spread_orig * fat_tails_knob
        tail_spread_match = self._match_relative(
            tail_spread_target, tail_spread_synth, self.tail_spread_tol_rel
        )

        fat_tail_match = 0.5 * kurt_match + 0.5 * tail_spread_match

        # =========================
        # Momentum (ACF lag-1)
        # =========================
        acf_orig = self._acf_lag1(orig_returns)
        acf_synth = self._acf_lag1(synth_returns)
        momentum_scale = momentum_knob / 0.5 if 0.5 != 0 else 1.0
        acf_target = acf_orig * momentum_scale
        momentum_match = self._match_absolute(
            acf_target, acf_synth, self.momentum_tol_abs
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
                "tail_spread_q95_q5": {
                    "original": tail_spread_orig,
                    "target": tail_spread_target,
                    "synthetic": tail_spread_synth,
                    "tolerance_rel": self.tail_spread_tol_rel,
                    "match_pct": tail_spread_match * 100.0,
                },
                "match_pct": fat_tail_match * 100.0,
            },
            "momentum": {
                "acf_lag1": {
                    "original": acf_orig,
                    "target": acf_target,
                    "synthetic": acf_synth,
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

    def _acf_lag1(self, returns: np.ndarray) -> float:
        if returns.size < 2:
            return float("nan")
        return self._to_scalar(acf(returns, nlags=1, fft=False)[1])

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
        return self._exp_match(delta_rel, tol_rel, scale=tol_rel * 2.0)

    def _match_absolute(self, orig: float, synth: float, tol_abs: float) -> float:
        if not np.isfinite(orig) or not np.isfinite(synth):
            return 0.0
        delta = abs(synth - orig)
        return self._exp_match(delta, tol_abs, scale=tol_abs * 2.0)

    def _exp_match(self, delta: float, tol: float, scale: float) -> float:
        if tol <= 0 or scale <= 0:
            return 0.0
        if delta <= tol:
            return 1.0
        return float(max(0.0, 1.0 - (delta - tol) / scale))

    def _to_scalar(self, value) -> float:
        if isinstance(value, (float, int)):
            return float(value)
        if hasattr(value, "item"):
            return float(value.item())
        if hasattr(value, "__len__") and len(value) == 1:
            return float(value[0])
        return float(value)
