"""
Validation service for DDPM generated paths.
Extends GARCH ValidationService with DDPM-specific metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DDPMValidationService:
    """
    Comprehensive validation for DDPM-generated scenarios.
    Combines standard statistical metrics (from GARCH ValidationService pattern)
    with DDPM-specific regime and volatility metrics.
    """

    def __init__(self, device="cpu"):
        """
        Args:
            device: torch device (for compatibility, not actively used here)
        """
        self.device = device

    def validate_scenarios(
        self,
        historical_data: np.ndarray,
        synthetic_data: np.ndarray,
        stage: int = 1,
        regime_segment_len: int = 63,
        vol_threshold_low: float = 0.25,
        vol_threshold_high: float = 0.50,
    ) -> Dict:
        """
        Validate synthetic paths against historical data.

        Args:
            historical_data: Array (num_windows, num_assets, seq_len) of historical returns
            synthetic_data: Array (num_paths, num_assets, seq_len) of generated returns
            stage: 1 or 2 (affects which metrics are included)
            regime_segment_len: Length of segments for regime analysis (default 63 = quarterly)
            vol_threshold_low: Volatility threshold for calm regime
            vol_threshold_high: Volatility threshold for high-vol regime

        Returns:
            Dictionary with all validation metrics
        """
        metrics = {}

        # ===== Standard Statistical Metrics (from GARCH pattern) =====
        historical_flat = historical_data.flatten()
        synthetic_flat = synthetic_data.flatten()

        # KS Test
        ks_stat, ks_pval = stats.ks_2samp(historical_flat, synthetic_flat)
        metrics["ks_statistic"] = float(ks_stat)
        metrics["ks_pvalue"] = float(ks_pval)

        # Kurtosis
        metrics["kurtosis_historical"] = float(stats.kurtosis(historical_flat))
        metrics["kurtosis_synthetic"] = float(stats.kurtosis(synthetic_flat))

        # Skewness
        metrics["skewness_historical"] = float(stats.skew(historical_flat))
        metrics["skewness_synthetic"] = float(stats.skew(synthetic_flat))

        # ACF lag-1
        metrics["acf_lag1_historical"] = float(self._compute_acf_lag1(historical_flat))
        metrics["acf_lag1_synthetic"] = float(self._compute_acf_lag1(synthetic_flat))

        # ACF of squared returns (volatility clustering)
        metrics["acf_vol_historical"] = float(
            self._compute_acf_lag1(historical_flat ** 2)
        )
        metrics["acf_vol_synthetic"] = float(self._compute_acf_lag1(synthetic_flat ** 2))

        # ===== DDPM-Specific Metrics =====

        # Terminal return statistics
        term_hist = self._compute_terminal_returns(historical_data)
        term_synth = self._compute_terminal_returns(synthetic_data)
        metrics["terminal_return_ratio"] = float(
            term_synth.mean() / max(term_hist.mean(), 1e-6)
        )

        # Rolling volatility metrics
        rolling_vol_hist = self._compute_rolling_volatility(historical_data)
        rolling_vol_synth = self._compute_rolling_volatility(synthetic_data)
        metrics["rolling_volatility_ratio"] = float(
            rolling_vol_synth.mean() / max(rolling_vol_hist.mean(), 1e-6)
        )

        # Regime hit rates (only for Stage 2 or when analyzing regime dynamics)
        if stage == 2:
            regime_metrics = self._compute_regime_hit_rates(
                synthetic_data, regime_segment_len, vol_threshold_low, vol_threshold_high
            )
            metrics.update(regime_metrics)

        metrics["validation_stage"] = stage

        logger.info(
            f"Validation complete (stage={stage}): KS={ks_stat:.4f}, "
            f"Kurtosis_synth={metrics['kurtosis_synthetic']:.4f}"
        )

        return metrics

    def _compute_acf_lag1(self, x: np.ndarray) -> float:
        """
        Compute ACF at lag 1.

        Args:
            x: Time series array

        Returns:
            ACF lag-1 value
        """
        x = x - x.mean()
        c0 = np.dot(x, x) / len(x)
        c1 = np.dot(x[:-1], x[1:]) / len(x)
        return c1 / max(c0, 1e-10)

    def _compute_terminal_returns(self, data: np.ndarray) -> np.ndarray:
        """
        Compute cumulative return at end of each path.

        Args:
            data: Array (num_paths, num_assets, seq_len) of log returns

        Returns:
            Terminal returns array (num_paths,)
        """
        # Sum log returns across time and then exponentiate
        log_cumsum = data.sum(axis=2)  # (num_paths, num_assets)
        terminal = np.exp(log_cumsum.mean(axis=1))  # (num_paths,)
        return terminal

    def _compute_rolling_volatility(self, data: np.ndarray, window: int = 63) -> np.ndarray:
        """
        Compute rolling volatility (std of log returns over quarterly windows).

        Args:
            data: Array (num_paths, num_assets, seq_len) of log returns
            window: Rolling window length (default 63 = quarterly)

        Returns:
            Mean rolling volatility per path (num_paths,)
        """
        num_paths = data.shape[0]
        rolling_vols = []

        for i in range(num_paths):
            path = data[i]  # (num_assets, seq_len)
            path_flat = path.flatten()
            rolling = np.array(
                [path_flat[j : j + window].std() for j in range(0, len(path_flat) - window, window)]
            )
            rolling_vols.append(rolling.mean())

        return np.array(rolling_vols)

    def _compute_regime_hit_rates(
        self,
        data: np.ndarray,
        segment_len: int = 63,
        vol_threshold_low: float = 0.25,
        vol_threshold_high: float = 0.50,
    ) -> Dict:
        """
        Compute regime hit rates for conditional paths.

        Regimes:
          - calm: rolling_vol < vol_threshold_low
          - highvol: vol_threshold_low <= rolling_vol < vol_threshold_high
          - crisis: rolling_vol >= vol_threshold_high

        Args:
            data: Array (num_paths, num_assets, seq_len) of log returns
            segment_len: Length of segments for rolling vol computation
            vol_threshold_low: Calm threshold
            vol_threshold_high: High-vol threshold

        Returns:
            Dictionary with regime hit rates
        """
        num_paths = data.shape[0]
        calm_count = 0
        highvol_count = 0
        crisis_count = 0
        total_segments = 0

        for i in range(num_paths):
            path = data[i]  # (num_assets, seq_len)
            path_flat = path.flatten()

            for j in range(0, len(path_flat) - segment_len, segment_len):
                segment = path_flat[j : j + segment_len]
                vol = segment.std()

                if vol < vol_threshold_low:
                    calm_count += 1
                elif vol < vol_threshold_high:
                    highvol_count += 1
                else:
                    crisis_count += 1

                total_segments += 1

        # Hit rates
        calm_rate = calm_count / max(total_segments, 1)
        highvol_rate = highvol_count / max(total_segments, 1)
        crisis_rate = crisis_count / max(total_segments, 1)

        return {
            "regime_calm_hit_rate": float(calm_rate),
            "regime_highvol_hit_rate": float(highvol_rate),
            "regime_crisis_hit_rate": float(crisis_rate),
            "target_calm_rate": 0.40,  # Expected distribution from training data
            "target_highvol_rate": 0.45,
            "target_crisis_rate": 0.15,
        }

    def assess_quality(self, metrics: Dict) -> str:
        """
        Simple quality assessment based on validation metrics.

        Returns:
            "PASS" if metrics are acceptable, "FAIL" otherwise
        """
        # KS test: p-value > 0.05 is good
        ks_pass = metrics.get("ks_pvalue", 0.0) > 0.05

        # Kurtosis difference: should be < 0.5 in absolute terms
        kurt_diff = abs(
            metrics.get("kurtosis_synthetic", 0.0)
            - metrics.get("kurtosis_historical", 0.0)
        )
        kurt_pass = kurt_diff < 0.5

        # ACF should match within 0.1
        acf_diff = abs(
            metrics.get("acf_lag1_synthetic", 0.0)
            - metrics.get("acf_lag1_historical", 0.0)
        )
        acf_pass = acf_diff < 0.1

        overall = ks_pass and kurt_pass and acf_pass
        return "PASS" if overall else "FAIL"
