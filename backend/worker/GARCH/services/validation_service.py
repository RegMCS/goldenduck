# services/validation_service.py
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf
import logging

logger = logging.getLogger(__name__)


class ValidationService:
    def __init__(self, historical_data: pd.DataFrame):
        # Extract returns and force to 1D numpy array
        returns = historical_data["Close"].pct_change().dropna()
        self.historical_returns = np.asarray(returns).flatten()

        # Compute statistics with safe scalar extraction (ddof=1 for sample std,
        # consistent with _series_stats in garch_worker.py)
        self.historical_volatility = self._to_scalar(
            np.std(self.historical_returns, ddof=1)
        )
        self.historical_kurtosis = self._to_scalar(
            stats.kurtosis(self.historical_returns)
        )
        self.historical_skewness = self._to_scalar(stats.skew(self.historical_returns))

    def _to_scalar(self, value):
        """
        Safely convert numpy array/scalar to Python float
        """
        # If it's already a Python float/int
        if isinstance(value, (float, int)):
            return float(value)

        # If it's a numpy scalar or 0-d array
        if hasattr(value, "item"):
            return float(value.item())

        # If it's a 1-element array
        if hasattr(value, "__len__") and len(value) == 1:
            return float(value[0])

        # Fallback: try to convert directly
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            logger.error(f"Could not convert {type(value)} to scalar: {value}")
            raise

    def _compute_mean_sign_match(
        self,
        synth_mean: float,
        desired_trend: float,
        synth_std: float,
        sample_size: int,
    ) -> float:
        """
        Trend match focused on DIRECTION (bull/bear), not exact mean magnitude.

        - desired_trend > 0: reward positive synthetic mean
        - desired_trend < 0: reward negative synthetic mean
        - desired_trend near 0: reward near-zero synthetic mean
        """
        neutral_trend_eps = 0.1
        # Neutral tolerance (daily mean units):
        # use max of a fixed floor and a noise-aware confidence band.
        fixed_floor_band = 0.05 / 252  # ±5% annualised
        if sample_size > 1 and synth_std > 0:
            se_mean = synth_std / np.sqrt(sample_size)
            noise_band = 2.0 * se_mean  # ~95% band
        else:
            noise_band = fixed_floor_band
        neutral_mean_band = max(fixed_floor_band, noise_band)

        if abs(desired_trend) < neutral_trend_eps:
            # Smooth score: 1 at 0, decays with distance from 0
            return float(np.exp(-abs(synth_mean) / (neutral_mean_band + 1e-12)))

        desired_sign = 1.0 if desired_trend > 0 else -1.0
        return 1.0 if (synth_mean * desired_sign) > 0 else 0.0

    def compute_positive_terminal_ratio_metrics(
        self, scenarios: list[pd.DataFrame], user_knobs: dict
    ) -> dict:
        """
        Evaluate trend direction at the PATH level.

        Metric: percentage of simulated price paths with positive terminal return.
        This directly answers "how often do paths end up higher than they start?".
        """
        terminal_returns = []
        for scenario in scenarios:
            if "Close" not in scenario.columns or len(scenario) < 2:
                continue
            first_close = float(scenario["Close"].iloc[0])
            last_close = float(scenario["Close"].iloc[-1])
            if first_close == 0:
                continue
            terminal_returns.append((last_close / first_close) - 1.0)

        if len(terminal_returns) == 0:
            positive_terminal_ratio = 0.5
            negative_terminal_ratio = 0.5
        else:
            terminal_returns_arr = np.asarray(terminal_returns, dtype=float)
            positive_terminal_ratio = float(np.mean(terminal_returns_arr > 0.0))
            negative_terminal_ratio = float(np.mean(terminal_returns_arr < 0.0))

        # Map trend knob to desired probability of ending positive:
        # -1 -> 0.0, 0 -> 0.5, +1 -> 1.0
        desired_trend = float(user_knobs.get("desired_trend", 0.0))
        desired_positive_terminal_ratio = float(
            np.clip(0.5 + 0.5 * desired_trend, 0.0, 1.0)
        )

        trend_match = max(
            0.0,
            1.0 - abs(positive_terminal_ratio - desired_positive_terminal_ratio),
        )

        return {
            "positive_terminal_ratio": positive_terminal_ratio,
            "negative_terminal_ratio": negative_terminal_ratio,
            "desired_positive_terminal_ratio": desired_positive_terminal_ratio,
            "trend_match": trend_match,
            "num_paths_evaluated": int(len(terminal_returns)),
        }

    def validate(self, synthetic_returns: np.ndarray) -> dict:
        """
        Validate synthetic data against historical data
        """
        # Ensure both are numpy arrays
        hist_returns = np.array(self.historical_returns).flatten()
        synth_returns = np.array(synthetic_returns).flatten()

        # Remove any NaN/inf values
        hist_returns = hist_returns[np.isfinite(hist_returns)]
        synth_returns = synth_returns[np.isfinite(synth_returns)]

        logger.info(
            f"Validating: hist={len(hist_returns)} samples, synth={len(synth_returns)} samples"
        )

        # KS test
        ks_stat, ks_pval = stats.ks_2samp(hist_returns, synth_returns)

        # Kurtosis (fat tails)
        hist_kurt = stats.kurtosis(hist_returns)
        synth_kurt = stats.kurtosis(synth_returns)

        # Skewness (asymmetry)
        hist_skew = stats.skew(hist_returns)
        synth_skew = stats.skew(synth_returns)

        # Autocorrelation
        hist_acf_returns = acf(hist_returns, nlags=10, fft=False)[1]
        synth_acf_returns = acf(synth_returns, nlags=10, fft=False)[1]

        hist_acf_vol = acf(hist_returns**2, nlags=10, fft=False)[1]
        synth_acf_vol = acf(synth_returns**2, nlags=10, fft=False)[1]

        return {
            "ks_statistic": self._to_scalar(ks_stat),
            "ks_pvalue": self._to_scalar(ks_pval),
            "kurtosis_historical": self._to_scalar(hist_kurt),
            "kurtosis_synthetic": self._to_scalar(synth_kurt),
            "skewness_historical": self._to_scalar(hist_skew),
            "skewness_synthetic": self._to_scalar(synth_skew),
            "acf_returns_historical": self._to_scalar(hist_acf_returns),
            "acf_returns_synthetic": self._to_scalar(synth_acf_returns),
            "acf_volatility_historical": self._to_scalar(hist_acf_vol),
            "acf_volatility_synthetic": self._to_scalar(synth_acf_vol),
        }

    def validate_against_desired(
        self, synthetic_returns: np.ndarray, user_knobs: dict
    ) -> dict:
        """
        Validate synthetic data against desired characteristics from user knobs.
        """
        # Clean synthetic returns
        synth_returns = np.array(synthetic_returns).flatten()
        synth_returns = synth_returns[np.isfinite(synth_returns)]

        # Calculate synthetic statistics
        synth_mean = self._to_scalar(np.mean(synth_returns))
        synth_volatility = self._to_scalar(np.std(synth_returns, ddof=1))
        synth_kurtosis = self._to_scalar(stats.kurtosis(synth_returns))
        synth_skewness = self._to_scalar(stats.skew(synth_returns))
        synth_acf = self._to_scalar(acf(synth_returns, nlags=10, fft=False)[1])

        # Extract user knobs
        desired_volatility = float(user_knobs.get("desired_volatility", 1.0))
        desired_fat_tails = float(user_knobs.get("desired_fat_tails", 1.0))
        desired_trend = float(user_knobs.get("desired_trend", 0.0))
        desired_momentum = float(user_knobs.get("desired_momentum", 0.5))

        # ============================================================
        # Compute desired characteristics
        # ============================================================

        # Volatility
        desired_volatility_value = self.historical_volatility * desired_volatility

        # Mean return (trend) — absolute bull/bear target
        # Piecewise annual drift mapping (symmetric for bearish):
        # +0.25 -> +10%/yr, +0.50 -> +25%/yr, +1.00 -> +50%/yr.
        historical_mean = self._to_scalar(np.mean(self.historical_returns))
        trend_mag = float(np.clip(abs(desired_trend), 0.0, 1.0))
        if trend_mag <= 0.25:
            annual_drift_mag = 0.40 * trend_mag
        elif trend_mag <= 0.50:
            annual_drift_mag = 0.10 + 0.60 * (trend_mag - 0.25)
        else:
            annual_drift_mag = 0.25 + 0.50 * (trend_mag - 0.50)

        annual_drift = np.sign(desired_trend) * annual_drift_mag
        desired_mean_value = annual_drift / 252

        # Kurtosis (with momentum adjustment)
        base_kurtosis = self.historical_kurtosis * desired_fat_tails
        momentum_boost = max(0.0, (desired_momentum - 0.7) * 2.0)
        desired_kurtosis_value = base_kurtosis + momentum_boost

        # Kurtosis (FIXED - use model-aware formula)
        # desired_kurtosis_value = self.compute_target_kurtosis(user_knobs)

        # Skewness (AR(1) + trend contributions)
        phi = -0.1 + 0.4 * desired_momentum

        if abs(phi) > 0.05:
            k = 0.5 + (desired_volatility - 1.0) * 0.2
            ar_skewness = k * phi
        else:
            ar_skewness = 0.0

        if abs(desired_trend) > 0.7:
            trend_skewness = desired_trend * 0.2
        else:
            trend_skewness = 0.0

        desired_skewness_value = ar_skewness + trend_skewness

        # ACF (with GARCH dampening)
        dampening = 0.9
        desired_acf_value = phi * dampening

        # ============================================================
        # Calculate matches
        # ============================================================

        volatility_match = 1.0 - abs(synth_volatility - desired_volatility_value) / (
            desired_volatility_value + 1e-6
        )
        kurtosis_match = 1.0 - min(
            abs(synth_kurtosis - desired_kurtosis_value)
            / (abs(desired_kurtosis_value) + 0.5),
            1.0,
        )
        mean_match = self._compute_mean_sign_match(
            synth_mean=synth_mean,
            desired_trend=desired_trend,
            synth_std=synth_volatility,
            sample_size=len(synth_returns),
        )
        acf_match = 1.0 - abs(synth_acf - desired_acf_value) / (
            abs(desired_acf_value) + 0.1
        )

        # ============================================================
        # Logging
        # ============================================================

        logger.info(
            f"Validation vs desired:"
            f"\n  Mean (directional): synth={synth_mean:.6f} vs desired={desired_mean_value:.6f} (match={mean_match:.2%})"
            f"\n  Volatility: synth={synth_volatility:.4f} vs desired={desired_volatility_value:.4f} (match={volatility_match:.2%})"
            f"\n  Kurtosis: synth={synth_kurtosis:.2f} vs desired={desired_kurtosis_value:.2f} (match={kurtosis_match:.2%})"
            f"\n  Skewness (diagnostic): synth={synth_skewness:.4f} vs desired={desired_skewness_value:.4f}"
            f"\n  ACF: synth={synth_acf:.4f} vs desired={desired_acf_value:.4f} (match={acf_match:.2%})"
        )

        return {
            "mean_historical": historical_mean,
            "mean_synthetic": synth_mean,
            "mean_desired": desired_mean_value,
            "mean_match": mean_match,
            "volatility_historical": self._to_scalar(self.historical_volatility),
            "volatility_synthetic": synth_volatility,
            "volatility_desired": desired_volatility_value,
            "volatility_match": volatility_match,
            "kurtosis_historical": self._to_scalar(self.historical_kurtosis),
            "kurtosis_synthetic": synth_kurtosis,
            "kurtosis_desired": desired_kurtosis_value,
            "kurtosis_match": kurtosis_match,
            "skewness_historical": self._to_scalar(self.historical_skewness),
            "skewness_synthetic": synth_skewness,
            "skewness_desired": desired_skewness_value,
            "acf_synthetic": synth_acf,
            "acf_desired": desired_acf_value,
            "acf_match": acf_match,
            "overall_match": (
                mean_match + volatility_match + kurtosis_match + acf_match
            )
            / 4.0,
        }

    def compute_target_kurtosis(self, user_knobs):
        """
        Compute target excess kurtosis based on the fat_tails knob.

        WHY NOT anchored to historical kurtosis:
        Historical sample kurtosis (e.g. AAPL ~12.5) is inflated by a handful
        of extreme events (COVID crash, flash crashes) over many years. It is a
        sample artefact, not a stable distributional property. Our model with
        t(6) shocks + GARCH clustering has a true achievable range of ~2-10,
        so anchoring the target to 12.5 would always produce a poor match at
        fat_tails=1.0, which is misleading.

        Knob semantics:
          fat_tails = 0.5  → ~2.0  (thin tails, near-normal)
          fat_tails = 1.0  → ~4.5  (model's natural t(6)+GARCH baseline)
          fat_tails = 1.5  → ~7.0  (noticeably fat tails)
          fat_tails = 2.0  → ~10.0 (very fat tails, near model ceiling)
        """
        desired_fat_tails = float(user_knobs.get("desired_fat_tails", 1.0))
        desired_momentum = float(user_knobs.get("desired_momentum", 0.5))

        # Model's natural output with t(6) shocks + GARCH (empirically ~4-5)
        baseline_kurtosis = 4.5
        # Achievable ceiling (empirically validated)
        max_kurtosis = 10.0

        # Linear interpolation across the full knob range [0.5, 2.0]
        # anchored: fat_tails=1.0 → baseline, fat_tails=2.0 → max
        if desired_fat_tails <= 1.0:
            # Scale from near-normal (2.0) up to baseline (4.5)
            kurtosis_from_fat_tails = 2.0 + (baseline_kurtosis - 2.0) * (
                desired_fat_tails / 1.0
            )
        else:
            # Scale from baseline (4.5) up to max (10.0)
            excess = desired_fat_tails - 1.0  # [0, 1.0]
            kurtosis_from_fat_tails = baseline_kurtosis + (
                max_kurtosis - baseline_kurtosis
            ) * min(excess, 1.0)

        # AR(1) persistence adds to unconditional tail heaviness
        momentum_boost = 0.0
        if desired_momentum > 0.7:
            momentum_boost = (desired_momentum - 0.7) / 0.3 * 2.0

        target_kurtosis = kurtosis_from_fat_tails + momentum_boost
        return float(min(target_kurtosis, max_kurtosis))
