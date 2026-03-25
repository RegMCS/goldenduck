"""GC-GARCH service with knob mappings and generator."""
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import skew as skew_fn, kurtosis as kurtosis_fn

from .gc_range import sample_gc_z, sample_gc_high_low_range

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GCGarchKnobs:
    """User-facing knobs for GC-GARCH generation.

    volatility: 0.5..2.0 (1.0 = no change)
    trend: -1.0..1.0 (0 = unchanged drift, 1 = amplify, -1 = reverse)
    fat_tails: 0.5..2.0 (1.0 = baseline kurtosis)
    momentum: 0.0..1.0 (0.5 = baseline persistence)
    """

    volatility: float = 1.0
    trend: float = 0.0
    fat_tails: float = 1.0
    momentum: float = 0.5


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_knobs(knobs: GCGarchKnobs) -> GCGarchKnobs:
    """Clamp knobs to agreed ranges."""
    return GCGarchKnobs(
        volatility=clamp(knobs.volatility, 0.5, 2.0),
        trend=clamp(knobs.trend, -1.0, 1.0),
        fat_tails=clamp(knobs.fat_tails, 0.5, 2.0),
        momentum=clamp(knobs.momentum, 0.0, 1.0),
    )


def apply_trend(mu_base: float, trend: float) -> float:
    """Map trend knob to drift adjustment.

    trend=0 -> unchanged; trend=1 -> 2x; trend=-1 -> -1x.
    """
    if trend >= 0:
        return mu_base * (1 + trend)
    return mu_base * (1 + 2 * trend)


def apply_momentum_to_persistence(
    alpha: float, beta: float, momentum: float
) -> Tuple[float, float]:
    """Adjust GARCH persistence (alpha+beta) using momentum knob.

    momentum=0.5 keeps persistence unchanged.
    momentum>0.5 increases persistence; momentum<0.5 decreases it.
    """
    p = alpha + beta
    if p <= 0:
        return alpha, beta

    scale = 1 + 2 * (momentum - 0.5)
    p_new = clamp(p * scale, 0.5, 0.995)
    ratio = p_new / p
    return alpha * ratio, beta * ratio


def adjust_omega_for_persistence(
    omega: float,
    alpha_old: float,
    beta_old: float,
    alpha_new: float,
    beta_new: float,
) -> float:
    """Adjust omega to keep long-run variance stable after persistence change."""
    denom_old = max(1e-8, 1.0 - alpha_old - beta_old)
    denom_new = max(1e-8, 1.0 - alpha_new - beta_new)
    return omega * (denom_new / denom_old)


class GCGarchGenerator:
    """Fit GARCH parameters and generate GC-GARCH synthetic OHLCV."""

    def __init__(self, scale_factor: float = 100.0) -> None:
        self.scale_factor = float(scale_factor)
        self.fitted_model = None
        self.params: Optional[dict] = None
        self.last_sigma: float = 0.0
        self.last_return: float = 0.0
        self.initial_price: float = 1.0
        self.base_mu: float = 0.0
        self.base_skew: float = 0.0
        self.base_kurt: float = 0.0
        self.base_volume: float = 1_000_000.0

    def fit(self, historical_data: pd.DataFrame, p: int = 1, q: int = 1) -> dict:
        df = historical_data.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]

        if "close" not in df.columns:
            raise ValueError("Input data must contain a 'Close' column")

        close = df["close"].astype(float).to_numpy()
        if close.size < 2:
            raise ValueError("Need at least 2 close prices to fit GC-GARCH")

        # Start synthetic series from the first historical close
        self.initial_price = float(close[0])

        returns = np.log(close[1:] / close[:-1])
        self.base_mu = float(np.mean(returns)) if returns.size else 0.0
        self.last_return = float(returns[-1]) if returns.size else 0.0

        if "volume" in df.columns:
            vol_values = df["volume"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vol_values) > 0:
                self.base_volume = float(np.median(vol_values))

        scaled_returns = returns * self.scale_factor
        model = arch_model(
            scaled_returns,
            vol="GARCH",
            p=p,
            q=q,
            mean="Zero",
            dist="normal",
            rescale=False,
        )
        fitted = model.fit(disp="off", options={"maxiter": 5000})
        self.fitted_model = fitted

        self.params = {
            "omega": float(fitted.params.get("omega", np.nan)),
            "alpha": float(fitted.params.get("alpha[1]", np.nan)),
            "beta": float(fitted.params.get("beta[1]", np.nan)),
        }

        conditional_vol = np.asarray(fitted.conditional_volatility)
        if conditional_vol.size > 0:
            self.last_sigma = float(conditional_vol[-1]) / self.scale_factor
        else:
            self.last_sigma = float(np.std(returns)) if returns.size else 0.0

        std_resid = np.asarray(fitted.std_resid)
        std_resid = std_resid[np.isfinite(std_resid)]
        if std_resid.size >= 10:
            self.base_skew = float(skew_fn(std_resid, bias=False))
            self.base_kurt = float(kurtosis_fn(std_resid, fisher=True, bias=False))
        else:
            self.base_skew = 0.0
            self.base_kurt = 0.0

        if not np.isfinite(self.base_skew):
            self.base_skew = 0.0
        if not np.isfinite(self.base_kurt):
            self.base_kurt = 0.0

        # Clamp to keep GC approximation stable
        self.base_skew = float(np.clip(self.base_skew, -1.5, 1.5))
        self.base_kurt = max(0.0, float(self.base_kurt))

        logger.info(
            "GC-GARCH fit complete: mu=%.6f, skew=%.3f, kurt=%.3f",
            self.base_mu,
            self.base_skew,
            self.base_kurt,
        )

        return self.params

    def generate(
        self,
        horizon: int,
        knobs: GCGarchKnobs,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        if self.params is None:
            raise ValueError("Must call fit() before generate().")
        if horizon <= 0:
            raise ValueError("Horizon must be > 0")

        knobs = normalize_knobs(knobs)

        alpha = float(self.params["alpha"])
        beta = float(self.params["beta"])
        omega = float(self.params["omega"]) / (self.scale_factor**2)

        alpha_new, beta_new = apply_momentum_to_persistence(alpha, beta, knobs.momentum)
        omega_new = adjust_omega_for_persistence(
            omega, alpha, beta, alpha_new, beta_new
        )

        # Apply volatility knob as a variance scale
        vol_scale = knobs.volatility
        omega_new *= vol_scale**2
        sigma2_prev = max(1e-12, (self.last_sigma**2) * (vol_scale**2))
        r_prev = float(self.last_return)

        # Trend knob: scale drift by volatility so it is visually meaningful.
        # Use last_sigma as a stable proxy before simulating the path.
        avg_sigma = max(1e-8, float(self.last_sigma) * vol_scale)
        trend_scale = 0.1
        mu = self.base_mu + knobs.trend * trend_scale * avg_sigma

        # AR(1) momentum on returns: r_t = mu + phi * r_{t-1} + sigma * z_t
        # Map momentum in [0,1] to phi in [-phi_max, +phi_max]
        phi_max = 0.3
        phi = (knobs.momentum - 0.5) * 2.0 * phi_max
        skew = float(np.clip(self.base_skew, -1.5, 1.5))
        # Clip kurtosis to keep GC approximation stable
        kurt = float(np.clip(self.base_kurt * knobs.fat_tails, 0.0, 10.0))

        rng = np.random.default_rng(seed)

        returns = np.zeros(horizon, dtype=float)
        sigmas = np.zeros(horizon, dtype=float)

        for t in range(horizon):
            sigma2 = omega_new + alpha_new * (r_prev**2) + beta_new * sigma2_prev
            if sigma2 <= 0:
                sigma2 = 1e-12
            sigma = float(np.sqrt(sigma2))

            z = float(sample_gc_z(rng, 1, skew, kurt)[0])
            r_t = mu + phi * r_prev + sigma * z
            # Cap returns to avoid explosive paths when kurtosis is large
            r_t = float(np.clip(r_t, -0.08, 0.08))

            returns[t] = r_t
            sigmas[t] = sigma

            r_prev = r_t
            sigma2_prev = sigma2

        close_prices = self.initial_price * np.exp(np.cumsum(returns))
        return self._generate_ohlcv_from_close(
            close_prices=close_prices,
            volatility=sigmas,
            skew=skew,
            kurt=kurt,
            fat_tails=knobs.fat_tails,
            rng=rng,
        )

    def _generate_ohlcv_from_close(
        self,
        close_prices: np.ndarray,
        volatility: np.ndarray,
        skew: float,
        kurt: float,
        fat_tails: float,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        n = len(close_prices)

        if len(volatility) != n:
            if len(volatility) < n:
                volatility = np.pad(volatility, (0, n - len(volatility)), mode="edge")
            else:
                volatility = volatility[:n]

        ohlcv_rows = []
        tail_scale = 0.5 + 0.5 * float(fat_tails)

        for i in range(n):
            c_price = float(close_prices[i])
            sigma = float(volatility[i])

            if i > 0:
                gap = rng.normal(0.0, sigma * 0.3)
                o_price = float(close_prices[i - 1] * (1 + gap))
            else:
                o_price = c_price

            high_mult, low_mult = sample_gc_high_low_range(
                sigma=sigma,
                skew=skew,
                excess_kurtosis=kurt,
                tail_scale=tail_scale,
                range_scale=1.5,
                rng=rng,
            )

            h_price = max(o_price, c_price) * (1 + high_mult)
            l_price = min(o_price, c_price) * (1 - low_mult)
            l_price = max(l_price, 1e-8)

            v_price = int(self.base_volume * (1 + sigma * rng.exponential(2)))

            ohlcv_rows.append(
                {
                    "Open": o_price,
                    "High": h_price,
                    "Low": l_price,
                    "Close": c_price,
                    "Volume": v_price,
                }
            )

        return pd.DataFrame(ohlcv_rows)
