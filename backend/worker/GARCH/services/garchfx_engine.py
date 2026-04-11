# services/garchfx_engine.py
import numpy as np
from typing import Optional, List, Dict
import logging
from arch.univariate.distribution import SkewStudent

logger = logging.getLogger(__name__)


class GARCHFXEngine:
    """
    GARCH-FX forecasting engine implementing the stochastic volatility framework.
    Based on: https://github.com/nitintonypaul/GARCH-FX
    Paper: GARCH-FX: A Modular Framework for Stochastic and Regime-Aware GARCH Forecasting
    """

    def __init__(self, volatility: float, params: dict, scale_factor: float = 100):
        """
        Initialize GARCH-FX engine.

        Parameters:
        -----------
        volatility : float
            Last conditional volatility from fitted GARCH model (unscaled)
        params : dict
            GARCH parameters: {'alpha': α, 'beta': β, 'omega': ω}
        scale_factor : float
            Scaling factor used in original GARCH fitting (default: 100)
        """
        self.initial_volatility = volatility * scale_factor  # Scale to match GARCH
        self.alpha = params["alpha"]
        self.beta = params["beta"]
        self.omega = params["omega"]
        self.params = params
        self.persistence = self.alpha + self.beta
        self.scale_factor = scale_factor

        # Store original parameters for potential modulation
        self.base_alpha = self.alpha
        self.base_beta = self.beta

        # Lazy-initialized true skew-t sampler (arch package)
        self._skewt_sampler = None

        logger.info(
            f"GARCH-FX initialized: σ₀={self.initial_volatility:.4f}, "
            f"α={self.alpha:.4f}, β={self.beta:.4f}, ω={self.omega:.6f}"
        )

    def forecast(
        self,
        horizon: int,
        theta: float = 0.005,
        theta_sequence: Optional[np.ndarray] = None,
        delta_sequence: Optional[np.ndarray] = None,
        regime_switching: bool = False,
        regime_states: Optional[np.ndarray] = None,
        regimes: Optional[List[float]] = None,
        user_knobs: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Generate GARCH-FX volatility forecast.

        Parameters:
        -----------
        horizon : int
            Forecast horizon
        theta : float
            Base stochasticity parameter (scale of Gamma distribution)
        theta_sequence : np.ndarray, optional
                Phase-aware absolute theta values for stress scenarios
        delta_sequence : np.ndarray, optional
            Predetermined delta multipliers (for stress scenarios)
        regime_switching : bool
            Enable stochastic Markov regime switching
        regime_states : np.ndarray, optional
            Transition probability matrix
        regimes : List[float], optional
            Delta multipliers for each regime
        user_knobs : dict, optional
            User knobs including desired_momentum to modulate persistence

        Returns:
        --------
        np.ndarray : Volatility forecast (unscaled, daily units)
        """
        delta = 1.0
        forecasts = [self.initial_volatility]
        previous_variance = self.initial_volatility**2

        for i in range(horizon - 1):
            # Step 1: Determine theta for this step (absolute value or from base)
            if theta_sequence is not None:
                theta_step = float(
                    theta_sequence[i] if i < len(theta_sequence) else theta_sequence[-1]
                )
            else:
                theta_step = float(theta)
            theta_step = max(theta_step, 1e-8)

            # Step 2: Calculate Gamma shape parameter
            shape = (previous_variance / theta_step) + 1

            # Step 3: Determine delta for this step
            if delta_sequence is not None:
                delta = (
                    delta_sequence[i] if i < len(delta_sequence) else delta_sequence[-1]
                )
            elif regime_switching:
                delta = self._regime_switcher(delta, regime_states, regimes)

            # Step 4: Sample stochastic variance from Gamma distribution
            stochastic_variance = np.random.gamma(
                shape=shape, scale=theta_step, size=1
            )[0]

            # Step 5: GARCH-FX equation
            forecasted_variance = (self.omega * delta) + (
                self.persistence * stochastic_variance
            )

            # Step 6: Update for next iteration
            previous_variance = forecasted_variance
            forecasts.append(np.sqrt(previous_variance))

        # Unscale volatility to daily units
        forecasts_array = np.array(forecasts) / self.scale_factor

        logger.debug(
            f"Generated forecast: mean_vol={np.mean(forecasts_array):.4f}, "
            f"max_vol={np.max(forecasts_array):.4f}"
        )

        return forecasts_array

    def _regime_switcher(
        self,
        current_delta: float,
        regime_states: Optional[np.ndarray],
        regimes: Optional[List[float]],
    ) -> float:
        """
        Stochastic regime switching using Markov chain.

        Parameters:
        -----------
        current_delta : float
            Current delta multiplier
        regime_states : np.ndarray
            Transition probability matrix
        regimes : List[float]
            Delta values for each state

        Returns:
        --------
        float : Next delta value
        """
        # Default 5-state regime system
        if regime_states is None:
            regime_states = np.array(
                [
                    [0.85, 0.12, 0.03, 0.00, 0.00],
                    [0.05, 0.75, 0.18, 0.02, 0.00],
                    [0.01, 0.10, 0.70, 0.17, 0.02],
                    [0.00, 0.02, 0.15, 0.70, 0.13],
                    [0.00, 0.00, 0.03, 0.10, 0.87],
                ]
            )

        if regimes is None:
            regimes = [0.5, 0.9, 1.0, 1.1, 1.5]

        # Find current regime index
        try:
            current_regime = regimes.index(current_delta)
        except ValueError:
            # If current_delta not in regimes, default to middle state
            current_regime = len(regimes) // 2
            logger.warning(
                f"Delta {current_delta} not in regimes list, defaulting to state {current_regime}"
            )

        # Get transition probabilities
        probs = regime_states[current_regime]

        # Draw random number and select next regime
        r = np.random.rand()
        new_regime = regimes[np.searchsorted(np.cumsum(probs), r)]

        return new_regime

    def generate_returns_from_volatility(
        self,
        volatility_forecast: np.ndarray,
        distribution: str,
        user_knobs: Optional[Dict] = None,
        historical_returns: Optional[np.ndarray] = None,
        drift_sequence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate returns from volatility path with proper trend and momentum control

        Args:
            volatility_forecast: Array of volatility values (from forecast())
            distribution: Distribution type ('normal', 't', 'skewt')
            user_knobs: User-specified desired characteristics
            historical_returns: Historical returns (optional; not required for trend target)

        Returns:
            Array of returns with correct trend and momentum
        """

        horizon = len(volatility_forecast)

        # ============================================================
        # 1. Compute target mean return (TREND)
        # ============================================================

        if user_knobs is not None:
            # Absolute trend target (not relative to historical drift)
            # Piecewise annual drift mapping (symmetric for bearish):
            #  +0.25 -> +10%/yr, +0.50 -> +25%/yr, +1.00 -> +50%/yr.
            desired_trend = float(user_knobs.get("desired_trend", 0.0))
            trend_mag = float(np.clip(abs(desired_trend), 0.0, 1.0))

            if trend_mag <= 0.25:
                annual_drift_mag = 0.40 * trend_mag
            elif trend_mag <= 0.50:
                annual_drift_mag = 0.10 + 0.60 * (trend_mag - 0.25)
            else:
                annual_drift_mag = 0.25 + 0.50 * (trend_mag - 0.50)

            annual_drift = np.sign(desired_trend) * annual_drift_mag
            base_mu = annual_drift / 252
        else:
            # Fallback: assume zero drift
            base_mu = 0.0

        if drift_sequence is not None:
            drift_arr = np.asarray(drift_sequence, dtype=float)
            if len(drift_arr) == 0:
                mu_sequence = np.full(horizon, float(base_mu), dtype=float)
            elif len(drift_arr) >= horizon:
                mu_sequence = drift_arr[:horizon]
            else:
                mu_sequence = np.concatenate(
                    [
                        drift_arr,
                        np.full(
                            horizon - len(drift_arr), float(drift_arr[-1]), dtype=float
                        ),
                    ]
                )
        else:
            mu_sequence = np.full(horizon, float(base_mu), dtype=float)

        # ============================================================
        # 2. Compute AR(1) coefficient (MOMENTUM)
        # ============================================================

        if user_knobs is not None:
            target_hurst = user_knobs.get("target_hurst")
            if target_hurst is None:
                # Legacy fallback if target_hurst was not precomputed upstream.
                momentum = float(user_knobs.get("desired_momentum", 0.5))
                target_hurst = float(np.clip(momentum, 0.0, 1.0))
            else:
                target_hurst = float(np.clip(float(target_hurst), 0.0, 1.0))

            # Derive AR(1) coefficient from target H:
            # Approximation around fractional-memory behavior: H ≈ 0.5 + phi/2
            # => phi ≈ 2H - 1 (clipped for stability)
            phi = float(np.clip(2.0 * target_hurst - 1.0, -0.95, 0.95))
        else:
            phi = 0.0  # No autocorrelation by default

        # ============================================================
        # 2b. A/B switch: optional skew-aware shocks
        # Default is False to preserve current behavior.
        # ============================================================
        use_skew_shocks = False
        if user_knobs is not None:
            use_skew_shocks = bool(user_knobs.get("use_skew_shocks", False))

        # ============================================================
        # 3. Generate returns with AR(1) + GARCH volatility
        # ============================================================

        returns = np.zeros(horizon)

        # First return (no previous return to reference)
        shock_0 = self._generate_shock(distribution, use_skew_shocks=use_skew_shocks)
        returns[0] = mu_sequence[0] + volatility_forecast[0] * shock_0

        # Subsequent returns with AR(1) component
        for t in range(1, horizon):
            # Generate shock (symmetric distribution for unbiased skewness)
            shock = self._generate_shock(distribution, use_skew_shocks=use_skew_shocks)

            mu_t = float(mu_sequence[t])
            mu_prev = float(mu_sequence[t - 1])

            # AR(1) term (creates momentum/autocorrelation)
            ar_component = phi * (returns[t - 1] - mu_prev)

            # Volatility term (GARCH dynamics)
            volatility_component = volatility_forecast[t] * shock

            # Combined return: drift + momentum + volatility
            returns[t] = mu_t + ar_component + volatility_component

        return returns

    def _generate_shock(
        self, distribution: str, use_skew_shocks: bool = False
    ) -> float:
        """
        Generate a random shock from specified distribution
        Uses SYMMETRIC distributions to avoid uncontrolled skewness
        """

        if distribution == "t":
            # Student's t (symmetric)
            nu = self.params.get("nu", 8)
            shock = np.random.standard_t(nu)
            # Standardize to unit variance
            shock = shock / np.sqrt(nu / (nu - 2)) if nu > 2 else shock

        elif distribution == "skewt":
            if use_skew_shocks:
                # True standardized skew-t draw using arch's SkewStudent.
                # Parameters: eta (df), lambda (skew), with constraints
                # eta > 4 for finite kurtosis and |lambda| < 1.
                eta = max(float(self.params.get("nu", 8.0)), 4.05)
                lam = float(self.params.get("lambda", 0.0))
                lam = float(np.clip(lam, -0.99, 0.99))

                # Build sampler lazily and refresh if parameters changed.
                if self._skewt_sampler is None:
                    dist = SkewStudent(seed=np.random.randint(0, 2**31 - 1))
                    self._skewt_sampler = (eta, lam, dist.simulate([eta, lam]))
                else:
                    cached_eta, cached_lam, sampler = self._skewt_sampler
                    if abs(cached_eta - eta) > 1e-12 or abs(cached_lam - lam) > 1e-12:
                        dist = SkewStudent(seed=np.random.randint(0, 2**31 - 1))
                        self._skewt_sampler = (eta, lam, dist.simulate([eta, lam]))

                _, _, sampler = self._skewt_sampler
                shock = float(sampler(1)[0])
            else:
                # Default behavior (safe): keep shocks symmetric
                # to avoid uncontrolled skewness interfering with trend.
                shock = np.random.randn()

        else:  # "normal"
            # Standard normal (symmetric)
            shock = np.random.randn()

        return shock
