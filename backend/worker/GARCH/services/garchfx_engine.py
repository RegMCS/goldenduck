# services/garchfx_engine.py
import numpy as np
from typing import Optional, List
import logging

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
        self.persistence = self.alpha + self.beta
        self.scale_factor = scale_factor

        logger.info(
            f"GARCH-FX initialized: σ₀={self.initial_volatility:.4f}, "
            f"α={self.alpha:.4f}, β={self.beta:.4f}, ω={self.omega:.6f}"
        )

    def forecast(
        self,
        horizon: int,
        theta: float = 0.005,
        delta_sequence: Optional[np.ndarray] = None,
        regime_switching: bool = False,
        regime_states: Optional[np.ndarray] = None,
        regimes: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Generate GARCH-FX volatility forecast.

        Parameters:
        -----------
        horizon : int
            Forecast horizon
        theta : float
            Stochasticity parameter (scale of Gamma distribution)
        delta_sequence : np.ndarray, optional
            Predetermined delta multipliers (for stress scenarios)
        regime_switching : bool
            Enable stochastic Markov regime switching
        regime_states : np.ndarray, optional
            Transition probability matrix
        regimes : List[float], optional
            Delta multipliers for each regime

        Returns:
        --------
        np.ndarray : Volatility forecast (unscaled, daily units)
        """
        delta = 1.0
        forecasts = [self.initial_volatility]
        previous_variance = self.initial_volatility**2

        for i in range(horizon - 1):
            # Step 1: Calculate Gamma shape parameter
            shape = (previous_variance / theta) + 1

            # Step 2: Determine delta for this step
            if delta_sequence is not None:
                delta = (
                    delta_sequence[i] if i < len(delta_sequence) else delta_sequence[-1]
                )
            elif regime_switching:
                delta = self._regime_switcher(delta, regime_states, regimes)

            # Step 3: Sample stochastic variance from Gamma distribution
            stochastic_variance = np.random.gamma(shape=shape, scale=theta, size=1)[0]

            # Step 4: GARCH-FX equation
            forecasted_variance = (self.omega * delta) + (
                self.persistence * stochastic_variance
            )

            # Step 5: Update for next iteration
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
        self, volatility: np.ndarray, distribution: str = "normal"
    ) -> np.ndarray:
        """
        Generate returns from forecasted volatility.

        Parameters:
        -----------
        volatility : np.ndarray
            Forecasted volatility path
        distribution : str
            Distribution for returns: 'normal', 't', 'skewt'

        Returns:
        --------
        np.ndarray : Returns
        """
        n = len(volatility)

        if distribution == "normal":
            returns = np.random.normal(0, volatility)
        elif distribution == "t":
            # Student-t with df=10 (typical for financial data)
            df = 10
            returns = (
                np.random.standard_t(df, size=n) * volatility / np.sqrt(df / (df - 2))
            )
        elif distribution == "skewt":
            # Simplified: use t-distribution as proxy
            df = 8
            returns = (
                np.random.standard_t(df, size=n) * volatility / np.sqrt(df / (df - 2))
            )
        else:
            logger.warning(f"Unknown distribution '{distribution}', using normal")
            returns = np.random.normal(0, volatility)

        return returns
