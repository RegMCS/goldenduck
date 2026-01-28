import numpy as np
import pandas as pd
from arch import arch_model
from typing import List, Dict, Optional, Literal
import logging

logger = logging.getLogger(__name__)


class GARCHService:
    """
    Service for GARCH-based synthetic data generation.
    Now supports both standard GARCH and GARCH-FX (stochastic + regime-aware) modes.
    """

    def __init__(self):
        self.fitted_model = None
        self.historical_data = None
        self.scale_factor = 100  # For numerical stability
        # NEW: Store GARCH parameters for GARCH-FX
        self.garch_params = None
        self.last_conditional_volatility = None
        self.distribution = "normal"

    # GARCH/services/garch_service.py

    def _fit_once(
        self,
        historical_data: pd.DataFrame,
        p: int,
        q: int,
        dist: str,
    ) -> Dict:
        """
        Single attempt to fit a GARCH model.
        """
        self.historical_data = historical_data

        # Calculate log returns
        close_prices = historical_data["Close"].values
        returns = np.log(close_prices[1:] / close_prices[:-1])

        # Scale for numerical stability
        scaled_returns = returns * self.scale_factor

        model = arch_model(
            scaled_returns,
            vol="GARCH",
            p=p,
            q=q,
            mean="Zero",
            dist=dist,
            rescale=False,
        )

        fitted = model.fit(disp="off", options={"maxiter": 5000})

        # Store fitted model (IMPORTANT)
        self.fitted_model = fitted

        params = {
            "omega": float(fitted.params.get("omega", np.nan)),
            "alpha": float(fitted.params.get("alpha[1]", np.nan)),
            "beta": float(fitted.params.get("beta[1]", np.nan)),
            "converged": fitted.convergence_flag == 0,
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "distribution": dist,
        }

        # Optional skew-t params
        if dist in ("t", "skewt") and "nu" in fitted.params:
            params["nu"] = float(fitted.params["nu"])
        if dist == "skewt" and "lambda" in fitted.params:
            params["lambda"] = float(fitted.params["lambda"])

        # Store GARCH parameters for GARCH-FX usage
        self.garch_params = {
            "alpha": params["alpha"],
            "beta": params["beta"],
            "omega": params["omega"],
        }

        # FIX: Store distribution type separately
        self.distribution = dist  # ADD THIS LINE

        # Handle both numpy array and pandas Series
        conditional_vol = fitted.conditional_volatility

        if isinstance(conditional_vol, np.ndarray):
            self.last_conditional_volatility = (
                float(conditional_vol[-1]) / self.scale_factor
            )
        else:
            self.last_conditional_volatility = (
                float(conditional_vol.iloc[-1]) / self.scale_factor
            )

        logger.info(
            f"Fit attempt: dist={dist}, converged={params['converged']}, "
            f"AIC={params['aic']:.2f}, last_vol={self.last_conditional_volatility:.6f}"
        )

        return params

    def fit_with_retry(
        self,
        historical_data: pd.DataFrame,
        p: int = 1,
        q: int = 1,
    ) -> Dict:
        """
        Fit GARCH model with bounded retries across distributions.
        Convergence is preferred but not forced.
        """

        best_result = None
        best_model = None

        for dist in ["t", "skewt"]:  # safe → expressive
            for attempt in range(2):  # bounded retries
                result = self._fit_once(
                    historical_data=historical_data,
                    p=p,
                    q=q,
                    dist=dist,
                )

                # Keep first result as baseline
                if best_result is None:
                    best_result = result
                    best_model = self.fitted_model

                # Prefer converged models immediately
                if result["converged"]:
                    logger.info(
                        f"GARCH converged using dist={dist} on attempt={attempt+1}"
                    )
                    return result

                # Otherwise keep the statistically better one
                if result["aic"] < best_result["aic"]:
                    best_result = result
                    best_model = self.fitted_model

        # Restore best non-converged model ONCE
        if best_model is not None:
            self.fitted_model = best_model

        logger.warning("No converged model found; returning best non-converged fit")
        return best_result

    # ========== NEW: GARCH-FX METHODS ==========

    def generate_scenarios_fx(
        self,
        num_scenarios: int = 1000,
        horizon: int = 252,
        theta: float = 0.005,
        scenario_type: Optional[str] = None,
        delta_sequence: Optional[np.ndarray] = None,
        regime_switching: bool = False,
        regime_states: Optional[np.ndarray] = None,
        regimes: Optional[List[float]] = None,
        seed_start: int = 42,
    ) -> List[pd.DataFrame]:
        """Generate synthetic scenarios using GARCH-FX framework"""
        if self.fitted_model is None or self.garch_params is None:
            raise ValueError("Must fit model first! Call fit_with_retry()")

        logger.info(
            f"Starting GARCH-FX generation: {num_scenarios} scenarios, "
            f"θ={theta}, scenario={scenario_type}"
        )

        from GARCH.services.garchfx_engine import GARCHFXEngine

        engine = GARCHFXEngine(
            volatility=self.last_conditional_volatility,
            params=self.garch_params,
            scale_factor=self.scale_factor,
        )

        if scenario_type and delta_sequence is None:
            from GARCH.services.scenarios import generate_scenario

            delta_sequence, _ = generate_scenario(scenario_type, horizon)
            logger.info(f"Using scenario: {scenario_type}")

        initial_price = float(self.historical_data["Close"].iloc[-1])
        scenarios = []

        for scenario_idx in range(num_scenarios):
            np.random.seed(seed_start + scenario_idx)

            volatility_forecast = engine.forecast(
                horizon=horizon,
                theta=theta,
                delta_sequence=delta_sequence,
                regime_switching=regime_switching,
                regime_states=regime_states,
                regimes=regimes,
            )

            # FIX: Use self.distribution instead of self.fitted_model.distribution
            returns = engine.generate_returns_from_volatility(
                volatility_forecast, self.distribution  # CHANGED THIS LINE
            )

            cumulative_returns = np.cumsum(returns)
            close_prices = initial_price * np.exp(cumulative_returns)

            ohlcv = self._generate_ohlcv_from_close(close_prices, volatility_forecast)
            scenarios.append(ohlcv)

            if (scenario_idx + 1) % 100 == 0:
                logger.info(f"Generated {scenario_idx + 1}/{num_scenarios} scenarios")

        logger.info(f"Successfully generated {num_scenarios} GARCH-FX scenarios")
        return scenarios

    # ========== ORIGINAL METHOD (PRESERVED) ==========

    def generate_scenarios(
        self,
        num_scenarios: int = 1000,
        horizon: int = 252,
        volatility_multiplier: float = 1.0,
    ) -> List[pd.DataFrame]:
        """
        Generate multiple synthetic OHLCV scenarios using standard GARCH.
        (Original method preserved for backward compatibility)
        """
        if self.fitted_model is None:
            raise ValueError("Must fit model first! Call fit_with_retry()")

        try:
            logger.info(
                f"Starting generation of {num_scenarios} scenarios (standard GARCH), "
                f"{horizon} days each"
            )

            initial_price_value = float(self.historical_data["Close"].iloc[-1].item())
            scenarios = []

            for scenario_idx in range(num_scenarios):
                # Generate ONE scenario at a time
                forecast = self.fitted_model.forecast(
                    horizon=horizon,
                    method="simulation",
                    simulations=1,
                    reindex=False,
                )

                # Extract and flatten returns
                returns_scaled = forecast.simulations.values.flatten()
                returns = returns_scaled / self.scale_factor

                # Extract and flatten volatility
                variance = forecast.variance.values.flatten()
                volatility = (
                    np.sqrt(variance) / self.scale_factor
                ) * volatility_multiplier

                # Ensure correct length
                returns = returns[:horizon]
                volatility = volatility[:horizon]

                logger.debug(
                    f"Scenario {scenario_idx}: returns={len(returns)}, vol={len(volatility)}"
                )

                # Calculate prices
                cumulative_returns = np.cumsum(returns)
                close_prices = initial_price_value * np.exp(cumulative_returns)

                # Generate OHLCV
                ohlcv = self._generate_ohlcv_from_close(close_prices, volatility)
                scenarios.append(ohlcv)

                if (scenario_idx + 1) % 100 == 0:
                    logger.info(
                        f"Generated {scenario_idx + 1}/{num_scenarios} scenarios"
                    )

            logger.info(f"Successfully generated {num_scenarios} scenarios")
            return scenarios

        except Exception as e:
            logger.error(f"Error generating scenarios: {e}", exc_info=True)
            raise

    def _generate_ohlcv_from_close(
        self, close_prices: np.ndarray, volatility: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate realistic OHLCV from close prices
        """
        n = len(close_prices)

        # Ensure volatility matches length
        if len(volatility) != n:
            logger.warning(
                f"Volatility length {len(volatility)} != close prices length {n}"
            )
            if len(volatility) < n:
                volatility = np.pad(volatility, (0, n - len(volatility)), mode="edge")
            else:
                volatility = volatility[:n]

        # Build OHLCV data row by row
        ohlcv_rows = []

        for i in range(n):
            C = float(close_prices[i])
            sigma = float(volatility[i])

            # Open: Previous close + gap
            if i > 0:
                gap = np.random.normal(0, sigma * 0.3)
                O = float(close_prices[i - 1] * (1 + gap))
            else:
                O = C

            # High/Low using Parkinson range
            hl_range = abs(np.random.normal(0, sigma * 1.5))
            H = max(O, C) * (1 + hl_range)
            L = min(O, C) * (1 - hl_range)

            # Volume: Correlated with volatility
            base_volume = 1_000_000
            V = int(base_volume * (1 + sigma * np.random.exponential(2)))

            ohlcv_rows.append({"Open": O, "High": H, "Low": L, "Close": C, "Volume": V})

        df = pd.DataFrame(ohlcv_rows)

        logger.debug(
            f"Created OHLCV DataFrame: {len(df)} rows x {len(df.columns)} columns"
        )

        return df

    def validate_scenarios(self, scenarios: List[pd.DataFrame]) -> Dict:
        """
        Validate synthetic data quality
        """
        try:
            from GARCH.services.validation_service import ValidationService

            validator = ValidationService(self.historical_data)

            # Sample first 100 scenarios and collect returns
            sample_size = min(100, len(scenarios))
            all_synthetic_returns = []

            for scenario in scenarios[:sample_size]:
                returns = scenario["Close"].pct_change().dropna().values
                all_synthetic_returns.extend(returns)

            synthetic_returns_array = np.array(all_synthetic_returns)
            metrics = validator.validate(synthetic_returns_array)

            logger.info(f"Validation complete: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error during validation: {e}", exc_info=True)
            return {
                "ks_statistic": 0.0,
                "ks_pvalue": 0.0,
                "kurtosis_historical": 0.0,
                "kurtosis_synthetic": 0.0,
                "acf_lag1_historical": 0.0,
                "acf_lag1_synthetic": 0.0,
                "error": str(e),
            }
