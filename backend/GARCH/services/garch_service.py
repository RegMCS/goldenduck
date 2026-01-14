# services/garch_service.py
import numpy as np
import pandas as pd
from arch import arch_model
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class GARCHService:
    """
    Service for GARCH-based synthetic data generation
    """
    
    def __init__(self):
        self.fitted_model = None
        self.historical_data = None
        self.scale_factor = 100  # For numerical stability
    
    def fit_model(self, historical_data: pd.DataFrame, p: int = 1, q: int = 1, dist='normal') -> Dict:
        """
        Fit GARCH model with optional Student-t distribution
        
        Args:
            dist: 'normal', 't' (Student-t), or 'skewt' (skewed Student-t)
        """
        try:
            self.historical_data = historical_data
            
            # Calculate returns
            close_prices = historical_data['Close'].values
            returns = np.log(close_prices[1:] / close_prices[:-1])
            
            # Scale returns by 100 for numerical stability
            scaled_returns = returns * self.scale_factor
            
            # Fit GARCH model
            model = arch_model(
                scaled_returns, 
                vol='GARCH', 
                p=p, 
                q=q, 
                mean='Zero',
                dist=dist,  # ← 'normal', 't', or 'skewt'
                rescale=False
            )
            self.fitted_model = model.fit(disp='off', options={'maxiter': 5000})
            
            # Check convergence
            if not self.fitted_model.convergence_flag:
                logger.warning("GARCH model did not converge")
            else:
                logger.info("GARCH model converged successfully")
            
            # Extract parameters
            params = {
                'omega': float(self.fitted_model.params['omega']),
                'alpha': float(self.fitted_model.params.get('alpha[1]', 0)),
                'beta': float(self.fitted_model.params.get('beta[1]', 0)),
                'converged': bool(self.fitted_model.convergence_flag),
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic)
            }
            
            logger.info(f"GARCH({p},{q}) fitted: {params}")
            return params
            
        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            raise

    def generate_scenarios(
        self, 
        num_scenarios: int = 1000, 
        horizon: int = 252,
        volatility_multiplier: float = 1.0
    ) -> List[pd.DataFrame]:
        """
        Generate multiple synthetic OHLCV scenarios (one at a time for reliability)
        """
        if self.fitted_model is None:
            raise ValueError("Must fit model first! Call fit_model()")
        
        try:
            logger.info(f"Starting generation of {num_scenarios} scenarios, {horizon} days each")
            
            initial_price_value = float(self.historical_data['Close'].iloc[-1])
            scenarios = []
            
            for scenario_idx in range(num_scenarios):
                # Generate ONE scenario at a time
                forecast = self.fitted_model.forecast(
                    horizon=horizon,
                    method='simulation',
                    simulations=1,  # ← Generate only 1 scenario per loop
                    reindex=False
                )
                
                # Extract and flatten returns
                returns_scaled = forecast.simulations.values.flatten()  # Always 1D
                returns = returns_scaled / self.scale_factor
                
                # Extract and flatten volatility
                variance = forecast.variance.values.flatten()
                volatility = (np.sqrt(variance) / self.scale_factor) * volatility_multiplier
                
                # Ensure correct length
                returns = returns[:horizon]
                volatility = volatility[:horizon]
                
                logger.debug(f"Scenario {scenario_idx}: returns={len(returns)}, vol={len(volatility)}")
                
                # Calculate prices
                cumulative_returns = np.cumsum(returns)
                close_prices = initial_price_value * np.exp(cumulative_returns)
                
                # Generate OHLCV
                ohlcv = self._generate_ohlcv_from_close(close_prices, volatility)
                scenarios.append(ohlcv)
                
                # Log progress
                if (scenario_idx + 1) % 100 == 0:
                    logger.info(f"Generated {scenario_idx + 1}/{num_scenarios} scenarios")
            
            logger.info(f"Successfully generated {num_scenarios} scenarios")
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating scenarios: {e}", exc_info=True)
            raise



    def _generate_ohlcv_from_close(
        self, 
        close_prices: np.ndarray, 
        volatility: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate realistic OHLCV from close prices
        """
        n = len(close_prices)
        
        # Ensure volatility matches length
        if len(volatility) != n:
            logger.warning(f"Volatility length {len(volatility)} != close prices length {n}")
            if len(volatility) < n:
                # Pad with last value
                volatility = np.pad(volatility, (0, n - len(volatility)), mode='edge')
            else:
                # Truncate
                volatility = volatility[:n]
        
        # Build OHLCV data row by row
        ohlcv_rows = []
        
        for i in range(n):
            C = float(close_prices[i])
            sigma = float(volatility[i])
            
            # Open: Previous close + gap
            if i > 0:
                gap = np.random.normal(0, sigma * 0.3)
                O = float(close_prices[i-1] * (1 + gap))
            else:
                O = C
            
            # High/Low using Parkinson range
            hl_range = abs(np.random.normal(0, sigma * 1.5))
            H = max(O, C) * (1 + hl_range)
            L = min(O, C) * (1 - hl_range)
            
            # Volume: Correlated with volatility
            base_volume = 1_000_000
            V = int(base_volume * (1 + sigma * np.random.exponential(2)))
            
            # Append as dictionary
            ohlcv_rows.append({
                'Open': O,
                'High': H,
                'Low': L,
                'Close': C,
                'Volume': V
            })
        
        # Create DataFrame from list of dictionaries
        df = pd.DataFrame(ohlcv_rows)
        
        logger.debug(f"Created OHLCV DataFrame: {len(df)} rows x {len(df.columns)} columns")
        
        return df
    
    def validate_scenarios(self, scenarios: List[pd.DataFrame]) -> Dict:
        """
        Validate synthetic data quality
        """
        try:
            from services.validation_service import ValidationService
            
            validator = ValidationService(self.historical_data)
            
            # Sample first 100 scenarios and collect returns
            sample_size = min(100, len(scenarios))
            all_synthetic_returns = []
            
            for scenario in scenarios[:sample_size]:
                returns = scenario['Close'].pct_change().dropna().values
                all_synthetic_returns.extend(returns)
            
            # Convert to numpy array
            synthetic_returns_array = np.array(all_synthetic_returns)
            
            # Validate
            metrics = validator.validate(synthetic_returns_array)
            
            logger.info(f"Validation complete: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during validation: {e}", exc_info=True)
            # Return empty metrics rather than failing
            return {
                "ks_statistic": 0.0,
                "ks_pvalue": 0.0,
                "kurtosis_historical": 0.0,
                "kurtosis_synthetic": 0.0,
                "acf_lag1_historical": 0.0,
                "acf_lag1_synthetic": 0.0
            }
