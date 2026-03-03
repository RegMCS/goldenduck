# services/msgarch_service.py
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)


@dataclass
class RegimeParams:
    omega: float
    alpha: float
    beta: float
    dist: str
    df: Optional[float] = None  # for Student-t if available


class MSGARCHService:
    """
    Two-regime Markov-Switching GARCH(1,1) service.

    Notes:
    - Uses an EM-ish loop:
      (E) compute regime probs via forward filter using regime-specific conditional likelihoods
      (M) update transition matrix + refit regime-specific GARCH on regime-assigned segments
    - Practical approximation because arch_model does not directly fit MS-GARCH.
    """

    def __init__(self):
        self.historical_data: Optional[pd.DataFrame] = None
        self.scale_factor: float = 100.0  # numerical stability
        # learned/fitted artifacts
        self.P: Optional[np.ndarray] = None  # 2x2 transition matrix
        self.regime_models = [None, None]  # arch_model fitted results per regime
        self.regime_params: Optional[List[RegimeParams]] = None
        self.filtered_probs: Optional[np.ndarray] = None  # T x 2

    # ----------------------------
    # Public API
    # ----------------------------
    def fit_model(
        self,
        historical_data: pd.DataFrame,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        n_iter: int = 5,
        init_split_quantile: float = 0.7,
        init_split_quantile_low: Optional[float] = None,
        init_split_quantile_high: Optional[float] = None,
        min_points_per_regime: int = 200,
    ) -> Dict:
        """
        Fit a 2-regime MS-GARCH(1,1) model.

        Args:
            dist: 'normal' or 't' or 'skewt' (skewt will fallback to 't' or 'normal' for simulation)
            n_iter: EM-ish iterations
            init_split_quantile: initial split uses rolling vol; top quantile treated as high-vol regime
            min_points_per_regime: safeguard to avoid degenerate refits
        Returns:
            dict with transition matrix, regime parameters, convergence diagnostics
        """
        self.historical_data = historical_data

        # close_prices = historical_data["Close"].astype(float).values
        close_prices = historical_data["Close"].astype(float).to_numpy().ravel()

        r = np.log(close_prices[1:] / close_prices[:-1])
        r_scaled = r * self.scale_factor
        T = len(r_scaled)

        # 1) Initialize regimes using a volatility proxy
        vol_proxy = self._rolling_vol_proxy(r_scaled)
        if (
            init_split_quantile_low is not None
            and init_split_quantile_high is not None
        ):
            q_low = float(np.quantile(vol_proxy, init_split_quantile_low))
            q_high = float(np.quantile(vol_proxy, init_split_quantile_high))
            if q_high <= q_low:
                raise ValueError("init_split_quantile_high must be > init_split_quantile_low")
            init_state = np.zeros_like(vol_proxy, dtype=int)
            hi_mask = vol_proxy >= q_high
            lo_mask = vol_proxy <= q_low
            init_state[hi_mask] = 1
            init_state[lo_mask] = 0
            mid_mask = ~(hi_mask | lo_mask)
            if np.any(mid_mask):
                dist_to_low = np.abs(vol_proxy[mid_mask] - q_low)
                dist_to_high = np.abs(vol_proxy[mid_mask] - q_high)
                init_state[mid_mask] = (dist_to_high < dist_to_low).astype(int)
        else:
            thresh = np.quantile(vol_proxy, init_split_quantile)
            init_state = (vol_proxy >= thresh).astype(int)  # 0=low, 1=high

        # Initial transition matrix from init_state with smoothing
        P = self._estimate_transition_matrix(init_state, smoothing=1.0)
        self.P = P

        # Initial regime fits (hard split)
        models = self._fit_regime_models_from_states(
            r_scaled, init_state, p=p, q=q, dist=dist, min_points=min_points_per_regime
        )
        self.regime_models = models

        # EM-ish loop - iteration algo - Expectation-Maximization - can read up if you want
        prev_ll = -np.inf
        converged = False

        prev_params = None
        param_diff = np.inf

        for it in range(n_iter):
            
            # E-step: compute conditional likelihoods under each regime model
            ll_tk = self._per_t_loglikelihoods(r_scaled, self.regime_models)
            # Forward filter to get filtered probs p(s_t=k | r_1..t)
            filtered_probs, ll = self._forward_filter(ll_tk, P)

            # M-step: update transition matrix using expected transitions
            P_new = self._m_step_transition(filtered_probs, P)

            # M-step: refit regime-specific GARCH by assigning observations
            # Hard assignment (simple, stable): pick regime with max prob at time t
            states = np.argmax(filtered_probs, axis=1).astype(int)

            models_new = self._fit_regime_models_from_states(
                r_scaled,
                states,
                p=p,
                q=q,
                dist=dist,
                min_points=min_points_per_regime,
                fallback_models=self.regime_models,
            )

            # --- Add after 'models_new =' inside the loop ---
            # Extract omega, alpha, beta for both regimes into one array
            current_params = np.array([[m.params['omega'], m.params.get('alpha[1]', 0), m.params.get('beta[1]', 0)] for m in models_new])
            
            if prev_params is not None:
                param_diff = np.mean(np.abs(current_params - prev_params))
            
            ll_diff = abs(ll - prev_ll) if np.isfinite(prev_ll) else np.inf

            # Update your convergence check
            if ll_diff < 1e-1 and param_diff < 1e-4:
                converged = True

            # Check improvement
            logger.info(f"MS-GARCH iter {it+1}/{n_iter}: loglik={ll:.2f}")
            if np.isfinite(prev_ll) and abs(ll - prev_ll) < 1e-1:
                converged = True
                P, self.regime_models = P_new, models_new
                self.filtered_probs = filtered_probs
                prev_ll = ll
                break

            P, self.regime_models = P_new, models_new
            self.filtered_probs = filtered_probs
            prev_ll = ll
            prev_params = current_params

        self.P = P
        self.regime_params = self._extract_regime_params(self.regime_models, dist=dist)

        out = {
            "num_regimes": 2,
            "converged": converged,
            "param_diff": float(param_diff),
            "transition_matrix": P.tolist(),
            "regimes": [
                {
                    "omega": rp.omega,
                    "alpha": rp.alpha,
                    "beta": rp.beta,
                    "dist": rp.dist,
                    "df": rp.df,
                    "garch_converged": bool(self.regime_models[i].convergence_flag == 0),
                    "garch_status": self.regime_models[i].fit_stop,
                }
                for i, rp in enumerate(self.regime_params)
            ],
            "loglik": float(prev_ll),
        }
        return out


    def generate_scenarios(
        self,
        num_scenarios: int = 1000,
        horizon: int = 252,
        volatility_multiplier: float = 1.0,
        random_seed: Optional[int] = None,
        include_regime: bool = False,
        min_run_length: Optional[int] = None,
        switch_scale: Optional[float] = None,
    ) -> List[pd.DataFrame]:
        """
        Simulate OHLCV scenarios using the fitted MS-GARCH model.
        """
        if self.historical_data is None or self.P is None or self.regime_params is None:
            raise ValueError("Must fit model first! Call fit_model().")

        if random_seed is not None:
            np.random.seed(random_seed)

        initial_price_value = float(self.historical_data["Close"].iloc[-1])

        scenarios: List[pd.DataFrame] = []
        P = self.P
        if switch_scale is not None and switch_scale > 0:
            # Reduce switching by scaling off-diagonal probabilities.
            # switch_scale < 1 => fewer switches; ==1 => no change.
            P_adj = P.copy()
            off01 = float(P_adj[0, 1]) * float(switch_scale)
            off10 = float(P_adj[1, 0]) * float(switch_scale)
            P_adj[0, 1] = min(max(off01, 0.0), 1.0)
            P_adj[1, 0] = min(max(off10, 0.0), 1.0)
            P_adj[0, 0] = 1.0 - P_adj[0, 1]
            P_adj[1, 1] = 1.0 - P_adj[1, 0]
            P = P_adj
        rp0, rp1 = self.regime_params

        for sidx in range(num_scenarios):
            states = self._simulate_markov_chain(
                P, horizon, start_state=None, min_run_length=min_run_length
            )

            # GARCH recursion per regime
            r, sigma = self._simulate_msgarch_path(
                states=states,
                rp0=rp0,
                rp1=rp1,
                horizon=horizon,
                volatility_multiplier=volatility_multiplier,
            )

            # Undo scaling back to log-return units
            r = r / self.scale_factor
            sigma = sigma / self.scale_factor  # volatility in log-return units

            # Build close prices
            close_prices = initial_price_value * np.exp(np.cumsum(r))

            # Convert to OHLCV using same helper style as your GARCHService
            ohlcv = self._generate_ohlcv_from_close(close_prices, sigma)
            if include_regime:
                ohlcv["Regime"] = states.astype(int)
            scenarios.append(ohlcv)

            if (sidx + 1) % 100 == 0:
                logger.info(f"Generated {sidx + 1}/{num_scenarios} MS-GARCH scenarios")

        return scenarios

    def validate_scenarios(self, scenarios: List[pd.DataFrame]) -> Dict:
        """
        Reuse your existing validation service for synthetic data quality checks.
        """
        try:
            from services.validation_service import ValidationService

            validator = ValidationService(self.historical_data)

            sample_size = min(100, len(scenarios))
            all_synthetic_returns = []

            for scenario in scenarios[:sample_size]:
                returns = scenario["Close"].pct_change().dropna().values
                all_synthetic_returns.extend(returns)

            synthetic_returns_array = np.array(all_synthetic_returns)
            metrics = validator.validate(synthetic_returns_array)
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
            }

    # ----------------------------
    # Internals: fitting
    # ----------------------------
    def _rolling_vol_proxy(self, r_scaled: np.ndarray, window: int = 20) -> np.ndarray:
        # simple rolling std proxy; pad to length T
        if len(r_scaled) < window + 2:
            return np.abs(r_scaled)  # fallback
        s = pd.Series(r_scaled)
        vol = s.rolling(window).std().fillna(method="bfill").values
        return vol

    def _estimate_transition_matrix(self, states: np.ndarray, smoothing: float = 1.0) -> np.ndarray:
        # counts with Laplace smoothing
        c = np.full((2, 2), smoothing, dtype=float)
        for t in range(1, len(states)):
            c[states[t - 1], states[t]] += 1.0
        P = c / c.sum(axis=1, keepdims=True)
        return P

    def _fit_regime_models_from_states(
        self,
        r_scaled: np.ndarray,
        states: np.ndarray,
        p: int,
        q: int,
        dist: str,
        min_points: int,
        fallback_models=None,
    ):
        models = [None, None]
        for k in [0, 1]:
            idx = np.where(states == k)[0]
            if len(idx) < min_points:
                logger.warning(
                    f"Regime {k} has only {len(idx)} points (<{min_points}). Using fallback model."
                )
                if fallback_models is not None and fallback_models[k] is not None:
                    models[k] = fallback_models[k]
                else:
                    # fallback: fit on all data
                    models[k] = self._fit_garch(r_scaled, p, q, dist)
                continue

            series = r_scaled[idx]
            models[k] = self._fit_garch(series, p, q, dist)
        return models

    def _fit_garch(self, series: np.ndarray, p: int, q: int, dist: str):
        # Keep mean="Zero" like your base service.
        model = arch_model(
            series,
            vol="GARCH",
            p=p,
            q=q,
            mean="Zero",
            dist=dist if dist in ("normal", "t", "skewt") else "normal",
            rescale=False,
        )
        res = model.fit(disp="off", options={"maxiter": 5000})
        return res

    def _per_t_loglikelihoods(self, r_scaled: np.ndarray, models) -> np.ndarray:
        """
        Approximate per-time log-likelihood under each regime:
        Use conditional variance from fitted model recursion applied to the full series.

        Because arch result doesn't expose a direct 'per t' likelihood for arbitrary new series,
        we approximate with:
            ll_t = log pdf of N(0, sigma_t^2) or t-dist using fitted params and sigma_t^2 recursion.
        """
        T = len(r_scaled)
        ll = np.zeros((T, 2), dtype=float)

        for k in [0, 1]:
            res = models[k]
            omega = float(res.params["omega"])
            alpha = float(res.params.get("alpha[1]", 0.0))
            beta = float(res.params.get("beta[1]", 0.0))

            # initialize variance with unconditional if possible
            denom = max(1e-8, (1.0 - alpha - beta))
            var = np.zeros(T, dtype=float)
            var[0] = omega / denom

            eps2_prev = r_scaled[0] ** 2
            for t in range(1, T):
                var[t] = omega + alpha * eps2_prev + beta * var[t - 1]
                eps2_prev = r_scaled[t] ** 2

            # compute log-likelihood under conditional normal
            # (keeps things stable and fast)
            ll[:, k] = self._logpdf_normal_zero_mean(r_scaled, var)

        return ll

    def _forward_filter(self, ll_tk: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Hamilton filter in log space for numerical stability.
        Inputs:
          ll_tk: T x 2 log-likelihood of observation at t in each state
          P: 2x2 transition matrix
        Returns:
          filtered_probs: T x 2
          total_loglik: float
        """
        T = ll_tk.shape[0]
        fp = np.zeros((T, 2), dtype=float)

        # initial state prob: stationary distribution approx
        pi = self._stationary_dist(P)

        log_alpha = np.log(pi + 1e-12) + ll_tk[0]
        c0 = self._logsumexp(log_alpha)
        log_alpha = log_alpha - c0
        fp[0] = np.exp(log_alpha)

        total_ll = c0

        for t in range(1, T):
            # predict: alpha_pred(j) = sum_i alpha(i) P(i->j)
            # operate in prob space here (2 states only; stable enough after normalization)
            alpha_prev = fp[t - 1]
            alpha_pred = alpha_prev @ P  # shape (2,)

            log_alpha = np.log(alpha_pred + 1e-12) + ll_tk[t]
            ct = self._logsumexp(log_alpha)
            log_alpha = log_alpha - ct
            fp[t] = np.exp(log_alpha)
            total_ll += ct

        return fp, float(total_ll)

    def _m_step_transition(self, filtered_probs: np.ndarray, P_old: np.ndarray) -> np.ndarray:
        """
        Update transition matrix using a simple approximation:
        Count expected transitions using filtered probs (not full xi from backward pass).
        Good practical approximation for 2-regime use.
        """
        T = filtered_probs.shape[0]
        c = np.zeros((2, 2), dtype=float) + 1e-3  # tiny smoothing
        for t in range(1, T):
            prev = filtered_probs[t - 1]
            curr = filtered_probs[t]
            # expected transitions approx outer(prev, curr) normalized by row totals
            c += np.outer(prev, curr)

        P_new = c / c.sum(axis=1, keepdims=True)
        return P_new

    def _extract_regime_params(self, models, dist: str) -> List[RegimeParams]:
        out = []
        for k in [0, 1]:
            res = models[k]
            omega = float(res.params["omega"])
            alpha = float(res.params.get("alpha[1]", 0.0))
            beta = float(res.params.get("beta[1]", 0.0))

            df = None
            # arch uses 'nu' for t / skewt degrees of freedom
            if "nu" in res.params.index:
                df = float(res.params["nu"])

            out.append(RegimeParams(omega=omega, alpha=alpha, beta=beta, dist=dist, df=df))
        return out

    def _stationary_dist(self, P: np.ndarray) -> np.ndarray:
        # stationary distribution for 2-state Markov chain
        a = P[0, 1]
        b = P[1, 0]
        denom = a + b
        if denom <= 0:
            return np.array([0.5, 0.5], dtype=float)
        pi0 = b / denom
        return np.array([pi0, 1.0 - pi0], dtype=float)

    @staticmethod
    def _logsumexp(x: np.ndarray) -> float:
        m = np.max(x)
        return float(m + np.log(np.sum(np.exp(x - m))))

    @staticmethod
    def _logpdf_normal_zero_mean(x: np.ndarray, var: np.ndarray) -> np.ndarray:
        var = np.maximum(var, 1e-12)
        return -0.5 * (np.log(2.0 * np.pi * var) + (x * x) / var)

    # ----------------------------
    # Internals: simulation
    # ----------------------------
    def _simulate_markov_chain(
        self,
        P: np.ndarray,
        T: int,
        start_state: Optional[int],
        min_run_length: Optional[int] = None,
    ) -> np.ndarray:
        if start_state is None:
            pi = self._stationary_dist(P)
            state = 0 if np.random.rand() < pi[0] else 1
        else:
            state = int(start_state)

        states = np.zeros(T, dtype=int)
        min_run = int(min_run_length) if min_run_length and min_run_length > 0 else 0
        remaining = min_run
        for t in range(T):
            states[t] = state
            if remaining > 0:
                remaining -= 1
                continue
            u = np.random.rand()
            next_state = 0 if u < P[state, 0] else 1
            if next_state != state and min_run > 0:
                remaining = min_run - 1
            state = next_state
        return states

    def _simulate_msgarch_path(
        self,
        states: np.ndarray,
        rp0: RegimeParams,
        rp1: RegimeParams,
        horizon: int,
        volatility_multiplier: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate scaled returns and scaled volatility sigma_t (sqrt(var_t)) under MS-GARCH.
        Uses per-regime (omega, alpha, beta). Innovations ~ N(0,1) or t.
        """
        r = np.zeros(horizon, dtype=float)
        var = np.zeros(horizon, dtype=float)

        # initialize with a reasonable starting variance
        # use regime 0 unconditional as base
        denom0 = max(1e-8, (1.0 - rp0.alpha - rp0.beta))
        var[0] = rp0.omega / denom0

        for t in range(horizon):
            rp = rp0 if states[t] == 0 else rp1

            if t > 0:
                eps2 = r[t - 1] ** 2
                var[t] = rp.omega + rp.alpha * eps2 + rp.beta * var[t - 1]

            sigma_t = np.sqrt(max(var[t], 1e-12)) * volatility_multiplier

            z = self._draw_innovation(rp)
            r[t] = sigma_t * z

        sigma = np.sqrt(np.maximum(var, 1e-12)) * volatility_multiplier
        return r, sigma

    def _draw_innovation(self, rp: RegimeParams) -> float:
        # keep it robust: support normal and t; skewt fallback
        if rp.dist == "t" and rp.df is not None and rp.df > 2:
            return float(np.random.standard_t(df=rp.df))
        if rp.dist == "skewt":
            # fallback: treat as t if df available else normal
            if rp.df is not None and rp.df > 2:
                return float(np.random.standard_t(df=rp.df))
            return float(np.random.normal())
        return float(np.random.normal())

    # ----------------------------
    # OHLCV helper - smth smth need OHLCV TT
    # ----------------------------
    def _generate_ohlcv_from_close(self, close_prices: np.ndarray, volatility: np.ndarray) -> pd.DataFrame:
        n = len(close_prices)

        if len(volatility) != n:
            if len(volatility) < n:
                volatility = np.pad(volatility, (0, n - len(volatility)), mode="edge")
            else:
                volatility = volatility[:n]

        rows = []
        for i in range(n):
            C = float(close_prices[i])
            sigma = float(volatility[i])

            if i > 0:
                gap = np.random.normal(0, sigma * 0.3)
                O = float(close_prices[i - 1] * (1.0 + gap))
            else:
                O = C

            hl_range = abs(np.random.normal(0, sigma * 1.5))
            H = max(O, C) * (1.0 + hl_range)
            L = min(O, C) * (1.0 - hl_range)

            base_volume = 1_000_000
            V = int(base_volume * (1.0 + sigma * np.random.exponential(2.0)))

            rows.append({"Open": O, "High": H, "Low": L, "Close": C, "Volume": V})

        return pd.DataFrame(rows)

