import logging
from typing import Dict, List

from .garch_service import GARCHService

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Runs controlled evaluation experiments for GARCH models.
    """

    def __init__(self, historical_data):
        self.historical_data = historical_data

    def evaluate_distribution(
        self,
        dist: str,
        p: int = 1,
        q: int = 1,
        num_scenarios: int = 500,
        horizon: int = 252,
    ) -> Dict:
        """
        Fit → Generate → Validate for a single distribution
        """
        logger.info(f"Evaluating GARCH with dist={dist}")

        garch = GARCHService()

        # ---- Fit (with retry if skew/t) ----
        if dist in ["t", "skewt"]:
            params = garch.fit_with_retry(
                historical_data=self.historical_data,
                p=p,
                q=q,
            )
        else:
            params = garch._fit_once(
                historical_data=self.historical_data,
                p=p,
                q=q,
                dist=dist,
            )

        # ---- Generate ----
        scenarios = garch.generate_scenarios(
            num_scenarios=num_scenarios,
            horizon=horizon,
        )

        # ---- Validate ----
        metrics = garch.validate_scenarios(scenarios)

        return {
            "distribution": dist,
            "parameters": params,
            "metrics": metrics,
            "num_scenarios": num_scenarios,
            "horizon": horizon,
        }

    def compare_distributions(
        self,
        distributions: List[str] = ["normal", "t", "skewt"],
    ) -> Dict:
        """
        Run evaluation across multiple distributions.
        """
        results = {}

        for dist in distributions:
            try:
                results[dist] = self.evaluate_distribution(dist)
            except Exception as e:
                logger.error(f"Evaluation failed for dist={dist}: {e}")
                results[dist] = {"error": str(e)}

        return results
