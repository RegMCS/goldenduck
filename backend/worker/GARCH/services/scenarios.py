# services/scenarios.py
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def generate_scenario(scenario_type: str, horizon: int) -> Tuple[np.ndarray, str]:
    """
    Generate predetermined delta sequences for stress testing.

    Parameters:
    -----------
    scenario_type : str
        Type of scenario
    horizon : int
        Forecast horizon

    Returns:
    --------
    Tuple[np.ndarray, str] : (delta_sequence, description)
    """
    scenarios = {
        "sudden_crisis": {
            "description": "Sudden crisis at day 200, lasting 300 days, then recovery",
            "sequence": [1.0] * 200 + [2.5] * 300 + [1.2] * (horizon - 500),
        },
        "gradual_escalation": {
            "description": "Gradual escalation from normal to extreme stress",
            "sequence": [1.0] * (horizon // 4)
            + [1.3] * (horizon // 4)
            + [1.7] * (horizon // 4)
            + [2.2] * (horizon // 4),
        },
        "crisis_waves": {
            "description": "Multiple crisis waves (5 cycles)",
            "sequence": ([1.0] * 100 + [2.0] * 100) * (horizon // 200),
        },
        "prolonged_stress": {
            "description": "Extended high-volatility period",
            "sequence": [1.0] * 100
            + [1.8] * 200
            + [2.5] * 400
            + [1.5] * (horizon - 700),
        },
        "flash_crash": {
            "description": "Brief extreme spike with quick recovery",
            "sequence": [1.0] * 300
            + [3.5] * 50
            + [1.8] * 100
            + [1.1] * (horizon - 450),
        },
    }

    if scenario_type not in scenarios:
        available = ", ".join(scenarios.keys())
        raise ValueError(f"Unknown scenario '{scenario_type}'. Available: {available}")

    scenario = scenarios[scenario_type]
    sequence = scenario["sequence"]

    # Ensure correct length
    if len(sequence) < horizon:
        sequence = sequence + [sequence[-1]] * (horizon - len(sequence))
    elif len(sequence) > horizon:
        sequence = sequence[:horizon]

    logger.info(f"Generated scenario: {scenario_type} - {scenario['description']}")

    return np.array(sequence), scenario["description"]


def list_scenarios():
    """Return list of available scenarios."""
    return [
        "sudden_crisis",
        "gradual_escalation",
        "crisis_waves",
        "prolonged_stress",
        "flash_crash",
    ]
