import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


SCENARIOS = {
    "sudden_crisis": {
        "description": "Sudden crisis at day 200, lasting 300 days, then recovery",
        "sequence": lambda horizon: [1.0] * 200 + [2.5] * 300 + [1.2] * max(0, horizon - 500),
    },
    "gradual_escalation": {
        "description": "Gradual escalation from normal to extreme stress",
        "sequence": lambda horizon: [1.0] * (horizon // 4)
        + [1.3] * (horizon // 4)
        + [1.7] * (horizon // 4)
        + [2.2] * (horizon // 4),
    },
    "crisis_waves": {
        "description": "Multiple crisis waves (5 cycles)",
        "sequence": lambda horizon: ([1.0] * 100 + [2.0] * 100) * (max(1, horizon // 200)),
    },
    "prolonged_stress": {
        "description": "Extended high-volatility period",
        "sequence": lambda horizon: [1.0] * 100
        + [1.8] * 200
        + [2.5] * 400
        + [1.5] * max(0, horizon - 700),
    },
    "flash_crash": {
        "description": "Brief extreme spike with quick recovery",
        # Phases expressed as (fraction_of_horizon, delta_value)
        "phases": [
            (0.60, 1.0),  # 60% calm buildup
            (0.10, 3.5),  # 10% extreme crash spike
            (0.20, 1.8),  # 20% elevated-vol recovery
            (0.10, 1.1),  # 10% return to near-normal
        ],
        # Preset knobs aligned to flash-crash behavior.
        "knobs": {
            "desired_trend": -0.2,
            "desired_volatility": 1.5,
            "desired_fat_tails": 2.0,
            "desired_momentum": 0.5,
        },
    },
}


def _build_phase_sequence(phases: list[tuple[float, float]], horizon: int) -> list[float]:
    sequence: list[float] = []
    remaining = int(horizon)

    for idx, (fraction, value) in enumerate(phases):
        if idx == len(phases) - 1:
            count = remaining
        else:
            count = int(round(horizon * float(fraction)))
            count = max(0, min(count, remaining))
        sequence.extend([float(value)] * count)
        remaining -= count

    if len(sequence) < horizon:
        sequence.extend([sequence[-1] if sequence else 1.0] * (horizon - len(sequence)))
    elif len(sequence) > horizon:
        sequence = sequence[:horizon]

    return sequence


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
    if scenario_type not in SCENARIOS:
        available = ", ".join(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_type}'. Available: {available}")

    scenario = SCENARIOS[scenario_type]
    if "phases" in scenario:
        sequence = _build_phase_sequence(scenario["phases"], horizon)
    else:
        sequence = scenario["sequence"](horizon)

    # Ensure correct length
    if len(sequence) < horizon:
        sequence = sequence + [sequence[-1]] * (horizon - len(sequence))
    elif len(sequence) > horizon:
        sequence = sequence[:horizon]

    logger.info(f"Generated scenario: {scenario_type} - {scenario['description']}")

    return np.array(sequence), scenario["description"]


def list_scenarios():
    """Return list of available scenarios."""
    return list(SCENARIOS.keys())


def get_scenario_knobs(scenario_type: str) -> dict:
    """Return preset knob values for a scenario if provided."""
    if scenario_type not in SCENARIOS:
        available = ", ".join(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_type}'. Available: {available}")
    return dict(SCENARIOS[scenario_type].get("knobs", {}))
