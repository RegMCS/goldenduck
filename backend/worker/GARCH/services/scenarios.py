import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


FLASH_CRASH_TRIGGER_SCALE = 0.008
FLASH_CRASH_TRIGGER_MIN_STEPS = 5
FLASH_CRASH_TRIGGER_MAX_STEPS = 20


# Flash-crash schedules use mixed phase sizing:
# - fractions (<= 1.0) are portions of the horizon
# - integers (> 1.0) are fixed step counts
# This keeps the trigger as a true flash event regardless of horizon length.
FLASH_CRASH_DELTA_PHASES = [
    (0.55, 1.0),  # Calm
    (5, 3.5),     # Trigger (resolved dynamically at runtime)
    (0.15, 2.5),  # Stress
    (0.15, 1.5),  # Recovery
    (0.10, 1.1),  # Tail
]

# Absolute theta values by phase.
FLASH_CRASH_THETA_PHASES = [
    (0.55, 1e-5),  # Calm
    (5, 5e-4),     # Trigger (resolved dynamically at runtime)
    (0.15, 3e-4),  # Stress
    (0.15, 8e-5),  # Recovery
    (0.10, 1e-5),  # Tail
]

# Absolute daily drift values by phase.
FLASH_CRASH_DRIFT_PHASES = [
    (0.55, 0.0002),   # Calm
    (5, -0.0015),     # Trigger (resolved dynamically at runtime)
    (0.15, -0.0008),  # Stress
    (0.15, 0.0008),   # Recovery
    (0.10, 0.0002),   # Tail
]


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
        "description": "Brief extreme spike with quick recovery (delta scheduler)",
        "phases": FLASH_CRASH_DELTA_PHASES,
        # Preset knobs aligned to flash-crash behavior.
        "knobs": {
            "desired_trend": -0.2,
            "desired_volatility": 1.5,
            "desired_fat_tails": 2.0,
            "desired_momentum": 0.5,
        },
    },
}


def _build_phase_sequence(phases: list[tuple[float | int, float]], horizon: int) -> list[float]:
    sequence: list[float] = []
    remaining = int(horizon)

    for idx, (portion_or_steps, value) in enumerate(phases):
        if idx == len(phases) - 1:
            count = remaining
        else:
            amount = float(portion_or_steps)
            # Fractions are interpreted as horizon portions; values > 1 are fixed steps.
            if amount <= 1.0:
                count = int(round(horizon * amount))
            else:
                count = int(round(amount))
            count = max(0, min(count, remaining))
        sequence.extend([float(value)] * count)
        remaining -= count

    if len(sequence) < horizon:
        sequence.extend([sequence[-1] if sequence else 1.0] * (horizon - len(sequence)))
    elif len(sequence) > horizon:
        sequence = sequence[:horizon]

    return sequence


def _get_flash_crash_trigger_steps(horizon: int) -> int:
    """Scale trigger length with horizon while keeping it within [5, 20]."""
    return int(
        np.clip(
            float(horizon) * FLASH_CRASH_TRIGGER_SCALE,
            FLASH_CRASH_TRIGGER_MIN_STEPS,
            FLASH_CRASH_TRIGGER_MAX_STEPS,
        )
    )


def _resolve_flash_crash_trigger_steps(
    phases: list[tuple[float | int, float]], horizon: int
) -> list[tuple[float | int, float]]:
    """Return phases with trigger step count replaced by horizon-aware value."""
    resolved = list(phases)
    if len(resolved) >= 2:
        trigger_steps = _get_flash_crash_trigger_steps(horizon)
        resolved[1] = (trigger_steps, resolved[1][1])
    return resolved


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
        phases = scenario["phases"]
        if scenario_type == "flash_crash":
            phases = _resolve_flash_crash_trigger_steps(phases, horizon)
        sequence = _build_phase_sequence(phases, horizon)
    else:
        sequence = scenario["sequence"](horizon)

    # Ensure correct length
    if len(sequence) < horizon:
        sequence = sequence + [sequence[-1]] * (horizon - len(sequence))
    elif len(sequence) > horizon:
        sequence = sequence[:horizon]

    logger.info(f"Generated scenario: {scenario_type} - {scenario['description']}")

    return np.array(sequence), scenario["description"]


def get_flash_crash_delta_schedule(horizon: int) -> np.ndarray:
    """Return the flash-crash delta schedule."""
    phases = _resolve_flash_crash_trigger_steps(FLASH_CRASH_DELTA_PHASES, horizon)
    return np.array(_build_phase_sequence(phases, horizon), dtype=float)


def get_flash_crash_theta_schedule(horizon: int) -> np.ndarray:
    """Return the flash-crash absolute theta schedule."""
    phases = _resolve_flash_crash_trigger_steps(FLASH_CRASH_THETA_PHASES, horizon)
    return np.array(_build_phase_sequence(phases, horizon), dtype=float)


def get_flash_crash_drift_schedule(horizon: int) -> np.ndarray:
    """Return the flash-crash absolute daily drift schedule."""
    phases = _resolve_flash_crash_trigger_steps(FLASH_CRASH_DRIFT_PHASES, horizon)
    return np.array(_build_phase_sequence(phases, horizon), dtype=float)


def list_scenarios():
    """Return list of available scenarios."""
    return list(SCENARIOS.keys())


def get_scenario_knobs(scenario_type: str) -> dict:
    """Return preset knob values for a scenario if provided."""
    if scenario_type not in SCENARIOS:
        available = ", ".join(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_type}'. Available: {available}")
    return dict(SCENARIOS[scenario_type].get("knobs", {}))
