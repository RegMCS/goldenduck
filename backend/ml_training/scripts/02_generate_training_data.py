"""
Generate 25,000 training samples (500 assets × 50 scenarios)
"""

import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import json

SCRIPT_DIR = Path(__file__).parent
ML_TRAINING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ML_TRAINING_DIR))
from config.training_config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    N_SCENARIOS_PER_ASSET,
)
from feature_extraction.catch22_extractor import extract_all_features
from parameter_optimization.grid_search import find_optimal_parameters


def main():
    # Load asset split
    with open(f"{PROCESSED_DATA_DIR}/assets_split.json", "r") as f:
        assets_split = json.load(f)

    train_assets = assets_split["train"]
    val_assets = assets_split["val"]
    test_assets = assets_split["test"]

    # Generate training samples
    train_samples = generate_samples_for_assets(train_assets, split="train")
    val_samples = generate_samples_for_assets(val_assets, split="val")
    test_samples = generate_samples_for_assets(test_assets, split="test")

    # Save
    with open(f"{PROCESSED_DATA_DIR}/train_samples.pkl", "wb") as f:
        pickle.dump(train_samples, f)

    with open(f"{PROCESSED_DATA_DIR}/val_samples.pkl", "wb") as f:
        pickle.dump(val_samples, f)

    with open(f"{PROCESSED_DATA_DIR}/test_samples.pkl", "wb") as f:
        pickle.dump(test_samples, f)

    print(f"✓ Generated {len(train_samples)} train samples")
    print(f"✓ Generated {len(val_samples)} val samples")
    print(f"✓ Generated {len(test_samples)} test samples")


def generate_samples_for_assets(asset_list, split="train"):
    samples = []

    for asset in tqdm(asset_list, desc=f"Generating {split} samples"):
        # Load historical data
        df = pd.read_csv(f"{RAW_DATA_DIR}/{asset}.csv")
        returns = df["returns"].values

        # Extract asset features
        asset_features = extract_all_features(returns)

        # Generate 50 scenarios
        for i in range(N_SCENARIOS_PER_ASSET):
            # Random user knobs
            user_knobs = {
                "desired_volatility": np.random.choice(
                    [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
                ),
                "desired_trend": np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0]),
                "desired_fat_tails": np.random.choice([0.8, 1.0, 1.2, 1.5]),
                "desired_momentum": np.random.choice([0.5, 0.75, 1.0, 1.25]),
            }

            # Find optimal parameters (ground truth)
            optimal_delta, optimal_theta = find_optimal_parameters(returns, user_knobs)

            # Create sample
            user_knob_features = np.array(list(user_knobs.values()))
            X = np.concatenate([asset_features, user_knob_features])

            samples.append(
                {
                    "X": X,
                    "y_delta": optimal_delta,
                    "y_theta": optimal_theta,
                    "asset": asset,
                    "scenario_id": i,
                }
            )

    return samples


if __name__ == "__main__":
    main()
