import numpy as np
import pandas as pd

# def compute_theta_from_knobs(user_knobs):
#     """
#     Compute theta (volatility-of-volatility) using a heuristic formula

#     theta controls how "jagged" the volatility path is:
#     - Higher theta → more volatile volatility (spiky paths)
#     - Lower theta → smoother volatility (gradual changes)
#     """

#     fat_tails = user_knobs['desired_fat_tails']
#     momentum = user_knobs['desired_momentum']

#     # Step 1: Base theta from fat_tails (primary driver)
#     # Log-linear relationship
#     base_theta = 10 ** (-3.7 + 1.6 * (fat_tails - 1.0))

#     # Step 2: Adjust for momentum (secondary driver)
#     momentum_factor = 0.7 + 0.6 * momentum

#     # Step 3: Combine
#     theta = base_theta * momentum_factor

#     # Step 4: Clip to valid range
#     theta = np.clip(theta, 1e-4, 1e-2)

#     return float(theta)


def compute_theta_hybrid(user_knobs, historical_returns):
    """
    Compute theta using theoretically-grounded approach
    """

    # Step 1: Estimate baseline theta from historical data
    historical_theta = calibrate_theta_from_data(historical_returns)

    # Step 2: Adjust for user's desired fat_tails
    desired_fat_tails = user_knobs["desired_fat_tails"]

    # Use square-root scaling (from Heston model)
    theta = historical_theta * np.sqrt(desired_fat_tails)

    # Step 3: Momentum adjustment (optional - weakly justified)
    # High momentum → expect smoother vol transitions
    momentum = user_knobs.get("desired_momentum", 0.5)
    if momentum > 0.8:
        # Reduce theta by up to 20% for very high momentum
        momentum_damping = 1.0 - 0.2 * (momentum - 0.8) / 0.2
        theta *= momentum_damping

    # Step 4: Clip to reasonable range
    # Upper bound 5e-2: allows high vol-of-vol needed to replicate
    # excess kurtosis > 6 in assets with very fat-tailed returns.
    theta = np.clip(theta, 1e-4, 5e-2)

    return float(theta)


def calibrate_theta_from_data(historical_returns):
    """
    Estimate theta from historical volatility-of-volatility
    """
    # Ensure 1D array for pandas Series
    historical_returns = np.asarray(historical_returns).ravel()

    # Compute rolling volatility
    window = 21  # 1 month
    rolling_vol = pd.Series(historical_returns).rolling(window).std()

    # Compute variance of volatility
    vol_of_vol = np.std(rolling_vol.dropna())

    # Approximate theta using method of moments
    # For Gamma distribution: Var(σ²) ≈ θ × E[σ²]
    mean_vol = np.mean(rolling_vol.dropna())
    theta = vol_of_vol / (mean_vol + 1e-12)

    return float(theta)
