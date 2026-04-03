"""
Data processing for DDPM Stage 1 & Stage 2.
Handles: download, returns, windowing, scaling, stress scoring.
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, List
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

# DDPM config will be imported at module level to avoid circular imports
# Import happens inside functions that need it


class DataProcessor:
    """
    Download market data, compute log returns, create rolling windows,
    and apply scaling/stress scoring.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.data = None
        self.returns = None
        self.scaler_mean = None
        self.scaler_std = None

    def download_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download daily OHLC from yfinance.

        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT', ...])
            start_date: Date string (YYYY-MM-DD)
            end_date: Date string (YYYY-MM-DD)

        Returns:
            DataFrame with columns [AAPL, MSFT, ...] and DatetimeIndex
        """
        logger.info(f"Downloading {tickers} from {start_date} to {end_date}")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)[
            "Adj Close"
        ]
        self.data = data
        logger.info(f"Downloaded {len(data)} days for {len(tickers)} assets")
        return data

    def compute_returns(self, data: pd.DataFrame = None, winsorize=True) -> pd.DataFrame:
        """
        Compute log returns and apply per-asset winsorization (±10σ).

        Args:
            data: DataFrame with prices (if None, use self.data)
            winsorize: Whether to winsorize at ±10 standard deviations

        Returns:
            DataFrame with log returns
        """
        if data is None:
            data = self.data
        assert data is not None, "No data provided. Call download_data first."

        returns = np.log(data / data.shift(1)).dropna()

        if winsorize:
            for col in returns.columns:
                mean = returns[col].mean()
                std = returns[col].std()
                lower = mean - 10 * std
                upper = mean + 10 * std
                returns[col] = returns[col].clip(lower, upper)

        self.returns = returns
        logger.info(f"Computed log returns: {returns.shape}")
        return returns

    def create_rolling_windows(
        self,
        returns: pd.DataFrame = None,
        window_len: int = 1260,
        stride: int = 5,
    ) -> np.ndarray:
        """
        Create rolling windows from returns (B, num_assets, window_len).

        Args:
            returns: DataFrame with log returns (if None, use self.returns)
            window_len: Window length (default 1260)
            stride: Step size between windows (default 5)

        Returns:
            Array of shape (num_windows, num_assets, window_len)
        """
        if returns is None:
            returns = self.returns
        assert returns is not None, "No returns provided. Call compute_returns first."

        num_assets = returns.shape[1]
        data_array = returns.values  # (time_steps, num_assets)
        windows = []

        for i in range(0, len(data_array) - window_len + 1, stride):
            window = data_array[i : i + window_len].T  # (num_assets, window_len)
            windows.append(window)

        windows = np.array(windows)  # (num_windows, num_assets, window_len)
        logger.info(f"Created {windows.shape[0]} rolling windows of length {window_len}")
        return windows

    def global_standardize(
        self, windows: np.ndarray, fit=True
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Global standardization: (X - mean) / std across all assets and windows.

        Args:
            windows: Array of shape (num_windows, num_assets, window_len)
            fit: If True, compute mean/std from windows (training)
                 If False, use stored self.scaler_mean/std (inference)

        Returns:
            Tuple of (standardized_windows, (mean, std))
        """
        if fit:
            # Compute global statistics across all data
            self.scaler_mean = windows.mean(axis=(0, 2), keepdims=True)  # (1, num_assets, 1)
            self.scaler_std = windows.std(axis=(0, 2), keepdims=True)  # (1, num_assets, 1)
            self.scaler_std = np.maximum(self.scaler_std, 1e-6)  # Avoid division by zero
        else:
            assert (
                self.scaler_mean is not None and self.scaler_std is not None
            ), "Scaler not fitted. Call with fit=True first."

        standardized = (windows - self.scaler_mean) / self.scaler_std
        logger.info(f"Standardized windows: mean={standardized.mean():.6f}, std={standardized.std():.6f}")
        return standardized, (self.scaler_mean, self.scaler_std)

    def stress_scoring(
        self, windows: np.ndarray, base_percentile: float = 0.95
    ) -> np.ndarray:
        """
        Compute exponential-weighted stress scores for regime identification.

        Stress is the negative of squared returns (volatility proxy).
        Exponential weights favor recent observations (decay towards past).

        Args:
            windows: Array of shape (num_windows, num_assets, window_len)
            base_percentile: Percentile for volatility threshold (default 0.95)

        Returns:
            Stress scores array of shape (num_windows,)
        """
        # Squared returns (volatility proxy)
        squared_returns = windows ** 2  # (num_windows, num_assets, window_len)

        # Compute percentile threshold
        threshold = np.percentile(squared_returns, base_percentile * 100)

        # Threshold: stress = 1 if squared_return > threshold, else 0
        stress_binary = (squared_returns > threshold).astype(float)  # (num_windows, num_assets, window_len)

        # Exponential weighting: recent observations have higher weight
        t = np.arange(stress_binary.shape[-1])  # Time indices [0, 1, ..., window_len-1]
        decay = np.exp(0.5 * t / stress_binary.shape[-1])  # Exponential decay
        decay = decay / decay.sum()  # Normalize to sum to 1

        # Apply exponential weighting: sum over assets and time
        stress_scores = (stress_binary * decay[None, None, :]).sum(axis=(1, 2))  # (num_windows,)

        logger.info(f"Computed stress scores: min={stress_scores.min():.4f}, max={stress_scores.max():.4f}")
        return stress_scores

    def to_tensor(self, windows: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor on device.

        Args:
            windows: Array of shape (num_windows, num_assets, window_len)

        Returns:
            Tensor on specified device
        """
        return torch.from_numpy(windows).float().to(self.device)
