"""
Label-conditioned TinyTimeMixer dataset (daily OHLCV).

Builds:
  - Base features: log_return, log_range, log_volume
  - Exogenous input: realized_vol
  - Control labels: future volatility, trend, fat_tails, momentum
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from backend.worker.TinyTimeMixer.services.ttm_controlled_dataset import (
    TARGET_FEATURES,
    EXOG_FEATURES,
    build_base_features,
    download_daily_ohlcv,
)

CONTROL_NAMES = [
    "volatility",
    "trend",
    "fat_tails",
    "momentum",
]


@dataclass
class TTMLabelConfig:
    context_length: int = 180
    prediction_length: int = 60
    detrend_returns: bool = True
    detrend_window: int = 20
    detrend_mode: str = "rolling"
    realized_vol_window: int = 20

    @property
    def num_target_features(self) -> int:
        return len(TARGET_FEATURES)

    @property
    def num_exogenous(self) -> int:
        return len(EXOG_FEATURES)

    @property
    def num_controls(self) -> int:
        return len(CONTROL_NAMES)

    @property
    def num_input_channels(self) -> int:
        return self.num_target_features + self.num_exogenous + self.num_controls

    @property
    def prediction_channel_indices(self) -> List[int]:
        return list(range(self.num_target_features))

    @property
    def exogenous_channel_indices(self) -> List[int]:
        start = self.num_target_features
        return list(range(start, start + self.num_exogenous + self.num_controls))

    @property
    def channel_names(self) -> List[str]:
        return TARGET_FEATURES + EXOG_FEATURES + CONTROL_NAMES


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + self.eps) + self.mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "mean": self.mean,
            "std": self.std,
            "eps": np.array(self.eps, dtype=np.float32),
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, np.ndarray]) -> "StandardScaler":
        return cls(mean=d["mean"], std=d["std"], eps=float(d.get("eps", 1e-6)))


@dataclass
class LabelScaler:
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def normalize(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / (self.std + self.eps)
        z = np.clip(z, -2.0, 2.0)
        return z / 2.0

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "mean": self.mean,
            "std": self.std,
            "eps": np.array(self.eps, dtype=np.float32),
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, np.ndarray]) -> "LabelScaler":
        return cls(mean=d["mean"], std=d["std"], eps=float(d.get("eps", 1e-6)))


@dataclass
class TimeSeriesBundle:
    ticker: str
    dates: np.ndarray
    features: np.ndarray  # [N, num_target_features]
    exog_features: np.ndarray  # [N, num_exogenous]
    close: np.ndarray


def fit_feature_scalers(
    series_list: Sequence[TimeSeriesBundle],
) -> Tuple[StandardScaler, StandardScaler]:
    if not series_list:
        raise ValueError("No series to fit scalers.")
    all_targets = np.concatenate([s.features for s in series_list], axis=0)
    all_exog = np.concatenate([s.exog_features for s in series_list], axis=0)

    targ_mean = all_targets.mean(axis=0)
    targ_std = all_targets.std(axis=0)
    targ_std = np.where(targ_std < 1e-8, 1.0, targ_std)

    exog_mean = all_exog.mean(axis=0)
    exog_std = all_exog.std(axis=0)
    exog_std = np.where(exog_std < 1e-8, 1.0, exog_std)

    return (
        StandardScaler(mean=targ_mean, std=targ_std, eps=1e-6),
        StandardScaler(mean=exog_mean, std=exog_std, eps=1e-6),
    )


def compute_label_stats(returns: np.ndarray) -> np.ndarray:
    if len(returns) < 3:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    vol = float(np.std(returns))
    trend = float(np.mean(returns))

    centered = returns - np.mean(returns)
    std = float(np.std(centered) + 1e-12)
    fat_tails = float(np.mean((centered / std) ** 4) - 3.0)

    if len(returns) >= 2:
        r0 = returns[:-1]
        r1 = returns[1:]
        if np.std(r0) < 1e-12 or np.std(r1) < 1e-12:
            ac1 = 0.0
        else:
            ac1 = float(np.corrcoef(r0, r1)[0, 1])
    else:
        ac1 = 0.0

    momentum = float(np.clip(ac1, 0.0, 1.0))
    return np.array([vol, trend, fat_tails, momentum], dtype=np.float32)


def build_label_scaler(label_stats: np.ndarray) -> LabelScaler:
    mean = label_stats.mean(axis=0)
    std = label_stats.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return LabelScaler(mean=mean, std=std, eps=1e-6)


class LabelWindowDataset(Dataset):
    def __init__(
        self,
        series_list: Sequence[TimeSeriesBundle],
        cfg: TTMLabelConfig,
        target_scaler: StandardScaler,
        exog_scaler: StandardScaler,
        *,
        label_scaler: Optional[LabelScaler] = None,
        target_date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        compute_weights: bool = False,
        weight_beta: float = 1.0,
        weight_min: float = 0.5,
        weight_max: float = 4.0,
    ) -> None:
        self.series_list = list(series_list)
        self.cfg = cfg
        self.target_scaler = target_scaler
        self.exog_scaler = exog_scaler
        self.target_date_range = target_date_range

        self._index: List[Tuple[int, int]] = []
        label_stats = []
        vol_stats = []

        for s_idx, series in enumerate(self.series_list):
            n = len(series.features)
            max_start = n - (cfg.context_length + cfg.prediction_length)
            if max_start <= 0:
                continue
            for start in range(max_start + 1):
                if self.target_date_range is not None:
                    mid = start + cfg.context_length
                    end = mid + cfg.prediction_length
                    end_date = pd.to_datetime(series.dates[end - 1])
                    if (
                        end_date < self.target_date_range[0]
                        or end_date > self.target_date_range[1]
                    ):
                        continue

                mid = start + cfg.context_length
                end = mid + cfg.prediction_length
                future_returns = series.features[mid:end, 0]
                stats = compute_label_stats(future_returns)
                label_stats.append(stats)
                vol_stats.append(float(stats[0]))
                self._index.append((s_idx, start))

        if not self._index:
            raise ValueError(
                "No usable windows found; check data length and context/pred lengths."
            )

        label_stats = np.asarray(label_stats, dtype=np.float32)
        self.label_scaler = label_scaler or build_label_scaler(label_stats)
        self._label_controls = self.label_scaler.normalize(label_stats)

        self._weights = None
        if compute_weights:
            vol_stats = np.asarray(vol_stats, dtype=np.float32)
            z = (vol_stats - self.label_scaler.mean[0]) / (
                self.label_scaler.std[0] + self.label_scaler.eps
            )
            z = np.clip(z, -2.0, 2.0)
            weights = np.exp(weight_beta * z)
            weights = np.clip(weights, weight_min, weight_max)
            self._weights = weights.astype(np.float32)

    def __len__(self) -> int:
        return len(self._index)

    def sampling_weights(self) -> Optional[np.ndarray]:
        return self._weights

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s_idx, start = self._index[idx]
        series = self.series_list[s_idx]
        mid = start + self.cfg.context_length
        end = mid + self.cfg.prediction_length

        past_raw = series.features[start:mid].copy()
        future_raw = series.features[mid:end].copy()
        past_exog = series.exog_features[start:mid].copy()

        past_scaled = self.target_scaler.transform(past_raw)
        future_scaled = self.target_scaler.transform(future_raw)
        past_exog_scaled = self.exog_scaler.transform(past_exog)

        ctrl = self._label_controls[idx]
        past_ctrl = np.repeat(ctrl[None, :], self.cfg.context_length, axis=0)

        past_values = np.concatenate([past_scaled, past_exog_scaled, past_ctrl], axis=1)
        future_targets = future_scaled.astype(np.float32)

        return torch.tensor(past_values, dtype=torch.float32), torch.tensor(
            future_targets, dtype=torch.float32
        )
