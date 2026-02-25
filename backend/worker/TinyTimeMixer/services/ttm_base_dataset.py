"""
Base TinyTimeMixer dataset for daily OHLCV without control channels.

Builds:
  - Base features: log_return, log_range, log_volume
  - Exogenous input: realized_vol
  - Sliding windows of past -> future targets (no synthetic control edits)
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class TTMBaseConfig:
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
    def num_input_channels(self) -> int:
        return self.num_target_features + self.num_exogenous

    @property
    def prediction_channel_indices(self) -> List[int]:
        return list(range(self.num_target_features))

    @property
    def exogenous_channel_indices(self) -> List[int]:
        start = self.num_target_features
        return list(range(start, start + self.num_exogenous))

    @property
    def channel_names(self) -> List[str]:
        return list(TARGET_FEATURES) + list(EXOG_FEATURES)


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
class TimeSeriesBundle:
    ticker: str
    dates: np.ndarray
    features: np.ndarray  # [N, num_target_features]
    exog_features: np.ndarray  # [N, num_exogenous]
    close: np.ndarray


def fit_feature_scaler(series_list: Sequence[TimeSeriesBundle]) -> StandardScaler:
    if not series_list:
        raise ValueError("No series to fit scaler.")
    all_feats = np.concatenate([s.features for s in series_list], axis=0)
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return StandardScaler(mean=mean, std=std, eps=1e-6)


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


class BaseWindowDataset(Dataset):
    def __init__(
        self,
        series_list: Sequence[TimeSeriesBundle],
        cfg: TTMBaseConfig,
        target_scaler: StandardScaler,
        exog_scaler: StandardScaler,
        *,
        target_date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ) -> None:
        self.series_list = list(series_list)
        self.cfg = cfg
        self.target_scaler = target_scaler
        self.exog_scaler = exog_scaler
        self.target_date_range = target_date_range

        self._index: List[Tuple[int, int]] = []
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
                self._index.append((s_idx, start))

        if not self._index:
            raise ValueError(
                "No usable windows found; check data length and context/pred lengths."
            )

    def __len__(self) -> int:
        return len(self._index)

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

        past_values = np.concatenate([past_scaled, past_exog_scaled], axis=1).astype(
            np.float32
        )
        future_targets = future_scaled.astype(np.float32)

        return torch.tensor(past_values, dtype=torch.float32), torch.tensor(
            future_targets, dtype=torch.float32
        )
