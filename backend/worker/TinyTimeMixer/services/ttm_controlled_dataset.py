"""
Controlled TinyTimeMixer (TTM) dataset + feature utilities for daily OHLCV.

This module builds:
  - Base features: log_return, log_range, log_volume
  - Control channels (exogenous): volatility, trend, fat_tails, momentum,
    mean_reversion, horizon

It also includes utilities to:
  - download daily OHLCV from yfinance (no backfill)
  - build sliding windows across multiple tickers
  - apply control transforms to future targets
  - reconstruct OHLCV from predicted features
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yfinance as yf


TARGET_FEATURES = ["log_return", "log_range", "log_volume"]
CONTROL_NAMES = [
    "volatility_mult",
    "trend",
    "fat_tails",
    "momentum",
    "mean_reversion",
    "horizon",
]


@dataclass
class ControlRanges:
    volatility_mult: Tuple[float, float] = (0.5, 2.0)
    trend: Tuple[float, float] = (-0.5, 0.5)  # scaled in units of sigma/day
    fat_tails: Tuple[float, float] = (0.5, 2.0)
    momentum: Tuple[float, float] = (0.0, 1.0)
    mean_reversion: Tuple[float, float] = (0.0, 1.0)
    horizon: Tuple[float, float] = (60.0, 500.0)


@dataclass
class ControlValues:
    volatility_mult: float = 1.0
    trend: float = 0.0
    fat_tails: float = 1.0
    momentum: float = 0.0
    mean_reversion: float = 0.0
    horizon: float = 60.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "volatility_mult": float(self.volatility_mult),
            "trend": float(self.trend),
            "fat_tails": float(self.fat_tails),
            "momentum": float(self.momentum),
            "mean_reversion": float(self.mean_reversion),
            "horizon": float(self.horizon),
        }


@dataclass
class TTMControlledConfig:
    context_length: int = 180
    prediction_length: int = 60
    control_ranges: ControlRanges = field(default_factory=ControlRanges)
    trend_sigma_scale: float = 0.1
    tail_clip_z: float = 3.0
    tail_range_coef: float = 0.25
    vol_volume_coef: float = 0.30

    @property
    def num_target_features(self) -> int:
        return len(TARGET_FEATURES)

    @property
    def num_controls(self) -> int:
        return len(CONTROL_NAMES)

    @property
    def num_input_channels(self) -> int:
        return self.num_target_features + self.num_controls

    @property
    def prediction_channel_indices(self) -> List[int]:
        return list(range(self.num_target_features))

    @property
    def exogenous_channel_indices(self) -> List[int]:
        start = self.num_target_features
        return list(range(start, start + self.num_controls))

    @property
    def channel_names(self) -> List[str]:
        return TARGET_FEATURES + CONTROL_NAMES


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
        return {"mean": self.mean, "std": self.std, "eps": np.array(self.eps, dtype=np.float32)}

    @classmethod
    def from_state_dict(cls, d: Dict[str, np.ndarray]) -> "StandardScaler":
        return cls(mean=d["mean"], std=d["std"], eps=float(d.get("eps", 1e-6)))


@dataclass
class TimeSeriesBundle:
    ticker: str
    dates: np.ndarray
    features: np.ndarray  # [N, 3] for TARGET_FEATURES
    close: np.ndarray


def download_daily_ohlcv(
    ticker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")

    # yfinance may return MultiIndex columns (field, ticker) even for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        level_values = df.columns.get_level_values(-1)
        if ticker in level_values:
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df = df.copy()
            df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "date"})
    else:
        df = df.rename(columns={df.columns[0]: "date"})

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker} missing columns {missing}.")

    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=required)
    return df


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out["log_range"] = np.log(out["High"] / out["Low"])
    out["log_volume"] = np.log1p(out["Volume"].astype(float))
    out = out.dropna().reset_index(drop=True)
    return out[["date", "Close"] + TARGET_FEATURES]


def split_by_year(
    df: pd.DataFrame,
    train_end_year: int,
    val_year: int,
    test_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    train = df[df["year"] <= train_end_year].copy()
    val = df[df["year"] == val_year].copy()
    test = df[df["year"] == test_year].copy()
    return train, val, test


def fit_feature_scaler(series_list: Sequence[TimeSeriesBundle]) -> StandardScaler:
    if not series_list:
        raise ValueError("No series to fit scaler.")
    all_feats = np.concatenate([s.features for s in series_list], axis=0)
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return StandardScaler(mean=mean, std=std, eps=1e-6)


def _scale_control(value: float, minv: float, maxv: float) -> float:
    if maxv <= minv:
        return 0.0
    return 2.0 * (value - minv) / (maxv - minv) - 1.0


def scale_controls(values: ControlValues, ranges: ControlRanges) -> np.ndarray:
    return np.array(
        [
            _scale_control(values.volatility_mult, *ranges.volatility_mult),
            _scale_control(values.trend, *ranges.trend),
            _scale_control(values.fat_tails, *ranges.fat_tails),
            _scale_control(values.momentum, *ranges.momentum),
            _scale_control(values.mean_reversion, *ranges.mean_reversion),
            _scale_control(values.horizon, *ranges.horizon),
        ],
        dtype=np.float32,
    )


def sample_controls(ranges: ControlRanges, rng: np.random.Generator) -> ControlValues:
    return ControlValues(
        volatility_mult=float(rng.uniform(*ranges.volatility_mult)),
        trend=float(rng.uniform(*ranges.trend)),
        fat_tails=float(rng.uniform(*ranges.fat_tails)),
        momentum=float(rng.uniform(*ranges.momentum)),
        mean_reversion=float(rng.uniform(*ranges.mean_reversion)),
        horizon=float(rng.uniform(*ranges.horizon)),
    )


def apply_controls_to_future(
    future_features: np.ndarray,
    past_returns: np.ndarray,
    controls: ControlValues,
    cfg: TTMControlledConfig,
) -> np.ndarray:
    out = future_features.copy()
    r = out[:, 0]
    log_range = out[:, 1]
    log_volume = out[:, 2]

    sigma = float(np.std(past_returns) + 1e-8)

    r = r * controls.volatility_mult
    r = r + controls.trend * cfg.trend_sigma_scale * sigma

    if controls.momentum > 0.0:
        r_mom = r.copy()
        prev = float(past_returns[-1])
        for i in range(len(r_mom)):
            r_mom[i] = (1.0 - controls.momentum) * r[i] + controls.momentum * prev
            prev = r_mom[i]
        r = r_mom

    if controls.mean_reversion > 0.0:
        r_mr = r.copy()
        prev = float(past_returns[-1])
        for i in range(len(r_mr)):
            r_mr[i] = (1.0 - controls.mean_reversion) * r[i] - controls.mean_reversion * prev
            prev = r_mr[i]
        r = r_mr

    tail_strength = controls.fat_tails - 1.0
    if abs(tail_strength) > 1e-6:
        z = r / sigma
        r = r * (1.0 + tail_strength * np.clip(np.abs(z), 0.0, cfg.tail_clip_z))

    if controls.volatility_mult > 0:
        log_range = log_range + np.log(controls.volatility_mult)
    log_range = log_range + cfg.tail_range_coef * (controls.fat_tails - 1.0)

    if controls.volatility_mult > 0:
        log_volume = log_volume + cfg.vol_volume_coef * np.log(controls.volatility_mult)

    out[:, 0] = r
    out[:, 1] = log_range
    out[:, 2] = log_volume
    return out


class ControlledWindowDataset(Dataset):
    def __init__(
        self,
        series_list: Sequence[TimeSeriesBundle],
        cfg: TTMControlledConfig,
        scaler: StandardScaler,
        *,
        control_mode: str = "random",
        fixed_controls: Optional[ControlValues] = None,
        target_date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        seed: int = 42,
    ) -> None:
        self.series_list = list(series_list)
        self.cfg = cfg
        self.scaler = scaler
        self.control_mode = control_mode
        self.fixed_controls = fixed_controls
        self.target_date_range = target_date_range
        self.rng = np.random.default_rng(seed)

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
                    if end_date < self.target_date_range[0] or end_date > self.target_date_range[1]:
                        continue
                self._index.append((s_idx, start))

        if not self._index:
            raise ValueError("No usable windows found; check data length and context/pred lengths.")

    def __len__(self) -> int:
        return len(self._index)

    def _get_controls(self) -> ControlValues:
        if self.control_mode == "fixed" and self.fixed_controls is not None:
            return self.fixed_controls
        return sample_controls(self.cfg.control_ranges, self.rng)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s_idx, start = self._index[idx]
        series = self.series_list[s_idx]
        mid = start + self.cfg.context_length
        end = mid + self.cfg.prediction_length

        past_raw = series.features[start:mid].copy()
        future_raw = series.features[mid:end].copy()

        controls = self._get_controls()
        future_raw = apply_controls_to_future(
            future_raw,
            past_raw[:, 0],
            controls,
            self.cfg,
        )

        past_scaled = self.scaler.transform(past_raw)
        future_scaled = self.scaler.transform(future_raw)

        ctrl_scaled = scale_controls(controls, self.cfg.control_ranges)
        past_ctrl = np.repeat(ctrl_scaled[None, :], self.cfg.context_length, axis=0)
        future_ctrl = np.repeat(ctrl_scaled[None, :], self.cfg.prediction_length, axis=0)

        past_values = np.concatenate([past_scaled, past_ctrl], axis=1)
        future_targets = future_scaled.astype(np.float32)

        return torch.tensor(past_values, dtype=torch.float32), torch.tensor(future_targets, dtype=torch.float32)


def reconstruct_ohlcv_from_features(
    last_close: float,
    pred_features: np.ndarray,
    start_date: datetime,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date) + pd.offsets.BDay(1)
    dates = pd.bdate_range(start=start, periods=len(pred_features))

    rows = []
    prev_close = float(last_close)
    for i in range(len(pred_features)):
        log_ret, log_range, log_vol = pred_features[i]

        close = prev_close * float(np.exp(log_ret))
        open_ = prev_close
        ratio = float(np.exp(log_range))
        base = float(np.sqrt(ratio))

        high = max(open_, close) * base
        low = min(open_, close) / base

        volume = float(np.expm1(log_vol))
        if volume < 0:
            volume = 0.0

        rows.append(
            {
                "Date": dates[i],
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )
        prev_close = close

    return pd.DataFrame(rows)
