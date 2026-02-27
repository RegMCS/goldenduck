"""
Controlled TinyTimeMixer (TTM) dataset + feature utilities for daily OHLCV.

This module builds:
  - Base features: log_return, log_range, log_volume
  - Control channels (exogenous): volatility, trend, fat_tails, momentum,
    horizon

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

TARGET_FEATURES = ["log_price"]
EXOG_FEATURES = ["realized_vol"]
CONTROL_NAMES = [
    "volatility_mult",
    "trend",
    "fat_tails",
    "momentum",
    "horizon",
]


@dataclass
class ControlRanges:
    volatility_mult: Tuple[float, float] = (0.5, 2.0)
    trend: Tuple[float, float] = (-0.5, 0.5)  # scaled in units of sigma/day
    fat_tails: Tuple[float, float] = (0.5, 2.0)
    momentum: Tuple[float, float] = (0.0, 1.0)
    horizon: Tuple[float, float] = (60.0, 500.0)


@dataclass
class ControlValues:
    volatility_mult: float = 1.0
    trend: float = 0.0
    fat_tails: float = 1.0
    momentum: float = 0.0
    horizon: float = 60.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "volatility_mult": float(self.volatility_mult),
            "trend": float(self.trend),
            "fat_tails": float(self.fat_tails),
            "momentum": float(self.momentum),
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
    train_noise_enabled: bool = True
    infer_noise_enabled: bool = True
    noise_df_min: float = 3.0
    noise_df_max: float = 30.0
    train_return_noise_scale: float = 0.15
    train_range_noise_scale: float = 0.05
    train_volume_noise_scale: float = 0.05
    infer_return_noise_scale: float = 0.15
    infer_range_noise_scale: float = 0.05
    infer_volume_noise_scale: float = 0.05
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
class TimeSeriesBundle:
    ticker: str
    dates: np.ndarray
    features: np.ndarray  # [N, num_target_features]
    exog_features: np.ndarray  # [N, num_exogenous]
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


def build_base_features(
    df: pd.DataFrame, cfg: Optional[TTMControlledConfig] = None
) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out["log_price"] = np.log(out["Close"])

    if cfg is not None and cfg.detrend_returns:
        if cfg.detrend_mode == "rolling":
            roll = out["log_return"].rolling(cfg.detrend_window, min_periods=1).mean()
            out["log_return"] = out["log_return"] - roll
        elif cfg.detrend_mode == "mean":
            out["log_return"] = out["log_return"] - out["log_return"].mean()
        else:
            raise ValueError(
                f"Invalid detrend_mode={cfg.detrend_mode}. Use 'rolling' or 'mean'."
            )

    # Realized volatility (exogenous) computed from returns only (no leakage)
    vol_window = cfg.realized_vol_window if cfg is not None else 20
    vol_window = int(vol_window) if vol_window and vol_window > 0 else 20
    out["realized_vol"] = (
        out["log_return"].rolling(vol_window, min_periods=2).std().shift(1)
    )

    out = out.dropna().reset_index(drop=True)
    return out[["date", "Close"] + TARGET_FEATURES + EXOG_FEATURES]


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
    """
    Backward-compatible: fits scaler on target features only.
    """
    if not series_list:
        raise ValueError("No series to fit scaler.")
    all_feats = np.concatenate([s.features for s in series_list], axis=0)
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return StandardScaler(mean=mean, std=std, eps=1e-6)


def fit_feature_scalers(
    series_list: Sequence[TimeSeriesBundle],
    *,
    center_targets: bool = True,
    center_exog: bool = True,
) -> Tuple[StandardScaler, StandardScaler]:
    """
    Fit separate scalers for target and exogenous features.
    Set center_targets=False to preserve return drift (no mean subtraction).
    """
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

    if not center_targets:
        targ_mean = np.zeros_like(targ_mean)
    if not center_exog:
        exog_mean = np.zeros_like(exog_mean)

    return (
        StandardScaler(mean=targ_mean, std=targ_std, eps=1e-6),
        StandardScaler(mean=exog_mean, std=exog_std, eps=1e-6),
    )


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
        horizon=float(rng.uniform(*ranges.horizon)),
    )


def _fat_tail_df(
    fat_tails: float, ranges: ControlRanges, df_min: float, df_max: float
) -> float:
    lo, hi = ranges.fat_tails
    if hi <= lo:
        return df_max
    alpha = (fat_tails - lo) / (hi - lo)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return float(df_max - alpha * (df_max - df_min))


def _apply_feature_noise(
    features: np.ndarray,
    *,
    sigma: float,
    fat_tails: float,
    cfg: TTMControlledConfig,
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    if features.shape[1] == 1:
        df = _fat_tail_df(
            fat_tails, cfg.control_ranges, cfg.noise_df_min, cfg.noise_df_max
        )
        eps = rng.standard_t(df, size=(features.shape[0], 1)).astype(np.float32)
        r_scale = (
            cfg.train_return_noise_scale * sigma
            if mode == "train"
            else cfg.infer_return_noise_scale * sigma
        )
        out = features.copy()
        out[:, 0] = out[:, 0] + eps[:, 0] * r_scale
        return out

    df = _fat_tail_df(fat_tails, cfg.control_ranges, cfg.noise_df_min, cfg.noise_df_max)
    eps = rng.standard_t(df, size=(features.shape[0], 3)).astype(np.float32)

    if mode == "train":
        r_scale = cfg.train_return_noise_scale * sigma
        rg_scale = cfg.train_range_noise_scale
        v_scale = cfg.train_volume_noise_scale
    else:
        r_scale = cfg.infer_return_noise_scale * sigma
        rg_scale = cfg.infer_range_noise_scale
        v_scale = cfg.infer_volume_noise_scale

    out = features.copy()
    out[:, 0] = out[:, 0] + eps[:, 0] * r_scale
    out[:, 1] = out[:, 1] + eps[:, 1] * rg_scale
    out[:, 2] = out[:, 2] + eps[:, 2] * v_scale
    return out


def apply_controls_to_future(
    future_features: np.ndarray,
    past_returns: np.ndarray,
    controls: ControlValues,
    cfg: TTMControlledConfig,
    rng: Optional[np.random.Generator] = None,
    past_last_log_price: Optional[float] = None,
) -> np.ndarray:
    out = future_features.copy()
    if rng is None:
        rng = np.random.default_rng()

    if future_features.shape[1] == 1:
        if past_last_log_price is None:
            raise ValueError("past_last_log_price is required for log_price targets.")
        last_log_price = float(past_last_log_price)
        raw_returns = np.diff(
            np.concatenate([[last_log_price], future_features[:, 0]])
        )
        sigma = float(np.std(past_returns) + 1e-8)

        r = raw_returns * controls.volatility_mult
        r = r + controls.trend * cfg.trend_sigma_scale * sigma

        if controls.momentum > 0.0 and len(r) > 0:
            r_mom = r.copy()
            prev = float(past_returns[-1]) if len(past_returns) > 0 else 0.0
            for i in range(len(r_mom)):
                r_mom[i] = (1.0 - controls.momentum) * r[i] + controls.momentum * prev
                prev = r_mom[i]
            r = r_mom

        tail_strength = controls.fat_tails - 1.0
        if abs(tail_strength) > 1e-6 and sigma > 0:
            z = r / sigma
            r = r * (1.0 + tail_strength * np.clip(np.abs(z), 0.0, cfg.tail_clip_z))

        if cfg.train_noise_enabled:
            r = _apply_feature_noise(
                r.reshape(-1, 1),
                sigma=sigma,
                fat_tails=controls.fat_tails,
                cfg=cfg,
                rng=rng,
                mode="train",
            ).reshape(-1)

        log_price = last_log_price + np.cumsum(r)
        out[:, 0] = log_price
        return out

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

    if cfg.train_noise_enabled:
        out = _apply_feature_noise(
            out,
            sigma=sigma,
            fat_tails=controls.fat_tails,
            cfg=cfg,
            rng=rng,
            mode="train",
        )
    return out


def apply_inference_noise(
    pred_features: np.ndarray,
    *,
    sigma: float,
    controls: ControlValues,
    cfg: TTMControlledConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if not cfg.infer_noise_enabled:
        return pred_features
    if rng is None:
        rng = np.random.default_rng()
    return _apply_feature_noise(
        pred_features,
        sigma=sigma,
        fat_tails=controls.fat_tails,
        cfg=cfg,
        rng=rng,
        mode="infer",
    )


class ControlledWindowDataset(Dataset):
    def __init__(
        self,
        series_list: Sequence[TimeSeriesBundle],
        cfg: TTMControlledConfig,
        target_scaler: StandardScaler,
        exog_scaler: StandardScaler,
        *,
        control_mode: str = "random",
        fixed_controls: Optional[ControlValues] = None,
        target_date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
        allowed_years: Optional[Sequence[int]] = None,
        excluded_years: Optional[Sequence[int]] = None,
        seed: int = 42,
    ) -> None:
        self.series_list = list(series_list)
        self.cfg = cfg
        self.target_scaler = target_scaler
        self.exog_scaler = exog_scaler
        self.control_mode = control_mode
        self.fixed_controls = fixed_controls
        self.target_date_range = target_date_range
        self.allowed_years = (
            set(int(y) for y in allowed_years) if allowed_years is not None else None
        )
        self.excluded_years = (
            set(int(y) for y in excluded_years) if excluded_years is not None else None
        )
        self.rng = np.random.default_rng(seed)

        self._index: List[Tuple[int, int]] = []
        for s_idx, series in enumerate(self.series_list):
            n = len(series.features)
            max_start = n - (cfg.context_length + cfg.prediction_length)
            if max_start <= 0:
                continue
            for start in range(max_start + 1):
                mid = start + cfg.context_length
                end = mid + cfg.prediction_length

                end_date = None
                if (
                    self.target_date_range is not None
                    or self.allowed_years is not None
                    or self.excluded_years is not None
                ):
                    end_date = pd.to_datetime(series.dates[end - 1])

                if self.target_date_range is not None:
                    if (
                        end_date < self.target_date_range[0]
                        or end_date > self.target_date_range[1]
                    ):
                        continue

                if self.allowed_years is not None or self.excluded_years is not None:
                    start_year = int(pd.to_datetime(series.dates[start]).year)
                    end_year = int(end_date.year)
                    min_year = min(start_year, end_year)
                    max_year = max(start_year, end_year)

                    if self.allowed_years is not None:
                        allowed = self.allowed_years
                        if any(y not in allowed for y in range(min_year, max_year + 1)):
                            continue

                    if self.excluded_years is not None:
                        excluded = self.excluded_years
                        if any(y in excluded for y in range(min_year, max_year + 1)):
                            continue
                self._index.append((s_idx, start))

        if not self._index:
            raise ValueError(
                "No usable windows found; check data length and context/pred lengths."
            )

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
        past_exog = series.exog_features[start:mid].copy()

        controls = self._get_controls()
        if future_raw.shape[1] == 1:
            past_returns = np.diff(past_raw[:, 0])
            future_raw = apply_controls_to_future(
                future_raw,
                past_returns,
                controls,
                self.cfg,
                rng=self.rng,
                past_last_log_price=float(past_raw[-1, 0]),
            )
        else:
            future_raw = apply_controls_to_future(
                future_raw,
                past_raw[:, 0],
                controls,
                self.cfg,
                rng=self.rng,
            )

        past_scaled = self.target_scaler.transform(past_raw)
        future_scaled = self.target_scaler.transform(future_raw)
        past_exog_scaled = self.exog_scaler.transform(past_exog)

        ctrl_scaled = scale_controls(controls, self.cfg.control_ranges)
        past_ctrl = np.repeat(ctrl_scaled[None, :], self.cfg.context_length, axis=0)
        future_ctrl = np.repeat(
            ctrl_scaled[None, :], self.cfg.prediction_length, axis=0
        )

        past_values = np.concatenate([past_scaled, past_exog_scaled, past_ctrl], axis=1)
        future_targets = future_scaled.astype(np.float32)

        return torch.tensor(past_values, dtype=torch.float32), torch.tensor(
            future_targets, dtype=torch.float32
        )


def reconstruct_ohlcv_from_features(
    last_close: float,
    pred_features: np.ndarray,
    start_date: datetime,
) -> pd.DataFrame:
    if pred_features.shape[1] == 1:
        dates = pd.bdate_range(start=pd.Timestamp(start_date) + pd.offsets.BDay(1), periods=len(pred_features))
        log_price = pred_features[:, 0].astype(float)
        close = np.exp(log_price)
        rows = []
        prev_close = float(last_close)
        for i, c in enumerate(close):
            open_ = prev_close
            high = max(open_, float(c))
            low = min(open_, float(c))
            rows.append(
                {
                    "Date": dates[i],
                    "Open": open_,
                    "High": high,
                    "Low": low,
                    "Close": float(c),
                    "Volume": 0.0,
                }
            )
            prev_close = float(c)
        return pd.DataFrame(rows)

    start = pd.Timestamp(start_date) + pd.offsets.BDay(1)
    dates = pd.bdate_range(start=start, periods=len(pred_features))

    rows = []
    prev_close = float(last_close)
    for i in range(len(pred_features)):
        log_ret, log_range, log_vol = pred_features[i]

        close = prev_close * float(np.exp(log_ret))
        open_ = prev_close
        ratio = float(np.exp(log_range))
        if ratio < 1.0:
            ratio = 1.0

        hi_anchor = max(open_, close)
        lo_anchor = min(open_, close)
        skew = float(np.tanh(float(log_ret) * 5.0) * 0.25)
        high = hi_anchor * float(ratio ** (0.5 - skew))
        low = lo_anchor / float(ratio ** (0.5 + skew))

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
