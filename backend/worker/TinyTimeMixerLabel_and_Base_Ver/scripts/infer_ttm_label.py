"""
Inference for label-conditioned TTM.

Run:
  python infer_ttm_label.py --ticker AAPL --horizon 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.worker.TinyTimeMixerNew.services.ttm_label_dataset import (
    LabelScaler,
    StandardScaler,
    TTMLabelConfig,
)
from backend.worker.TinyTimeMixer.services.ttm_controlled_dataset import (
    build_base_features,
    download_daily_ohlcv,
    reconstruct_ohlcv_from_features,
)
from backend.worker.GARCH.services.validation_service import ValidationService

try:
    from tsfm_public.toolkit.time_series_preprocessor import DEFAULT_FREQUENCY_MAPPING
except Exception:
    DEFAULT_FREQUENCY_MAPPING = {
        "min": 1,
        "2min": 2,
        "5min": 3,
        "10min": 4,
        "15min": 5,
        "30min": 6,
        "h": 7,
        "H": 7,
        "d": 8,
        "D": 8,
        "W": 9,
    }


DEFAULT_TICKER = "AAPL"
DEFAULT_OUTPUT_CSV_TEMPLATE = (
    "backend/worker/TinyTimeMixerNew/outputs/ttm_label/{ticker}_synthetic.csv"
)
ANNUALIZE_METRICS = True
TRADING_DAYS = 252

# ----------------------------
# Editable defaults (controls)
# ----------------------------
DEFAULT_VOL_MULT = 2.0
DEFAULT_TREND_MULT = 1.0
DEFAULT_FAT_TAILS_MULT = 1.0
DEFAULT_MOMENTUM_MULT = 1.0


def extract_predictions(outputs) -> torch.Tensor:
    if hasattr(outputs, "predictions"):
        y_hat = outputs.predictions
    elif hasattr(outputs, "prediction_outputs"):
        y_hat = outputs.prediction_outputs
    elif hasattr(outputs, "logits"):
        y_hat = outputs.logits
    elif (
        isinstance(outputs, (tuple, list))
        and len(outputs) > 0
        and torch.is_tensor(outputs[0])
    ):
        y_hat = outputs[0]
    elif torch.is_tensor(outputs):
        y_hat = outputs
    else:
        raise RuntimeError(
            f"Could not extract predictions from outputs type: {type(outputs)}"
        )
    if not torch.is_tensor(y_hat):
        raise RuntimeError("Extracted predictions is not a tensor.")
    return y_hat


def maybe_fix_pred_shape(y_hat: torch.Tensor, num_channels: int) -> torch.Tensor:
    if y_hat.ndim != 3:
        raise RuntimeError(f"Expected 3D predictions, got {tuple(y_hat.shape)}")
    _, d1, d2 = y_hat.shape
    if d1 == num_channels and d2 != num_channels:
        return y_hat.transpose(1, 2).contiguous()
    return y_hat


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def load_label_scaler(path: Path) -> LabelScaler:
    data = json.loads(path.read_text())
    return LabelScaler(
        mean=np.array(data["mean"], dtype=np.float32),
        std=np.array(data["std"], dtype=np.float32),
        eps=float(data.get("eps", 1e-6)),
    )


def load_input_ohlcv(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    lookback_days: int,
    input_csv: Optional[Path],
) -> pd.DataFrame:
    if input_csv is None:
        if start is None or end is None:
            end_ts = pd.Timestamp.utcnow().normalize()
            start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
            start = start_ts.strftime("%Y-%m-%d")
            end = end_ts.strftime("%Y-%m-%d")
        return download_daily_ohlcv(ticker, start=start, end=end)

    df = pd.read_csv(input_csv)
    if "date" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "date"})
        else:
            raise ValueError("Input CSV missing a date column (date/Date/Datetime).")

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns {missing}.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=required)
    return df


def compute_metrics(
    df: pd.DataFrame,
    *,
    annualize: bool = False,
    trading_days: int = 252,
) -> Dict[str, float]:
    close = df["Close"].astype(float)
    ret = np.log(close / close.shift(1)).dropna().values
    if len(ret) < 3:
        return {
            "volatility": float("nan"),
            "trend": float("nan"),
            "fat_tails": float("nan"),
            "momentum": float("nan"),
        }

    vol = float(np.std(ret))
    trend = float(np.mean(ret))
    if annualize and trading_days > 0:
        vol = vol * float(np.sqrt(trading_days))
        trend = trend * float(trading_days)

    centered = ret - np.mean(ret)
    std = np.std(centered) + 1e-12
    kurt = float(np.mean((centered / std) ** 4) - 3.0)

    if len(ret) >= 2:
        r0 = ret[:-1]
        r1 = ret[1:]
        if np.std(r0) < 1e-12 or np.std(r1) < 1e-12:
            ac1 = 0.0
        else:
            ac1 = float(np.corrcoef(r0, r1)[0, 1])
    else:
        ac1 = 0.0

    momentum = max(0.0, ac1)

    return {
        "volatility": vol,
        "trend": trend,
        "fat_tails": kurt,
        "momentum": momentum,
    }


def plot_validation_bars(
    ax: plt.Axes,
    validation_metrics: Dict[str, float],
    *,
    title: str = "Validation Comparison",
) -> None:
    acf_ret_hist = validation_metrics.get(
        "acf_returns_historical", validation_metrics.get("acf_lag1_historical", np.nan)
    )
    acf_ret_synth = validation_metrics.get(
        "acf_returns_synthetic", validation_metrics.get("acf_lag1_synthetic", np.nan)
    )
    acf_vol_hist = validation_metrics.get("acf_volatility_historical", np.nan)
    acf_vol_synth = validation_metrics.get("acf_volatility_synthetic", np.nan)

    labels = ["kurtosis", "skewness", "acf_returns_lag1", "acf_vol_lag1"]
    hist = [
        validation_metrics["kurtosis_historical"],
        validation_metrics["skewness_historical"],
        acf_ret_hist,
        acf_vol_hist,
    ]
    synth = [
        validation_metrics["kurtosis_synthetic"],
        validation_metrics["skewness_synthetic"],
        acf_ret_synth,
        acf_vol_synth,
    ]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, hist, width, label="Input")
    ax.bar(x + width / 2, synth, width, label="Synthetic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_title(title)
    ax.legend()


def plot_return_distribution(
    ax: plt.Axes,
    hist_returns: np.ndarray,
    synth_returns: np.ndarray,
    *,
    title: str = "Return Distribution",
) -> None:
    if len(hist_returns) == 0 or len(synth_returns) == 0:
        ax.text(0.5, 0.5, "No returns available", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.hist(
        hist_returns * 100,
        bins=50,
        alpha=0.6,
        label="Input",
        density=True,
        color="black",
    )
    ax.hist(
        synth_returns * 100,
        bins=50,
        alpha=0.6,
        label="Synthetic",
        density=True,
        color="blue",
    )
    ax.set_xlabel("Daily Returns (%)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_nonlog_feature(
    input_feats: np.ndarray,
    pred_feats: np.ndarray,
    *,
    ax: plt.Axes,
    idx: int,
    title: str,
    y_label: str,
) -> None:
    x_in = np.arange(len(input_feats))
    x_pred = np.arange(len(pred_feats))

    if idx == 0:
        input_series = np.expm1(input_feats[:, 0])
        pred_series = np.expm1(pred_feats[:, 0])
    elif idx == 1:
        input_series = np.exp(input_feats[:, 1])
        pred_series = np.exp(pred_feats[:, 1])
    else:
        input_series = np.expm1(input_feats[:, 2])
        pred_series = np.expm1(pred_feats[:, 2])

    ax.plot(x_in, input_series, label="Input")
    ax.plot(x_pred, pred_series, label="Pred")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.legend(fontsize=8)


def compute_quality_scores(
    historical_returns: np.ndarray,
    synthetic_returns: np.ndarray,
    *,
    vol_mult: float,
    fat_tails_mult: float,
    momentum_mult: float,
) -> Dict[str, float]:
    hist = np.asarray(historical_returns).flatten()
    synth = np.asarray(synthetic_returns).flatten()

    hist = hist[np.isfinite(hist)]
    synth = synth[np.isfinite(synth)]

    if len(hist) < 100 or len(synth) < 100:
        return {
            "vol_score": 0.0,
            "kurtosis_score": 0.0,
            "momentum_score": 0.0,
            "distribution_score": 0.0,
            "total_score": 0.0,
        }

    synthetic_vol = float(np.std(synth) * np.sqrt(252))
    synthetic_kurtosis = float(stats.kurtosis(synth))

    if len(synth) > 1:
        try:
            synthetic_autocorr = float(np.corrcoef(synth[:-1], synth[1:])[0, 1])
            if not np.isfinite(synthetic_autocorr):
                synthetic_autocorr = 0.0
        except Exception:
            synthetic_autocorr = 0.0
    else:
        synthetic_autocorr = 0.0

    historical_vol = float(np.std(hist) * np.sqrt(252))
    historical_kurtosis = float(stats.kurtosis(hist))

    target_vol = historical_vol * float(vol_mult)
    target_kurtosis = historical_kurtosis * float(fat_tails_mult)
    target_autocorr = float(momentum_mult) - 0.5

    vol_error = abs(synthetic_vol - target_vol) / max(target_vol, 0.01)
    vol_score = max(0.0, 1.0 - vol_error)

    kurtosis_error = abs(synthetic_kurtosis - target_kurtosis) / max(
        abs(target_kurtosis), 3.0
    )
    kurtosis_score = max(0.0, 1.0 - kurtosis_error)

    momentum_error = abs(synthetic_autocorr - target_autocorr)
    momentum_score = max(0.0, 1.0 - momentum_error)

    try:
        ks_stat = stats.ks_2samp(hist, synth).statistic
        distribution_score = max(0.0, 1.0 - float(ks_stat))
    except Exception:
        distribution_score = 0.5

    total_score = (
        0.35 * vol_score
        + 0.25 * kurtosis_score
        + 0.20 * momentum_score
        + 0.20 * distribution_score
    )

    return {
        "vol_score": float(vol_score),
        "kurtosis_score": float(kurtosis_score),
        "momentum_score": float(momentum_score),
        "distribution_score": float(distribution_score),
        "total_score": float(np.clip(total_score, 0.0, 1.0)),
    }


def plot_quality_score(
    quality_scores: Dict[str, float],
    path: Path,
    *,
    title: str = "Synthetic Data Quality Score",
) -> None:
    labels = [
        "volatility",
        "kurtosis",
        "momentum",
        "distribution",
        "total",
    ]
    values = [
        quality_scores.get("vol_score", 0.0),
        quality_scores.get("kurtosis_score", 0.0),
        quality_scores.get("momentum_score", 0.0),
        quality_scores.get("distribution_score", 0.0),
        quality_scores.get("total_score", 0.0),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=["#444", "#666", "#888", "#aaa", "#1f77b4"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score (0-1)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, min(1.0, v + 0.03), f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_end_to_end_quality(
    total_score: float,
    path: Path,
    *,
    title: str = "End-to-End Synthetic Data Quality",
    target: float = 0.7,
    minimum: float = 0.5,
) -> None:
    score = float(np.clip(total_score, 0.0, 1.0))
    color = "red" if score < minimum else "orange" if score < target else "green"

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(["Quality\nScore"], [score], color=color, alpha=0.7)
    ax.axvline(
        x=target, color="green", linestyle="--", linewidth=2, label="Target (0.7)"
    )
    ax.axvline(
        x=minimum, color="orange", linestyle="--", linewidth=2, label="Minimum (0.5)"
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.text(score, 0, f"  {score:.3f}", va="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_end_to_end_quality_ax(
    ax: plt.Axes,
    total_score: float,
    *,
    title: str = "End-to-End Synthetic Data Quality",
    target: float = 0.7,
    minimum: float = 0.5,
) -> None:
    score = float(np.clip(total_score, 0.0, 1.0))
    color = "red" if score < minimum else "orange" if score < target else "green"

    ax.barh(["Quality\nScore"], [score], color=color, alpha=0.7)
    ax.axvline(
        x=target, color="green", linestyle="--", linewidth=2, label="Target (0.7)"
    )
    ax.axvline(
        x=minimum, color="orange", linestyle="--", linewidth=2, label="Minimum (0.5)"
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.text(score, 0, f"  {score:.3f}", va="center", fontsize=12, fontweight="bold")


def plot_all_charts(
    input_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    metrics_input: Dict[str, float],
    metrics_synth: Dict[str, float],
    validation_metrics: Dict[str, float],
    historical_returns: np.ndarray,
    synthetic_returns: np.ndarray,
    input_feats: np.ndarray,
    pred_feats: np.ndarray,
    total_quality: float,
    path: Path,
    *,
    metrics_title_suffix: str = "",
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.plot(synth_df["Date"], synth_df["Close"], label="Close", linewidth=1.5)
    ax.fill_between(
        synth_df["Date"],
        synth_df["Low"],
        synth_df["High"],
        alpha=0.2,
        label="High-Low Range",
    )
    ax.set_title("Synthetic OHLCV (Close + Range)")
    ax.legend()

    ax = axes[0, 1]
    plot_end_to_end_quality_ax(ax, total_quality)

    ax = axes[1, 0]
    ax.plot(np.arange(len(input_df)), input_df["Volume"].values, label="Input Volume")
    ax.plot(
        np.arange(len(synth_df)), synth_df["Volume"].values, label="Synthetic Volume"
    )
    ax.set_title("Volume (Input vs Synthetic, aligned by index)")
    ax.legend()

    ax = axes[1, 1]
    plot_nonlog_feature(
        input_feats,
        pred_feats,
        ax=ax,
        idx=0,
        title="Returns (Input vs Synthetic)",
        y_label="Return",
    )

    ax = axes[2, 0]
    plot_nonlog_feature(
        input_feats,
        pred_feats,
        ax=ax,
        idx=1,
        title="Range (Input vs Synthetic)",
        y_label="High/Low Ratio",
    )

    ax = axes[2, 1]
    plot_return_distribution(ax, historical_returns, synthetic_returns)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_context(
    df: pd.DataFrame,
    cfg: TTMLabelConfig,
    target_scaler: StandardScaler,
    exog_scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray, float, pd.Timestamp, float]:
    feats = build_base_features(df, cfg)
    if len(feats) < cfg.context_length:
        raise ValueError(
            f"Not enough rows for context_length={cfg.context_length}. "
            f"Have {len(feats)} rows after feature construction."
        )
    feats = feats.iloc[-cfg.context_length :].copy()

    past_raw = feats[["log_return", "log_range", "log_volume"]].values.astype(
        np.float32
    )
    past_exog = feats[["realized_vol"]].values.astype(np.float32)
    past_sigma = float(np.std(past_raw[:, 0]) + 1e-8)
    past_scaled = target_scaler.transform(past_raw)
    past_exog_scaled = exog_scaler.transform(past_exog)

    last_close = float(feats["Close"].iloc[-1])
    last_date = pd.to_datetime(feats["date"].iloc[-1])
    past_values = np.concatenate([past_scaled, past_exog_scaled], axis=1)
    return past_values, past_raw, last_close, last_date, past_sigma


def controls_from_multipliers(
    label_scaler: LabelScaler,
    *,
    vol_mult: float,
    trend_mult: float,
    fat_tails_mult: float,
    momentum_mult: float,
) -> np.ndarray:
    mean = label_scaler.mean
    std = label_scaler.std

    targets = np.zeros_like(mean)

    # volatility and fat_tails use mean multiplier (fallback to mean+std scaling)
    if mean[0] > 0:
        targets[0] = mean[0] * float(vol_mult)
    else:
        targets[0] = mean[0] + (float(vol_mult) - 1.0) * std[0]

    if mean[2] > 0:
        targets[2] = mean[2] * float(fat_tails_mult)
    else:
        targets[2] = mean[2] + (float(fat_tails_mult) - 1.0) * std[2]

    # trend and momentum use mean + std scaling
    targets[1] = mean[1] + (float(trend_mult) - 1.0) * std[1]
    targets[3] = mean[3] + (float(momentum_mult) - 1.0) * std[3]

    targets[3] = float(np.clip(targets[3], 0.0, 1.0))

    return label_scaler.normalize(targets.astype(np.float32))


@torch.no_grad()
def rollout_forecast(
    model: torch.nn.Module,
    past_scaled: np.ndarray,
    past_raw: np.ndarray,
    cfg: TTMLabelConfig,
    target_scaler: StandardScaler,
    exog_scaler: StandardScaler,
    controls_norm: np.ndarray,
    *,
    horizon: int,
    roll_step: int,
    device: str,
    freq_token_value: Optional[int] = None,
) -> np.ndarray:
    remaining = int(horizon)
    current = past_scaled[:, : cfg.num_target_features].copy()
    current_raw = past_raw.copy()
    outputs_scaled = []

    past_ctrl = np.repeat(controls_norm[None, :], cfg.context_length, axis=0)

    while remaining > 0:
        step = min(roll_step, remaining)
        returns_series = pd.Series(current_raw[:, 0])
        vol_window = int(cfg.realized_vol_window)
        realized_vol = returns_series.rolling(vol_window, min_periods=2).std().shift(1)
        realized_vol = realized_vol.bfill().fillna(0.0).values.astype(np.float32)
        past_exog_scaled = exog_scaler.transform(realized_vol.reshape(-1, 1))

        past_values = np.concatenate([current, past_exog_scaled, past_ctrl], axis=1)
        x = torch.tensor(past_values, dtype=torch.float32, device=device).unsqueeze(0)
        if freq_token_value is not None:
            freq_token = torch.full(
                (x.shape[0],), int(freq_token_value), device=device, dtype=torch.long
            )
            out = model(past_values=x, freq_token=freq_token)
        else:
            out = model(past_values=x)
        y_hat = extract_predictions(out)
        y_hat = maybe_fix_pred_shape(y_hat, num_channels=past_values.shape[-1])
        if y_hat.shape[1] != cfg.prediction_length:
            y_hat = y_hat[:, : cfg.prediction_length, :]

        y_hat = y_hat[:, :step, : cfg.num_target_features]
        pred_scaled = y_hat.squeeze(0).detach().cpu().numpy()
        outputs_scaled.append(pred_scaled)

        pred_raw = target_scaler.inverse_transform(pred_scaled)

        current = np.concatenate([current, pred_scaled], axis=0)[-cfg.context_length :]
        current_raw = np.concatenate([current_raw, pred_raw], axis=0)[
            -cfg.context_length :
        ]
        remaining -= step

    pred_scaled_all = np.concatenate(outputs_scaled, axis=0)
    pred_raw = target_scaler.inverse_transform(pred_scaled_all)
    return pred_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label-conditioned TTM inference.")
    parser.add_argument("--ticker", default=None, help="Ticker symbol (e.g., AAPL).")
    parser.add_argument(
        "--start", default=None, help="Start date (YYYY-MM-DD). Optional."
    )
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Optional.")
    parser.add_argument(
        "--lookback-days", type=int, default=365, help="Lookback days if start not set."
    )
    parser.add_argument(
        "--horizon", type=int, default=60, help="Forecast horizon (days)."
    )
    parser.add_argument(
        "--roll-step",
        type=int,
        default=1,
        help="Rollout step size (days). Use 1 for day-by-day generation.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV (synthetic OHLCV).",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Disable metrics/validation CSVs and charts.",
    )
    parser.add_argument(
        "--charts-dir",
        type=str,
        default=None,
        help="Optional charts output directory (defaults to <output_dir>/charts).",
    )
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument(
        "--weights", type=str, default=None, help="Path to finetuned weights."
    )
    parser.add_argument("--scaler", type=str, default=None, help="Path to scaler.")
    parser.add_argument("--label-scaler", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON.")

    parser.add_argument("--vol-mult", type=float, default=None)
    parser.add_argument("--trend-mult", type=float, default=None)
    parser.add_argument("--fat-tails-mult", type=float, default=None)
    parser.add_argument("--momentum-mult", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ticker:
        args.ticker = DEFAULT_TICKER
    if not args.output_csv:
        args.output_csv = DEFAULT_OUTPUT_CSV_TEMPLATE.format(
            ticker=str(args.ticker).lower()
        )

    default_dir = Path(__file__).resolve().parents[1] / "outputs" / "ttm_label"
    config_path = (
        Path(args.config) if args.config else (default_dir / "ttm_label_config.json")
    )
    weights_path = (
        Path(args.weights) if args.weights else (default_dir / "ttm_label_weights.pt")
    )
    scaler_path = (
        Path(args.scaler) if args.scaler else (default_dir / "ttm_label_scaler.pt")
    )
    label_scaler_path = (
        Path(args.label_scaler)
        if args.label_scaler
        else (default_dir / "ttm_label_label_scaler.json")
    )

    config = load_config(config_path)
    cfg = TTMLabelConfig(
        context_length=int(config["context_length"]),
        prediction_length=int(config["prediction_length"]),
    )
    cfg.detrend_returns = bool(config.get("detrend_returns", cfg.detrend_returns))
    cfg.detrend_window = int(config.get("detrend_window", cfg.detrend_window))
    cfg.detrend_mode = str(config.get("detrend_mode", cfg.detrend_mode))
    cfg.realized_vol_window = int(
        config.get("realized_vol_window", cfg.realized_vol_window)
    )

    scaler_state = torch.load(scaler_path, weights_only=False)
    if (
        isinstance(scaler_state, dict)
        and "targets" in scaler_state
        and "exog" in scaler_state
    ):
        target_scaler = StandardScaler.from_state_dict(scaler_state["targets"])
        exog_scaler = StandardScaler.from_state_dict(scaler_state["exog"])
    else:
        target_scaler = StandardScaler.from_state_dict(scaler_state)
        exog_scaler = StandardScaler(
            mean=np.array([0.0], dtype=np.float32),
            std=np.array([1.0], dtype=np.float32),
            eps=1e-6,
        )

    label_scaler = load_label_scaler(label_scaler_path)

    raw = load_input_ohlcv(
        args.ticker,
        start=args.start,
        end=args.end,
        lookback_days=args.lookback_days,
        input_csv=(Path(args.input_csv) if args.input_csv else None),
    )

    past_values, past_raw, last_close, last_date, _past_sigma = build_context(
        raw, cfg, target_scaler, exog_scaler
    )

    vol_mult = DEFAULT_VOL_MULT if args.vol_mult is None else float(args.vol_mult)
    trend_mult = (
        DEFAULT_TREND_MULT if args.trend_mult is None else float(args.trend_mult)
    )
    fat_tails_mult = (
        DEFAULT_FAT_TAILS_MULT
        if args.fat_tails_mult is None
        else float(args.fat_tails_mult)
    )
    momentum_mult = (
        DEFAULT_MOMENTUM_MULT
        if args.momentum_mult is None
        else float(args.momentum_mult)
    )

    controls_norm = controls_from_multipliers(
        label_scaler,
        vol_mult=vol_mult,
        trend_mult=trend_mult,
        fat_tails_mult=fat_tails_mult,
        momentum_mult=momentum_mult,
    )

    try:
        from tsfm_public.toolkit.get_model import get_model
    except Exception as e:
        raise ImportError("Failed to import get_model from tsfm_public.") from e

    model = get_model(
        config["model_id"],
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        freq=config.get("freq", "D"),
        freq_prefix_tuning=True,
        num_input_channels=cfg.num_input_channels,
        prediction_channel_indices=cfg.prediction_channel_indices,
        exogenous_channel_indices=cfg.exogenous_channel_indices,
        decoder_mode="mix_channel",
        scaling=None,
    )

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    use_freq_token = bool(getattr(model.config, "resolution_prefix_tuning", False))
    freq_str = config.get("freq", "D")
    freq_token_value = (
        DEFAULT_FREQUENCY_MAPPING.get(freq_str, None) if use_freq_token else None
    )

    pred_features = rollout_forecast(
        model,
        past_values,
        past_raw,
        cfg,
        target_scaler,
        exog_scaler,
        controls_norm,
        horizon=int(args.horizon),
        roll_step=int(args.roll_step),
        device=device,
        freq_token_value=freq_token_value,
    )

    synth_df = reconstruct_ohlcv_from_features(last_close, pred_features, last_date)

    out_path = Path(args.output_csv)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(out_path, index=False)
    print(f"Saved synthetic CSV: {out_path}")

    if args.no_charts:
        return

    charts_dir = Path(args.charts_dir) if args.charts_dir else (out_dir / "charts")
    charts_dir.mkdir(parents=True, exist_ok=True)

    input_df = raw.copy()
    input_df = input_df.rename(columns={"date": "Date"}).copy()
    if len(input_df) > len(synth_df):
        input_df = input_df.tail(len(synth_df)).reset_index(drop=True)

    try:
        input_for_metrics = pd.read_csv(args.input_csv) if args.input_csv else raw
    except Exception:
        input_for_metrics = raw

    metrics_title_suffix = (
        f"annualized {TRADING_DAYS}d" if ANNUALIZE_METRICS else "daily"
    )
    metrics_input = compute_metrics(
        input_for_metrics,
        annualize=ANNUALIZE_METRICS,
        trading_days=TRADING_DAYS,
    )
    metrics_synth = compute_metrics(
        synth_df,
        annualize=ANNUALIZE_METRICS,
        trading_days=TRADING_DAYS,
    )

    metrics_table = pd.DataFrame(
        [
            {"series": "input", **metrics_input},
            {"series": "synthetic", **metrics_synth},
        ]
    )
    metrics_csv = out_dir / f"{str(args.ticker).lower()}_metrics.csv"
    metrics_table.to_csv(metrics_csv, index=False)

    validation = ValidationService(input_for_metrics)
    synthetic_returns = synth_df["Close"].pct_change().dropna().values
    validation_metrics = validation.validate(synthetic_returns)
    validation_csv = out_dir / f"{str(args.ticker).lower()}_validation.csv"
    pd.DataFrame([validation_metrics]).to_csv(validation_csv, index=False)

    historical_returns = input_for_metrics["Close"].pct_change().dropna().values
    quality_scores = compute_quality_scores(
        historical_returns,
        synthetic_returns,
        vol_mult=vol_mult,
        fat_tails_mult=fat_tails_mult,
        momentum_mult=momentum_mult,
    )

    quality_score_chart = charts_dir / "synthetic_quality_score.png"
    e2e_quality_chart = charts_dir / "end_to_end_quality.png"
    plot_quality_score(quality_scores, quality_score_chart)
    plot_end_to_end_quality(quality_scores.get("total_score", 0.0), e2e_quality_chart)

    feats_for_compare = build_base_features(raw, cfg)
    input_feats = (
        feats_for_compare[["log_return", "log_range", "log_volume"]]
        .tail(len(pred_features))
        .values.astype(np.float32)
    )
    pred_feats = pred_features

    combined_chart = charts_dir / "all_charts.png"
    plot_all_charts(
        input_df,
        synth_df,
        metrics_input,
        metrics_synth,
        validation_metrics,
        historical_returns,
        synthetic_returns,
        input_feats,
        pred_feats,
        quality_scores.get("total_score", 0.0),
        combined_chart,
        metrics_title_suffix=metrics_title_suffix,
    )


if __name__ == "__main__":
    main()
