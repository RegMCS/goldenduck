"""
Test inference + visualization for controlled TTM (daily).

This script:
  - lets you choose ticker + time range
  - selects prediction length (rolling if > model pred_len)
  - applies control parameters
  - generates OHLCV CSV
  - visualizes input vs synthetic
  - computes metrics for user-selected parameters
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict
import json
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.worker.TinyTimeMixer.services.ttm_controlled_dataset import (
    ControlRanges,
    ControlValues,
    StandardScaler,
    TTMControlledConfig,
    apply_inference_noise,
    build_base_features,
    download_daily_ohlcv,
    reconstruct_ohlcv_from_features,
    scale_controls,
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


# ----------------------------
# User-configurable section
# ----------------------------
TICKER = "KO"
INPUT_START = "2025-01-01"
INPUT_END = "2025-12-29"
INPUT_CSV = None  # Optional Path to OHLCV CSV used for inference

PREDICTION_LENGTH = 120
ROLL_STEP = 1
ANCHOR_BLEND = 0.25  # 0 = no anchoring, 1 = full match to first input close

CONTROLS = ControlValues(
    volatility_mult=1.0, #0.5-2.0
    trend= 0.0, 
    fat_tails=1.0,
    momentum=0.0,
    horizon=float(PREDICTION_LENGTH),
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "ttm_controlled_test"
OUTPUT_CSV = OUTPUT_DIR / f"{TICKER.lower()}_synthetic.csv"
CHART_DIR = OUTPUT_DIR / "charts"
METRICS_CSV = OUTPUT_DIR / f"{TICKER.lower()}_metrics.csv"
VALIDATION_CSV = OUTPUT_DIR / f"{TICKER.lower()}_validation.csv"
VALIDATION_CHART = CHART_DIR / "validation_compare.png"
COMBINED_CHART = CHART_DIR / "all_charts.png"
SAVE_INDIVIDUAL_CHARTS = False
ANNUALIZE_METRICS = True
TRADING_DAYS = 252


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


def load_input_ohlcv(
    ticker: str,
    start: str,
    end: str,
    input_csv: Optional[Path],
) -> pd.DataFrame:
    if input_csv is None:
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


def build_context(
    df: pd.DataFrame,
    cfg: TTMControlledConfig,
    scaler: StandardScaler,
    controls: ControlValues,
) -> Tuple[np.ndarray, float, pd.Timestamp, float]:
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
    past_sigma = float(np.std(past_raw[:, 0]) + 1e-8)
    past_scaled = scaler.transform(past_raw)

    ctrl_scaled = scale_controls(controls, cfg.control_ranges)
    past_ctrl = np.repeat(ctrl_scaled[None, :], cfg.context_length, axis=0)
    past_values = np.concatenate([past_scaled, past_ctrl], axis=1)

    last_close = float(feats["Close"].iloc[-1])
    last_date = pd.to_datetime(feats["date"].iloc[-1])
    return past_values, last_close, last_date, past_sigma


@torch.no_grad()
def rollout_forecast(
    model: torch.nn.Module,
    past_scaled: np.ndarray,
    cfg: TTMControlledConfig,
    scaler: StandardScaler,
    controls: ControlValues,
    *,
    horizon: int,
    roll_step: int,
    device: str,
    freq_token_value: Optional[int] = None,
) -> np.ndarray:
    ctrl_scaled = scale_controls(controls, cfg.control_ranges)
    remaining = int(horizon)
    current = past_scaled.copy()
    outputs_scaled = []

    while remaining > 0:
        step = min(roll_step, remaining)
        past_ctrl = np.repeat(ctrl_scaled[None, :], cfg.context_length, axis=0)
        past_values = np.concatenate([current, past_ctrl], axis=1)

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

        current = np.concatenate([current, pred_scaled], axis=0)[-cfg.context_length :]
        remaining -= step

    pred_scaled_all = np.concatenate(outputs_scaled, axis=0)
    pred_raw = scaler.inverse_transform(pred_scaled_all)
    return pred_raw


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


def plot_close_compare(
    input_df: pd.DataFrame, synth_df: pd.DataFrame, path: Path
) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(input_df)), input_df["Close"].values, label="Input Close")
    plt.plot(
        np.arange(len(synth_df)), synth_df["Close"].values, label="Synthetic Close"
    )
    plt.title("Close (Input vs Synthetic, aligned by index)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_ohlcv_synth(synth_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(synth_df["Date"], synth_df["Close"], label="Close", linewidth=1.5)
    plt.fill_between(
        synth_df["Date"],
        synth_df["Low"],
        synth_df["High"],
        alpha=0.2,
        label="High-Low Range",
    )
    plt.title("Synthetic OHLC (Close + Range)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_volume_compare(
    input_df: pd.DataFrame, synth_df: pd.DataFrame, path: Path
) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(input_df)), input_df["Volume"].values, label="Input Volume")
    plt.plot(
        np.arange(len(synth_df)), synth_df["Volume"].values, label="Synthetic Volume"
    )
    plt.title("Volume (Input vs Synthetic, aligned by index)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_metrics(
    metrics_input: Dict[str, float],
    metrics_synth: Dict[str, float],
    path: Path,
    *,
    title_suffix: str = "",
) -> None:
    labels = ["volatility", "trend", "fat_tails", "momentum"]
    x = np.arange(len(labels))
    inp = [metrics_input[k] for k in labels]
    syn = [metrics_synth[k] for k in labels]

    plt.figure(figsize=(10, 4))
    plt.bar(x - 0.2, inp, width=0.4, label="Input")
    plt.bar(x + 0.2, syn, width=0.4, label="Synthetic")
    plt.xticks(x, labels, rotation=15)
    title = "Metrics Comparison"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_validation(
    validation_metrics: Dict[str, float],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_validation_bars(ax, validation_metrics)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_validation_bars(
    ax: plt.Axes,
    validation_metrics: Dict[str, float],
    *,
    title: str = "Validation Comparison",
) -> None:
    labels = ["kurtosis", "skewness", "acf_lag1"]
    hist = [
        validation_metrics["kurtosis_historical"],
        validation_metrics["skewness_historical"],
        validation_metrics["acf_lag1_historical"],
    ]
    synth = [
        validation_metrics["kurtosis_synthetic"],
        validation_metrics["skewness_synthetic"],
        validation_metrics["acf_lag1_synthetic"],
    ]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, hist, width, label="Input")
    ax.bar(x + width / 2, synth, width, label="Synthetic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_title(title)
    ax.legend()


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


def plot_all_charts(
    input_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    metrics_input: Dict[str, float],
    metrics_synth: Dict[str, float],
    validation_metrics: Dict[str, float],
    input_feats: np.ndarray,
    pred_feats: np.ndarray,
    path: Path,
    *,
    metrics_title_suffix: str = "",
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 13))

    ax = axes[0, 0]
    ax.plot(np.arange(len(input_df)), input_df["Close"].values, label="Input Close")
    ax.plot(np.arange(len(synth_df)), synth_df["Close"].values, label="Synthetic Close")
    ax.set_title("Close (Input vs Synthetic, aligned by index)")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(np.arange(len(input_df)), input_df["Volume"].values, label="Input Volume")
    ax.plot(np.arange(len(synth_df)), synth_df["Volume"].values, label="Synthetic Volume")
    ax.set_title("Volume (Input vs Synthetic, aligned by index)")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(synth_df["Date"], synth_df["Close"], label="Close", linewidth=1.5)
    ax.fill_between(
        synth_df["Date"],
        synth_df["Low"],
        synth_df["High"],
        alpha=0.2,
        label="High-Low Range",
    )
    ax.set_title("Synthetic OHLC (Close + Range)")
    ax.legend()

    ax = axes[1, 1]
    labels = ["volatility", "trend", "fat_tails", "momentum"]
    x = np.arange(len(labels))
    inp = [metrics_input[k] for k in labels]
    syn = [metrics_synth[k] for k in labels]
    ax.bar(x - 0.2, inp, width=0.4, label="Input")
    ax.bar(x + 0.2, syn, width=0.4, label="Synthetic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    title = "Metrics Comparison"
    if metrics_title_suffix:
        title = f"{title} ({metrics_title_suffix})"
    ax.set_title(title)
    ax.legend()

    ax = axes[2, 0]
    plot_validation_bars(ax, validation_metrics)
    ax = axes[2, 1]
    plot_nonlog_feature(
        input_feats,
        pred_feats,
        ax=ax,
        idx=0,
        title="Raw (Non-Log) Return",
        y_label="Return",
    )

    ax = axes[3, 0]
    plot_nonlog_feature(
        input_feats,
        pred_feats,
        ax=ax,
        idx=1,
        title="Raw (Non-Log) Range Ratio",
        y_label="High/Low Ratio",
    )
    ax = axes[3, 1]
    plot_nonlog_feature(
        input_feats,
        pred_feats,
        ax=ax,
        idx=2,
        title="Raw (Non-Log) Volume",
        y_label="Volume",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    default_dir = Path(__file__).resolve().parents[1] / "outputs" / "ttm_controlled"
    config_path = default_dir / "ttm_controlled_config.json"
    weights_path = default_dir / "ttm_controlled_weights.pt"
    scaler_path = default_dir / "ttm_controlled_scaler.pt"

    config = load_config(config_path)
    cfg = TTMControlledConfig(
        context_length=int(config["context_length"]),
        prediction_length=int(config["prediction_length"]),
    )
    cfg.control_ranges = ControlRanges(**config["control_ranges"])
    cfg.detrend_returns = bool(config.get("detrend_returns", cfg.detrend_returns))
    cfg.detrend_window = int(config.get("detrend_window", cfg.detrend_window))
    cfg.detrend_mode = str(config.get("detrend_mode", cfg.detrend_mode))

    scaler_state = torch.load(scaler_path, weights_only=False)
    scaler = StandardScaler.from_state_dict(scaler_state)

    raw = load_input_ohlcv(
        TICKER,
        start=INPUT_START,
        end=INPUT_END,
        input_csv=(Path(INPUT_CSV) if INPUT_CSV else None),
    )
    input_df = raw.rename(columns={"date": "Date"}).copy()
    if len(input_df) > PREDICTION_LENGTH:
        input_df = input_df.tail(PREDICTION_LENGTH).reset_index(drop=True)

    past_values, last_close, last_date, past_sigma = build_context(
        raw, cfg, scaler, CONTROLS
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
        past_values[:, : cfg.num_target_features],
        cfg,
        scaler,
        CONTROLS,
        horizon=int(PREDICTION_LENGTH),
        roll_step=int(ROLL_STEP),
        device=device,
        freq_token_value=freq_token_value,
    )

    pred_features = apply_inference_noise(
        pred_features,
        sigma=past_sigma,
        controls=CONTROLS,
        cfg=cfg,
    )

    feats_for_compare = build_base_features(raw, cfg)
    input_feats = (
        feats_for_compare[["log_return", "log_range", "log_volume"]]
        .tail(len(pred_features))
        .values.astype(np.float32)
    )

    synth_df = reconstruct_ohlcv_from_features(last_close, pred_features, last_date)
    if not input_df.empty and not synth_df.empty:
        anchor_close = float(input_df["Close"].iloc[0])
        first_synth_close = float(synth_df["Close"].iloc[0])
        if abs(first_synth_close) > 1e-12 and ANCHOR_BLEND > 0:
            raw_scale = anchor_close / first_synth_close
            scale = 1.0 + ANCHOR_BLEND * (raw_scale - 1.0)
            price_cols = ["Open", "High", "Low", "Close"]
            synth_df[price_cols] = synth_df[price_cols] * scale

        anchor_date = pd.to_datetime(input_df["Date"].iloc[0])
        if len(input_df) == len(synth_df):
            synth_df["Date"] = input_df["Date"].values
        else:
            synth_df["Date"] = pd.bdate_range(start=anchor_date, periods=len(synth_df))
    synth_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved synthetic CSV: {OUTPUT_CSV}")

    try:
        input_for_metrics = pd.read_csv(INPUT_CSV) if INPUT_CSV else raw
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
    try:
        synth_for_metrics = pd.read_csv(OUTPUT_CSV)
    except Exception:
        synth_for_metrics = synth_df
    metrics_synth = compute_metrics(
        synth_for_metrics,
        annualize=ANNUALIZE_METRICS,
        trading_days=TRADING_DAYS,
    )

    metrics_table = pd.DataFrame(
        [
            {"series": "input", **metrics_input},
            {"series": "synthetic", **metrics_synth},
        ]
    )
    metrics_table.to_csv(METRICS_CSV, index=False)
    print(f"Saved metrics CSV: {METRICS_CSV}")

    validation = ValidationService(input_for_metrics)
    synthetic_returns = (
        pd.to_numeric(synth_for_metrics["Close"], errors="coerce")
        .pct_change()
        .dropna()
        .values
    )
    validation_metrics = validation.validate(synthetic_returns)
    pd.DataFrame([validation_metrics]).to_csv(VALIDATION_CSV, index=False)
    print(f"Saved validation CSV: {VALIDATION_CSV}")
    plot_validation(validation_metrics, VALIDATION_CHART)
    print(f"Saved validation chart: {VALIDATION_CHART}")

    plot_all_charts(
        input_df,
        synth_df,
        metrics_input,
        metrics_synth,
        validation_metrics,
        input_feats,
        pred_features,
        COMBINED_CHART,
        metrics_title_suffix=metrics_title_suffix,
    )
    if SAVE_INDIVIDUAL_CHARTS:
        plot_close_compare(input_df, synth_df, CHART_DIR / "close_compare.png")
        plot_volume_compare(input_df, synth_df, CHART_DIR / "volume_compare.png")
        plot_ohlcv_synth(synth_df, CHART_DIR / "ohlcv_synthetic.png")
        plot_metrics(
            metrics_input,
            metrics_synth,
            CHART_DIR / "metrics_compare.png",
            title_suffix=metrics_title_suffix,
        )
    print(f"Charts saved to: {CHART_DIR}")


if __name__ == "__main__":
    main()
