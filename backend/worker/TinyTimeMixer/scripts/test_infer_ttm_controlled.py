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
TICKER = "AAPL"
INPUT_START = "2023-01-01"
INPUT_END = "2023-12-31"

PREDICTION_LENGTH = 180
ROLL_STEP = 1

CONTROLS = ControlValues(
    volatility_mult=1.3,
    trend=0.1,
    fat_tails=1.2,
    momentum=0.25,
    mean_reversion=0.1,
    horizon=float(PREDICTION_LENGTH),
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "ttm_controlled_test"
OUTPUT_CSV = OUTPUT_DIR / f"{TICKER.lower()}_synthetic.csv"
CHART_DIR = OUTPUT_DIR / "charts"
METRICS_CSV = OUTPUT_DIR / f"{TICKER.lower()}_metrics.csv"


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


def build_context(
    df: pd.DataFrame,
    cfg: TTMControlledConfig,
    scaler: StandardScaler,
    controls: ControlValues,
) -> Tuple[np.ndarray, float, pd.Timestamp, float]:
    feats = build_base_features(df)
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


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    close = df["Close"].astype(float)
    ret = np.log(close / close.shift(1)).dropna().values
    if len(ret) < 3:
        return {
            "volatility": float("nan"),
            "trend": float("nan"),
            "fat_tails": float("nan"),
            "momentum": float("nan"),
            "mean_reversion": float("nan"),
        }

    vol = float(np.std(ret))
    trend = float(np.mean(ret))

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
    mean_rev = max(0.0, -ac1)

    return {
        "volatility": vol,
        "trend": trend,
        "fat_tails": kurt,
        "momentum": momentum,
        "mean_reversion": mean_rev,
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
    metrics_input: Dict[str, float], metrics_synth: Dict[str, float], path: Path
) -> None:
    labels = ["volatility", "trend", "fat_tails", "momentum", "mean_reversion"]
    x = np.arange(len(labels))
    inp = [metrics_input[k] for k in labels]
    syn = [metrics_synth[k] for k in labels]

    plt.figure(figsize=(10, 4))
    plt.bar(x - 0.2, inp, width=0.4, label="Input")
    plt.bar(x + 0.2, syn, width=0.4, label="Synthetic")
    plt.xticks(x, labels, rotation=15)
    plt.title("Metrics Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


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

    scaler_state = torch.load(scaler_path, weights_only=False)
    scaler = StandardScaler.from_state_dict(scaler_state)

    raw = download_daily_ohlcv(TICKER, start=INPUT_START, end=INPUT_END)
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

    synth_df = reconstruct_ohlcv_from_features(last_close, pred_features, last_date)
    synth_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved synthetic CSV: {OUTPUT_CSV}")

    input_df = raw.rename(columns={"date": "Date"}).copy()
    if len(input_df) > PREDICTION_LENGTH:
        input_df = input_df.tail(PREDICTION_LENGTH).reset_index(drop=True)

    metrics_input = compute_metrics(input_df)
    metrics_synth = compute_metrics(synth_df)

    metrics_table = pd.DataFrame(
        [
            {"series": "input", **metrics_input},
            {"series": "synthetic", **metrics_synth},
        ]
    )
    metrics_table.to_csv(METRICS_CSV, index=False)
    print(f"Saved metrics CSV: {METRICS_CSV}")

    plot_close_compare(input_df, synth_df, CHART_DIR / "close_compare.png")
    plot_volume_compare(input_df, synth_df, CHART_DIR / "volume_compare.png")
    plot_ohlcv_synth(synth_df, CHART_DIR / "ohlcv_synthetic.png")
    plot_metrics(metrics_input, metrics_synth, CHART_DIR / "metrics_compare.png")
    print(f"Charts saved to: {CHART_DIR}")


if __name__ == "__main__":
    main()
