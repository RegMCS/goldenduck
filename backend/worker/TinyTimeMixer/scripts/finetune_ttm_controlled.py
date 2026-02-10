"""
Fine-tune TTM (daily) with controllable synthetic parameters for OHLCV generation.

Outputs:
  - weights: outputs/ttm_controlled/ttm_controlled_weights.pt
  - scaler:  outputs/ttm_controlled/ttm_controlled_scaler.pt
  - config:  outputs/ttm_controlled/ttm_controlled_config.json

Run:
  python finetune_ttm_controlled.py
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

# optional: map frequency string to token id for prefix tuning
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

# local imports
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.worker.GARCH.config.tickers import US_TICKERS, INDEX_TICKERS
from backend.worker.TinyTimeMixer.services.ttm_controlled_dataset import (
    CONTROL_NAMES,
    TARGET_FEATURES,
    ControlValues,
    TTMControlledConfig,
    TimeSeriesBundle,
    build_base_features,
    download_daily_ohlcv,
    fit_feature_scaler,
    ControlledWindowDataset,
)

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
FREQ = "D"

CONTEXT_LEN = 180
PRED_LEN = 60

START_DATE = "2000-01-01"
END_DATE = "2026-01-01"  # end-exclusive, includes all of 2025

TRAIN_END_YEAR = 2023
VAL_YEAR = 2024
TEST_YEAR = 2025

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
LOG_EVERY_N_BATCHES = 200

FULL_FINETUNE = False 
UNFREEZE_KEYWORDS = [
    "head",
    "decoder",
    "prediction",
    "pred",
    "proj",
    "output",
    "lm_head",
]

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "ttm_controlled"
WEIGHTS_PATH = OUTPUT_DIR / "ttm_controlled_weights.pt"
SCALER_PATH = OUTPUT_DIR / "ttm_controlled_scaler.pt"
CONFIG_PATH = OUTPUT_DIR / "ttm_controlled_config.json"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def freeze_for_fewshot(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in UNFREEZE_KEYWORDS):
            p.requires_grad = True


def count_trainable_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def to_bundle(ticker: str, df) -> TimeSeriesBundle:
    features = df[["log_return", "log_range", "log_volume"]].values.astype(np.float32)
    dates = df["date"].values
    close = df["Close"].values.astype(np.float32)
    return TimeSeriesBundle(ticker=ticker, dates=dates, features=features, close=close)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    *,
    device: str,
    pred_len: int,
    freq_token_value: Optional[int] = None,
    log_every: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_batches = len(loader)
    for batch_idx, (past, future_tgt) in enumerate(loader, start=1):
        past = past.to(device)
        future_tgt = future_tgt.to(device)

        freq_token = None
        if freq_token_value is not None:
            freq_token = torch.full(
                (past.shape[0],), int(freq_token_value), device=device, dtype=torch.long
            )
        if freq_token is not None:
            outputs = model(past_values=past, freq_token=freq_token)
        else:
            outputs = model(past_values=past)
        y_hat = extract_predictions(outputs)
        y_hat = maybe_fix_pred_shape(y_hat, num_channels=past.shape[-1])

        if y_hat.shape[1] != pred_len:
            y_hat = y_hat[:, :pred_len, :]

        y_hat_tgt = y_hat[:, :, : future_tgt.shape[-1]]
        loss = loss_fn(y_hat_tgt, future_tgt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=GRAD_CLIP_NORM,
            )
        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1

        if log_every and (batch_idx % log_every == 0 or batch_idx == total_batches):
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  batch {batch_idx:04d}/{total_batches} | avg_loss={avg_loss:.6f}")
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    *,
    device: str,
    pred_len: int,
    freq_token_value: Optional[int] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for past, future_tgt in loader:
        past = past.to(device)
        future_tgt = future_tgt.to(device)

        freq_token = None
        if freq_token_value is not None:
            freq_token = torch.full(
                (past.shape[0],), int(freq_token_value), device=device, dtype=torch.long
            )
        if freq_token is not None:
            outputs = model(past_values=past, freq_token=freq_token)
        else:
            outputs = model(past_values=past)
        y_hat = extract_predictions(outputs)
        y_hat = maybe_fix_pred_shape(y_hat, num_channels=past.shape[-1])

        if y_hat.shape[1] != pred_len:
            y_hat = y_hat[:, :pred_len, :]

        y_hat_tgt = y_hat[:, :, : future_tgt.shape[-1]]
        loss = loss_fn(y_hat_tgt, future_tgt)

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = list(US_TICKERS) + list(INDEX_TICKERS)

    cfg = TTMControlledConfig(context_length=CONTEXT_LEN, prediction_length=PRED_LEN)

    all_series = []

    print(f"Tickers: {len(tickers)}")
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        raw = download_daily_ohlcv(ticker, start=START_DATE, end=END_DATE)
        print(f"  rows downloaded: {len(raw)}")
        feats = build_base_features(raw, cfg)
        print(f"  rows after features: {len(feats)}")
        if len(feats) < (CONTEXT_LEN + PRED_LEN + 1):
            print(f"[skip] {ticker}: not enough rows ({len(feats)}).")
            continue
        all_series.append(to_bundle(ticker, feats))

    if not all_series:
        raise RuntimeError("No training series available after filtering.")

    scaler = fit_feature_scaler(all_series)
    torch.save(scaler.state_dict(), SCALER_PATH)

    fixed_controls = ControlValues(
        volatility_mult=1.0,
        trend=0.0,
        fat_tails=1.0,
        momentum=0.0,
        mean_reversion=0.0,
        horizon=float(PRED_LEN),
    )

    train_range = (
        np.datetime64("2000-01-01"),
        np.datetime64(f"{TRAIN_END_YEAR}-12-31"),
    )
    val_range = (np.datetime64(f"{VAL_YEAR}-01-01"), np.datetime64(f"{VAL_YEAR}-12-31"))
    test_range = (
        np.datetime64(f"{TEST_YEAR}-01-01"),
        np.datetime64(f"{TEST_YEAR}-12-31"),
    )

    train_ds = ControlledWindowDataset(
        all_series,
        cfg,
        scaler,
        control_mode="random",
        target_date_range=train_range,
        seed=SEED,
    )
    val_ds = ControlledWindowDataset(
        all_series,
        cfg,
        scaler,
        control_mode="fixed",
        fixed_controls=fixed_controls,
        target_date_range=val_range,
        seed=SEED,
    )
    test_ds = ControlledWindowDataset(
        all_series,
        cfg,
        scaler,
        control_mode="fixed",
        fixed_controls=fixed_controls,
        target_date_range=test_range,
        seed=SEED,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    print(
        f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)} | Test windows: {len(test_ds)}"
    )
    print(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}"
    )

    try:
        from tsfm_public.toolkit.get_model import get_model
    except Exception as e:
        raise ImportError("Failed to import get_model from tsfm_public.") from e

    model = get_model(
        MODEL_ID,
        context_length=CONTEXT_LEN,
        prediction_length=PRED_LEN,
        freq=FREQ,
        freq_prefix_tuning=True,
        num_input_channels=cfg.num_input_channels,
        prediction_channel_indices=cfg.prediction_channel_indices,
        exogenous_channel_indices=cfg.exogenous_channel_indices,
        decoder_mode="mix_channel",
        scaling=None,
    )

    model = model.to(DEVICE)

    if not FULL_FINETUNE:
        freeze_for_fewshot(model)

    trainable, total = count_trainable_params(model)
    print(
        f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.4f}%)"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = torch.nn.HuberLoss(delta=1.0)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    use_freq_token = bool(getattr(model.config, "resolution_prefix_tuning", False))
    freq_token_value = (
        DEFAULT_FREQUENCY_MAPPING.get(FREQ, None) if use_freq_token else None
    )
    if use_freq_token and freq_token_value is None:
        raise ValueError(f"Frequency token not found for freq={FREQ}.")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch:02d} starting...")
        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=DEVICE,
            pred_len=PRED_LEN,
            freq_token_value=freq_token_value,
            log_every=LOG_EVERY_N_BATCHES,
        )
        va_loss = eval_one_epoch(
            model,
            val_loader,
            loss_fn,
            device=DEVICE,
            pred_len=PRED_LEN,
            freq_token_value=freq_token_value,
        )
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        torch.save(best_state, WEIGHTS_PATH)
        model.load_state_dict(best_state)
        print(f"Saved best weights: {WEIGHTS_PATH} (val_loss={best_val:.6f})")

    te_loss = eval_one_epoch(
        model,
        test_loader,
        loss_fn,
        device=DEVICE,
        pred_len=PRED_LEN,
        freq_token_value=freq_token_value,
    )
    print(f"Test loss: {te_loss:.6f}")

    config_payload = {
        "model_id": MODEL_ID,
        "freq": FREQ,
        "context_length": CONTEXT_LEN,
        "prediction_length": PRED_LEN,
        "channels": cfg.channel_names,
        "target_features": TARGET_FEATURES,
        "controls": CONTROL_NAMES,
        "control_ranges": asdict(cfg.control_ranges),
        "trend_sigma_scale": cfg.trend_sigma_scale,
        "tail_clip_z": cfg.tail_clip_z,
        "tail_range_coef": cfg.tail_range_coef,
        "vol_volume_coef": cfg.vol_volume_coef,
        "train_noise_enabled": cfg.train_noise_enabled,
        "infer_noise_enabled": cfg.infer_noise_enabled,
        "noise_df_min": cfg.noise_df_min,
        "noise_df_max": cfg.noise_df_max,
        "train_return_noise_scale": cfg.train_return_noise_scale,
        "train_range_noise_scale": cfg.train_range_noise_scale,
        "train_volume_noise_scale": cfg.train_volume_noise_scale,
        "infer_return_noise_scale": cfg.infer_return_noise_scale,
        "infer_range_noise_scale": cfg.infer_range_noise_scale,
        "infer_volume_noise_scale": cfg.infer_volume_noise_scale,
        "detrend_returns": cfg.detrend_returns,
        "detrend_window": cfg.detrend_window,
        "detrend_mode": cfg.detrend_mode,
        "train_end_year": TRAIN_END_YEAR,
        "val_year": VAL_YEAR,
        "test_year": TEST_YEAR,
    }
    CONFIG_PATH.write_text(json.dumps(config_payload, indent=2))
    print(f"Saved config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
