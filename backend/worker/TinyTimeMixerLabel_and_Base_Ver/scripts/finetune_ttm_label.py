"""
Fine-tune TTM with label conditioning (future stats as controls).

Outputs:
  - weights: outputs/ttm_label/ttm_label_weights.pt
  - scaler:  outputs/ttm_label/ttm_label_scaler.pt
  - label:   outputs/ttm_label/ttm_label_label_scaler.json
  - config:  outputs/ttm_label/ttm_label_config.json

Run:
  python finetune_ttm_label.py
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

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
from backend.worker.TinyTimeMixerNew.services.ttm_label_dataset import (
    CONTROL_NAMES,
    TARGET_FEATURES,
    EXOG_FEATURES,
    LabelWindowDataset,
    LabelScaler,
    StandardScaler,
    TTMLabelConfig,
    TimeSeriesBundle,
    build_base_features,
    download_daily_ohlcv,
    fit_feature_scalers,
)

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
FREQ = "D"

CONTEXT_LEN = 180
PRED_LEN = 60
DETREND_RETURNS = False

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

# Loss config (Student-t NLL)
RETURN_DF = 5.0
RETURN_SCALE = 1.0

# Aux loss
LAMBDA_AUX_MAX = 0.1
LAMBDA_WARMUP_EPOCHS = 2

# Weighted sampler
SAMPLE_BETA = 1.0
SAMPLE_MIN = 0.5
SAMPLE_MAX = 4.0

# Full finetune disabled: only head/decoder/prediction layers are trainable.
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

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "ttm_label"
WEIGHTS_PATH = OUTPUT_DIR / "ttm_label_weights.pt"
SCALER_PATH = OUTPUT_DIR / "ttm_label_scaler.pt"
LABEL_SCALER_PATH = OUTPUT_DIR / "ttm_label_label_scaler.json"
CONFIG_PATH = OUTPUT_DIR / "ttm_label_config.json"


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


def student_t_nll(
    err: torch.Tensor,
    *,
    df: float,
    scale: float,
) -> torch.Tensor:
    if df <= 2.0:
        raise ValueError("RETURN_DF must be > 2 for stable Student-t variance.")
    scale = float(scale)
    df = float(df)
    return 0.5 * (df + 1.0) * torch.log1p((err**2) / (df * scale * scale)) + np.log(
        scale
    )


class StudentTLoss:
    def __init__(self, *, df: float, scale: float) -> None:
        self.df = float(df)
        self.scale = float(scale)

    def __call__(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        err = y_hat - y_true
        return student_t_nll(err, df=self.df, scale=self.scale).mean()


def to_bundle(ticker: str, df) -> TimeSeriesBundle:
    features = df[TARGET_FEATURES].values.astype(np.float32)
    exog_features = df[EXOG_FEATURES].values.astype(np.float32)
    dates = df["date"].values
    close = df["Close"].values.astype(np.float32)
    return TimeSeriesBundle(
        ticker=ticker,
        dates=dates,
        features=features,
        exog_features=exog_features,
        close=close,
    )


def _grad_norm(params) -> float:
    grads = []
    for p in params:
        if p.grad is None:
            continue
        grads.append(p.grad.detach().norm(2))
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack(grads), 2).cpu().item())


def _compute_pred_stats(
    pred_raw: torch.Tensor,
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    returns = pred_raw[..., 0]

    vol = torch.std(returns, dim=1, unbiased=False)
    trend = torch.mean(returns, dim=1)

    centered = returns - trend.unsqueeze(1)
    std = torch.std(centered, dim=1, unbiased=False) + eps
    kurt = torch.mean((centered / std.unsqueeze(1)) ** 4, dim=1) - 3.0

    if returns.shape[1] >= 2:
        r0 = returns[:, :-1]
        r1 = returns[:, 1:]
        r0m = r0 - r0.mean(dim=1, keepdim=True)
        r1m = r1 - r1.mean(dim=1, keepdim=True)
        cov = torch.mean(r0m * r1m, dim=1)
        std0 = torch.std(r0, dim=1, unbiased=False) + eps
        std1 = torch.std(r1, dim=1, unbiased=False) + eps
        ac1 = cov / (std0 * std1)
    else:
        ac1 = torch.zeros_like(vol)
    momentum = torch.clamp(ac1, 0.0, 1.0)

    stats = torch.stack([vol, trend, kurt, momentum], dim=1)
    z = (stats - label_mean) / (label_std + 1e-6)
    z = torch.clamp(z, -2.0, 2.0)
    return z / 2.0


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    *,
    device: str,
    pred_len: int,
    freq_token_value: Optional[int],
    label_mean: torch.Tensor,
    label_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    lambda_aux: float,
    log_every: int = 0,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_aux = 0.0
    n_batches = 0
    total_grad_norm = 0.0
    total_batches = len(loader)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    mse = torch.nn.MSELoss()

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
        loss_pred = loss_fn(y_hat_tgt, future_tgt)

        # Auxiliary loss (labels are in input controls)
        ctrl = past[:, 0, -len(CONTROL_NAMES) :]
        pred_raw = y_hat_tgt * (target_std + 1e-6) + target_mean
        pred_stats = _compute_pred_stats(pred_raw, label_mean, label_std)
        loss_aux = mse(pred_stats, ctrl)

        loss = loss_pred + float(lambda_aux) * loss_aux

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=GRAD_CLIP_NORM,
            )
            total_grad_norm += float(grad_norm)
        else:
            total_grad_norm += _grad_norm(trainable_params)
        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_aux += float(loss_aux.detach().cpu().item())
        n_batches += 1

        if log_every and (batch_idx % log_every == 0 or batch_idx == total_batches):
            avg_loss = total_loss / max(n_batches, 1)
            avg_aux = total_aux / max(n_batches, 1)
            print(
                f"  batch {batch_idx:04d}/{total_batches} | avg_loss={avg_loss:.6f} | avg_aux={avg_aux:.6f}"
            )

    avg_loss = total_loss / max(n_batches, 1)
    avg_aux = total_aux / max(n_batches, 1)
    avg_grad_norm = total_grad_norm / max(n_batches, 1)
    return avg_loss, avg_aux, avg_grad_norm


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    *,
    device: str,
    pred_len: int,
    freq_token_value: Optional[int],
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

    cfg = TTMLabelConfig(context_length=CONTEXT_LEN, prediction_length=PRED_LEN)
    cfg.detrend_returns = DETREND_RETURNS

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

    target_scaler, exog_scaler = fit_feature_scalers(all_series)
    scaler_state = {
        "targets": target_scaler.state_dict(),
        "exog": exog_scaler.state_dict(),
        "type": "target_exog_v1",
    }
    torch.save(scaler_state, SCALER_PATH)

    train_range = (
        np.datetime64("2000-01-01"),
        np.datetime64(f"{TRAIN_END_YEAR}-12-31"),
    )
    val_range = (np.datetime64(f"{VAL_YEAR}-01-01"), np.datetime64(f"{VAL_YEAR}-12-31"))
    test_range = (
        np.datetime64(f"{TEST_YEAR}-01-01"),
        np.datetime64(f"{TEST_YEAR}-12-31"),
    )

    train_ds = LabelWindowDataset(
        all_series,
        cfg,
        target_scaler,
        exog_scaler,
        target_date_range=train_range,
        compute_weights=True,
        weight_beta=SAMPLE_BETA,
        weight_min=SAMPLE_MIN,
        weight_max=SAMPLE_MAX,
    )
    label_scaler = train_ds.label_scaler

    val_ds = LabelWindowDataset(
        all_series,
        cfg,
        target_scaler,
        exog_scaler,
        label_scaler=label_scaler,
        target_date_range=val_range,
    )
    test_ds = LabelWindowDataset(
        all_series,
        cfg,
        target_scaler,
        exog_scaler,
        label_scaler=label_scaler,
        target_date_range=test_range,
    )

    weights = train_ds.sampling_weights()
    if weights is None:
        raise RuntimeError("Sampling weights not computed.")
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True
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
    loss_fn = StudentTLoss(df=RETURN_DF, scale=RETURN_SCALE)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    use_freq_token = bool(getattr(model.config, "resolution_prefix_tuning", False))
    freq_token_value = (
        DEFAULT_FREQUENCY_MAPPING.get(FREQ, None) if use_freq_token else None
    )
    if use_freq_token and freq_token_value is None:
        raise ValueError(f"Frequency token not found for freq={FREQ}.")

    label_mean = torch.tensor(label_scaler.mean, dtype=torch.float32, device=DEVICE)
    label_std = torch.tensor(label_scaler.std, dtype=torch.float32, device=DEVICE)
    target_mean = torch.tensor(target_scaler.mean, dtype=torch.float32, device=DEVICE)
    target_std = torch.tensor(target_scaler.std, dtype=torch.float32, device=DEVICE)

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch:02d} starting...")
        warm = min(1.0, epoch / float(LAMBDA_WARMUP_EPOCHS))
        lambda_aux = LAMBDA_AUX_MAX * warm

        tr_loss, tr_aux, tr_grad_norm = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=DEVICE,
            pred_len=PRED_LEN,
            freq_token_value=freq_token_value,
            label_mean=label_mean,
            label_std=label_std,
            target_mean=target_mean,
            target_std=target_std,
            lambda_aux=lambda_aux,
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
        print(
            f"Epoch {epoch:02d} | train_loss={tr_loss:.6f} | aux_loss={tr_aux:.6f} | val_loss={va_loss:.6f} | grad_norm={tr_grad_norm:.6f} | lambda_aux={lambda_aux:.4f}"
        )

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

    label_payload = {
        "mean": label_scaler.mean.tolist(),
        "std": label_scaler.std.tolist(),
        "eps": float(label_scaler.eps),
    }
    LABEL_SCALER_PATH.write_text(json.dumps(label_payload, indent=2))

    config_payload = {
        "model_id": MODEL_ID,
        "freq": FREQ,
        "context_length": CONTEXT_LEN,
        "prediction_length": PRED_LEN,
        "channels": cfg.channel_names,
        "target_features": TARGET_FEATURES,
        "exogenous_features": EXOG_FEATURES,
        "controls": CONTROL_NAMES,
        "detrend_returns": cfg.detrend_returns,
        "detrend_window": cfg.detrend_window,
        "detrend_mode": cfg.detrend_mode,
        "realized_vol_window": cfg.realized_vol_window,
        "train_end_year": TRAIN_END_YEAR,
        "val_year": VAL_YEAR,
        "test_year": TEST_YEAR,
        "label_scaler": str(LABEL_SCALER_PATH),
        "sampler": {
            "beta": SAMPLE_BETA,
            "min_weight": SAMPLE_MIN,
            "max_weight": SAMPLE_MAX,
        },
        "aux_loss": {
            "lambda_max": LAMBDA_AUX_MAX,
            "warmup_epochs": LAMBDA_WARMUP_EPOCHS,
        },
    }
    CONFIG_PATH.write_text(json.dumps(config_payload, indent=2))
    print(f"Saved config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
