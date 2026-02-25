"""
Fine-tune TTM (daily) with controllable synthetic parameters for OHLCV generation
using local GARCH-generated CSVs (no yfinance download).

Outputs:
  - weights: model/ttm_controlled/ttm_controlled_weights.pt
  - scaler:  model/ttm_controlled/ttm_controlled_scaler.pt
  - config:  model/ttm_controlled/ttm_controlled_config.json

Run:
  python finetune_ttm_controlled.py
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

from backend.worker.TinyTimeMixerSyn.dataset.tickers import US_TICKERS, INDEX_TICKERS
from backend.worker.TinyTimeMixer.services.ttm_controlled_dataset import (
    CONTROL_NAMES,
    TARGET_FEATURES,
    EXOG_FEATURES,
    ControlValues,
    TTMControlledConfig,
    TimeSeriesBundle,
    build_base_features,
    fit_feature_scalers,
    ControlledWindowDataset,
)

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
FREQ = "D"

CONTEXT_LEN = 180
PRED_LEN = 60
DETREND_RETURNS = False
CENTER_TARGETS = False
CENTER_EXOG = True

VAL_YEAR = None  # set to an int to override
TEST_YEAR = None  # set to an int to override
SCENARIO_ID = 1  # set to None to use all scenarios

TRAIN_CONTROL_MODE = "fixed"  # "random" or "fixed"
TRAIN_CONTROLS = ControlValues(
    volatility_mult=1.0,
    trend=0.0,
    fat_tails=1.0,
    momentum=0.0,
    horizon=float(PRED_LEN),
)
USE_TICKER_FILTER = True
DISABLE_HORIZON_CONTROL = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
LOG_EVERY_N_BATCHES = 200

# Loss config (Student-t NLL for all channels)
RETURN_DF = 5.0
RETURN_SCALE = 1.0
RETURN_LOSS_WEIGHT = 1.0
OTHER_LOSS_WEIGHT = 1.0

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

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "dataset" / "GARCH_generated_data"
OUTPUT_DIR = BASE_DIR / "outputs" / "ttm_controlled"
MODEL_DIR = BASE_DIR / "model" / "ttm_controlled"
WEIGHTS_PATH = MODEL_DIR / "ttm_controlled_weights.pt"
SCALER_PATH = MODEL_DIR / "ttm_controlled_scaler.pt"
CONFIG_PATH = MODEL_DIR / "ttm_controlled_config.json"


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
    # Negative log-likelihood up to a constant; robust to extremes.
    if df <= 2.0:
        raise ValueError("RETURN_DF must be > 2 for stable Student-t variance.")
    scale = float(scale)
    df = float(df)
    return 0.5 * (df + 1.0) * torch.log1p((err**2) / (df * scale * scale)) + np.log(
        scale
    )


class CombinedLoss:
    def __init__(
        self,
        *,
        df: float,
        scale: float,
        return_weight: float,
        other_weight: float,
    ) -> None:
        self.df = float(df)
        self.scale = float(scale)
        self.return_weight = float(return_weight)
        self.other_weight = float(other_weight)
        self.huber = torch.nn.HuberLoss(delta=1.0, reduction="mean")

    def __call__(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Return channel (index 0) -> Student-t NLL
        err_ret = y_hat[..., 0] - y_true[..., 0]
        loss_ret = student_t_nll(err_ret, df=self.df, scale=self.scale).mean()

        # Range + volume channels -> Huber
        loss_other = self.huber(y_hat[..., 1:], y_true[..., 1:])

        return self.return_weight * loss_ret + self.other_weight * loss_other


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


def _infer_ticker_from_csv(csv_path: Path) -> str:
    stem = csv_path.stem
    marker = "_garch_fx_training"
    if marker in stem:
        stem = stem.split(marker)[0]
    return stem.upper()


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        return df
    if "Date" in df.columns:
        return df.rename(columns={"Date": "date"})
    if "Datetime" in df.columns:
        return df.rename(columns={"Datetime": "date"})
    raise ValueError("CSV is missing a date/Date/Datetime column.")


def load_garch_csv_bundles(
    csv_path: Path,
    cfg: TTMControlledConfig,
    *,
    scenario_id_filter: Optional[int] = None,
) -> List[TimeSeriesBundle]:
    df = pd.read_csv(csv_path)
    df = _ensure_date_column(df)

    required = ["date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} missing columns {missing}.")

    df = df.dropna(subset=required)
    ticker = _infer_ticker_from_csv(csv_path)

    bundles: List[TimeSeriesBundle] = []
    if "scenario_id" in df.columns:
        df = df.sort_values(["scenario_id", "date"]).reset_index(drop=True)
        if scenario_id_filter is not None:
            df = df[df["scenario_id"] == scenario_id_filter].reset_index(drop=True)
        grouped = df.groupby("scenario_id", sort=True)
    else:
        df = df.sort_values("date").reset_index(drop=True)
        grouped = [(None, df)]

    for scenario_id, group in grouped:
        base = build_base_features(group[required].copy(), cfg)
        if len(base) < (CONTEXT_LEN + PRED_LEN + 1):
            label = f"{ticker}" if scenario_id is None else f"{ticker}_s{int(scenario_id)}"
            print(f"[skip] {label}: not enough rows ({len(base)}).")
            continue
        suffix = "" if scenario_id is None else f"_s{int(scenario_id):03d}"
        bundles.append(to_bundle(f"{ticker}{suffix}", base))

    return bundles


def list_available_years(series_list: List[TimeSeriesBundle]) -> List[int]:
    years_set = set()
    for series in series_list:
        if len(series.dates) == 0:
            continue
        start_year = pd.to_datetime(series.dates[0]).year
        end_year = pd.to_datetime(series.dates[-1]).year
        if start_year > end_year:
            start_year, end_year = end_year, start_year
        years_set.update(range(start_year, end_year + 1))
    years = sorted(years_set)
    if not years:
        raise ValueError("No dates found to resolve train/val/test years.")
    return years


def year_window_counts(
    series_list: List[TimeSeriesBundle], cfg: TTMControlledConfig
) -> Dict[int, int]:
    window_len = cfg.context_length + cfg.prediction_length
    counts: Dict[int, int] = {}
    for series in series_list:
        if len(series.dates) == 0:
            continue
        years = pd.to_datetime(series.dates).year
        year_counts = pd.Series(years).value_counts()
        for year, count in year_counts.items():
            windows = int(count) - window_len + 1
            if windows > 0:
                y = int(year)
                counts[y] = counts.get(y, 0) + windows
    return counts


def pick_split_years(
    all_years: List[int],
    eligible_years: List[int],
    *,
    seed: int,
    val_year: Optional[int] = None,
    test_year: Optional[int] = None,
) -> Tuple[List[int], int, int]:
    if len(eligible_years) < 3:
        raise ValueError(
            "Need at least 3 eligible years with enough windows for train/val/test."
        )

    rng = np.random.default_rng(seed)
    if test_year is None:
        test_year = int(rng.choice(eligible_years))
    elif test_year not in eligible_years:
        raise ValueError(
            f"TEST_YEAR {test_year} has no usable windows. Eligible years: {eligible_years}"
        )

    remaining = [y for y in eligible_years if y != test_year]
    if val_year is None:
        preferred = test_year - 1
        if preferred in remaining:
            val_year = preferred
        else:
            preferred = test_year + 1
            if preferred in remaining:
                val_year = preferred
            else:
                val_year = int(rng.choice(remaining))
    elif val_year not in remaining:
        raise ValueError(
            f"VAL_YEAR {val_year} has no usable windows. Eligible years: {remaining}"
        )

    train_years = [y for y in all_years if y not in (val_year, test_year)]
    if not train_years:
        raise ValueError("No training years left after selecting val/test years.")

    return train_years, int(val_year), int(test_year)


def _grad_norm(params) -> float:
    grads = []
    for p in params:
        if p.grad is None:
            continue
        grads.append(p.grad.detach().norm(2))
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack(grads), 2).cpu().item())


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
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_grad_norm = 0.0
    total_batches = len(loader)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
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
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=GRAD_CLIP_NORM,
            )
            total_grad_norm += float(grad_norm)
        else:
            total_grad_norm += _grad_norm(trainable_params)
        optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1

        if log_every and (batch_idx % log_every == 0 or batch_idx == total_batches):
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  batch {batch_idx:04d}/{total_batches} | avg_loss={avg_loss:.6f}")
    avg_loss = total_loss / max(n_batches, 1)
    avg_grad_norm = total_grad_norm / max(n_batches, 1)
    return avg_loss, avg_grad_norm


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
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    cfg = TTMControlledConfig(context_length=CONTEXT_LEN, prediction_length=PRED_LEN)
    cfg.detrend_returns = False  # force off to preserve drift
    if DISABLE_HORIZON_CONTROL:
        cfg.control_ranges.horizon = (float(PRED_LEN), float(PRED_LEN))

    all_series = []

    csv_paths = sorted(DATA_DIR.glob("*.csv"))
    if not csv_paths:
        raise RuntimeError(f"No CSV files found in {DATA_DIR}.")

    allowed_tickers = set(t.upper() for t in (list(US_TICKERS) + list(INDEX_TICKERS)))
    print(f"Loading GARCH CSVs: {len(csv_paths)}")
    for csv_path in csv_paths:
        ticker = _infer_ticker_from_csv(csv_path)
        if USE_TICKER_FILTER and ticker not in allowed_tickers:
            print(f"[skip] {csv_path.name}: ticker {ticker} not in tickers.py.")
            continue
        print(f"Processing {csv_path.name}...")
        bundles = load_garch_csv_bundles(
            csv_path, cfg, scenario_id_filter=SCENARIO_ID
        )
        print(f"  series added: {len(bundles)}")
        all_series.extend(bundles)

    if not all_series:
        raise RuntimeError("No training series available after filtering.")

    all_years = list_available_years(all_series)
    window_counts = year_window_counts(all_series, cfg)
    eligible_years = sorted(window_counts.keys())
    print(f"Year window counts: {window_counts}")
    train_years, val_year, test_year = pick_split_years(
        all_years,
        eligible_years,
        seed=SEED,
        val_year=VAL_YEAR,
        test_year=TEST_YEAR,
    )
    print(
        f"Split years | train: {train_years} | val: {val_year} | test: {test_year}"
    )

    target_scaler, exog_scaler = fit_feature_scalers(
        all_series,
        center_targets=CENTER_TARGETS,
        center_exog=CENTER_EXOG,
    )
    scaler_state = {
        "targets": target_scaler.state_dict(),
        "exog": exog_scaler.state_dict(),
        "type": "target_exog_v1",
    }
    torch.save(scaler_state, SCALER_PATH)

    fixed_controls = ControlValues(
        volatility_mult=1.0,
        trend=0.0,
        fat_tails=1.0,
        momentum=0.0,
        horizon=float(PRED_LEN),
    )

    train_ds = ControlledWindowDataset(
        all_series,
        cfg,
        target_scaler,
        exog_scaler,
        control_mode=TRAIN_CONTROL_MODE,
        fixed_controls=(
            TRAIN_CONTROLS if TRAIN_CONTROL_MODE == "fixed" else None
        ),
        allowed_years=train_years,
        seed=SEED,
    )
    val_ds = ControlledWindowDataset(
        all_series,
        cfg,
        target_scaler,
        exog_scaler,
        control_mode="fixed",
        fixed_controls=fixed_controls,
        allowed_years=[val_year],
        seed=SEED,
    )
    test_ds = ControlledWindowDataset(
        all_series,
        cfg,
        target_scaler,
        exog_scaler,
        control_mode="fixed",
        fixed_controls=fixed_controls,
        allowed_years=[test_year],
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

    # Always freeze for few-shot training (no full-parameter finetuning)
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

    train_losses = []
    val_losses = []
    grad_norms = []

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch:02d} starting...")
        tr_loss, tr_grad_norm = train_one_epoch(
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
        train_losses.append(float(tr_loss))
        val_losses.append(float(va_loss))
        grad_norms.append(float(tr_grad_norm))
        print(
            f"Epoch {epoch:02d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} | grad_norm={tr_grad_norm:.6f}"
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

    # Save loss curve + grad norms
    loss_csv = OUTPUT_DIR / "loss_curve.csv"
    loss_payload = {
        "epoch": list(range(1, EPOCHS + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "grad_norm": grad_norms,
    }
    pd.DataFrame(loss_payload).to_csv(loss_csv, index=False)
    print(f"Saved loss curve CSV: {loss_csv}")

    loss_png = OUTPUT_DIR / "loss_curve.png"
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(loss_payload["epoch"], train_losses, label="train")
    axes[0].plot(loss_payload["epoch"], val_losses, label="val")
    axes[0].set_title("Loss Curve")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        loss_payload["epoch"], grad_norms, label="grad_norm", color="tab:orange"
    )
    axes[1].set_title("Gradient Norm (avg per epoch)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Grad Norm")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(loss_png, dpi=150)
    plt.close(fig)
    print(f"Saved loss curve PNG: {loss_png}")

    config_payload = {
        "model_id": MODEL_ID,
        "freq": FREQ,
        "context_length": CONTEXT_LEN,
        "prediction_length": PRED_LEN,
        "channels": cfg.channel_names,
        "target_features": TARGET_FEATURES,
        "exogenous_features": EXOG_FEATURES,
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
        "realized_vol_window": cfg.realized_vol_window,
        "center_targets": CENTER_TARGETS,
        "center_exog": CENTER_EXOG,
        "train_control_mode": TRAIN_CONTROL_MODE,
        "disable_horizon_control": DISABLE_HORIZON_CONTROL,
        "train_years": train_years,
        "val_year": val_year,
        "test_year": test_year,
    }
    CONFIG_PATH.write_text(json.dumps(config_payload, indent=2))
    print(f"Saved config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
