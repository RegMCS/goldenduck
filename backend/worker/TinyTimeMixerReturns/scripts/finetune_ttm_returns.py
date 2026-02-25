"""
Fine-tune TTM on log-returns only (no controls, no exogenous features).

Data source:
  - yfinance OHLCV for tickers in tickers.py
  - past N years (default: 10)

Outputs:
  - weights: model/ttm_returns/ttm_returns_weights.pt
  - scaler:  model/ttm_returns/ttm_returns_scaler.pt
  - config:  model/ttm_returns/ttm_returns_config.json

Run:
  python finetune_ttm_returns.py
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

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

import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.worker.TinyTimeMixerSyn.dataset.tickers import US_TICKERS, INDEX_TICKERS
from backend.worker.TinyTimeMixer.services.ttm_controlled_dataset import (
    StandardScaler,
    download_daily_ohlcv,
)


# ----------------------------
# Config
# ----------------------------
MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
FREQ = "D"

CONTEXT_LEN = 180
PRED_LEN = 60
YEARS_LOOKBACK = 10

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
OUTPUT_DIR = BASE_DIR / "outputs" / "ttm_returns"
MODEL_DIR = BASE_DIR / "model" / "ttm_returns"
WEIGHTS_PATH = MODEL_DIR / "ttm_returns_weights.pt"
SCALER_PATH = MODEL_DIR / "ttm_returns_scaler.pt"
CONFIG_PATH = MODEL_DIR / "ttm_returns_config.json"


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


def student_t_nll(err: torch.Tensor, *, df: float, scale: float) -> torch.Tensor:
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


class ReturnsSeries:
    def __init__(self, ticker: str, dates: np.ndarray, returns: np.ndarray) -> None:
        self.ticker = ticker
        self.dates = dates
        self.returns = returns  # [N, 1]


def build_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out = out.dropna().reset_index(drop=True)
    return out[["date", "log_return"]]


def load_series_for_ticker(
    ticker: str, *, start: str, end: str
) -> Optional[ReturnsSeries]:
    try:
        raw = download_daily_ohlcv(ticker, start=start, end=end)
    except Exception as e:
        print(f"[skip] {ticker}: download failed ({e}).")
        return None
    feats = build_log_returns(raw)
    if len(feats) < (CONTEXT_LEN + PRED_LEN + 1):
        print(f"[skip] {ticker}: not enough rows ({len(feats)}).")
        return None
    stats = feats["log_return"].astype(float)
    print(
        f"{ticker}: n={len(stats)} mean={stats.mean():.6f} std={stats.std():.6f} "
        f"min={stats.min():.6f} max={stats.max():.6f}"
    )
    returns = feats["log_return"].values.astype(np.float32).reshape(-1, 1)
    dates = feats["date"].values
    return ReturnsSeries(ticker=ticker, dates=dates, returns=returns)


def list_available_years(series_list: List[ReturnsSeries]) -> List[int]:
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


def pick_split_years(years: List[int]) -> Tuple[List[int], int, int]:
    if len(years) < 3:
        raise ValueError("Need at least 3 distinct years for train/val/test.")
    test_year = years[-1]
    val_year = years[-2]
    train_years = years[:-2]
    return train_years, val_year, test_year


class ReturnsWindowDataset(Dataset):
    def __init__(
        self,
        series_list: List[ReturnsSeries],
        scaler: StandardScaler,
        *,
        context_length: int,
        prediction_length: int,
        allowed_years: Optional[List[int]] = None,
    ) -> None:
        self.series_list = list(series_list)
        self.scaler = scaler
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.allowed_years = set(int(y) for y in allowed_years) if allowed_years else None

        self._index: List[Tuple[int, int]] = []
        for s_idx, series in enumerate(self.series_list):
            n = len(series.returns)
            max_start = n - (self.context_length + self.prediction_length)
            if max_start <= 0:
                continue
            for start in range(max_start + 1):
                end = start + self.context_length + self.prediction_length
                end_date = pd.to_datetime(series.dates[end - 1])
                if self.allowed_years is not None:
                    if int(end_date.year) not in self.allowed_years:
                        continue
                self._index.append((s_idx, start))

        if not self._index:
            raise ValueError("No usable windows found; check data length.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s_idx, start = self._index[idx]
        series = self.series_list[s_idx]
        mid = start + self.context_length
        end = mid + self.prediction_length

        past_raw = series.returns[start:mid].copy()
        future_raw = series.returns[mid:end].copy()

        past_scaled = self.scaler.transform(past_raw)
        future_scaled = self.scaler.transform(future_raw)

        return (
            torch.tensor(past_scaled, dtype=torch.float32),
            torch.tensor(future_scaled, dtype=torch.float32),
        )


def fit_returns_scaler(series_list: List[ReturnsSeries]) -> StandardScaler:
    if not series_list:
        raise ValueError("No series to fit scaler.")
    all_returns = np.concatenate([s.returns for s in series_list], axis=0)
    mean = all_returns.mean(axis=0)
    std = all_returns.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return StandardScaler(mean=mean, std=std, eps=1e-6)


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
    loss_fn: StudentTLoss,
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

        loss = loss_fn(y_hat, future_tgt)

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
    loss_fn: StudentTLoss,
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

        loss = loss_fn(y_hat, future_tgt)

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    end_date = pd.Timestamp.utcnow().normalize()
    start_date = end_date - pd.DateOffset(years=YEARS_LOOKBACK)

    tickers = list(US_TICKERS) + list(INDEX_TICKERS)
    print(f"Tickers: {len(tickers)}")
    print(f"Downloading {YEARS_LOOKBACK}y from {start_date.date()} to {end_date.date()}")

    all_series: List[ReturnsSeries] = []
    for ticker in tickers:
        series = load_series_for_ticker(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )
        if series is None:
            continue
        all_series.append(series)

    if not all_series:
        raise RuntimeError("No training series available after filtering.")

    all_years = list_available_years(all_series)
    train_years, val_year, test_year = pick_split_years(all_years)
    print(
        f"Split years | train: {min(train_years)}-{max(train_years)} "
        f"({len(train_years)} yrs) | val: {val_year} | test: {test_year}"
    )

    scaler = fit_returns_scaler(all_series)
    torch.save(scaler.state_dict(), SCALER_PATH)

    train_ds = ReturnsWindowDataset(
        all_series,
        scaler,
        context_length=CONTEXT_LEN,
        prediction_length=PRED_LEN,
        allowed_years=train_years,
    )
    val_ds = ReturnsWindowDataset(
        all_series,
        scaler,
        context_length=CONTEXT_LEN,
        prediction_length=PRED_LEN,
        allowed_years=[val_year],
    )
    test_ds = ReturnsWindowDataset(
        all_series,
        scaler,
        context_length=CONTEXT_LEN,
        prediction_length=PRED_LEN,
        allowed_years=[test_year],
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
        num_input_channels=1,
        prediction_channel_indices=[0],
        exogenous_channel_indices=[],
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

    config_payload = {
        "model_id": MODEL_ID,
        "freq": FREQ,
        "context_length": CONTEXT_LEN,
        "prediction_length": PRED_LEN,
        "tickers": tickers,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "train_years": train_years,
        "val_year": val_year,
        "test_year": test_year,
        "return_df": RETURN_DF,
        "return_scale": RETURN_SCALE,
    }
    CONFIG_PATH.write_text(json.dumps(config_payload, indent=2))
    print(f"Saved config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
