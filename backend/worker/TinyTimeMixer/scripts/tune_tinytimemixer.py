"""
finetune_ttm_aapl_hourly.py

Few-shot fine-tuning for IBM Granite TinyTimeMixer (TTM) on AAPL HOURLY OHLCV (~last 3 years).

IMPORTANT (yfinance limits):
- yfinance typically DOES NOT allow 1h bars for a full 3y window in one request.
- This script fetches hourly data in CHUNKS (default: 700 days per chunk) and concatenates.
- If Yahoo throttles / returns empty chunks, reduce CHUNK_DAYS.

Outputs:
  - tinytimemixer_aapl_hourly_head.pt  (fine-tuned weights for model)
  - scaler_aapl_hourly.pt              (train mean/std)

Run:
  python finetune_ttm_aapl_hourly.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yfinance as yf

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = None  # e.g. torch.float16 if you know what you're doing

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Windowing (hourly => you can afford longer context)
# 512
CONTEXT_LEN = 512
PRED_LEN = 96

# Features / channels
FEATURE_COLS = ["ret", "log_hl", "log_oc", "log_vol"]
TARGET_COLS = ["ret"]  # subset of FEATURE_COLS

# Train split (chronological)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Training
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0

# Freeze policy (few-shot):
UNFREEZE_KEYWORDS = [
    "head",
    "decoder",
    "prediction",
    "pred",
    "proj",
    "output",
    "lm_head",
]

# Saving
OUT_WEIGHTS = "tinytimemixer_aapl_hourly_head.pt"
OUT_SCALER = "scaler_aapl_hourly.pt"

# yfinance hourly download chunking
TICKER = "AAPL"
INTERVAL = "1d"
LOOKBACK_YEARS = 20
# LOOKBACK_DAYS = 700
CHUNK_DAYS = 700  # adjust down if Yahoo throttles (e.g. 365)
PAUSE_BETWEEN_CHUNKS_SEC = 0.0  # set to >0 if you get rate limited


# ----------------------------
# Utilities
# ----------------------------
@dataclass
class StandardScaler:
    mean: torch.Tensor  # [C]
    std: torch.Tensor  # [C]
    eps: float = 1e-6

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "StandardScaler":
        return cls(mean=d["mean"], std=d["std"], eps=float(d.get("eps", 1e-6)))


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def extract_predictions(outputs: Any) -> torch.Tensor:
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
        raise RuntimeError(
            f"Expected 3D predictions [B,H,C] or [B,C,H], got {tuple(y_hat.shape)}"
        )
    _, d1, d2 = y_hat.shape
    if d1 == num_channels and d2 != num_channels:
        return y_hat.transpose(1, 2).contiguous()
    return y_hat


# ----------------------------
# Data preparation (hourly)
# ----------------------------
def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def download_hourly_in_chunks(
    ticker: str,
    years: int,
    interval: str,
    chunk_days: int,
    pause_sec: float = 0.0,
) -> pd.DataFrame:
    """
    Downloads hourly data in multiple requests to work around yfinance period/interval limits.
    Uses [start, end) slicing with timezone-aware UTC datetimes.

    Returns dataframe with DatetimeIndex (from yfinance) then reset_index with 'date' column.
    """
    import time

    end = _to_utc(datetime.utcnow())
    start = end - timedelta(days=int(years * 365.25))

    chunks = []
    cur_start = start

    print(
        f"Downloading {ticker} {interval} from {start.date()} to {end.date()} in ~{chunk_days}d chunks..."
    )

    while cur_start < end:
        cur_end = min(cur_start + timedelta(days=chunk_days), end)

        df = yf.download(
            ticker,
            start=cur_start,
            end=cur_end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            prepost=False,  # regular session; set True if you want extended hours
            threads=True,
        )

        if df is None or df.empty:
            print(f"  - chunk {cur_start.date()} -> {cur_end.date()}: EMPTY (skipping)")
        else:
            print(f"  - chunk {cur_start.date()} -> {cur_end.date()}: {len(df)} rows")
            chunks.append(df)

        cur_start = cur_end
        if pause_sec and pause_sec > 0:
            time.sleep(pause_sec)

    if not chunks:
        raise RuntimeError(
            "All yfinance chunks returned empty. Try smaller CHUNK_DAYS or different interval."
        )

    out = pd.concat(chunks, axis=0)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out = out.reset_index()

    # yfinance names the datetime column differently depending on interval; normalize to "date"
    if "Datetime" in out.columns:
        out = out.rename(columns={"Datetime": "date"})
    elif "Date" in out.columns:
        out = out.rename(columns={"Date": "date"})
    else:
        # fallback: first column is usually the index name
        out = out.rename(columns={out.columns[0]: "date"})

    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create stationary-ish features suitable for TTM few-shot.
    For hourly bars, these are still sensible:
      - ret: log return of Close
      - log_hl: log(High/Low)
      - log_oc: log(Close/Open)
      - log_vol: log1p(Volume)
    """
    out = df.copy()

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Got columns: {list(out.columns)}")

    out = out.sort_values("date").reset_index(drop=True)

    out["ret"] = np.log(out["Close"] / out["Close"].shift(1))
    out["log_hl"] = np.log(out["High"] / out["Low"])
    out["log_oc"] = np.log(out["Close"] / out["Open"])
    out["log_vol"] = np.log1p(out["Volume"].astype(float))

    out = out.dropna().reset_index(drop=True)

    # Keep only what we use
    return out[["date"] + FEATURE_COLS]


def chronological_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()
    return train, val, test


def fit_scaler(train_df: pd.DataFrame, cols: List[str]) -> StandardScaler:
    x = torch.tensor(train_df[cols].values, dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return StandardScaler(mean=mean, std=std, eps=1e-6)


class WindowedTimeSeriesDataset(Dataset):
    """
    Produces:
      past_values: [L, C_in]
      future_targets: [H, C_tgt]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: List[str],
        target_cols: List[str],
        context_len: int,
        pred_len: int,
        scaler: StandardScaler,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.context_len = context_len
        self.pred_len = pred_len
        self.scaler = scaler

        self.x_all = torch.tensor(
            self.df[self.feature_cols].values, dtype=torch.float32
        )
        self.x_all = self.scaler.transform(self.x_all)

        tgt_idx = [self.feature_cols.index(c) for c in self.target_cols]
        self.tgt_idx = torch.tensor(tgt_idx, dtype=torch.long)

        self.max_start = len(self.df) - (self.context_len + self.pred_len)
        if self.max_start <= 0:
            raise ValueError(
                f"Not enough rows ({len(self.df)}) for context_len={context_len} + pred_len={pred_len}."
            )

    def __len__(self) -> int:
        return self.max_start + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        mid = start + self.context_len
        end = mid + self.pred_len

        past = self.x_all[start:mid]  # [L, C]
        future = self.x_all[mid:end]  # [H, C]
        future_tgt = future.index_select(1, self.tgt_idx)  # [H, C_tgt]
        return past, future_tgt


# ----------------------------
# Model loading (trainable)
# ----------------------------


def load_ttm_for_training(
    model_id: str,
    *,
    num_input_channels: int,
    prediction_length: int,
    decoder_mode: str = "mix_channel",
    torch_dtype: Optional[torch.dtype] = None,
) -> torch.nn.Module:
    try:
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
    except Exception as e:
        raise ImportError(
            "Failed to import TinyTimeMixerForPrediction from tsfm_public. "
            "Install IBM Granite TSFM library (granite-tsfm / tsfm_public)."
        ) from e

    extra_kwargs: Dict[str, Any] = {
        "num_input_channels": int(num_input_channels),
        "prediction_length": int(prediction_length),
        "decoder_mode": decoder_mode,
    }
    if torch_dtype is not None:
        extra_kwargs["torch_dtype"] = torch_dtype

    model = TinyTimeMixerForPrediction.from_pretrained(model_id, **extra_kwargs)
    return model


def freeze_for_fewshot(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        lname = name.lower()
        if any(k in lname for k in UNFREEZE_KEYWORDS):
            p.requires_grad = True


# ----------------------------
# Training / evaluation
# ----------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    *,
    device: str,
    num_input_channels: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for past, future_tgt in loader:
        past = past.to(device)
        future_tgt = future_tgt.to(device)

        outputs = model(past_values=past)
        y_hat = extract_predictions(outputs)
        y_hat = maybe_fix_pred_shape(y_hat, num_channels=num_input_channels)

        tgt_idx = loader.dataset.tgt_idx.to(device)
        y_hat_tgt = y_hat.index_select(dim=2, index=tgt_idx)

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

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    *,
    device: str,
    num_input_channels: int,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for past, future_tgt in loader:
        past = past.to(device)
        future_tgt = future_tgt.to(device)

        outputs = model(past_values=past)
        y_hat = extract_predictions(outputs)
        y_hat = maybe_fix_pred_shape(y_hat, num_channels=num_input_channels)

        tgt_idx = loader.dataset.tgt_idx.to(device)
        y_hat_tgt = y_hat.index_select(dim=2, index=tgt_idx)

        loss = loss_fn(y_hat_tgt, future_tgt)

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    set_seed(SEED)

    print(f"Device: {DEVICE}")
    print(
        f"Downloading {TICKER} hourly (~last {LOOKBACK_YEARS}y) via yfinance (chunked)..."
    )

    raw = download_hourly_in_chunks(
        ticker=TICKER,
        years=LOOKBACK_YEARS,
        interval=INTERVAL,
        chunk_days=CHUNK_DAYS,
        pause_sec=PAUSE_BETWEEN_CHUNKS_SEC,
    )

    feat = build_features(raw)
    train_df, val_df, test_df = chronological_split(feat)

    print(f"Rows: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Safety: ensure split is large enough for windowing
    context_len = CONTEXT_LEN
    pred_len = PRED_LEN

    min_needed = context_len + pred_len + 1
    if len(train_df) < min_needed:
        new_context = max(64, len(train_df) - pred_len - 1)
        print(
            f"WARNING: train rows={len(train_df)} < {min_needed}. "
            f"Adjusting CONTEXT_LEN {context_len} -> {new_context}."
        )
        context_len = new_context

    scaler = fit_scaler(train_df, FEATURE_COLS)
    torch.save(scaler.state_dict(), OUT_SCALER)
    print(f"Saved scaler to: {OUT_SCALER}")

    train_ds = WindowedTimeSeriesDataset(
        train_df,
        feature_cols=FEATURE_COLS,
        target_cols=TARGET_COLS,
        context_len=CONTEXT_LEN,
        pred_len=PRED_LEN,
        scaler=scaler,
    )
    val_ds = WindowedTimeSeriesDataset(
        val_df,
        feature_cols=FEATURE_COLS,
        target_cols=TARGET_COLS,
        context_len=CONTEXT_LEN,
        pred_len=PRED_LEN,
        scaler=scaler,
    )
    test_ds = WindowedTimeSeriesDataset(
        test_df,
        feature_cols=FEATURE_COLS,
        target_cols=TARGET_COLS,
        context_len=CONTEXT_LEN,
        pred_len=PRED_LEN,
        scaler=scaler,
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

    num_input_channels = len(FEATURE_COLS)

    print(f"Loading TTM from: {MODEL_ID}")
    model = load_ttm_for_training(
        MODEL_ID,
        num_input_channels=num_input_channels,
        prediction_length=PRED_LEN,
        decoder_mode="mix_channel",
        torch_dtype=TORCH_DTYPE,
    ).to(DEVICE)

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
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=DEVICE,
            num_input_channels=num_input_channels,
        )
        va_loss = eval_one_epoch(
            model,
            val_loader,
            loss_fn,
            device=DEVICE,
            num_input_channels=num_input_channels,
        )

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        torch.save(best_state, OUT_WEIGHTS)
        print(f"Saved best weights to: {OUT_WEIGHTS} (best val={best_val:.6f})")
        model.load_state_dict(best_state)

    te_loss = eval_one_epoch(
        model,
        test_loader,
        loss_fn,
        device=DEVICE,
        num_input_channels=num_input_channels,
    )
    print(f"Test loss: {te_loss:.6f}")
    print("Done.")


if __name__ == "__main__":
    main()
