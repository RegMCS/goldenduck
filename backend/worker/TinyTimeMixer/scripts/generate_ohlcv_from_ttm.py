"""
generate_ohlcv_from_ttm.py

Load fine-tuned TinyTimeMixer weights + scaler, predict future feature channels,
then reconstruct synthetic OHLCV.

This version implements Option A:
- Instantiate the model with the checkpoint's native head horizon (MODEL_PRED_LEN=96),
  so load_state_dict succeeds.
- Slice the model output to the desired horizon (PRED_LEN=32).

Files expected (relative to where you run this script):
- tinytimemixer_aapl_hourly_head.pt
- scaler_aapl_hourly.pt

Output:
- synthetic_aapl_ohlcv.csv
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import yfinance as yf

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Must match training-time feature engineering
FEATURE_COLS = ["ret", "log_hl", "log_oc", "log_vol"]

# Desired output horizon (how many future rows to generate)
PRED_LEN = 32

# Checkpoint-native horizon (what your saved head expects, based on the error: 96)
MODEL_PRED_LEN = 96

# Context length fed into the model (must be >= model.config.context_length)
CONTEXT_LEN = 512

WEIGHTS_PATH = "tinytimemixer_aapl_hourly_head.pt"
SCALER_PATH = "scaler_aapl_hourly.pt"

TICKER = "AAPL"
INTERVAL = "1d"
LOOKBACK_YEARS = 10


# ----------------------------
# Scaler
# ----------------------------
class StandardScaler:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "StandardScaler":
        return cls(
            mean=d["mean"].float(), std=d["std"].float(), eps=float(d.get("eps", 1e-6))
        )


# ----------------------------
# Data + features
# ----------------------------
def download_daily(ticker: str, years: int) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=f"{years}y",
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError("yfinance returned empty dataframe.")

    df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "date"})
    else:
        df = df.rename(columns={df.columns[0]: "date"})

    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    out["ret"] = np.log(out["Close"] / out["Close"].shift(1))
    out["log_hl"] = np.log(out["High"] / out["Low"])
    out["log_oc"] = np.log(out["Close"] / out["Open"])
    out["log_vol"] = np.log1p(out["Volume"].astype(float))

    out = out.dropna().reset_index(drop=True)
    return out


# ----------------------------
# Model helpers
# ----------------------------
def load_ttm(num_input_channels: int, prediction_length: int) -> torch.nn.Module:
    """
    Load TTM model instance. For inference with your saved weights:
    - prediction_length MUST match the head horizon in WEIGHTS_PATH (MODEL_PRED_LEN=96)
    """
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

    model = TinyTimeMixerForPrediction.from_pretrained(
        MODEL_ID,
        num_input_channels=int(num_input_channels),
        prediction_length=int(prediction_length),
        decoder_mode="mix_channel",
        ignore_mismatched_sizes=False,  # we want exact match to load your saved weights
        low_cpu_mem_usage=False,  # avoids meta-loading quirks
    )
    return model


def extract_predictions(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "predictions"):
        return outputs.predictions
    if hasattr(outputs, "prediction_outputs"):
        return outputs.prediction_outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    if (
        isinstance(outputs, (tuple, list))
        and len(outputs)
        and torch.is_tensor(outputs[0])
    ):
        return outputs[0]
    if torch.is_tensor(outputs):
        return outputs
    raise RuntimeError(f"Cannot extract predictions from: {type(outputs)}")


def ensure_bhc(y: torch.Tensor, channels: int) -> torch.Tensor:
    """
    Ensure y is [B, H, C]. Some implementations may return [B, C, H].
    """
    if y.ndim != 3:
        raise RuntimeError(f"Expected 3D output, got {tuple(y.shape)}")

    b, d1, d2 = y.shape
    if d1 == channels and d2 != channels:
        y = y.transpose(1, 2).contiguous()
    return y


# ----------------------------
# Feature -> OHLCV reconstruction
# ----------------------------
def reconstruct_ohlcv(
    last_ohlcv: pd.Series,
    pred_feats: pd.DataFrame,  # columns: ret, log_hl, log_oc, log_vol
) -> pd.DataFrame:
    """
    Reconstruct OHLCV forward using:
      ret    = log(C_t / C_{t-1})
      log_oc = log(C / O)  => O = C / exp(log_oc)
      log_hl = log(H / L)  => ratio r = exp(log_hl)
      log_vol= log1p(V)    => V = exp(log_vol) - 1

    For H/L, we set a symmetric range around M = (O + C)/2:
      L = M / sqrt(r)
      H = M * sqrt(r)
    Then clamp so H >= max(O,C) and L <= min(O,C).
    """
    C_prev = float(last_ohlcv["Close"])

    rows = []
    for i in range(len(pred_feats)):
        ret = float(pred_feats.loc[i, "ret"])
        log_hl = float(pred_feats.loc[i, "log_hl"])
        log_oc = float(pred_feats.loc[i, "log_oc"])
        log_vol = float(pred_feats.loc[i, "log_vol"])

        C = C_prev * float(np.exp(ret))
        O = C / float(np.exp(log_oc))

        r = float(np.exp(log_hl))
        r = max(r, 1.000001)

        M = 0.5 * (O + C)
        L = M / np.sqrt(r)
        H = M * np.sqrt(r)

        hi_body = max(O, C)
        lo_body = min(O, C)
        H = max(H, hi_body)
        L = min(L, lo_body)

        V = float(np.exp(log_vol) - 1.0)
        V = max(V, 0.0)

        rows.append({"Open": O, "High": H, "Low": L, "Close": C, "Volume": V})
        C_prev = C

    return pd.DataFrame(rows)


def make_future_dates(last_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    """
    Business-day future dates. For daily equities, this is usually what you want.
    """
    last_date = pd.to_datetime(last_date)
    return pd.bdate_range(last_date, periods=n + 1)[1:]


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    print(f"Device: {DEVICE}")

    raw = download_daily(TICKER, LOOKBACK_YEARS)
    feat = build_features(raw)

    # Load scaler fitted on training feature space
    scaler_state = torch.load(SCALER_PATH, map_location="cpu")
    scaler = StandardScaler.from_state_dict(scaler_state)

    # Prepare context window in scaled feature space
    x = torch.tensor(feat[FEATURE_COLS].values, dtype=torch.float32)
    x_scaled = scaler.transform(x)  # [T, C]

    if x_scaled.shape[0] < CONTEXT_LEN:
        raise RuntimeError(
            f"Not enough rows ({x_scaled.shape[0]}) for CONTEXT_LEN={CONTEXT_LEN}"
        )

    ctx = x_scaled[-CONTEXT_LEN:, :].unsqueeze(0).to(DEVICE)  # [1, L, C]

    # Load model with checkpoint-native horizon so state_dict loads cleanly
    model = load_ttm(
        num_input_channels=len(FEATURE_COLS), prediction_length=MODEL_PRED_LEN
    ).to(DEVICE)

    # Load fine-tuned weights
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        outputs = model(past_values=ctx)
        y_hat = extract_predictions(outputs)
        y_hat = ensure_bhc(y_hat, channels=len(FEATURE_COLS))  # [1, H_model, C]

    # Slice to desired horizon (Option A)
    y_hat = y_hat[:, :PRED_LEN, :]  # [1, 32, C]

    # Inverse-scale predictions back to original feature units
    y_hat = y_hat.squeeze(0).cpu()  # [H, C]
    y_hat_inv = scaler.inverse_transform(y_hat)  # [H, C]

    pred_df = pd.DataFrame(y_hat_inv.numpy(), columns=FEATURE_COLS)

    # Reconstruct OHLCV
    last_bar = raw.iloc[-1][["Open", "High", "Low", "Close", "Volume"]]
    synth = reconstruct_ohlcv(last_bar, pred_df[["ret", "log_hl", "log_oc", "log_vol"]])

    # Add dates
    last_date = raw["date"].iloc[-1]
    synth["date"] = make_future_dates(pd.to_datetime(last_date), len(synth))
    synth = synth[["date", "Open", "High", "Low", "Close", "Volume"]]

    print(synth.head())
    out_path = "synthetic_aapl_ohlcv.csv"
    synth.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
