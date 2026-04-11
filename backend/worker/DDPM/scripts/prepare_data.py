import os, argparse
import numpy as np
import pandas as pd
import yfinance as yf
from worker.DDPM.utils import build_conditioning_matrix, normalise_windows

TICKERS = ["SPY", "QQQ", "GLD", "USO", "DX-Y.NYB", "EURUSD=X", "BTC-USD"]
START   = "2019-01-01"
END     = "2024-12-31"
SEQ_LEN = 1260
STEP    = 21


def download_data(tickers=TICKERS, start=START, end=END) -> pd.DataFrame:
    raw     = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    raw     = raw.dropna()
    returns = np.log(raw / raw.shift(1)).dropna()
    return returns


def build_windows(returns: pd.DataFrame,
                  seq_len: int = SEQ_LEN,
                  step: int    = STEP) -> np.ndarray:
    arr         = returns.values.T
    n_assets, T = arr.shape
    windows     = []
    for start in range(0, T - seq_len, step):
        windows.append(arr[:, start:start + seq_len])
    return np.array(windows, dtype=np.float32)


def slice_context_prediction(returns: pd.DataFrame,
                              context_len: int = 252,
                              pred_len: int    = 1260,
                              step: int        = 21):
    """
    Slice returns into (context, prediction) window pairs for fine-tuning.
    Returns:
        context_windows    (N, n_assets, context_len)
        prediction_windows (N, n_assets, pred_len)
        anchor_indices     (N,)
    """
    arr         = returns.values.T
    N_a, T      = arr.shape
    total_len   = context_len + pred_len
    ctx_wins, pred_wins, anchors = [], [], []
    for start in range(0, T - total_len, step):
        ctx_wins.append(arr[:, start              : start + context_len])
        pred_wins.append(arr[:, start+context_len : start + total_len])
        anchors.append(start + context_len - 1)
    return (np.array(ctx_wins,  dtype=np.float32),
            np.array(pred_wins, dtype=np.float32),
            np.array(anchors))


def main(out_dir: str = "artefacts"):
    os.makedirs(out_dir, exist_ok=True)

    print("Downloading data ...")
    returns = download_data()
    print(f"Returns shape: {returns.shape}")

    print("Building windows ...")
    windows = build_windows(returns)
    print(f"Windows shape: {windows.shape}")

    print("Building conditioning matrix ...")
    C_raw, C_norm, cond_norm_params = build_conditioning_matrix(windows)

    print("Normalising windows ...")
    windows_norm, window_scales_saved = normalise_windows(windows)

    rv_per_window  = C_raw[:, 0]
    vol_threshold  = np.percentile(rv_per_window, 75)
    sample_weights = np.where(rv_per_window > vol_threshold, 5.0, 1.0)
    sample_weights = sample_weights / sample_weights.sum()

    np.save(f"{out_dir}/windows.npy",       windows)
    np.save(f"{out_dir}/windows_norm.npy",  windows_norm)
    np.save(f"{out_dir}/window_scales.npy", window_scales_saved)
    np.save(f"{out_dir}/C_raw.npy",         C_raw)
    np.save(f"{out_dir}/C_norm.npy",        C_norm)
    np.save(f"{out_dir}/sample_weights.npy",sample_weights)
    np.save(f"{out_dir}/cond_norm_min.npy", cond_norm_params["min"])
    np.save(f"{out_dir}/cond_norm_max.npy", cond_norm_params["max"])
    returns.to_csv(f"{out_dir}/returns.csv")

    print(f"Artefacts saved to ./{out_dir}/")
    print(f"  windows        : {windows.shape}")
    print(f"  windows_norm   : {windows_norm.shape}")
    print(f"  C_norm         : {C_norm.shape}")
    print(f"  sample_weights : {sample_weights.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artefacts")
    args = ap.parse_args()
    main(args.out_dir)