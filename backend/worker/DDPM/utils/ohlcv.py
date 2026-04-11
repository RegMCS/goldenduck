
import numpy as np
import pandas as pd


def load_ohlcv(filepath: str):
    """Load OHLCV CSV and compute log-returns from Close prices."""
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()
    first_close = float(df["Close"].iloc[-1])
    return df, first_close


def returns_to_ohlcv(log_returns: np.ndarray, first_close: float,
                     df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct a single OHLCV DataFrame from a 1D log-return path.

    log_returns : (seq_len,)
    first_close : starting price anchor
    df_input    : reference OHLCV for spread ratio and volume sampling
    """
    seq_len = len(log_returns)

    close = first_close * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate([[first_close], close[:-1]])

    hist_spread  = ((df_input["High"] - df_input["Low"]) / df_input["Close"]).median()
    spread_noise = np.random.uniform(0.5, 1.5, size=seq_len)
    half_spread  = (np.abs(close) * hist_spread * spread_noise) / 2

    high = np.maximum(open_, close) + half_spread
    low  = np.minimum(open_, close) - half_spread

    hist_vol  = df_input["Volume"].values
    volume    = np.random.choice(hist_vol, size=seq_len, replace=True)
    daily_rv  = np.abs(log_returns)
    vol_scale = 1 + (daily_rv / (daily_rv.mean() + 1e-8) - 1) * 0.5
    volume    = (volume * vol_scale).astype(int)

    start_date = df_input.index[-1] + pd.offsets.BDay(1)
    dates      = pd.bdate_range(start=start_date, periods=seq_len)

    return pd.DataFrame({
        "Open"  : open_,
        "High"  : high,
        "Low"   : low,
        "Close" : close,
        "Volume": volume,
    }, index=dates)