"""
Download 5 years of OHLCV data for a non-equity asset and save it as a CSV
compatible with the GARCH-FX worker (Date, Open, High, Low, Close, Volume).

Suggested assets by class:
  Gold futures     : GC=F   (commodity — safe haven, low equity correlation)
  Crude oil        : CL=F   (commodity — supply-shock driven, spiky vol)
  EUR/USD FX       : EURUSD=X (currency — mean-reverting, low kurtosis)
  US 20yr Treasuries: TLT   (fixed income — negatively correlated to equities)
  Bitcoin          : BTC-USD (crypto — extreme fat tails, high vol-of-vol)
  VIX futures ETF  : VIXY   (fear index — strong negative skew, regime spikes)

Usage:
  python fetch_test_asset.py              # downloads Gold (default)
  python fetch_test_asset.py BTC-USD
  python fetch_test_asset.py TLT
"""

import sys
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
TICKER    = sys.argv[1] if len(sys.argv) > 1 else "GC=F"
PERIOD    = "5y"
OUTPUT_DIR = Path(__file__).parent
# ──────────────────────────────────────────────────────────────────────────────


def download(ticker: str) -> pd.DataFrame:
    print(f"\nDownloading {ticker} ({PERIOD})…")
    df = yf.download(ticker, period=PERIOD, auto_adjust=True, progress=False)

    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    # Flatten MultiIndex columns (yfinance ≥0.2 behaviour for single tickers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns {missing}. Available: {list(df.columns)}")

    df = df[required].dropna(subset=["Close"])
    df.index.name = "Date"
    return df


def print_stats(df: pd.DataFrame, ticker: str):
    closes  = df["Close"].values
    returns = np.log(closes[1:] / closes[:-1])
    returns = returns[np.isfinite(returns)]

    ann_vol    = np.std(returns) * np.sqrt(252)
    ann_return = (1 + np.mean(returns)) ** 252 - 1
    kurtosis   = stats.kurtosis(returns)      # excess kurtosis
    skewness   = stats.skew(returns)
    acf1       = float(pd.Series(returns).autocorr(lag=1))

    print(f"\n{'─'*50}")
    print(f"  Asset            : {ticker}")
    print(f"  Period           : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Trading days     : {len(df)}")
    print(f"  Annualised vol   : {ann_vol:.2%}")
    print(f"  Annualised return: {ann_return:.2%}")
    print(f"  Excess kurtosis  : {kurtosis:.2f}  (AAPL ≈ 12.5)")
    print(f"  Skewness         : {skewness:.3f}")
    print(f"  Lag-1 ACF        : {acf1:.4f}")
    print(f"{'─'*50}\n")


def save(df: pd.DataFrame, ticker: str) -> Path:
    # Sanitise ticker for filename (GC=F → GC_F)
    safe_name = ticker.replace("=", "_").replace("/", "_")
    out_path  = OUTPUT_DIR / f"{safe_name}_5y.csv"
    df.to_csv(out_path)
    print(f"Saved → {out_path}")
    return out_path


if __name__ == "__main__":
    df = download(TICKER)
    print_stats(df, TICKER)
    save(df, TICKER)
    print("Done. Upload this CSV to the GARCH-FX frontend to test.")
