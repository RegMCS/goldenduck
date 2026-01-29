"""
Download historical stock data for TESTING (50 assets instead of 500)
Saves as CSV files in data/raw/
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from tqdm import tqdm
import time

SCRIPT_DIR = Path(__file__).parent
ML_TRAINING_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ML_TRAINING_DIR))

from config.training_config import (
    RAW_DATA_DIR,
    N_ASSETS,
    PROCESSED_DATA_DIR,
    TESTING_MODE,
)


def get_testing_asset_list():
    """
    Get 100 diverse assets for TESTING
    Mix of high/medium/low volatility across sectors
    """

    assets = []

    # ============================================================
    # High volatility (tech/growth) - 30 assets
    # ============================================================
    tech_stocks = [
        # Mega-cap tech
        "NVDA",
        "TSLA",
        "AMD",
        "AAPL",
        "MSFT",
        "GOOGL",
        "META",
        "AMZN",
        "NFLX",
        "PYPL",
        "SHOP",
        "SQ",
        "COIN",
        "RBLX",
        "PLTR",
        # Additional tech
        "INTC",
        "QCOM",
        "AVGO",
        "CRM",
        "ADBE",
        "NOW",
        "SNOW",
        "ZM",
        "DOCU",
        "UBER",
        "LYFT",
        "ABNB",
        "DASH",
        "DKNG",
        "ROKU",
    ]
    assets.extend(tech_stocks)

    # ============================================================
    # Medium volatility (blue chips) - 40 assets
    # ============================================================
    # Financials (10)
    financials = ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"]
    assets.extend(financials)

    # Healthcare (10)
    healthcare = [
        "JNJ",
        "PFE",
        "ABBV",
        "MRK",
        "TMO",
        "UNH",
        "LLY",
        "ABT",
        "BMY",
        "AMGN",
    ]
    assets.extend(healthcare)

    # Energy (10)
    energy = ["XOM", "CVX", "COP", "SLB", "PSX", "MPC", "VLO", "EOG", "OXY", "HAL"]
    assets.extend(energy)

    # Consumer/Retail (10)
    consumer = ["WMT", "HD", "COST", "TGT", "LOW", "NKE", "MCD", "SBUX", "DIS", "CMCSA"]
    assets.extend(consumer)

    # ============================================================
    # Low volatility (defensive + ETFs) - 30 assets
    # ============================================================
    # Consumer Staples (8)
    staples = ["KO", "PEP", "PG", "KMB", "CL", "WMT", "COST", "MDLZ"]
    assets.extend(staples)

    # Utilities (7)
    utilities = ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE"]
    assets.extend(utilities)

    # ETFs - Equities (8)
    equity_etfs = ["SPY", "QQQ", "IWM", "VTI", "VOO", "DIA", "VEA", "VWO"]
    assets.extend(equity_etfs)

    # ETFs - Fixed Income & Commodities (7)
    defensive_etfs = ["TLT", "IEF", "SHY", "AGG", "BND", "GLD", "SLV"]
    assets.extend(defensive_etfs)

    return assets[:N_ASSETS]


def download_single_asset(ticker, start_date="2015-01-01", end_date="2024-12-31"):
    """
    Download data for a single asset
    Returns: DataFrame with returns, or None if failed
    """
    try:
        print(f"  Downloading {ticker}...", end=" ")

        # ============================================================
        # FIX: Download single ticker and flatten columns immediately
        # ============================================================
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            # Removed show_errors parameter - doesn't exist
        )

        # Check if download succeeded
        if df is None or df.empty:
            print(f"❌ No data")
            return None

        # ============================================================
        # FIX: Flatten MultiIndex columns
        # ============================================================
        if isinstance(df.columns, pd.MultiIndex):
            # df.columns looks like [('Close', 'AAPL'), ('High', 'AAPL'), ...]
            # We want just ['Close', 'High', ...]
            df.columns = df.columns.get_level_values(0)

        # More lenient: accept 750+ days (3 years)
        if len(df) < 750:
            print(f"❌ Only {len(df)} days")
            return None

        # With auto_adjust=True, column should be 'Close' (already adjusted)
        if "Close" not in df.columns:
            print(f"❌ No 'Close'. Available: {list(df.columns)}")
            return None

        # Calculate returns
        df["returns"] = df["Close"].pct_change()
        df = df.dropna()

        # Validation
        if len(df) < 750:
            print(f"❌ Only {len(df)} days after cleaning")
            return None

        if df["returns"].std() == 0:
            print(f"❌ Zero volatility")
            return None

        # Check for NaN or Inf
        if df["returns"].isnull().sum() > 0:
            print(f"❌ {df['returns'].isnull().sum()} NaN values")
            return None

        if np.isinf(df["returns"]).sum() > 0:
            print(f"❌ {np.isinf(df['returns']).sum()} Inf values")
            return None

        # ============================================================
        # FIX: Use .to_numpy() instead of .values to avoid issues
        # ============================================================
        output_df = pd.DataFrame(
            {
                "date": df.index.to_numpy(),
                "close": df["Close"].to_numpy(),
                "returns": df["returns"].to_numpy(),
            }
        )

        print(f"✓ {len(output_df)} days")
        return output_df

    except Exception as e:
        print(f"❌ {type(e).__name__}: {str(e)}")
        return None


def test_yfinance_connection():
    """
    Test if yfinance is working
    """
    print("\n" + "=" * 80)
    print("TESTING YFINANCE CONNECTION")
    print("=" * 80)

    try:
        print("\nTrying to download AAPL as a test...")
        df = yf.download(
            "AAPL",
            start="2024-01-01",
            end="2024-12-31",
            auto_adjust=True,
            progress=False,
        )

        if df is None or df.empty:
            print("❌ FAILED: yfinance returned no data")
            print("\nPossible causes:")
            print("  1. No internet connection")
            print("  2. Yahoo Finance API is down")
            print("  3. yfinance library needs update: pip install --upgrade yfinance")
            return False
        else:
            # Flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            print(f"✓ SUCCESS: Downloaded {len(df)} days of AAPL data")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Columns: {list(df.columns)}")
            return True

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("\nTry updating yfinance:")
        print("  pip install --upgrade yfinance")
        return False


def main():
    """
    Download historical data (TESTING MODE: 50 assets)
    """

    # Create directories
    Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    if TESTING_MODE:
        print("TESTING MODE: DOWNLOADING 50 ASSETS (not 500)")
    else:
        print("DOWNLOADING HISTORICAL DATA FOR 500 ASSETS")
    print("=" * 80)

    # Test connection first
    if not test_yfinance_connection():
        print("\n❌ yfinance connection test failed. Aborting.")
        print("\nTroubleshooting steps:")
        print("1. Check internet connection")
        print("2. Update yfinance: pip install --upgrade yfinance")
        print("3. Try again in a few minutes (Yahoo Finance might be rate-limiting)")
        return

    # Get asset list
    assets = get_testing_asset_list()
    print(f"\n📋 Asset list: {len(assets)} tickers")
    print(f"   First 10: {', '.join(assets[:10])}")

    print("\n" + "=" * 80)
    print("DOWNLOADING ASSETS")
    print("=" * 80)

    # Download data
    successful_downloads = []
    failed_downloads = []

    for ticker in assets:
        df = download_single_asset(ticker)

        if df is not None:
            # Save to CSV
            output_path = f"{RAW_DATA_DIR}/{ticker}.csv"
            df.to_csv(output_path, index=False)
            successful_downloads.append(ticker)
        else:
            failed_downloads.append(ticker)

        # Rate limiting - be nice to Yahoo Finance
        time.sleep(0.5)  # Increased from 0.1 to 0.5 seconds

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✓ Successful: {len(successful_downloads)}/{len(assets)}")
    print(f"❌ Failed: {len(failed_downloads)}/{len(assets)}")

    if successful_downloads:
        print(f"\n✓ Successfully downloaded:")
        print(f"   {', '.join(successful_downloads[:10])}")
        if len(successful_downloads) > 10:
            print(f"   ... and {len(successful_downloads) - 10} more")

    if failed_downloads:
        print(f"\n❌ Failed tickers:")
        print(f"   {', '.join(failed_downloads[:10])}")
        if len(failed_downloads) > 10:
            print(f"   ... and {len(failed_downloads) - 10} more")

    if len(successful_downloads) == 0:
        print("\n" + "=" * 80)
        print("⚠️  CRITICAL: NO ASSETS DOWNLOADED")
        print("=" * 80)
        print("\nTroubleshooting:")
        print("1. Check yfinance version: pip show yfinance")
        print("2. Update: pip install --upgrade yfinance")
        print("3. Check internet connection")
        print("4. Try manual test:")
        print(
            "   python3 -c \"import yfinance as yf; print(yf.download('AAPL', start='2024-01-01'))\""
        )
        return

    # Save successful asset list
    assets_info = {
        "total_assets": len(successful_downloads),
        "assets": successful_downloads,
        "failed": failed_downloads,
        "download_date": pd.Timestamp.now().isoformat(),
        "testing_mode": TESTING_MODE,
    }

    with open(f"{PROCESSED_DATA_DIR}/downloaded_assets.json", "w") as f:
        json.dump(assets_info, f, indent=2)

    print(f"\n✓ Asset list saved to {PROCESSED_DATA_DIR}/downloaded_assets.json")

    # Split into train/val/test
    if len(successful_downloads) >= 30:  # Need at least 30 for reasonable split
        split_assets(successful_downloads)

        print("\n" + "=" * 80)
        print("✓ DATA DOWNLOAD COMPLETE")
        print("=" * 80)
        print(f"\nNext step: Run 02_generate_training_data.py")
        print(f"Estimated time: ~{len(successful_downloads) * 0.6:.0f} minutes")
    else:
        print(f"\n⚠️  Warning: Only {len(successful_downloads)} assets downloaded")
        print("   Need at least 30 for proper training/validation/test split")


def split_assets(assets):
    """
    Split assets into train (70%), val (15%), test (15%)
    """
    np.random.seed(42)
    assets_shuffled = assets.copy()
    np.random.shuffle(assets_shuffled)

    n_train = int(0.70 * len(assets_shuffled))
    n_val = int(0.15 * len(assets_shuffled))

    train_assets = assets_shuffled[:n_train]
    val_assets = assets_shuffled[n_train : n_train + n_val]
    test_assets = assets_shuffled[n_train + n_val :]

    assets_split = {
        "train": train_assets,
        "val": val_assets,
        "test": test_assets,
        "split_date": pd.Timestamp.now().isoformat(),
        "train_size": len(train_assets),
        "val_size": len(val_assets),
        "test_size": len(test_assets),
        "testing_mode": TESTING_MODE,
    }

    with open(f"{PROCESSED_DATA_DIR}/assets_split.json", "w") as f:
        json.dump(assets_split, f, indent=2)

    print("\n" + "=" * 80)
    print("ASSET SPLIT")
    print("=" * 80)
    print(f"Train:      {len(train_assets)} assets (70%)")
    print(f"Validation: {len(val_assets)} assets (15%)")
    print(f"Test:       {len(test_assets)} assets (15%)")
    print(f"\n✓ Split saved to {PROCESSED_DATA_DIR}/assets_split.json")


if __name__ == "__main__":
    main()
