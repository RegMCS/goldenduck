# """
# Download historical stock data for TESTING (50 assets instead of 500)
# Saves as CSV files in data/raw/
# """

# import yfinance as yf
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import sys
# import json
# from tqdm import tqdm
# import time

# SCRIPT_DIR = Path(__file__).parent
# ML_TRAINING_DIR = SCRIPT_DIR.parent
# sys.path.insert(0, str(ML_TRAINING_DIR))

# from config.training_config import (
#     RAW_DATA_DIR,
#     N_ASSETS,
#     PROCESSED_DATA_DIR,
#     TESTING_MODE,
# )


# def get_testing_asset_list():
#     """
#     Get 100 diverse assets for TESTING
#     Mix of high/medium/low volatility across sectors
#     """

#     assets = []

#     # ============================================================
#     # High volatility (tech/growth) - 30 assets
#     # ============================================================
#     tech_stocks = [
#         # Mega-cap tech
#         "NVDA",
#         "TSLA",
#         "AMD",
#         "AAPL",
#         "MSFT",
#         "GOOGL",
#         "META",
#         "AMZN",
#         "NFLX",
#         "PYPL",
#         "SHOP",
#         "SQ",
#         "COIN",
#         "RBLX",
#         "PLTR",
#         # Additional tech
#         "INTC",
#         "QCOM",
#         "AVGO",
#         "CRM",
#         "ADBE",
#         "NOW",
#         "SNOW",
#         "ZM",
#         "DOCU",
#         "UBER",
#         "LYFT",
#         "ABNB",
#         "DASH",
#         "DKNG",
#         "ROKU",
#     ]
#     assets.extend(tech_stocks)

#     # ============================================================
#     # Medium volatility (blue chips) - 40 assets
#     # ============================================================
#     # Financials (10)
#     financials = ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"]
#     assets.extend(financials)

#     # Healthcare (10)
#     healthcare = [
#         "JNJ",
#         "PFE",
#         "ABBV",
#         "MRK",
#         "TMO",
#         "UNH",
#         "LLY",
#         "ABT",
#         "BMY",
#         "AMGN",
#     ]
#     assets.extend(healthcare)

#     # Energy (10)
#     energy = ["XOM", "CVX", "COP", "SLB", "PSX", "MPC", "VLO", "EOG", "OXY", "HAL"]
#     assets.extend(energy)

#     # Consumer/Retail (10)
#     consumer = ["WMT", "HD", "COST", "TGT", "LOW", "NKE", "MCD", "SBUX", "DIS", "CMCSA"]
#     assets.extend(consumer)

#     # ============================================================
#     # Low volatility (defensive + ETFs) - 30 assets
#     # ============================================================
#     # Consumer Staples (8)
#     staples = ["KO", "PEP", "PG", "KMB", "CL", "WMT", "COST", "MDLZ"]
#     assets.extend(staples)

#     # Utilities (7)
#     utilities = ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE"]
#     assets.extend(utilities)

#     # ETFs - Equities (8)
#     equity_etfs = ["SPY", "QQQ", "IWM", "VTI", "VOO", "DIA", "VEA", "VWO"]
#     assets.extend(equity_etfs)

#     # ETFs - Fixed Income & Commodities (7)
#     defensive_etfs = ["TLT", "IEF", "SHY", "AGG", "BND", "GLD", "SLV"]
#     assets.extend(defensive_etfs)

#     return assets[:N_ASSETS]


# def download_single_asset(ticker, start_date="2015-01-01", end_date="2024-12-31"):
#     """
#     Download data for a single asset
#     Returns: DataFrame with returns, or None if failed
#     """
#     try:
#         print(f"  Downloading {ticker}...", end=" ")

#         # ============================================================
#         # FIX: Download single ticker and flatten columns immediately
#         # ============================================================
#         df = yf.download(
#             ticker,
#             start=start_date,
#             end=end_date,
#             auto_adjust=True,
#             progress=False,
#             # Removed show_errors parameter - doesn't exist
#         )

#         # Check if download succeeded
#         if df is None or df.empty:
#             print(f"❌ No data")
#             return None

#         # ============================================================
#         # FIX: Flatten MultiIndex columns
#         # ============================================================
#         if isinstance(df.columns, pd.MultiIndex):
#             # df.columns looks like [('Close', 'AAPL'), ('High', 'AAPL'), ...]
#             # We want just ['Close', 'High', ...]
#             df.columns = df.columns.get_level_values(0)

#         # More lenient: accept 750+ days (3 years)
#         if len(df) < 750:
#             print(f"❌ Only {len(df)} days")
#             return None

#         # With auto_adjust=True, column should be 'Close' (already adjusted)
#         if "Close" not in df.columns:
#             print(f"❌ No 'Close'. Available: {list(df.columns)}")
#             return None

#         # Calculate returns
#         df["returns"] = df["Close"].pct_change()
#         df = df.dropna()

#         # Validation
#         if len(df) < 750:
#             print(f"❌ Only {len(df)} days after cleaning")
#             return None

#         if df["returns"].std() == 0:
#             print(f"❌ Zero volatility")
#             return None

#         # Check for NaN or Inf
#         if df["returns"].isnull().sum() > 0:
#             print(f"❌ {df['returns'].isnull().sum()} NaN values")
#             return None

#         if np.isinf(df["returns"]).sum() > 0:
#             print(f"❌ {np.isinf(df['returns']).sum()} Inf values")
#             return None

#         # ============================================================
#         # FIX: Use .to_numpy() instead of .values to avoid issues
#         # ============================================================
#         output_df = pd.DataFrame(
#             {
#                 "date": df.index.to_numpy(),
#                 "close": df["Close"].to_numpy(),
#                 "returns": df["returns"].to_numpy(),
#             }
#         )

#         print(f"✓ {len(output_df)} days")
#         return output_df

#     except Exception as e:
#         print(f"❌ {type(e).__name__}: {str(e)}")
#         return None


# def test_yfinance_connection():
#     """
#     Test if yfinance is working
#     """
#     print("\n" + "=" * 80)
#     print("TESTING YFINANCE CONNECTION")
#     print("=" * 80)

#     try:
#         print("\nTrying to download AAPL as a test...")
#         df = yf.download(
#             "AAPL",
#             start="2024-01-01",
#             end="2024-12-31",
#             auto_adjust=True,
#             progress=False,
#         )

#         if df is None or df.empty:
#             print("❌ FAILED: yfinance returned no data")
#             print("\nPossible causes:")
#             print("  1. No internet connection")
#             print("  2. Yahoo Finance API is down")
#             print("  3. yfinance library needs update: pip install --upgrade yfinance")
#             return False
#         else:
#             # Flatten MultiIndex if present
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = df.columns.get_level_values(0)

#             print(f"✓ SUCCESS: Downloaded {len(df)} days of AAPL data")
#             print(f"  Date range: {df.index[0]} to {df.index[-1]}")
#             print(f"  Columns: {list(df.columns)}")
#             return True

#     except Exception as e:
#         print(f"❌ ERROR: {str(e)}")
#         print("\nTry updating yfinance:")
#         print("  pip install --upgrade yfinance")
#         return False


# def main():
#     """
#     Download historical data (TESTING MODE: 50 assets)
#     """

#     # Create directories
#     Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
#     Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

#     print("=" * 80)
#     if TESTING_MODE:
#         print("TESTING MODE: DOWNLOADING 50 ASSETS (not 500)")
#     else:
#         print("DOWNLOADING HISTORICAL DATA FOR 500 ASSETS")
#     print("=" * 80)

#     # Test connection first
#     if not test_yfinance_connection():
#         print("\n❌ yfinance connection test failed. Aborting.")
#         print("\nTroubleshooting steps:")
#         print("1. Check internet connection")
#         print("2. Update yfinance: pip install --upgrade yfinance")
#         print("3. Try again in a few minutes (Yahoo Finance might be rate-limiting)")
#         return

#     # Get asset list
#     assets = get_testing_asset_list()
#     print(f"\n📋 Asset list: {len(assets)} tickers")
#     print(f"   First 10: {', '.join(assets[:10])}")

#     print("\n" + "=" * 80)
#     print("DOWNLOADING ASSETS")
#     print("=" * 80)

#     # Download data
#     successful_downloads = []
#     failed_downloads = []

#     for ticker in assets:
#         df = download_single_asset(ticker)

#         if df is not None:
#             # Save to CSV
#             output_path = f"{RAW_DATA_DIR}/{ticker}.csv"
#             df.to_csv(output_path, index=False)
#             successful_downloads.append(ticker)
#         else:
#             failed_downloads.append(ticker)

#         # Rate limiting - be nice to Yahoo Finance
#         time.sleep(0.5)  # Increased from 0.1 to 0.5 seconds

#     # Summary
#     print("\n" + "=" * 80)
#     print("DOWNLOAD SUMMARY")
#     print("=" * 80)
#     print(f"✓ Successful: {len(successful_downloads)}/{len(assets)}")
#     print(f"❌ Failed: {len(failed_downloads)}/{len(assets)}")

#     if successful_downloads:
#         print(f"\n✓ Successfully downloaded:")
#         print(f"   {', '.join(successful_downloads[:10])}")
#         if len(successful_downloads) > 10:
#             print(f"   ... and {len(successful_downloads) - 10} more")

#     if failed_downloads:
#         print(f"\n❌ Failed tickers:")
#         print(f"   {', '.join(failed_downloads[:10])}")
#         if len(failed_downloads) > 10:
#             print(f"   ... and {len(failed_downloads) - 10} more")

#     if len(successful_downloads) == 0:
#         print("\n" + "=" * 80)
#         print("⚠️  CRITICAL: NO ASSETS DOWNLOADED")
#         print("=" * 80)
#         print("\nTroubleshooting:")
#         print("1. Check yfinance version: pip show yfinance")
#         print("2. Update: pip install --upgrade yfinance")
#         print("3. Check internet connection")
#         print("4. Try manual test:")
#         print(
#             "   python3 -c \"import yfinance as yf; print(yf.download('AAPL', start='2024-01-01'))\""
#         )
#         return

#     # Save successful asset list
#     assets_info = {
#         "total_assets": len(successful_downloads),
#         "assets": successful_downloads,
#         "failed": failed_downloads,
#         "download_date": pd.Timestamp.now().isoformat(),
#         "testing_mode": TESTING_MODE,
#     }

#     with open(f"{PROCESSED_DATA_DIR}/downloaded_assets.json", "w") as f:
#         json.dump(assets_info, f, indent=2)

#     print(f"\n✓ Asset list saved to {PROCESSED_DATA_DIR}/downloaded_assets.json")

#     # Split into train/val/test
#     if len(successful_downloads) >= 30:  # Need at least 30 for reasonable split
#         split_assets(successful_downloads)

#         print("\n" + "=" * 80)
#         print("✓ DATA DOWNLOAD COMPLETE")
#         print("=" * 80)
#         print(f"\nNext step: Run 02_generate_training_data.py")
#         print(f"Estimated time: ~{len(successful_downloads) * 0.6:.0f} minutes")
#     else:
#         print(f"\n⚠️  Warning: Only {len(successful_downloads)} assets downloaded")
#         print("   Need at least 30 for proper training/validation/test split")


# def split_assets(assets):
#     """
#     Split assets into train (70%), val (15%), test (15%)
#     """
#     np.random.seed(42)
#     assets_shuffled = assets.copy()
#     np.random.shuffle(assets_shuffled)

#     n_train = int(0.70 * len(assets_shuffled))
#     n_val = int(0.15 * len(assets_shuffled))

#     train_assets = assets_shuffled[:n_train]
#     val_assets = assets_shuffled[n_train : n_train + n_val]
#     test_assets = assets_shuffled[n_train + n_val :]

#     assets_split = {
#         "train": train_assets,
#         "val": val_assets,
#         "test": test_assets,
#         "split_date": pd.Timestamp.now().isoformat(),
#         "train_size": len(train_assets),
#         "val_size": len(val_assets),
#         "test_size": len(test_assets),
#         "testing_mode": TESTING_MODE,
#     }

#     with open(f"{PROCESSED_DATA_DIR}/assets_split.json", "w") as f:
#         json.dump(assets_split, f, indent=2)

#     print("\n" + "=" * 80)
#     print("ASSET SPLIT")
#     print("=" * 80)
#     print(f"Train:      {len(train_assets)} assets (70%)")
#     print(f"Validation: {len(val_assets)} assets (15%)")
#     print(f"Test:       {len(test_assets)} assets (15%)")
#     print(f"\n✓ Split saved to {PROCESSED_DATA_DIR}/assets_split.json")


# if __name__ == "__main__":
#     main()


"""
Download historical stock data for 500 assets (PRODUCTION MODE)
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


def get_production_asset_list():
    """
    Get 500+ diverse assets for PRODUCTION
    Mix of stocks, ETFs, and international exposure
    """

    assets = []

    # ============================================================
    # US STOCKS - 350 assets
    # ============================================================

    # Technology (80)
    tech_stocks = [
        # Mega-cap tech (10)
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "NFLX",
        "ADBE",
        "CRM",
        # Semiconductors (15)
        "AMD",
        "INTC",
        "QCOM",
        "AVGO",
        "MU",
        "AMAT",
        "LRCX",
        "KLAC",
        "ASML",
        "TSM",
        "MRVL",
        "MCHP",
        "ADI",
        "TXN",
        "NXPI",
        # Software & Cloud (20)
        "NOW",
        "ORCL",
        "SAP",
        "INTU",
        "PANW",
        "SNOW",
        "DDOG",
        "NET",
        "ZS",
        "CRWD",
        "WDAY",
        "TEAM",
        "ZM",
        "DOCU",
        "TWLO",
        "SHOP",
        "SQ",
        "PYPL",
        "V",
        "MA",
        # E-commerce & Fintech (15)
        "BABA",
        "JD",
        "PDD",
        "MELI",
        "SE",
        "COIN",
        "HOOD",
        "SOFI",
        "AFRM",
        "UPST",
        "UBER",
        "LYFT",
        "DASH",
        "ABNB",
        "EXPE",
        # Gaming & Entertainment (10)
        "RBLX",
        "U",
        "EA",
        "TTWO",
        "ATVI",
        "DKNG",
        "PENN",
        "ROKU",
        "SPOT",
        "WBD",
        # Hardware & Other (10)
        "HPQ",
        "DELL",
        "WDC",
        "STX",
        "SNPS",
        "CDNS",
        "ANSS",
        "ADSK",
        "FTNT",
        "CHKP",
    ]

    # Financials (60)
    financials = [
        # Banks (20)
        "JPM",
        "BAC",
        "WFC",
        "C",
        "USB",
        "PNC",
        "TFC",
        "COF",
        "KEY",
        "FITB",
        "RF",
        "CFG",
        "HBAN",
        "MTB",
        "ZION",
        "CMA",
        "FHN",
        "SNV",
        "WTFC",
        "UBSI",
        # Investment Banks & Asset Managers (15)
        "GS",
        "MS",
        "BLK",
        "SCHW",
        "TROW",
        "BEN",
        "IVZ",
        "AMG",
        "APAM",
        "JHG",
        "VIRT",
        "LPLA",
        "RJF",
        "SF",
        "MKTX",
        # Insurance (15)
        "BRK.B",
        "AIG",
        "PRU",
        "MET",
        "AFL",
        "ALL",
        "TRV",
        "PGR",
        "CB",
        "AJG",
        "MMC",
        "AON",
        "WRB",
        "RLI",
        "CINF",
        # REITs & Real Estate (10)
        "AMT",
        "PLD",
        "CCI",
        "EQIX",
        "PSA",
        "DLR",
        "O",
        "WELL",
        "AVB",
        "EQR",
    ]

    # Healthcare (60)
    healthcare = [
        # Pharma (20)
        "JNJ",
        "PFE",
        "ABBV",
        "MRK",
        "LLY",
        "BMY",
        "AMGN",
        "GILD",
        "BIIB",
        "REGN",
        "VRTX",
        "ALXN",
        "INCY",
        "NBIX",
        "SRPT",
        "EXEL",
        "BMRN",
        "BLUE",
        "RGEN",
        "UTHR",
        # Biotech (15)
        "MRNA",
        "BNTX",
        "NVAX",
        "CRSP",
        "EDIT",
        "NTLA",
        "BEAM",
        "VCYT",
        "PACB",
        "ILMN",
        "QDEL",
        "TDOC",
        "VEEV",
        "DGX",
        "LH",
        # Medical Devices (15)
        "TMO",
        "DHR",
        "ABT",
        "SYK",
        "BSX",
        "MDT",
        "ISRG",
        "EW",
        "HOLX",
        "BAX",
        "BDX",
        "ZBH",
        "ALGN",
        "IDXX",
        "RMD",
        # Healthcare Services (10)
        "UNH",
        "CVS",
        "CI",
        "HUM",
        "ELV",
        "CNC",
        "MOH",
        "ANTM",
        "WCG",
        "ENSG",
    ]

    # Energy (40)
    energy = [
        # Oil & Gas Majors (15)
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "PXD",
        "PSX",
        "MPC",
        "VLO",
        "OXY",
        "HAL",
        "BKR",
        "DVN",
        "FANG",
        "MRO",
        # Midstream (10)
        "EPD",
        "ET",
        "WMB",
        "KMI",
        "OKE",
        "TRGP",
        "LNG",
        "ENLC",
        "AM",
        "PAA",
        # Renewables (10)
        "NEE",
        "AEP",
        "DUK",
        "SO",
        "D",
        "EXC",
        "SRE",
        "PEG",
        "ED",
        "FE",
        # Alternative Energy (5)
        "ENPH",
        "SEDG",
        "RUN",
        "PLUG",
        "BE",
    ]

    # Consumer Discretionary (40)
    consumer_disc = [
        # Retail (15)
        "AMZN",
        "WMT",
        "HD",
        "TGT",
        "LOW",
        "COST",
        "TJX",
        "ROST",
        "DG",
        "DLTR",
        "BBY",
        "FIVE",
        "BURL",
        "GPS",
        "ANF",
        # Restaurants (10)
        "MCD",
        "SBUX",
        "YUM",
        "QSR",
        "DPZ",
        "CMG",
        "WEN",
        "JACK",
        "BLMN",
        "TXRH",
        # Automotive (10)
        "TSLA",
        "F",
        "GM",
        "RIVN",
        "LCID",
        "NIO",
        "LI",
        "XPEV",
        "TM",
        "HMC",
        # Leisure (5)
        "DIS",
        "CMCSA",
        "NFLX",
        "LYV",
        "MAR",
    ]

    # Consumer Staples (30)
    consumer_staples = [
        # Food & Beverage (15)
        "KO",
        "PEP",
        "PG",
        "KMB",
        "CL",
        "MDLZ",
        "GIS",
        "K",
        "CPB",
        "CAG",
        "SJM",
        "HSY",
        "MKC",
        "CHD",
        "CLX",
        # Tobacco & Alcohol (5)
        "MO",
        "PM",
        "BTI",
        "TAP",
        "STZ",
        # Household Products (10)
        "WMT",
        "COST",
        "KR",
        "SYY",
        "DLTR",
        "DG",
        "BJ",
        "INGR",
        "GO",
        "USFD",
    ]

    # Industrials (40)
    industrials = [
        # Aerospace & Defense (10)
        "BA",
        "LMT",
        "RTX",
        "NOC",
        "GD",
        "LHX",
        "TDG",
        "HWM",
        "TXT",
        "HII",
        # Machinery (15)
        "CAT",
        "DE",
        "CMI",
        "EMR",
        "ETN",
        "ITW",
        "PH",
        "ROK",
        "DOV",
        "XYL",
        "FLS",
        "IR",
        "AME",
        "ROP",
        "IEX",
        # Transportation (10)
        "UPS",
        "FDX",
        "CHRW",
        "JBHT",
        "ODFL",
        "XPO",
        "KNX",
        "R",
        "EXPD",
        "LSTR",
        # Construction (5)
        "VMC",
        "MLM",
        "NUE",
        "X",
        "STLD",
    ]

    assets.extend(tech_stocks)
    assets.extend(financials)
    assets.extend(healthcare)
    assets.extend(energy)
    assets.extend(consumer_disc)
    assets.extend(consumer_staples)
    assets.extend(industrials)

    # ============================================================
    # ETFs - 100 assets
    # ============================================================

    # Equity ETFs - US (25)
    us_equity_etfs = [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "VEA",
        "VWO",
        "EEM",
        "IEMG",
        "VGK",
        "EWJ",
        "EWZ",
        "EWU",
        "EWH",
        "EWY",
        "EWC",
        "EWA",
        "EWS",
        "EWT",
        "VNQ",
        "IYR",
        "XLRE",
        "SCHH",
        "RWR",
    ]

    # Sector ETFs (11)
    sector_etfs = [
        "XLK",
        "XLF",
        "XLV",
        "XLE",
        "XLY",
        "XLP",
        "XLI",
        "XLU",
        "XLB",
        "XLRE",
        "XLC",
    ]

    # International Equity (15)
    intl_equity_etfs = [
        "IEFA",
        "IEMG",
        "VEA",
        "VWO",
        "IXUS",
        "ACWI",
        "VXUS",
        "EFA",
        "EEM",
        "FXI",
        "MCHI",
        "INDA",
        "EWJ",
        "EWU",
        "EZU",
    ]

    # Fixed Income (20)
    fixed_income_etfs = [
        "TLT",
        "IEF",
        "SHY",
        "AGG",
        "BND",
        "LQD",
        "HYG",
        "JNK",
        "MUB",
        "TIP",
        "VCIT",
        "VCSH",
        "VGIT",
        "VGSH",
        "EMB",
        "BNDX",
        "IAGG",
        "GOVT",
        "IGIB",
        "USIG",
    ]

    # Commodities (10)
    commodity_etfs = [
        "GLD",
        "SLV",
        "USO",
        "UNG",
        "DBA",
        "DBC",
        "PDBC",
        "GSG",
        "BCI",
        "GCC",
    ]

    # Factor/Smart Beta (10)
    factor_etfs = [
        "MTUM",
        "VLUE",
        "QUAL",
        "SIZE",
        "USMV",
        "SPHD",
        "VIG",
        "VYM",
        "DGRO",
        "SCHD",
    ]

    # Volatility & Alternatives (9)
    volatility_etfs = [
        "VXX",
        "UVXY",
        "SVXY",
        "VIXY",
        "VIXM",
        "VXZ",
        "ZIV",
        "XIV",
        "TVIX",
    ]

    assets.extend(us_equity_etfs)
    assets.extend(sector_etfs)
    assets.extend(intl_equity_etfs)
    assets.extend(fixed_income_etfs)
    assets.extend(commodity_etfs)
    assets.extend(factor_etfs)
    assets.extend(volatility_etfs)

    # ============================================================
    # International Stocks - 50 assets
    # ============================================================

    international = [
        # China (10)
        "BABA",
        "JD",
        "PDD",
        "BIDU",
        "NIO",
        "LI",
        "XPEV",
        "BILI",
        "NTES",
        "TME",
        # Europe (15)
        "ASML",
        "SAP",
        "NVO",
        "SHEL",
        "BP",
        "HSBC",
        "UL",
        "NVS",
        "AZN",
        "GSK",
        "RIO",
        "BHP",
        "RELX",
        "DEO",
        "BTI",
        # Japan (10)
        "TM",
        "SONY",
        "HMC",
        "MUFG",
        "SMFG",
        "NMR",
        "FUJIY",
        "NSANY",
        "BZUN",
        "KYOCY",
        # Emerging Markets (10)
        "TSM",
        "ITUB",
        "VALE",
        "PBR",
        "SAN",
        "BBVA",
        "ING",
        "ABN",
        "KB",
        "SHG",
        # Canada (5)
        "TD",
        "RY",
        "BMO",
        "BNS",
        "CM",
    ]

    assets.extend(international)

    # Remove duplicates and return
    assets = list(set(assets))

    print(f"\n📋 Generated asset universe: {len(assets)} unique tickers")

    return assets[:N_ASSETS]  # Return up to N_ASSETS


def download_single_asset(ticker, start_date="2015-01-01", end_date="2024-12-31"):
    """
    Download data for a single asset
    Returns: DataFrame with returns, or None if failed
    """
    try:
        # Download single ticker
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

        # Check if download succeeded
        if df is None or df.empty:
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # More lenient: accept 750+ days (3 years)
        if len(df) < 750:
            return None

        if "Close" not in df.columns:
            return None

        # Calculate returns
        df["returns"] = df["Close"].pct_change()
        df = df.dropna()

        # Validation
        if len(df) < 750:
            return None

        if df["returns"].std() == 0:
            return None

        # Check for NaN or Inf
        if df["returns"].isnull().sum() > 0:
            return None

        if np.isinf(df["returns"]).sum() > 0:
            return None

        # Create output dataframe
        output_df = pd.DataFrame(
            {
                "date": df.index.to_numpy(),
                "close": df["Close"].to_numpy(),
                "returns": df["returns"].to_numpy(),
            }
        )

        return output_df

    except Exception as e:
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
    Download historical data
    """

    # Create directories
    Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    if TESTING_MODE:
        print(f"TESTING MODE: DOWNLOADING {N_ASSETS} ASSETS")
    else:
        print(f"PRODUCTION MODE: DOWNLOADING {N_ASSETS} ASSETS")
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
    assets = get_production_asset_list()
    print(f"\n📋 Target: {N_ASSETS} assets")
    print(f"   Generated: {len(assets)} tickers")
    print(f"   First 20: {', '.join(assets[:20])}")

    print("\n" + "=" * 80)
    print("DOWNLOADING ASSETS")
    print("=" * 80)
    print("(This will take ~45 minutes for 500 assets)")

    # Download data with progress bar
    successful_downloads = []
    failed_downloads = []

    for ticker in tqdm(assets, desc="Downloading", unit="asset"):
        df = download_single_asset(ticker)

        if df is not None:
            # Save to CSV
            output_path = f"{RAW_DATA_DIR}/{ticker}.csv"
            df.to_csv(output_path, index=False)
            successful_downloads.append(ticker)
        else:
            failed_downloads.append(ticker)

        # Rate limiting - be nice to Yahoo Finance
        time.sleep(0.3)  # 0.3 seconds between requests

    # Summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(
        f"✓ Successful: {len(successful_downloads)}/{len(assets)} ({len(successful_downloads)/len(assets)*100:.1f}%)"
    )
    print(
        f"❌ Failed: {len(failed_downloads)}/{len(assets)} ({len(failed_downloads)/len(assets)*100:.1f}%)"
    )

    if successful_downloads:
        print(f"\n✓ Sample successful downloads:")
        print(f"   {', '.join(successful_downloads[:15])}")
        if len(successful_downloads) > 15:
            print(f"   ... and {len(successful_downloads) - 15} more")

    if failed_downloads:
        print(f"\n❌ Failed tickers (first 20):")
        print(f"   {', '.join(failed_downloads[:20])}")
        if len(failed_downloads) > 20:
            print(f"   ... and {len(failed_downloads) - 20} more")

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
        "target_assets": N_ASSETS,
        "assets": successful_downloads,
        "failed": failed_downloads,
        "download_date": pd.Timestamp.now().isoformat(),
        "testing_mode": TESTING_MODE,
        "success_rate": len(successful_downloads) / len(assets),
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
        print(f"\nSuccessfully downloaded: {len(successful_downloads)} assets")
        print(f"Next step: Run 02_generate_training_data.py")

        # Estimate training time
        # Assume 10 scenarios per asset, 30 seconds per scenario
        estimated_minutes = len(successful_downloads) * 10 * 30 / 60
        print(
            f"Estimated training data generation time: ~{estimated_minutes:.0f} minutes ({estimated_minutes/60:.1f} hours)"
        )
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
