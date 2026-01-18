import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,   # use DEBUG if you want everything
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

PROJECT_ROOT = "/app"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pprint
import yfinance as yf

from services.evaluation_runner import EvaluationRunner
from config.tickers import SUPPORTED_TICKERS

pp = pprint.PrettyPrinter(indent=2)

all_results = {}

for ticker in SUPPORTED_TICKERS:
    print(f"\n=== Evaluating {ticker} ===")

    historical_data = yf.download(ticker, period="2y", progress=False)

    if historical_data.empty:
        print(f"Skipping {ticker}: no data")
        continue

    runner = EvaluationRunner(historical_data)
    results = runner.compare_distributions(distributions=["normal", "t", "skewt"])

    all_results[ticker] = results

pp.pprint(all_results)
