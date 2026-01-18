import time
import argparse
import requests

BASE_URL = "http://localhost:8000"
POLL_INTERVAL = 3


def main():
    parser = argparse.ArgumentParser(description="Run GARCH job end-to-end")

    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Ticker symbol (e.g. AAPL, GLD, BTC-USD)",
    )

    parser.add_argument(
        "--scenarios",
        type=int,
        default=100,
        help="Number of synthetic scenarios",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=252,
        help="Forecast horizon (days)",
    )

    parser.add_argument("--p", type=int, default=1, help="GARCH p order")
    parser.add_argument("--q", type=int, default=1, help="GARCH q order")

    args = parser.parse_args()

    # 1️⃣ Submit job
    payload = {
        "ticker": args.ticker,
        "num_scenarios": args.scenarios,
        "horizon": args.horizon,
        "p": args.p,
        "q": args.q,
    }

    print("Submitting GARCH job:")
    print(payload)

    resp = requests.post(f"{BASE_URL}/api/generate", json=payload)
    resp.raise_for_status()

    job = resp.json()
    job_id = job["job_id"]

    print(f"Job started: {job_id}")

    # 2️⃣ Poll status
    while True:
        resp = requests.get(f"{BASE_URL}/api/status/{job_id}")
        resp.raise_for_status()
        data = resp.json()

        status = data["status"]
        print(f"Status: {status}")

        if status == "completed":
            print("\nJob completed")

            # ---- Print GARCH parameters ----
            params = data.get("parameters", {})
            print("\nGARCH Parameters:")
            for k, v in params.items():
                print(f"  {k}: {v}")

            # ---- Print validation metrics ----
            metrics = data.get("validation_metrics", {})
            print("\nValidation Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")

            # ---- Download CSV ----
            download_url = BASE_URL + data["download_url"]
            print("\nDownloading results...")

            out = requests.get(download_url)
            out.raise_for_status()

            filename = f"{args.ticker}_{args.scenarios}.csv"
            with open(filename, "wb") as f:
                f.write(out.content)

            print(f"Saved results to {filename}")
            break


        if status == "failed":
            print(f"Job failed: {data.get('error')}")
            break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
