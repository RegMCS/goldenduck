import time
import argparse
import requests
import sys

BASE_URL = "http://localhost:8000"
POLL_INTERVAL = 3
USER_ID = "9b8f3f0c-7b9e-4c6e-9f6e-2d9d6f9a3c21"


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

    payload = {
        "ticker": args.ticker,
        "num_scenarios": args.scenarios,
        "horizon": args.horizon,
        "p": args.p,
        "q": args.q,
    }

    submit_url = f"{BASE_URL}/api/generate/user/{USER_ID}"

    print("Submitting GARCH job:")
    print(payload)

    try:
        resp = requests.post(submit_url, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to submit job: {e}", file=sys.stderr)
        sys.exit(1)

    job = resp.json()
    job_id = job["job_id"]

    print(f"Job queued: {job_id}")

    status_url = f"{BASE_URL}/api/status/user/{USER_ID}/{job_id}"

    while True:
        try:
            resp = requests.get(status_url, timeout=10)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Status check failed: {e}", file=sys.stderr)
            time.sleep(POLL_INTERVAL)
            continue

        data = resp.json()
        status = data["status"]

        print(f"Status: {status}")

        if status == "completed":
            print("\nJob completed")

            download_url = f"{BASE_URL}{data['download_url']}"
            print("\nDownloading results...")

            try:
                out = requests.get(download_url, timeout=60)
                out.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Download failed: {e}", file=sys.stderr)
                sys.exit(1)

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
