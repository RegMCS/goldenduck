import time
import argparse
import requests
import sys

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

    try:
        resp = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(
            f"Error: Unable to connect to the API at {BASE_URL}. Please ensure the service is running.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(
            f"Error: Request to {BASE_URL}/api/generate timed out. The service may be overloaded.",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(
            f"Error: API returned an error: {e.response.status_code} - {e.response.text}",
            file=sys.stderr,
        )
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(
            f"Error: An unexpected error occurred while submitting the job: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    job = resp.json()
    job_id = job["job_id"]

    print(f"Job started: {job_id}")

    # Poll status
    while True:
        try:
            resp = requests.get(f"{BASE_URL}/api/status/{job_id}", timeout=10)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(
                f"Error: Unable to connect to the API at {BASE_URL}. Please ensure the service is running.",
                file=sys.stderr,
            )
            sys.exit(1)
        except requests.exceptions.Timeout:
            print(
                f"Error: Request to {BASE_URL}/api/status/{job_id} timed out. Retrying...",
                file=sys.stderr,
            )
            time.sleep(POLL_INTERVAL)
            continue
        except requests.exceptions.HTTPError as e:
            print(
                f"Error: API returned an error: {e.response.status_code} - {e.response.text}",
                file=sys.stderr,
            )
            sys.exit(1)
        except requests.exceptions.RequestException as e:
            print(
                f"Error: An unexpected error occurred while checking job status: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        data = resp.json()

        status = data["status"]
        print(f"Status: {status}")

        if status == "completed":
            print("\nJob completed")

            params = data.get("parameters", {})
            print("\nParameters:")
            for k, v in params.items():
                print(f"  {k}: {v}")

            metrics = data.get("metrics", {})
            print("\nMetrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")

            # ---- Download CSV ----
            download_url = BASE_URL + data["download_url"]
            print("\nDownloading results...")

            try:
                out = requests.get(download_url, timeout=60)
                out.raise_for_status()
            except requests.exceptions.ConnectionError:
                print(
                    f"Error: Unable to connect to {download_url}. Please ensure the service is running.",
                    file=sys.stderr,
                )
                sys.exit(1)
            except requests.exceptions.Timeout:
                print(f"Error: Request to download results timed out.", file=sys.stderr)
                sys.exit(1)
            except requests.exceptions.HTTPError as e:
                print(
                    f"Error: Failed to download results: {e.response.status_code} - {e.response.text}",
                    file=sys.stderr,
                )
                sys.exit(1)
            except requests.exceptions.RequestException as e:
                print(
                    f"Error: An unexpected error occurred while downloading results: {e}",
                    file=sys.stderr,
                )
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
