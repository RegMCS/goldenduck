import json
import os
import time
import logging
import redis
import yfinance as yf
import pandas as pd

from GARCH.services.garch_service import GARCHService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("garch-worker")

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

def create_redis_client(retry_delay: int = 5) -> redis.Redis:
    """
    Create a Redis client and validate the connection.
    Retries indefinitely with a delay if the connection cannot be established.
    """
    while True:
        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_keepalive=True,
                retry_on_timeout=True,
            )
            # Force a connection attempt to validate that Redis is reachable
            client.ping()
            logger.info("Successfully connected to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
            return client
        except Exception as e:
            logger.error(
                "Failed to connect to Redis at %s:%s: %s. Retrying in %s seconds...",
                REDIS_HOST,
                REDIS_PORT,
                e,
                retry_delay,
            )
            time.sleep(retry_delay)

redis_client = create_redis_client()
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info("GARCH worker started, waiting for jobs...")

while True:
    try:
        # Block until a job arrives
        _, job_id = redis_client.brpop("queue:garch")
        logger.info(f"Picked up job {job_id}")
        redis_client.set(f"job:{job_id}:status", "running")

        meta_raw = redis_client.get(f"job:{job_id}:meta")
        if not meta_raw:
            raise ValueError("Job metadata not found")

        meta = json.loads(meta_raw)
        req = meta["request"]

        ticker = req["ticker"]
        p = int(req.get("p", 1))
        q = int(req.get("q", 1))
        num_scenarios = int(req.get("num_scenarios", 100))
        horizon = int(req.get("horizon", 252))
        volatility_multiplier = float(req.get("volatility_multiplier", 1.0))

        logger.info(
            f"Running GARCH for {ticker} "
            f"(p={p}, q={q}, scenarios={num_scenarios}, horizon={horizon})"
        )

        data = yf.download(
            ticker,
            period="2y",
            progress=False,
            threads=False,
        )

        if data.empty:
            raise ValueError(f"No market data returned for ticker {ticker}")

        garch = GARCHService()
        params = garch.fit_with_retry(data, p=p, q=q)

        scenarios = garch.generate_scenarios(
            num_scenarios=num_scenarios,
            horizon=horizon,
            volatility_multiplier=volatility_multiplier,
        )

        metrics = garch.validate_scenarios(scenarios)

        logger.info(f"Validation metrics for {job_id}: {metrics}")
        all_rows = []
        for i, scenario_df in enumerate(scenarios):
            df = scenario_df.copy()
            df["scenario_id"] = i + 1
            all_rows.append(df)

        combined = pd.concat(all_rows, ignore_index=True)

        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.csv")
        combined.to_csv(output_path, index=False)

        redis_client.set(f"job:{job_id}:parameters", json.dumps(params))
        redis_client.set(f"job:{job_id}:metrics", json.dumps(metrics))
        redis_client.set(f"job:{job_id}:output_file", output_path)
        redis_client.set(f"job:{job_id}:status", "completed")

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")

        redis_client.set(f"job:{job_id}:status", "failed")
        redis_client.set(f"job:{job_id}:error", str(e))

        time.sleep(1)
