import os
import time
import logging

import yfinance as yf
import pandas as pd

from job_scheduler.redis_client import redis_client
from job_scheduler.services.job_store import job_store
from worker.GARCH.services.garch_service import GARCHService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("garch-worker")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info("GARCH worker started, waiting for jobs...")

while True:
    job_id = None
    try:
        # wait for job
        _, job_id = redis_client.brpop("queue:garch")
        logger.info("Picked up job %s", job_id)

        job = job_store.get_job(job_id)
        if not job:
            raise ValueError("Job metadata not found")

        job_store.set_status(job_id, "running")

        params = job["parameters"]

        ticker = params["ticker"]
        p = int(params.get("p", 1))
        q = int(params.get("q", 1))
        num_scenarios = int(params.get("num_scenarios", 100))
        horizon = int(params.get("horizon", 252))
        volatility_multiplier = float(params.get("volatility_multiplier", 1.0))

        logger.info(
            "Running GARCH for %s (p=%s, q=%s, scenarios=%s, horizon=%s)",
            ticker,
            p,
            q,
            num_scenarios,
            horizon,
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

        fitted_params = garch.fit_with_retry(data, p=p, q=q)

        scenarios = garch.generate_scenarios(
            num_scenarios=num_scenarios,
            horizon=horizon,
            volatility_multiplier=volatility_multiplier,
        )

        metrics = garch.validate_scenarios(scenarios)

        all_rows = []
        for i, scenario_df in enumerate(scenarios):
            df = scenario_df.copy()
            df["scenario_id"] = i + 1
            all_rows.append(df)

        combined = pd.concat(all_rows, ignore_index=True)

        output_path = os.path.join(OUTPUT_DIR, f"{job_id}.csv")
        combined.to_csv(output_path, index=False)

        # Persist results
        job_store.set_parameters(job_id, fitted_params)
        job_store.set_metrics(job_id, metrics)
        job_store.set_output_file(job_id, output_path)
        job_store.set_status(job_id, "completed")

        logger.info("Job %s completed successfully", job_id)

    except Exception as e:
        if job_id:
            logger.exception("Job %s failed", job_id)
            job_store.set_status(job_id, "failed")
            job_store.set_error(job_id, str(e))
        else:
            logger.exception("Worker error before job pickup")
        time.sleep(1)
