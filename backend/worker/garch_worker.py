import os
import time
import logging

import yfinance as yf
import pandas as pd

import io
import boto3
from job_scheduler.redis_client import redis_client
from job_scheduler.services.job_store import job_store
from job_scheduler.models.enums import JobStatus
from worker.GARCH.services.garch_service import GARCHService
from job_scheduler.db.session import SessionLocal
from job_scheduler.services.job_service import update_job_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("garch-worker")


S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "goldenduck-results")
s3_client = boto3.client("s3")


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

        # Upload to S3
        csv_buffer = io.StringIO()
        combined.to_csv(csv_buffer, index=False)
        s3_key = f"garch/{job_id}.csv"

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME, Key=s3_key, Body=csv_buffer.getvalue()
        )

        s3_url = f"s3://{S3_BUCKET_NAME}/{s3_key}"

        # Persist results in Redis
        job_store.set_parameters(job_id, fitted_params)
        job_store.set_metrics(job_id, metrics)
        job_store.set_output_file(job_id, s3_url)
        job_store.set_status(job_id, "completed")

        # Update DB
        try:
            db = SessionLocal()
            update_job_status(db, job_id, JobStatus.completed, s3_url=s3_url)
            logger.info("Updated DB status for job %s", job_id)
        except Exception as db_exc:
            logger.error("Failed to update DB for job %s: %s", job_id, db_exc)
        finally:
            db.close()

        logger.info("Job %s completed successfully", job_id)

    except Exception as e:
        if job_id:
            logger.exception("Job %s failed", job_id)
            job_store.set_status(job_id, "failed")
            job_store.set_error(job_id, str(e))
        else:
            logger.exception("Worker error before job pickup")
        time.sleep(1)
