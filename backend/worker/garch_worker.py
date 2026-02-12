import os
import time
import logging
import sys
import importlib.util
from pathlib import Path

import yfinance as yf
import numpy as np
import pandas as pd

import io
import boto3
from job_scheduler.redis_client import redis_client
from job_scheduler.services.job_store import job_store
from job_scheduler.models.enums import JobStatus
from worker.GARCH.services.garch_service import GARCHService
from job_scheduler.db.session import SessionLocal
from job_scheduler.services.job_service import update_job_status

# Add ml_training to path for importing ParameterPredictor
ML_TRAINING_PATH = Path(__file__).parent.parent / "ml_training"
sys.path.insert(0, str(ML_TRAINING_PATH))

PREDICTOR_PATH = ML_TRAINING_PATH / "scripts" / "06_predict_parameters.py"
spec = importlib.util.spec_from_file_location(
    "predict_parameters_module", PREDICTOR_PATH
)
predict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_module)
predict_parameters = predict_module.predict_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("garch-worker")

# Use absolute path - Docker WORKDIR is /app, output is mounted at /app/output
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Output directory: {OUTPUT_DIR}")

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
        horizon = int(params.get("horizon", 252))

        # Use fixed defaults for GARCH fitting (not exposed to user)
        p = 1
        q = 1
        num_scenarios = 100
        volatility_multiplier = 1.0

        # Extract user knobs for ML parameter prediction
        user_knobs = {
            "desired_volatility": float(params.get("desired_volatility", 1.0)),
            "desired_trend": float(params.get("desired_trend", 0.0)),
            "desired_fat_tails": float(params.get("desired_fat_tails", 1.0)),
            "desired_momentum": float(params.get("desired_momentum", 0.5)),
        }

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

        # Predict delta and theta using ML
        logger.info(f"Predicting GARCH-FX parameters from user knobs...")
        returns = np.log(data["Close"].values[1:] / data["Close"].values[:-1])
        pred_params = predict_parameters(
            historical_returns=returns, user_knobs=user_knobs
        )

        logger.info(f"  Delta (ML): {pred_params['delta']:.4f}")
        logger.info(f"  Theta (heuristic): {pred_params['theta']:.6f}")

        delta_sequence = np.full(horizon, float(pred_params["delta"]))
        scenarios = garch.generate_scenarios_fx(
            num_scenarios=num_scenarios,
            horizon=horizon,
            theta=float(pred_params["theta"]),
            delta_sequence=delta_sequence,
            user_knobs=user_knobs,
        )

        metrics = garch.validate_scenarios(scenarios, user_knobs=user_knobs)

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

        # Generate visualization plots
        try:
            from worker.GARCH.services.visualization_service import VisualizationService

            viz_service = VisualizationService(
                data
            )  # Use 'data' (historical data downloaded above)

            # Plot 1: Price comparison
            price_plot_path = os.path.join(OUTPUT_DIR, f"{job_id}_prices.png")
            viz_service.plot_price_comparison(
                scenarios=scenarios,
                output_path=price_plot_path,
                title=f"Historical vs Synthetic Prices - {ticker}",
                num_scenarios_to_plot=50,
            )

            # Plot 2: Statistics comparison
            stats_plot_path = os.path.join(OUTPUT_DIR, f"{job_id}_stats.png")
            viz_service.plot_statistics_comparison(
                scenarios=scenarios,
                output_path=stats_plot_path,
                user_knobs=user_knobs,
                title=f"Synthetic vs Desired Characteristics - {ticker}",
            )

            logger.info(f"Generated plots: {price_plot_path}, {stats_plot_path}")
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")

        # Persist results (now including predicted parameters)
        results_with_predictions = {
            **fitted_params,
            "delta_predicted": float(pred_params["delta"]),
            "theta_predicted": float(pred_params["theta"]),
            "delta_confidence": pred_params["delta_confidence"],
        }
        job_store.set_parameters(job_id, results_with_predictions)
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
