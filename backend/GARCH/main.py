# main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import os
import uuid
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Optional, List

from worker.GARCH.models.schemas import (
    GenerateRequest,
    GenerateResponse,
    GARCHParameters,
    ValidationMetrics,
    # NEW SCHEMAS
    GenerateFXRequest,
    GenerateFXResponse,
    ScenarioListResponse,
)
from worker.GARCH.services.garch_service import GARCHService
from worker.GARCH.services.scenarios import list_scenarios  # NEW

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GARCH Synthetic Data Generator",
    description="Generate synthetic financial time series using GARCH and GARCH-FX models",
    version="2.0.0",  # Updated version
)

# Database connection
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "goldenduck")
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@db:5432/{POSTGRES_DB}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Job storage (in-memory for PoC, move to Redis/DB for production)
jobs = {}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ========== ROOT & HEALTH ==========


@app.get("/")
def read_root():
    return {
        "message": "GARCH Synthetic Data Generator API",
        "version": "2.0.0",
        "endpoints": {
            "standard_garch": {
                "generate": "/api/generate",
                "status": "/api/status/{job_id}",
                "download": "/api/download/{job_id}",
            },
            "garch_fx": {
                "generate": "/api/fx/generate",
                "scenarios": "/api/fx/scenarios",
                "status": "/api/status/{job_id}",
                "download": "/api/download/{job_id}",
            },
            "utility": {
                "jobs": "/api/jobs",
                "health": "/health",
            },
        },
    }


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": str(e)}


# ========== STANDARD GARCH ENDPOINTS (EXISTING) ==========


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_synthetic_data(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Generate synthetic OHLCV data using standard GARCH model

    This endpoint:
    1. Fetches historical data for the ticker
    2. Fits GARCH model
    3. Generates synthetic scenarios in background
    4. Returns job_id for status tracking
    """
    job_id = str(uuid.uuid4())

    try:
        # Initialize job status
        jobs[job_id] = {
            "status": "initializing",
            "progress": 0,
            "total": request.num_scenarios,
            "created_at": datetime.now().isoformat(),
            "model_type": "standard_garch",  # NEW: Track model type
            "parameters": None,
            "error": None,
        }

        # Fetch historical data
        logger.info(f"Fetching historical data for {request.ticker}")
        historical_data = yf.download(request.ticker, period="2y", progress=False)

        if historical_data.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for ticker {request.ticker}"
            )

        # Submit generation task to background
        background_tasks.add_task(
            generate_garch_background, job_id, historical_data, request
        )

        return GenerateResponse(
            job_id=job_id,
            status="queued",
            message=f"Generation job submitted. Track progress at /api/status/{job_id}",
        )

    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


def generate_garch_background(job_id: str, historical_data, request: GenerateRequest):
    """
    Background task for GARCH generation
    """
    try:
        jobs[job_id]["status"] = "fitting_model"

        # Initialize GARCH service
        garch = GARCHService()

        # Fit model
        logger.info(f"Fitting GARCH({request.p},{request.q}) model")
        params = garch.fit_with_retry(historical_data, p=request.p, q=request.q)
        jobs[job_id]["parameters"] = params

        # Generate scenarios
        jobs[job_id]["status"] = "generating"
        logger.info(f"Generating {request.num_scenarios} scenarios")

        scenarios = garch.generate_scenarios(
            num_scenarios=request.num_scenarios,
            horizon=request.horizon,
            volatility_multiplier=request.volatility_multiplier,
        )

        # Validate
        jobs[job_id]["status"] = "validating"
        validation_metrics = garch.validate_scenarios(scenarios)
        jobs[job_id]["validation_metrics"] = validation_metrics

        # Save to CSV
        jobs[job_id]["status"] = "saving"
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Save all scenarios to single CSV with scenario ID column
        import pandas as pd

        all_data = []
        for i, scenario in enumerate(scenarios):
            scenario_df = scenario.copy()
            scenario_df["scenario_id"] = i + 1
            scenario_df["day"] = range(1, len(scenario_df) + 1)
            all_data.append(scenario_df)

        combined = pd.concat(all_data, ignore_index=True)
        output_path = f"{output_dir}/{job_id}.csv"
        combined.to_csv(output_path, index=False)

        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_file"] = output_path
        jobs[job_id]["num_scenarios"] = len(scenarios)

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Error in background task: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# ========== NEW: GARCH-FX ENDPOINTS ==========


@app.get("/api/fx/scenarios", response_model=ScenarioListResponse)
async def get_available_scenarios():
    """
    Get list of available predefined stress testing scenarios

    Returns scenario names and descriptions for use with /api/fx/generate
    """
    scenarios = list_scenarios()

    scenario_details = {
        "sudden_crisis": "Sudden crisis at day 200, lasting 300 days, then recovery",
        "gradual_escalation": "Gradual escalation from normal to extreme stress over time",
        "crisis_waves": "Multiple crisis waves with 5 cycles of calm-to-crisis transitions",
        "prolonged_stress": "Extended high-volatility period simulating prolonged market stress",
        "flash_crash": "Brief extreme volatility spike followed by quick recovery",
    }

    return ScenarioListResponse(
        scenarios=[
            {"name": name, "description": scenario_details.get(name, "No description")}
            for name in scenarios
        ]
    )


@app.post("/api/fx/generate", response_model=GenerateFXResponse)
async def generate_garchfx_data(
    request: GenerateFXRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Generate synthetic OHLCV data using GARCH-FX framework with stress testing scenarios
    """
    job_id = str(uuid.uuid4())

    try:
        # Validate scenario if provided
        if request.scenario_type:
            from worker.GARCH.services.scenarios import list_scenarios

            available_scenarios = list_scenarios()
            if request.scenario_type not in available_scenarios:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid scenario. Available: {', '.join(available_scenarios)}",
                )

        # Initialize job status
        jobs[job_id] = {
            "status": "initializing",
            "progress": 0,
            "total": request.num_scenarios,
            "created_at": datetime.now().isoformat(),
            "model_type": "garch_fx",
            "scenario_type": request.scenario_type,
            "theta": request.theta,
            "regime_switching": request.regime_switching,
            "parameters": None,
            "error": None,
        }

        # Fetch historical data
        logger.info(f"[GARCH-FX] Fetching historical data for {request.ticker}")

        # Download data with proper structure
        historical_data = yf.download(
            request.ticker,
            period="2y",
            progress=False,
            auto_adjust=True,  # Simplifies column structure
        )

        if historical_data.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for ticker {request.ticker}"
            )

        # Ensure it's a DataFrame with 'Close' column
        if isinstance(historical_data.columns, pd.MultiIndex):
            # Flatten multi-index columns
            historical_data.columns = [
                "_".join(col).strip() if isinstance(col, tuple) else col
                for col in historical_data.columns.values
            ]

        # Find Close column (case-insensitive)
        close_cols = [col for col in historical_data.columns if "close" in col.lower()]
        if not close_cols:
            raise HTTPException(
                status_code=500,
                detail=f"No 'Close' column found. Available: {list(historical_data.columns)}",
            )

        # Rename to standard 'Close' if needed
        if close_cols[0] != "Close":
            historical_data = historical_data.rename(columns={close_cols[0]: "Close"})

        # Keep only Close column and reset index
        historical_data = historical_data[["Close"]].reset_index()

        logger.info(f"[GARCH-FX] Downloaded {len(historical_data)} days of data")

        # Submit generation task to background
        background_tasks.add_task(
            generate_garchfx_background, job_id, historical_data, request
        )

        return GenerateFXResponse(
            job_id=job_id,
            status="queued",
            message=f"GARCH-FX generation job submitted. Track progress at /api/status/{job_id}",
            scenario_type=request.scenario_type,
            theta=request.theta,
        )

    except Exception as e:
        logger.error(f"Error submitting GARCH-FX job: {e}", exc_info=True)
        if job_id in jobs:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))


def generate_garchfx_background(
    job_id: str, historical_data, request: GenerateFXRequest
):
    """
    Background task for GARCH-FX generation
    """
    try:
        jobs[job_id]["status"] = "fitting_model"

        # Initialize GARCH service
        garch = GARCHService()

        if not isinstance(historical_data, pd.DataFrame):
            logger.warning("Converting historical_data to DataFrame")
            historical_data = pd.DataFrame(historical_data)

        # Ensure 'Close' column exists
        if "Close" not in historical_data.columns:
            # If yfinance returns multi-index columns, flatten them
            if isinstance(historical_data.columns, pd.MultiIndex):
                historical_data.columns = historical_data.columns.droplevel(1)

            # If still no 'Close', try to find it
            close_col = [
                col for col in historical_data.columns if "close" in str(col).lower()
            ]
            if close_col:
                historical_data = historical_data.rename(
                    columns={close_col[0]: "Close"}
                )
            else:
                raise ValueError("No 'Close' column found in historical data")

        # Fit model
        logger.info(f"[GARCH-FX] Fitting GARCH({request.p},{request.q}) model")
        params = garch.fit_with_retry(historical_data, p=request.p, q=request.q)
        jobs[job_id]["parameters"] = params

        # Generate scenarios using GARCH-FX
        jobs[job_id]["status"] = "generating"
        logger.info(
            f"[GARCH-FX] Generating {request.num_scenarios} scenarios "
            f"(theta={request.theta}, scenario={request.scenario_type})"
        )

        scenarios = garch.generate_scenarios_fx(
            num_scenarios=request.num_scenarios,
            horizon=request.horizon,
            theta=request.theta,
            scenario_type=request.scenario_type,
            delta_sequence=request.delta_sequence,
            regime_switching=request.regime_switching,
            regime_states=request.regime_states,
            regimes=request.regimes,
            seed_start=request.seed_start,
        )

        # Validate
        jobs[job_id]["status"] = "validating"
        validation_metrics = garch.validate_scenarios(scenarios)
        jobs[job_id]["validation_metrics"] = validation_metrics

        # Save to CSV
        jobs[job_id]["status"] = "saving"
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        all_data = []
        for i, scenario in enumerate(scenarios):
            scenario_df = scenario.copy()
            scenario_df["scenario_id"] = i + 1
            scenario_df["day"] = range(1, len(scenario_df) + 1)
            all_data.append(scenario_df)

        combined = pd.concat(all_data, ignore_index=True)
        output_path = f"{output_dir}/{job_id}.csv"
        combined.to_csv(output_path, index=False)

        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_file"] = output_path
        jobs[job_id]["num_scenarios"] = len(scenarios)

        logger.info(f"[GARCH-FX] Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"[GARCH-FX] Error in background task: {e}", exc_info=True)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# ========== SHARED ENDPOINTS ==========


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get status of generation job (works for both GARCH and GARCH-FX)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    response = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "model_type": job.get("model_type", "standard_garch"),  # NEW
    }

    # Add GARCH-FX specific info
    if job.get("model_type") == "garch_fx":
        response["scenario_type"] = job.get("scenario_type")
        response["theta"] = job.get("theta")
        response["regime_switching"] = job.get("regime_switching")

    if job["status"] == "completed":
        response["download_url"] = f"/api/download/{job_id}"
        response["parameters"] = job.get("parameters")
        response["validation_metrics"] = job.get("validation_metrics")
        response["num_scenarios"] = job.get("num_scenarios")
    elif job["status"] == "failed":
        response["error"] = job.get("error")
    elif job["status"] == "generating":
        response["progress"] = job.get("progress", 0)
        response["total"] = job.get("total", 0)

    return response


@app.get("/api/download/{job_id}")
async def download_results(job_id: str):
    """
    Download generated synthetic data as CSV (works for both GARCH and GARCH-FX)
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, detail=f"Job not completed. Status: {job['status']}"
        )

    output_file = job.get("output_file")
    if not output_file or not os.path.exists(output_file):
        raise HTTPException(status_code=404, detail="Output file not found")

    model_type = job.get("model_type", "garch")
    return FileResponse(
        output_file,
        media_type="text/csv",
        filename=f"synthetic_{model_type}_{job_id}.csv",
    )


@app.get("/api/jobs")
async def list_jobs():
    """
    List all generation jobs
    """
    return {
        "total_jobs": len(jobs),
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "model_type": job.get("model_type", "standard_garch"),
                "scenario_type": job.get("scenario_type"),
                "created_at": job["created_at"],
            }
            for job_id, job in jobs.items()
        ],
    }
