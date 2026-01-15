# main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import os
import uuid
import yfinance as yf
import logging
from datetime import datetime

from models.schemas import (
    GenerateRequest,
    GenerateResponse,
    GARCHParameters,
    ValidationMetrics,
)
from services.garch_service import GARCHService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GARCH Synthetic Data Generator",
    description="Generate synthetic financial time series using GARCH models",
    version="1.0.0",
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


# ========== EXISTING ENDPOINTS ==========


@app.get("/")
def read_root():
    return {
        "message": "GARCH Synthetic Data Generator API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/generate",
            "status": "/api/status/{job_id}",
            "download": "/api/download/{job_id}",
        },
    }


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "database": str(e)}


# ========== NEW GARCH ENDPOINTS ==========


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_synthetic_data(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Generate synthetic OHLCV data using GARCH model

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
        params = garch.fit_model(
            historical_data, p=request.p, q=request.q, dist="skewt"
        )
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


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get status of generation job
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    response = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
    }

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
    Download generated synthetic data as CSV
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

    return FileResponse(
        output_file, media_type="text/csv", filename=f"synthetic_garch_{job_id}.csv"
    )


@app.get("/api/jobs")
async def list_jobs():
    """
    List all generation jobs
    """
    return {
        "total_jobs": len(jobs),
        "jobs": [
            {"job_id": job_id, "status": job["status"], "created_at": job["created_at"]}
            for job_id, job in jobs.items()
        ],
    }
