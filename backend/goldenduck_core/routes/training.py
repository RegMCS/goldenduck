"""
Training API Routes
===================
All routes are admin-only (require_admin dependency).

Endpoints:
  POST  /api/training/start                        — trigger a new training run
  GET   /api/training/runs                         — list all past runs
  GET   /api/training/runs/{run_id}                — status + live progress for one run
  GET   /api/training/uploads                      — list manually uploaded models
  POST  /api/training/models/upload                — upload .pkl, version name, store in S3
  POST  /api/training/models/{model_name}/activate — set a saved model as active
  GET   /api/training/active-model/evaluation      — evaluation report for active model
"""

import asyncio
import json
import os
import tempfile
import uuid as uuid_lib
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Body, File, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from goldenduck_core.db.session import get_db
from goldenduck_core.models.training_job import TrainingJob
from goldenduck_core.models.uploaded_model import UploadedModel
from goldenduck_core.models.user import User
from goldenduck_core.services.auth_service import require_admin
from goldenduck_core.services.training_store import training_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"])

# ─── Paths ────────────────────────────────────────────────────────────────────

ML_TRAINING_DIR = Path(__file__).parent.parent.parent / "ml_training"
MODEL_SAVE_DIR = ML_TRAINING_DIR / "models" / "saved_models"
EVALUATION_DIR = ML_TRAINING_DIR / "models" / "evaluation"
ACTIVE_MODEL_NAME = "rf_delta.pkl"

# Max upload size for manual model files (bytes)
MAX_MODEL_UPLOAD_BYTES = 256 * 1024 * 1024
# Read upload in chunks; spool rolls from RAM to disk after this many bytes
UPLOAD_CHUNK_BYTES = 1024 * 1024
UPLOAD_SPOOL_MAX_IN_MEMORY = 16 * 1024 * 1024


# ─── Schemas ─────────────────────────────────────────────────────────────────


class StartTrainingRequest(BaseModel):
    testing_mode: bool = True
    n_assets: int = Field(50, ge=5, le=500)
    n_scenarios: int = Field(10, ge=5, le=50)
    run_evaluation: bool = True


class TrainingRunResponse(BaseModel):
    id: str
    status: str
    config: dict
    triggered_by: str
    started_at: str
    completed_at: Optional[str]
    step: Optional[str]
    step_index: Optional[int]
    model_name: Optional[str]
    is_active: bool
    evaluation_report: Optional[dict]
    error: Optional[str]

    model_config = {"from_attributes": True}


class TrainingRunsListResponse(BaseModel):
    runs: List[TrainingRunResponse]
    total: int


class UploadedModelResponse(BaseModel):
    id: str
    model_name: str
    s3_key: str
    uploaded_by: str
    created_at: str
    file_size_bytes: Optional[int]
    is_active: bool

    model_config = {"from_attributes": True}


class UploadedModelsListResponse(BaseModel):
    uploads: List[UploadedModelResponse]
    total: int


class ModelUploadResult(BaseModel):
    id: str
    model_name: str
    s3_key: str


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _allocate_upload_model_name(db: Session) -> str:
    """
    Same pattern as training worker: rf_delta_YYYYMMDD_HHMMSS.pkl.
    If that name is taken (upload or training job), append a short random suffix.
    """
    tz = timezone(timedelta(hours=8))
    ts = datetime.now(tz=tz).strftime("%Y%m%d_%H%M%S")
    for _ in range(20):
        suffix = uuid_lib.uuid4().hex[:6] if _ else ""
        name = f"rf_delta_{ts}.pkl" if not suffix else f"rf_delta_{ts}_{suffix}.pkl"
        taken_up = (
            db.query(UploadedModel).filter(UploadedModel.model_name == name).first()
        )
        taken_tj = db.query(TrainingJob).filter(TrainingJob.model_name == name).first()
        if not taken_up and not taken_tj:
            return name
    return f"rf_delta_{ts}_{uuid_lib.uuid4().hex}.pkl"


def _uploaded_model_to_response(row: UploadedModel) -> UploadedModelResponse:
    return UploadedModelResponse(
        id=str(row.id),
        model_name=row.model_name,
        s3_key=row.s3_key,
        uploaded_by=str(row.uploaded_by),
        created_at=row.created_at.isoformat(),
        file_size_bytes=row.file_size_bytes,
        is_active=row.is_active,
    )


# ─── Helpers (runs) ─────────────────────────────────────────────────────────


def _build_run_response(run: TrainingJob, run_id_str: str) -> TrainingRunResponse:
    """API payload from the persisted `training_jobs` row only (no Redis overlay)."""
    return TrainingRunResponse(
        id=run_id_str,
        status=run.status,
        config=run.config or {},
        triggered_by=str(run.triggered_by),
        started_at=run.started_at.isoformat(),
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        step=run.step,
        step_index=run.step_index,
        model_name=run.model_name,
        is_active=run.is_active,
        evaluation_report=run.evaluation_report,
        error=run.error,
    )


# ─── Routes ───────────────────────────────────────────────────────────────────


@router.post("/start", response_model=TrainingRunResponse)
def start_training(
    body: StartTrainingRequest = Body(...),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Create a new training job in the DB and push it onto the Redis queue
    for the ml-training-worker to pick up.
    """
    run = TrainingJob(
        status="queued",
        triggered_by=admin.id,
        started_at=datetime.now(tz=timezone(timedelta(hours=8))),
        config={
            "testing_mode": body.testing_mode,
            "n_assets": body.n_assets,
            "n_scenarios": body.n_scenarios,
            "run_evaluation": body.run_evaluation,
        },
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    run_id_str = str(run.id)
    training_store.set_status(run_id_str, "queued")
    training_store.enqueue(run_id_str)

    logger.info("Training run %s queued by admin %s", run_id_str, admin.username)
    return _build_run_response(run, run_id_str)


@router.get("/runs", response_model=TrainingRunsListResponse)
def list_runs(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Return all training runs, newest first."""
    runs = db.query(TrainingJob).order_by(TrainingJob.started_at.desc()).all()
    items = [_build_run_response(r, str(r.id)) for r in runs]
    return TrainingRunsListResponse(runs=items, total=len(items))


@router.get("/uploads", response_model=UploadedModelsListResponse)
def list_uploads(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Return all manually uploaded model files, newest first."""
    rows = db.query(UploadedModel).order_by(UploadedModel.created_at.desc()).all()
    return UploadedModelsListResponse(
        uploads=[_uploaded_model_to_response(r) for r in rows],
        total=len(rows),
    )


@router.post("/models/upload", response_model=ModelUploadResult)
async def upload_model_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Accept a .pkl file, assign rf_delta_YYYYMMDD_HHMMSS(.pkl) (with suffix if needed),
    upload to S3 under models/, and record in uploaded_models.

    Streams the body in chunks (bounded RAM) and uploads via upload_fileobj so the
    full file is not loaded into a single bytes object.
    """
    import boto3

    spool = None
    try:
        if not file.filename or not file.filename.lower().endswith(".pkl"):
            raise HTTPException(status_code=400, detail="File must be a .pkl file")

        spool = tempfile.SpooledTemporaryFile(
            max_size=UPLOAD_SPOOL_MAX_IN_MEMORY, mode="w+b"
        )
        total = 0
        await file.seek(0)
        while True:
            chunk = await file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_MODEL_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"File too large (max {MAX_MODEL_UPLOAD_BYTES // (1024 * 1024)} MB)"
                    ),
                )
            spool.write(chunk)
        spool.seek(0)

        model_name = _allocate_upload_model_name(db)
        s3_key = f"models/{model_name}"
        bucket_name = os.environ["S3_BUCKET_NAME"]

        try:
            s3_client = boto3.client("s3")
            await asyncio.to_thread(
                s3_client.upload_fileobj, spool, bucket_name, s3_key
            )
        except Exception as e:
            logger.exception("S3 upload failed for model_name=%s", model_name)
            raise HTTPException(
                status_code=502,
                detail="Storage upload failed. Please try again later.",
                headers={"X-Error-Code": "STORAGE_UPLOAD_FAILED"},
            ) from e

        row = UploadedModel(
            model_name=model_name,
            s3_key=s3_key,
            uploaded_by=admin.id,
            file_size_bytes=total,
        )
        db.add(row)
        db.commit()
        db.refresh(row)

        logger.info(
            "Admin %s uploaded model %s (%d bytes)",
            admin.username,
            model_name,
            total,
        )
        return ModelUploadResult(
            id=str(row.id), model_name=row.model_name, s3_key=row.s3_key
        )
    finally:
        if spool is not None:
            spool.close()
        await file.close()


@router.get("/runs/{run_id}", response_model=TrainingRunResponse)
def get_run(
    run_id: str,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Return status + live progress for a single run. Used by the progress poller."""
    import uuid as _uuid

    try:
        run_uuid = _uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run_id format")

    run = db.query(TrainingJob).filter(TrainingJob.id == run_uuid).first()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    # If the worker has finished and we haven't synced back to DB yet, do it now
    live_status = training_store.get_status(run_id)
    if live_status in ("completed", "failed") and run.status not in (
        "completed",
        "failed",
    ):
        run.status = live_status
        run.model_name = training_store.get_model_name(run_id) or run.model_name
        run.error = training_store.get_error(run_id) or run.error
        if live_status == "completed":
            run.completed_at = datetime.now(tz=timezone(timedelta(hours=8)))
            # Load evaluation report from disk if available
            report_path = EVALUATION_DIR / "evaluation_report.json"
            if report_path.exists():
                try:
                    run.evaluation_report = json.loads(report_path.read_text())
                except Exception:
                    pass
        db.commit()
        db.refresh(run)

    return _build_run_response(run, run_id)


@router.post("/models/{model_name}/activate")
def activate_model(
    model_name: str,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Copy a versioned model file to the active inference path (rf_delta.pkl).
    Clears is_active on any currently active training job / upload, then marks
    the resolved row as active.
    """
    import shutil
    import os
    import boto3

    # Sanitise the model_name to prevent path traversal
    if "/" in model_name or "\\" in model_name or ".." in model_name:
        raise HTTPException(status_code=400, detail="Invalid model name")

    src = MODEL_SAVE_DIR / model_name
    dst = MODEL_SAVE_DIR / ACTIVE_MODEL_NAME

    run = db.query(TrainingJob).filter(TrainingJob.model_name == model_name).first()
    upload = (
        None
        if run
        else db.query(UploadedModel)
        .filter(UploadedModel.model_name == model_name)
        .first()
    )
    if not run and not upload:
        raise HTTPException(
            status_code=404,
            detail="No training run or uploaded model with this name",
        )

    if src.exists():
        shutil.copy2(src, dst)
    else:
        try:
            s3_client = boto3.client("s3")
            bucket_name = os.environ["S3_BUCKET_NAME"]
            s3_key = f"models/{model_name}"
            s3_client.download_file(bucket_name, s3_key, str(dst))
        except Exception as e:
            logger.error("Failed to download model %s from S3: %s", model_name, e)
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found locally or on S3: {model_name}",
            ) from e

    logger.info(
        "Admin %s activated model %s → %s",
        admin.username,
        model_name,
        ACTIVE_MODEL_NAME,
    )

    db.query(TrainingJob).filter(TrainingJob.is_active.is_(True)).update(
        {"is_active": False}
    )
    db.query(UploadedModel).filter(UploadedModel.is_active.is_(True)).update(
        {"is_active": False}
    )
    if run:
        run.is_active = True
    else:
        upload.is_active = True
    db.commit()

    return {
        "message": f"Model {model_name} is now active",
        "active_model": ACTIVE_MODEL_NAME,
    }


@router.get("/active-model/evaluation")
def get_active_model_evaluation(
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Return the evaluation report for the training run marked is_active.

    Prefer the JSON stored on that TrainingJob row (captured when the run
    completed). The shared evaluation_report.json on disk always reflects the
    *last* worker run, so it would be wrong after activating an older model.
    """
    run = db.query(TrainingJob).filter(TrainingJob.is_active.is_(True)).first()
    if run:
        if run.evaluation_report:
            return run.evaluation_report
        raise HTTPException(
            status_code=404,
            detail="No evaluation report stored for the active model.",
        )

    upload = db.query(UploadedModel).filter(UploadedModel.is_active.is_(True)).first()
    if upload:
        raise HTTPException(
            status_code=404,
            detail=(
                "The active model was uploaded manually; no evaluation report is available."
            ),
        )

    raise HTTPException(
        status_code=404,
        detail="No active model. Activate a training run or uploaded model first.",
    )
