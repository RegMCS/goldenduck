"""
ML Training Worker
==================
Listens on the Redis queue "queue:training" and runs the full training pipeline:

  Step 1 — Download historical data    (01_download_data.py main())
  Step 2 — Generate training samples   (02_generate_training_data.py main())
  Step 3 — Train Random Forest models  (03_train_models.py main())
  Step 4 — Evaluate models             (04_evaluate_models.py main())  [if requested]

After step 3, the trained model is saved as a timestamped file
(e.g. rf_delta_20260402_143000.pkl) AND copied to rf_delta.pkl (active).

Progress is broadcast to Redis so the API can surface it to the frontend.
All terminal state is synced back to Postgres via the training_jobs table.
"""

import json
import logging
import shutil
import sys
import time
import importlib.util
from datetime import datetime, timezone, timedelta
from pathlib import Path

from job_scheduler.redis_client import redis_client
from job_scheduler.db.session import SessionLocal
from job_scheduler.models.training_job import TrainingJob
from job_scheduler.services.training_store import training_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("training-worker")

# ─── Paths ────────────────────────────────────────────────────────────────────

BACKEND_DIR = Path(__file__).parent.parent
ML_TRAINING_DIR = BACKEND_DIR / "ml_training"
SCRIPTS_DIR = ML_TRAINING_DIR / "scripts"
MODEL_SAVE_DIR = ML_TRAINING_DIR / "models" / "saved_models"
EVALUATION_DIR = ML_TRAINING_DIR / "models" / "evaluation"

# Make sure the ml_training package is importable
sys.path.insert(0, str(ML_TRAINING_DIR))


# ─── Script loader ────────────────────────────────────────────────────────────


def _run_script_main(script_path: Path) -> None:
    """
    Dynamically import a script module and call its main() function.
    This avoids spawning subprocesses, which would lose our env vars.
    """
    spec = importlib.util.spec_from_file_location("_script_module", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "main"):
        mod.main()


# ─── DB helpers ───────────────────────────────────────────────────────────────


def _db_update(run_id_str: str, **kwargs) -> None:
    """Update a training_job row without holding the session open."""
    import uuid as _uuid

    run_uuid = _uuid.UUID(run_id_str)
    db = SessionLocal()
    try:
        db.query(TrainingJob).filter(TrainingJob.id == run_uuid).update(kwargs)
        db.commit()
    except Exception as exc:
        logger.error("DB update failed for run %s: %s", run_id_str, exc)
    finally:
        db.close()


def _db_get_config(run_id_str: str) -> dict:
    import uuid as _uuid

    run_uuid = _uuid.UUID(run_id_str)
    db = SessionLocal()
    try:
        run = db.query(TrainingJob).filter(TrainingJob.id == run_uuid).first()
        return run.config if run else {}
    finally:
        db.close()


# ─── Step helpers ─────────────────────────────────────────────────────────────


def _set_step(run_id: str, index: int, label: str) -> None:
    msg = f"Step {index} of 4: {label}"
    training_store.set_step(run_id, msg, index)
    _db_update(run_id, step=msg, step_index=index)
    logger.info("[Run %s] %s", run_id[:8], msg)


# ─── Main training loop ───────────────────────────────────────────────────────

logger.info("ML Training Worker started, waiting for jobs on queue:training …")

while True:
    run_id = None
    try:
        _, run_id = redis_client.brpop("queue:training")
        logger.info("Picked up training run %s", run_id)

        config = _db_get_config(run_id)
        testing_mode = config.get("testing_mode", True)
        n_assets = int(config.get("n_assets", 50))
        n_scenarios = int(config.get("n_scenarios", 10))
        run_evaluation = bool(config.get("run_evaluation", True))

        # ── Mark as running ──────────────────────────────────────────────────
        training_store.set_status(run_id, "running")
        _db_update(
            run_id,
            status="running",
            started_at=datetime.now(tz=timezone(timedelta(hours=8))),
        )

        # ── Patch training config to respect the UI settings ─────────────────
        from config import training_config as tc

        tc.TESTING_MODE = testing_mode
        tc.N_ASSETS = n_assets
        tc.N_SCENARIOS_PER_ASSET = n_scenarios

        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Download data ────────────────────────────────────────────
        _set_step(run_id, 1, "Downloading historical data")
        _run_script_main(SCRIPTS_DIR / "01_download_data.py")

        # ── Ensure assets_split.json exists (fallback for small asset counts) ─
        # The download script only calls split_assets() when >= 30 assets succeed.
        # In testing mode we may have fewer, so we build the split ourselves.
        from config.training_config import PROCESSED_DATA_DIR
        import numpy as np

        assets_split_path = Path(PROCESSED_DATA_DIR) / "assets_split.json"
        downloaded_assets_path = Path(PROCESSED_DATA_DIR) / "downloaded_assets.json"

        if not assets_split_path.exists() and downloaded_assets_path.exists():
            logger.info("assets_split.json missing — creating fallback split")
            with open(downloaded_assets_path) as f:
                dl_info = json.load(f)
            assets = dl_info.get("assets", [])
            if not assets:
                raise RuntimeError(
                    "No assets were downloaded — cannot proceed to step 2"
                )
            np.random.seed(42)
            shuffled = assets.copy()
            np.random.shuffle(shuffled)
            n_train = max(1, int(0.70 * len(shuffled)))
            n_val = max(1, int(0.15 * len(shuffled)))
            split = {
                "train": shuffled[:n_train],
                "val": shuffled[n_train : n_train + n_val],
                "test": shuffled[n_train + n_val :],
            }
            with open(assets_split_path, "w") as f:
                json.dump(split, f, indent=2)
            logger.info(
                "Fallback split created: %d train / %d val / %d test",
                len(split["train"]),
                len(split["val"]),
                len(split["test"]),
            )

        # ── Step 2: Generate training samples ────────────────────────────────
        _set_step(run_id, 2, "Generating training samples (grid search)")
        _run_script_main(SCRIPTS_DIR / "02_generate_training_data.py")

        # ── Step 3: Train Random Forest ───────────────────────────────────────
        _set_step(run_id, 3, "Training Random Forest model")
        _run_script_main(SCRIPTS_DIR / "03_train_models.py")

        # ── Version the trained model ─────────────────────────────────────────
        # Order: copy locally → upload to S3 → record name → delete local copy.
        # model_name is only set to the versioned filename after both the local
        # copy and the S3 upload succeed, so a failed upload never leaves the
        # run pointing at a model file that doesn't exist.
        import boto3
        import os

        ts = datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        versioned_name = f"rf_delta_{ts}.pkl"
        src = MODEL_SAVE_DIR / "rf_delta.pkl"
        local_versioned = MODEL_SAVE_DIR / versioned_name

        if not src.exists():
            logger.warning(
                "rf_delta.pkl not found after training — skipping versioning"
            )
            versioned_name = "rf_delta.pkl"
        else:
            # Step 1: write local versioned copy (atomic on most filesystems)
            try:
                shutil.copy2(src, local_versioned)
                logger.info("Local versioned copy written: %s", local_versioned)
            except Exception as copy_exc:
                logger.error(
                    "Failed to write local versioned copy for run %s: %s",
                    run_id,
                    copy_exc,
                )
                raise  # propagate — model_name stays unset rather than pointing nowhere

            # Step 2: upload the local copy to S3
            try:
                s3_client = boto3.client("s3")
                bucket_name = os.environ["S3_BUCKET_NAME"]
                s3_key = f"models/{versioned_name}"
                s3_client.upload_file(str(local_versioned), bucket_name, s3_key)
                logger.info(
                    "Versioned model uploaded to S3: s3://%s/%s", bucket_name, s3_key
                )
            except Exception as upload_exc:
                logger.error(
                    "S3 upload failed for run %s — keeping local copy, "
                    "model_name will not be updated: %s",
                    run_id,
                    upload_exc,
                )
                raise  # propagate — model_name stays unset rather than pointing nowhere

            # Step 3: record versioned name only after both steps succeeded
            training_store.set_model_name(run_id, versioned_name)
            _db_update(run_id, model_name=versioned_name)

            # Step 4: remove the local versioned copy (S3 is the source of truth)
            try:
                local_versioned.unlink()
                logger.info("Local versioned copy removed: %s", local_versioned)
            except Exception as unlink_exc:
                logger.warning(
                    "Could not remove local versioned copy %s: %s",
                    local_versioned,
                    unlink_exc,
                )

        if versioned_name == "rf_delta.pkl":
            # Fallback path: src was missing, record the unversioned name
            training_store.set_model_name(run_id, versioned_name)
            _db_update(run_id, model_name=versioned_name)

        # ── Step 4: Evaluate (optional but auto by default) ───────────────────
        evaluation_report = None
        if run_evaluation:
            _set_step(run_id, 4, "Running model evaluation (~10 min)")
            try:
                _run_script_main(SCRIPTS_DIR / "04_evaluate_models.py")
                report_path = EVALUATION_DIR / "evaluation_report.json"
                if report_path.exists():
                    evaluation_report = json.loads(report_path.read_text())
                    logger.info("Evaluation report loaded for run %s", run_id)
            except Exception as eval_exc:
                logger.warning("Evaluation failed for run %s: %s", run_id, eval_exc)

        # ── Mark active (this run's model is now the inference model) ─────────
        # De-activate any previous active run first
        db = SessionLocal()
        try:
            db.query(TrainingJob).update({"is_active": False})
            import uuid as _uuid

            run_uuid = _uuid.UUID(run_id)
            run = db.query(TrainingJob).filter(TrainingJob.id == run_uuid).first()
            if run:
                run.status = "completed"
                run.is_active = True
                run.completed_at = datetime.now(tz=timezone(timedelta(hours=8)))
                run.evaluation_report = evaluation_report
            db.commit()
        finally:
            db.close()

        training_store.set_status(run_id, "completed")
        logger.info("Training run %s completed successfully", run_id)

    except Exception as exc:
        logger.exception("Training run %s failed", run_id)
        if run_id:
            training_store.set_status(run_id, "failed")
            training_store.set_error(run_id, str(exc))
            _db_update(
                run_id,
                status="failed",
                error=str(exc),
                completed_at=datetime.now(tz=timezone(timedelta(hours=8))),
            )
        time.sleep(2)
