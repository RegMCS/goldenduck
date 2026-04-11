"""
Training job store — thin Redis wrapper for training run progress.
Mirrors the pattern of job_store.py used by the GARCH worker.

Keys used:
  training:{run_id}:status   → "queued" | "running" | "completed" | "failed"
  training:{run_id}:step     → "Step 2 of 4: Generating samples"
  training:{run_id}:step_index → "2"
  training:{run_id}:error    → error message string
  training:{run_id}:model_name → "rf_delta_20260402_143000.pkl"
"""

from typing import Optional
from goldenduck_core.redis_client import redis_client

_PREFIX = "training"


class TrainingStore:
    # ─── setters (called by worker) ──────────────────────────────────────────

    def set_status(self, run_id: str, status: str) -> None:
        redis_client.set(f"{_PREFIX}:{run_id}:status", status)

    def set_step(self, run_id: str, step: str, step_index: int) -> None:
        redis_client.set(f"{_PREFIX}:{run_id}:step", step)
        redis_client.set(f"{_PREFIX}:{run_id}:step_index", str(step_index))

    def set_error(self, run_id: str, error: str) -> None:
        redis_client.set(f"{_PREFIX}:{run_id}:error", error)

    def set_model_name(self, run_id: str, model_name: str) -> None:
        redis_client.set(f"{_PREFIX}:{run_id}:model_name", model_name)

    # ─── getters (called by API) ──────────────────────────────────────────────

    def get_status(self, run_id: str) -> Optional[str]:
        return redis_client.get(f"{_PREFIX}:{run_id}:status")

    def get_step(self, run_id: str) -> Optional[str]:
        return redis_client.get(f"{_PREFIX}:{run_id}:step")

    def get_step_index(self, run_id: str) -> Optional[int]:
        raw = redis_client.get(f"{_PREFIX}:{run_id}:step_index")
        return int(raw) if raw else None

    def get_error(self, run_id: str) -> Optional[str]:
        return redis_client.get(f"{_PREFIX}:{run_id}:error")

    def get_model_name(self, run_id: str) -> Optional[str]:
        return redis_client.get(f"{_PREFIX}:{run_id}:model_name")

    # ─── queue ────────────────────────────────────────────────────────────────

    def enqueue(self, run_id: str) -> None:
        """Push the run_id onto the training queue for the worker to pick up."""
        redis_client.lpush("queue:training", run_id)


training_store = TrainingStore()
