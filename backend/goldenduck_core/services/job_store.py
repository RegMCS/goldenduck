import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from goldenduck_core.redis_client import redis_client


class JobStore:
    def create_job(
        self,
        job_id: str,
        user_id: str,
        parameters: dict,
        status: str,
    ):
        job = {
            "job_id": job_id,
            "user_id": user_id,
            "status": status,
            "parameters": parameters,
            "created_at": datetime.now(tz=timezone(timedelta(hours=8))).isoformat(),
        }

        redis_client.set(f"job:{job_id}:job", json.dumps(job))
        redis_client.set(f"job:{job_id}:status", status)

    def enqueue(self, job_id: str):
        redis_client.lpush("queue:garch", job_id)

    # -------------------------
    # Core getters
    # -------------------------
    def get_job(self, job_id: str) -> Optional[dict]:
        raw = redis_client.get(f"job:{job_id}:job")
        return json.loads(raw) if raw else None

    def get_status(self, job_id: str) -> Optional[str]:
        return redis_client.get(f"job:{job_id}:status")

    # -------------------------
    # Result data
    # -------------------------
    def get_parameters(self, job_id: str) -> dict:
        raw = redis_client.get(f"job:{job_id}:parameters")
        return json.loads(raw) if raw else {}

    def get_metrics(self, job_id: str) -> dict:
        raw = redis_client.get(f"job:{job_id}:metrics")
        return json.loads(raw) if raw else {}

    def get_error(self, job_id: str) -> Optional[str]:
        return redis_client.get(f"job:{job_id}:error")

    def get_output_file(self, job_id: str) -> Optional[str]:
        return redis_client.get(f"job:{job_id}:output_file")

    # -------------------------
    # Worker-side setters
    # -------------------------
    def set_status(self, job_id: str, status: str):
        redis_client.set(f"job:{job_id}:status", status)

        job = self.get_job(job_id)
        if job:
            job["status"] = status
            redis_client.set(f"job:{job_id}:job", json.dumps(job))

    def set_parameters(self, job_id: str, parameters: dict):
        redis_client.set(f"job:{job_id}:parameters", json.dumps(parameters))

    def set_metrics(self, job_id: str, metrics: dict):
        redis_client.set(f"job:{job_id}:metrics", json.dumps(metrics))

    def set_error(self, job_id: str, error: str):
        redis_client.set(f"job:{job_id}:error", error)

    def set_output_file(self, job_id: str, path: str):
        redis_client.set(f"job:{job_id}:output_file", path)

    def get_chart_data(self, job_id: str) -> dict:
        raw = redis_client.get(f"job:{job_id}:chart_data")
        return json.loads(raw) if raw else {}

    def set_chart_data(self, job_id: str, chart_data: dict):
        redis_client.set(f"job:{job_id}:chart_data", json.dumps(chart_data))


job_store = JobStore()
