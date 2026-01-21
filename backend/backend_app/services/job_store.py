import json
from datetime import datetime
from backend_app.redis_client import redis_client


class JobStore:
    def create_job(self, job_id: str, request_data: dict):
        meta = {
            "created_at": datetime.utcnow().isoformat(),
            "request": request_data,
        }
        redis_client.set(f"job:{job_id}:meta", json.dumps(meta))
        redis_client.set(f"job:{job_id}:status", "queued")

    def enqueue(self, job_id: str):
        redis_client.lpush("queue:garch", job_id)

    def get_status(self, job_id: str):
        return redis_client.get(f"job:{job_id}:status")

    def get_parameters(self, job_id: str):
        raw = redis_client.get(f"job:{job_id}:parameters")
        return json.loads(raw) if raw else {}

    def get_metrics(self, job_id: str):
        raw = redis_client.get(f"job:{job_id}:metrics")
        return json.loads(raw) if raw else {}

    def get_error(self, job_id: str):
        return redis_client.get(f"job:{job_id}:error")

    def get_output_file(self, job_id: str):
        return redis_client.get(f"job:{job_id}:output_file")


job_store = JobStore()
