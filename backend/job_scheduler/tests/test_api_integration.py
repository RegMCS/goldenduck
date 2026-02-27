import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from job_scheduler.db.base import Base
from job_scheduler.db.session import SessionLocal, engine
from job_scheduler.main import app
from job_scheduler.models.ai_model_job import AIModelJob
from job_scheduler.redis_client import redis_client
from job_scheduler.services.job_store import job_store

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def ensure_tables_exist():
    # Integration tests use the real DB connection configured for the app.
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture(autouse=True)
def clean_state():
    with SessionLocal() as db:
        db.execute(text("DELETE FROM ai_model_jobs"))
        db.execute(text("DELETE FROM users"))
        db.commit()

    keys = list(redis_client.scan_iter("job:*"))
    if keys:
        redis_client.delete(*keys)
    redis_client.delete("queue:garch")


def _generate_payload():
    return {
        "ticker": "AAPL",
        "horizon": 252,
        "desired_volatility": 1.1,
        "desired_trend": 0.1,
        "desired_fat_tails": 1.2,
        "desired_momentum": 0.7,
    }


def test_generate_creates_db_job_and_redis_queue_entry():
    user_id = str(uuid.uuid4())
    with patch.object(job_store, "enqueue", wraps=job_store.enqueue) as enqueue_spy:
        response = client.post(
            f"/api/generate/user/{user_id}", json=_generate_payload()
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]
    enqueue_spy.assert_called_once_with(job_id)

    with SessionLocal() as db:
        db_job = db.query(AIModelJob).filter_by(id=uuid.UUID(job_id)).first()
        assert db_job is not None
        assert str(db_job.requestor) == user_id
        assert db_job.status.value == "queued"

    raw_job = redis_client.get(f"job:{job_id}:job")
    assert raw_job is not None
    job_payload = json.loads(raw_job)
    assert job_payload["job_id"] == job_id
    assert job_payload["user_id"] == user_id
    assert job_payload["status"] == "queued"
    assert redis_client.get(f"job:{job_id}:status") == "queued"


def test_status_completed_and_download_redirect_flow():
    user_id = str(uuid.uuid4())
    response = client.post(f"/api/generate/user/{user_id}", json=_generate_payload())
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    job_store.set_status(job_id, "completed")
    job_store.set_parameters(job_id, {"omega": 0.1, "alpha": 0.2, "beta": 0.7})
    job_store.set_metrics(job_id, {"ks_statistic": 0.05})
    job_store.set_output_file(job_id, "s3://my-bucket/garch/test-job.csv")

    status_response = client.get(f"/api/status/user/{user_id}/{job_id}")
    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["status"] == "completed"
    assert status_body["parameters"]["omega"] == 0.1
    assert status_body["metrics"]["ks_statistic"] == 0.05
    assert status_body["download_url"] == f"/api/download/user/{user_id}/{job_id}"

    with patch("job_scheduler.routes.jobs.boto3.client") as mock_boto:
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.generate_presigned_url.return_value = "https://example.com/file.csv"

        download_response = client.get(
            f"/api/download/user/{user_id}/{job_id}", follow_redirects=False
        )

    assert download_response.status_code == 307
    assert download_response.headers["location"] == "https://example.com/file.csv"


def test_status_returns_404_for_wrong_user():
    owner_user_id = str(uuid.uuid4())
    other_user_id = str(uuid.uuid4())

    response = client.post(
        f"/api/generate/user/{owner_user_id}", json=_generate_payload()
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    status_response = client.get(f"/api/status/user/{other_user_id}/{job_id}")
    assert status_response.status_code == 404
    assert status_response.json()["detail"] == "Job not found"


def test_download_returns_404_when_output_not_ready():
    user_id = str(uuid.uuid4())
    response = client.post(f"/api/generate/user/{user_id}", json=_generate_payload())
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    download_response = client.get(f"/api/download/user/{user_id}/{job_id}")
    assert download_response.status_code == 404
    assert download_response.json()["detail"] == "File not ready"
