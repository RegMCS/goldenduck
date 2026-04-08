from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import tempfile
import os
import uuid
import pytest
from goldenduck_core.db.session import get_db
from goldenduck_core.main import app
from goldenduck_core.services.auth_service import get_current_user

client = TestClient(app)

USER_ID = str(uuid.uuid4())


@pytest.fixture(autouse=True)
def override_db_dependency():
    def fake_db():
        yield MagicMock()  # fake SQLAlchemy session

    app.dependency_overrides[get_db] = fake_db
    yield
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def override_auth():
    """Override get_current_user so API calls succeed without a real JWT."""

    def fake_current_user(request=None, db=None):
        return MagicMock(id=USER_ID, username="test")

    app.dependency_overrides[get_current_user] = fake_current_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


# Global mock: replace Redis-backed job_store everywhere in routes
@pytest.fixture(autouse=True)
def mock_job_store():
    with patch("goldenduck_core.routes.jobs.job_store") as store:
        store.create_job.return_value = None
        store.enqueue.return_value = None

        # Default: job exists & belongs to user
        store.get_job.return_value = {
            "user_id": USER_ID,
            "status": "completed",
        }

        store.get_parameters.return_value = {
            "omega": 0.1,
            "alpha": 0.2,
            "beta": 0.7,
        }
        store.get_metrics.return_value = {"ks_statistic": 0.05}
        store.get_error.return_value = None
        store.get_output_file.return_value = None

        yield store


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_api_success(mock_job_store):
    response = client.post(
        f"/api/generate/user/{USER_ID}",
        json={
            "ticker": "AAPL",
            "horizon": 500,
            "desired_volatility": 1.5,
            "desired_trend": 0.0,
            "desired_fat_tails": 1.2,
            "desired_momentum": 0.8,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "job_id" in data
    assert data["status"] == "queued"
    assert "Poll /api/status/user/" in data["message"]

    mock_job_store.create_job.assert_called_once()
    mock_job_store.enqueue.assert_called_once()


def test_generate_api_invalid_parameters():
    response = client.post(
        f"/api/generate/user/{USER_ID}",
        json={
            "ticker": "AAPL",
            "horizon": 500,
            "desired_volatility": 3.0,  # invalid: > 2.0
            "desired_trend": 0.0,
            "desired_fat_tails": 1.2,
            "desired_momentum": 0.8,
        },
    )

    assert response.status_code == 422


def test_status_api_existing_completed_job(mock_job_store):
    job_id = "test-job-123"

    response = client.get(f"/api/status/user/{USER_ID}/{job_id}")

    assert response.status_code == 200
    data = response.json()

    assert data["job_id"] == job_id
    assert data["status"] == "completed"
    assert "parameters" in data
    assert "metrics" in data
    assert "download_url" in data


def test_status_api_failed_job(mock_job_store):
    mock_job_store.get_job.return_value = {
        "user_id": USER_ID,
        "status": "failed",
    }
    mock_job_store.get_error.return_value = "Model fitting failed"

    response = client.get(f"/api/status/user/{USER_ID}/test-job-failed")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "failed"
    assert data["error"] == "Model fitting failed"


def test_status_api_nonexistent_job(mock_job_store):
    mock_job_store.get_job.return_value = None

    response = client.get(f"/api/status/user/{USER_ID}/nonexistent-job")

    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_download_api_s3_redirect(mock_job_store):
    mock_job_store.get_output_file.return_value = "s3://my-bucket/garch/test-job.csv"

    with patch("goldenduck_core.routes.jobs.boto3.client") as mock_boto:
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        mock_s3.generate_presigned_url.return_value = "https://fake-s3-url.com/file.csv"

        response = client.get(
            f"/api/download/user/{USER_ID}/test-job", follow_redirects=False
        )

        assert response.status_code == 307
        assert response.headers["location"] == "https://fake-s3-url.com/file.csv"
