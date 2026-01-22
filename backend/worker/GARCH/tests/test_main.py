from fastapi.testclient import TestClient
from unittest.mock import patch
import tempfile
import os
import uuid
import pytest

from job_scheduler.main import app

client = TestClient(app)

USER_ID = str(uuid.uuid4())


# Global mock: replace Redis-backed job_store everywhere in routes
@pytest.fixture(autouse=True)
def mock_job_store():
    with patch("job_scheduler.routes.jobs.job_store") as store:
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
            "num_scenarios": 10,
            "horizon": 50,
            "volatility_multiplier": 1.0,
            "p": 1,
            "q": 1,
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
            "num_scenarios": 10,
            "horizon": 50,
            "volatility_multiplier": -1.0,  # invalid
            "p": 1,
            "q": 1,
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


def test_download_api_success(mock_job_store):
    os.makedirs("output", exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv", dir="output"
    ) as f:
        f.write("a,b,c\n1,2,3\n")
        temp_file = f.name

    try:
        mock_job_store.get_output_file.return_value = temp_file

        response = client.get(f"/api/download/user/{USER_ID}/test-job")

        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "a,b,c" in response.text
    finally:
        os.unlink(temp_file)


def test_download_api_file_not_ready(mock_job_store):
    mock_job_store.get_output_file.return_value = None

    response = client.get(f"/api/download/user/{USER_ID}/test-job")

    assert response.status_code == 404
    assert response.json()["detail"] == "File not ready"


def test_download_api_path_traversal_attack(mock_job_store):
    mock_job_store.get_output_file.return_value = "/etc/passwd"

    response = client.get(f"/api/download/user/{USER_ID}/test-job")

    assert response.status_code == 403
    assert response.json()["detail"] == "Access denied"


def test_download_api_path_traversal_with_relative_path(mock_job_store):
    mock_job_store.get_output_file.return_value = "output/../../etc/passwd"

    response = client.get(f"/api/download/user/{USER_ID}/test-job")

    assert response.status_code == 403
    assert response.json()["detail"] == "Access denied"


def test_download_api_valid_path_in_output_dir(mock_job_store):
    os.makedirs("output", exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv", dir="output"
    ) as f:
        f.write("scenario,value\n1,100\n")
        temp_file = f.name

    try:
        mock_job_store.get_output_file.return_value = temp_file

        response = client.get(f"/api/download/user/{USER_ID}/test-job")

        assert response.status_code == 200
        assert "scenario,value" in response.text
    finally:
        os.unlink(temp_file)
