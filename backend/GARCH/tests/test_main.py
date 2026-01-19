from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import os
import tempfile
import pytest
from GARCH.main import app, jobs

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_jobs():
    """Fixture to clean up jobs dictionary before and after each test"""
    jobs.clear()
    yield
    jobs.clear()


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "GARCH Synthetic Data Generator API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/generate",
            "status": "/api/status/{job_id}",
            "download": "/api/download/{job_id}",
        },
    }


def test_health_check_mock_db():
    # Note: This checks the endpoint wrapper itself.
    # For a true DB integration test, we would need a test DB environment.
    # Here we expect a 500 or 200 depending on if DB is actually reachable during test.
    # To make this robust unit test, we should mock the dependency.
    pass


# ========== GARCH API TESTS ==========


def test_generate_api_success():
    """Test successful job submission for synthetic data generation"""
    # Use fixed seed for reproducible test data
    np.random.seed(42)
    mock_data = pd.DataFrame(
        {
            "Close": np.random.uniform(100, 200, 500),
            "Open": np.random.uniform(100, 200, 500),
            "High": np.random.uniform(100, 200, 500),
            "Low": np.random.uniform(100, 200, 500),
            "Volume": np.random.uniform(1000000, 10000000, 500),
        }
    )

    # Patch where yfinance is actually used in the main module
    with patch("GARCH.main.yf.download", return_value=mock_data):
        response = client.post(
            "/api/generate",
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
    assert "Track progress" in data["message"]


def test_generate_api_invalid_ticker():
    """Test generation with invalid ticker (no data available)"""
    # Mock yfinance to return empty dataframe
    empty_data = pd.DataFrame()

    with patch("GARCH.main.yf.download", return_value=empty_data):
        response = client.post(
            "/api/generate",
            json={
                "ticker": "INVALID",
                "num_scenarios": 10,
                "horizon": 50,
                "p": 1,
                "q": 1,
            },
        )

    # The code catches HTTPException and re-raises as 500, so expect 500
    assert response.status_code == 500
    assert "No data found" in response.json()["detail"]


def test_generate_api_invalid_parameters():
    """Test generation with invalid parameters (boundary validation)"""
    # Test with num_scenarios out of range
    response = client.post(
        "/api/generate",
        json={
            "ticker": "AAPL",
            "num_scenarios": 20000,  # Max is 10000
            "horizon": 50,
            "p": 1,
            "q": 1,
        },
    )
    assert response.status_code == 422  # Validation error

    # Test with negative volatility multiplier
    response = client.post(
        "/api/generate",
        json={
            "ticker": "AAPL",
            "num_scenarios": 10,
            "horizon": 50,
            "volatility_multiplier": -1.0,  # Min is 0.1
            "p": 1,
            "q": 1,
        },
    )
    assert response.status_code == 422  # Validation error


def test_status_api_existing_job():
    """Test retrieving status for an existing job"""
    # Create a mock job
    job_id = "test-job-123"
    jobs[job_id] = {
        "status": "completed",
        "created_at": "2024-01-01T00:00:00",
        "parameters": {
            "omega": 0.1,
            "alpha": 0.2,
            "beta": 0.7,
            "converged": True,
            "aic": -100.0,
            "bic": -95.0,
        },
        "validation_metrics": {
            "ks_statistic": 0.05,
            "ks_pvalue": 0.8,
            "kurtosis_historical": 3.5,
            "kurtosis_synthetic": 3.4,
            "acf_lag1_historical": 0.02,
            "acf_lag1_synthetic": 0.03,
        },
        "num_scenarios": 10,
        "output_file": "/tmp/test.csv",
    }

    response = client.get(f"/api/status/{job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "completed"
    assert "download_url" in data
    assert "parameters" in data
    assert "validation_metrics" in data


def test_status_api_nonexistent_job():
    """Test retrieving status for a non-existent job (404)"""
    response = client.get("/api/status/nonexistent-job-id")
    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]


def test_status_api_different_statuses():
    """Test status endpoint with different job statuses"""
    # Test with failed job
    job_id_failed = "test-job-failed"
    jobs[job_id_failed] = {
        "status": "failed",
        "created_at": "2024-01-01T00:00:00",
        "error": "Model fitting failed",
    }

    response = client.get(f"/api/status/{job_id_failed}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "failed"
    assert "error" in data
    assert data["error"] == "Model fitting failed"

    # Test with generating job
    job_id_generating = "test-job-generating"
    jobs[job_id_generating] = {
        "status": "generating",
        "created_at": "2024-01-01T00:00:00",
        "progress": 5,
        "total": 10,
    }

    response = client.get(f"/api/status/{job_id_generating}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "generating"
    assert data["progress"] == 5
    assert data["total"] == 10


def test_download_api_success():
    """Test successful download of generated data"""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("day,scenario_id,Open,High,Low,Close,Volume\n")
        f.write("1,1,100,105,99,103,1000000\n")
        temp_file = f.name

    try:
        job_id = "test-job-download"
        jobs[job_id] = {
            "status": "completed",
            "created_at": "2024-01-01T00:00:00",
            "output_file": temp_file,
        }

        response = client.get(f"/api/download/{job_id}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "scenario_id" in response.text
    finally:
        os.unlink(temp_file)


def test_download_api_nonexistent_job():
    """Test download for non-existent job (404)"""
    response = client.get("/api/download/nonexistent-job-id")
    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]


def test_download_api_incomplete_job():
    """Test download for incomplete job (400)"""
    job_id = "test-job-incomplete"
    jobs[job_id] = {"status": "generating", "created_at": "2024-01-01T00:00:00"}

    response = client.get(f"/api/download/{job_id}")
    assert response.status_code == 400
    assert "not completed" in response.json()["detail"]


def test_jobs_api_list_all():
    """Test listing all jobs"""
    # Create multiple mock jobs
    jobs["job-1"] = {"status": "completed", "created_at": "2024-01-01T00:00:00"}
    jobs["job-2"] = {"status": "generating", "created_at": "2024-01-01T01:00:00"}
    jobs["job-3"] = {"status": "failed", "created_at": "2024-01-01T02:00:00"}

    response = client.get("/api/jobs")
    assert response.status_code == 200
    data = response.json()
    assert data["total_jobs"] == 3
    assert len(data["jobs"]) == 3

    job_ids = [job["job_id"] for job in data["jobs"]]
    assert "job-1" in job_ids
    assert "job-2" in job_ids
    assert "job-3" in job_ids


def test_jobs_api_empty_list():
    """Test listing jobs when no jobs exist"""
    response = client.get("/api/jobs")
    assert response.status_code == 200
    data = response.json()
    assert data["total_jobs"] == 0
    assert data["jobs"] == []
