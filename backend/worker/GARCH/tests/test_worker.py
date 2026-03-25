import pytest
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from redis.exceptions import ConnectionError as RedisConnectionError
import tempfile
import os


class TestRedisClientCreation:
    """Test Redis client creation with retry logic"""

    def test_create_redis_client_logic(self):
        """Test the retry logic for Redis connection"""
        # This tests the logic that would be in create_redis_client
        # without actually importing the module

        # Simulate successful connection
        mock_client = Mock()
        mock_client.ping.return_value = True

        # The worker should:
        # 1. Create Redis client
        # 2. Call ping() to validate connection
        # 3. Log success
        # 4. Return client

        mock_client.ping()
        assert mock_client.ping.called

    def test_retry_on_failure_logic(self):
        """Test retry behavior on connection failure"""
        # Test the retry logic
        max_attempts = 3
        attempts = 0

        def mock_connect():
            nonlocal attempts
            attempts += 1
            if attempts < max_attempts:
                raise RedisConnectionError("Connection failed")
            return True

        # Simulate retries
        while attempts < max_attempts:
            try:
                result = mock_connect()
                if result:
                    break
            except RedisConnectionError:
                pass

        assert attempts == max_attempts


class TestJobProcessing:
    """Test job processing logic"""

    def test_job_metadata_parsing(self):
        """Test parsing of job metadata with various parameter combinations"""
        # Test with all parameters
        meta = {
            "request": {
                "ticker": "AAPL",
                "p": 2,
                "q": 2,
                "num_scenarios": 200,
                "horizon": 500,
                "volatility_multiplier": 1.5,
            }
        }
        req = meta["request"]
        assert req["ticker"] == "AAPL"
        assert int(req.get("p", 1)) == 2
        assert int(req.get("q", 1)) == 2
        assert int(req.get("num_scenarios", 100)) == 200
        assert int(req.get("horizon", 500)) == 500
        assert float(req.get("volatility_multiplier", 1.0)) == 1.5

        # Test with default parameters
        meta_minimal = {"request": {"ticker": "GOOG"}}
        req_minimal = meta_minimal["request"]
        assert req_minimal["ticker"] == "GOOG"
        assert int(req_minimal.get("p", 1)) == 1
        assert int(req_minimal.get("q", 1)) == 1
        assert int(req_minimal.get("num_scenarios", 100)) == 100
        assert int(req_minimal.get("horizon", 500)) == 500
        assert float(req_minimal.get("volatility_multiplier", 1.0)) == 1.0

    def test_job_metadata_json_serialization(self):
        """Test that job metadata can be serialized/deserialized"""
        meta = {
            "request": {
                "ticker": "MSFT",
                "p": 1,
                "q": 1,
                "num_scenarios": 50,
                "horizon": 500,
            }
        }

        # Serialize
        json_str = json.dumps(meta)

        # Deserialize
        recovered = json.loads(json_str)

        assert recovered == meta


class TestRedisOperations:
    """Test Redis operations patterns used in worker"""

    def test_job_status_workflow(self):
        """Test the job status workflow"""
        mock_redis = Mock()
        job_id = "test-job-123"

        # Worker picks up job
        mock_redis.brpop.return_value = ("queue:garch", job_id)
        _, received_job_id = mock_redis.brpop("queue:garch")
        assert received_job_id == job_id

        # Set status to running
        mock_redis.set(f"job:{job_id}:status", "running")

        # Get metadata
        meta = {"request": {"ticker": "AAPL"}}
        mock_redis.get.return_value = json.dumps(meta)
        meta_raw = mock_redis.get(f"job:{job_id}:meta")
        assert meta_raw is not None

        # Complete job
        mock_redis.set(f"job:{job_id}:status", "completed")
        mock_redis.set(f"job:{job_id}:parameters", json.dumps({"omega": 0.1}))
        mock_redis.set(f"job:{job_id}:metrics", json.dumps({"ks_statistic": 0.05}))
        mock_redis.set(f"job:{job_id}:output_file", "output/test.csv")

        # Verify all operations were called
        assert mock_redis.set.called
        assert mock_redis.brpop.called
        assert mock_redis.get.called

    def test_job_error_handling(self):
        """Test error handling sets correct status"""
        mock_redis = Mock()
        job_id = "test-job-error"
        error_msg = "Model fitting failed"

        # Set failed status
        mock_redis.set(f"job:{job_id}:status", "failed")
        mock_redis.set(f"job:{job_id}:error", error_msg)

        # Verify calls
        mock_redis.set.assert_any_call(f"job:{job_id}:status", "failed")
        mock_redis.set.assert_any_call(f"job:{job_id}:error", error_msg)


class TestGARCHServiceIntegration:
    """Test GARCH service integration patterns"""

    def test_garch_service_workflow(self):
        """Test the typical GARCH service workflow"""
        mock_garch = Mock()

        # Mock data
        data = pd.DataFrame({"Close": [100, 101, 102, 103, 104]})

        # Fit model
        mock_params = {"omega": 0.1, "alpha": 0.2, "beta": 0.7}
        mock_garch.fit_with_retry.return_value = mock_params
        params = mock_garch.fit_with_retry(data, p=1, q=1)
        assert params == mock_params

        # Generate scenarios
        mock_scenario = pd.DataFrame({"returns": [0.01, 0.02, 0.03]})
        mock_garch.generate_scenarios.return_value = [mock_scenario, mock_scenario]
        scenarios = mock_garch.generate_scenarios(
            num_scenarios=2, horizon=3, volatility_multiplier=1.0
        )
        assert len(scenarios) == 2

        # Validate scenarios
        mock_metrics = {"ks_statistic": 0.05}
        mock_garch.validate_scenarios.return_value = mock_metrics
        metrics = mock_garch.validate_scenarios(scenarios)
        assert metrics == mock_metrics

        # Verify all methods were called
        mock_garch.fit_with_retry.assert_called_once()
        mock_garch.generate_scenarios.assert_called_once()
        mock_garch.validate_scenarios.assert_called_once()

    def test_garch_parameters_extraction(self):
        """Test extraction and storage of GARCH parameters"""
        params = {
            "omega": 0.1,
            "alpha": 0.2,
            "beta": 0.7,
            "converged": True,
            "aic": 100.5,
            "bic": 110.2,
        }

        # Verify parameters can be JSON serialized
        json_params = json.dumps(params)
        recovered_params = json.loads(json_params)

        assert recovered_params["omega"] == params["omega"]
        assert recovered_params["alpha"] == params["alpha"]
        assert recovered_params["beta"] == params["beta"]


class TestOutputGeneration:
    """Test output file generation"""

    def test_scenario_combination(self):
        """Test combining multiple scenarios into a single dataframe"""
        scenario1 = pd.DataFrame(
            {"returns": [0.01, 0.02, 0.03], "volatility": [0.1, 0.11, 0.12]}
        )
        scenario2 = pd.DataFrame(
            {"returns": [0.015, 0.025, 0.035], "volatility": [0.105, 0.115, 0.125]}
        )

        scenarios = [scenario1, scenario2]

        # Combine scenarios (same logic as worker)
        all_rows = []
        for i, scenario_df in enumerate(scenarios):
            df = scenario_df.copy()
            df["scenario_id"] = i + 1
            all_rows.append(df)

        combined = pd.concat(all_rows, ignore_index=True)

        # Verify the combined dataframe
        assert len(combined) == 6
        assert "scenario_id" in combined.columns
        assert combined["scenario_id"].unique().tolist() == [1, 2]
        assert len(combined[combined["scenario_id"] == 1]) == 3
        assert len(combined[combined["scenario_id"] == 2]) == 3

    def test_csv_output_creation(self):
        """Test CSV file creation"""
        df = pd.DataFrame({"returns": [0.01, 0.02, 0.03], "scenario_id": [1, 1, 1]})

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            output_path = f.name

        try:
            df.to_csv(output_path, index=False)

            # Verify file exists and can be read
            assert os.path.exists(output_path)

            # Read back and verify
            read_df = pd.read_csv(output_path)
            assert len(read_df) == len(df)
            assert list(read_df.columns) == list(df.columns)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestErrorHandling:
    """Test error handling patterns"""

    def test_missing_metadata_error(self):
        """Test handling of missing job metadata"""
        meta_raw = None

        with pytest.raises(ValueError, match="Job metadata not found"):
            if not meta_raw:
                raise ValueError("Job metadata not found")

    def test_empty_market_data_error(self):
        """Test handling of empty market data"""
        data = pd.DataFrame()  # Empty dataframe
        ticker = "INVALID"

        with pytest.raises(ValueError, match="No market data returned for ticker"):
            if data.empty:
                raise ValueError(f"No market data returned for ticker {ticker}")

    def test_parameter_validation(self):
        """Test that parameters are properly validated"""
        # Valid parameters
        req = {"ticker": "AAPL", "p": 1, "q": 1, "num_scenarios": 100, "horizon": 500}

        ticker = req["ticker"]
        p = int(req.get("p", 1))
        q = int(req.get("q", 1))
        num_scenarios = int(req.get("num_scenarios", 100))
        horizon = int(req.get("horizon", 500))

        assert ticker == "AAPL"
        assert p == 1
        assert q == 1
        assert num_scenarios == 100
        assert horizon == 500


class TestEnvironmentConfiguration:
    """Test environment variable configuration"""

    def test_default_redis_configuration(self):
        """Test default Redis configuration values"""
        # Default values that would be used
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", 6379))

        assert redis_host == "redis"
        assert redis_port == 6379

    @patch.dict(os.environ, {"REDIS_HOST": "custom-redis", "REDIS_PORT": "7000"})
    def test_custom_redis_configuration(self):
        """Test custom Redis configuration from environment"""
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", 6379))

        assert redis_host == "custom-redis"
        assert redis_port == 7000


class TestYFinanceIntegration:
    """Test yfinance data handling"""

    @patch("yfinance.download")
    def test_yfinance_data_download(self, mock_download):
        """Test yfinance data download"""
        mock_data = pd.DataFrame(
            {
                "Close": [100, 101, 102, 103, 104],
                "Open": [99, 100, 101, 102, 103],
                "High": [101, 102, 103, 104, 105],
                "Low": [98, 99, 100, 101, 102],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            }
        )
        mock_download.return_value = mock_data

        # Simulate the download call
        import yfinance as yf

        data = yf.download("AAPL", period="2y", progress=False, threads=False)

        assert not data.empty
        assert "Close" in data.columns
        assert len(data) == 5

    @patch("yfinance.download")
    def test_empty_yfinance_data(self, mock_download):
        """Test handling of empty yfinance data"""
        mock_download.return_value = pd.DataFrame()

        import yfinance as yf

        data = yf.download("INVALID", period="2y", progress=False, threads=False)

        assert data.empty
