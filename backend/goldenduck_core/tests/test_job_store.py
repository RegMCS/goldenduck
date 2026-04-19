import json

from goldenduck_core.services import job_store as job_store_module


def test_create_and_enqueue_job(monkeypatch):
    set_calls = []
    lpush_calls = []

    class RedisStub:
        @staticmethod
        def set(key, value):
            set_calls.append((key, value))

        @staticmethod
        def lpush(queue, job_id):
            lpush_calls.append((queue, job_id))

    monkeypatch.setattr(job_store_module, "redis_client", RedisStub())
    store = job_store_module.JobStore()

    store.create_job("job-1", "user-1", {"ticker": "AAPL"}, "queued")
    store.enqueue("job-1")

    assert len(set_calls) == 2
    assert set_calls[0][0] == "job:job-1:job"
    payload = json.loads(set_calls[0][1])
    assert payload["job_id"] == "job-1"
    assert payload["user_id"] == "user-1"
    assert payload["status"] == "queued"
    assert payload["parameters"] == {"ticker": "AAPL"}
    assert "created_at" in payload
    assert set_calls[1] == ("job:job-1:status", "queued")
    assert lpush_calls == [("queue:garch", "job-1")]


def test_getters_and_setters(monkeypatch):
    state = {
        "job:job-1:job": json.dumps(
            {"job_id": "job-1", "user_id": "u1", "status": "queued"}
        ),
        "job:job-1:status": "queued",
        "job:job-1:parameters": json.dumps({"omega": 0.1}),
        "job:job-1:metrics": json.dumps({"ks_statistic": 0.05}),
        "job:job-1:error": "boom",
        "job:job-1:output_file": "s3://b/k.csv",
    }

    class RedisStub:
        @staticmethod
        def get(key):
            return state.get(key)

        @staticmethod
        def set(key, value):
            state[key] = value

    monkeypatch.setattr(job_store_module, "redis_client", RedisStub())
    store = job_store_module.JobStore()

    assert store.get_job("job-1") == {
        "job_id": "job-1",
        "user_id": "u1",
        "status": "queued",
    }
    assert store.get_status("job-1") == "queued"
    assert store.get_parameters("job-1") == {"omega": 0.1}
    assert store.get_metrics("job-1") == {"ks_statistic": 0.05}
    assert store.get_error("job-1") == "boom"
    assert store.get_output_file("job-1") == "s3://b/k.csv"

    store.set_status("job-1", "completed")
    assert store.get_status("job-1") == "completed"
    assert store.get_job("job-1")["status"] == "completed"

    store.set_parameters("job-1", {"alpha": 0.2})
    store.set_metrics("job-1", {"acf": 0.3})
    store.set_error("job-1", "err")
    store.set_output_file("job-1", "s3://bucket/key")

    assert store.get_parameters("job-1") == {"alpha": 0.2}
    assert store.get_metrics("job-1") == {"acf": 0.3}
    assert store.get_error("job-1") == "err"
    assert store.get_output_file("job-1") == "s3://bucket/key"


def test_defaults_for_missing_values(monkeypatch):
    class RedisStub:
        @staticmethod
        def get(_key):
            return None

        @staticmethod
        def set(_key, _value):
            return None

    monkeypatch.setattr(job_store_module, "redis_client", RedisStub())
    store = job_store_module.JobStore()

    assert store.get_job("missing") is None
    assert store.get_status("missing") is None
    assert store.get_parameters("missing") == {}
    assert store.get_metrics("missing") == {}
    assert store.get_error("missing") is None
    assert store.get_output_file("missing") is None

    # Missing job should not crash status update path.
    store.set_status("missing", "failed")
