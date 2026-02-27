import importlib

import job_scheduler.redis_client as redis_client_module


def test_redis_client_uses_default_port_on_invalid_env(monkeypatch):
    calls = {}

    def fake_redis(*, host, port, decode_responses):
        calls["host"] = host
        calls["port"] = port
        calls["decode_responses"] = decode_responses
        return object()

    monkeypatch.setenv("REDIS_HOST", "my-redis")
    monkeypatch.setenv("REDIS_PORT", "invalid-port")
    monkeypatch.setattr("redis.Redis", fake_redis)

    importlib.reload(redis_client_module)

    assert calls["host"] == "my-redis"
    assert calls["port"] == 6379
    assert calls["decode_responses"] is True


def test_redis_client_uses_env_port_when_valid(monkeypatch):
    calls = {}

    def fake_redis(*, host, port, decode_responses):
        calls["host"] = host
        calls["port"] = port
        calls["decode_responses"] = decode_responses
        return object()

    monkeypatch.setenv("REDIS_HOST", "redis-prod")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setattr("redis.Redis", fake_redis)

    importlib.reload(redis_client_module)

    assert calls["host"] == "redis-prod"
    assert calls["port"] == 6380
    assert calls["decode_responses"] is True
