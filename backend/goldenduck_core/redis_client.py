import redis
import os

_redis_host = os.getenv("REDIS_HOST", "redis")
_redis_port_str = os.getenv("REDIS_PORT", "6379")
try:
    _redis_port = int(_redis_port_str)
except (TypeError, ValueError):
    _redis_port = 6379

redis_client = redis.Redis(host=_redis_host, port=_redis_port, decode_responses=True)
