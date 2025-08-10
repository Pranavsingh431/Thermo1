import time
import threading
from typing import Optional

from app.config import settings

_lock = threading.Lock()
_inmemory_store: dict[str, tuple[int, float]] = {}


def _redis_client():
    try:
        import redis
        return redis.Redis.from_url(settings.__dict__.get('REDIS_URL', 'redis://localhost:6379'), decode_responses=True)
    except Exception:
        return None


def rate_limit(key: str, limit: int, window_seconds: int) -> bool:
    """Simple fixed window rate limit. Returns True if allowed, False if limited."""
    client = _redis_client()
    now = int(time.time())
    window = now // window_seconds
    rl_key = f"ratelimit:{key}:{window}"

    if client:
        try:
            current = client.incr(rl_key)
            if current == 1:
                client.expire(rl_key, window_seconds)
            return current <= limit
        except Exception:
            pass

    # Fallback to in-memory
    with _lock:
        count, exp = _inmemory_store.get(rl_key, (0, time.time() + window_seconds))
        count += 1
        _inmemory_store[rl_key] = (count, exp)
        # Cleanup
        for k, (_, e) in list(_inmemory_store.items()):
            if e < time.time():
                _inmemory_store.pop(k, None)
        return count <= limit

