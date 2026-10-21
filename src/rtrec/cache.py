"""Recommendation cache with a Redis backend and a no-op fallback.

Cache failures never fail a request. A Redis error is counted and the request falls
through to the model.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import redis

log = logging.getLogger(__name__)


class NullCache:
    available = False

    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any, ttl: int) -> None:
        return None

    def delete_matching(self, pattern: str) -> int:
        return 0


class RedisCache:
    available = True

    def __init__(self, client: redis.Redis) -> None:
        self.client = client
        self.errors = 0

    @classmethod
    def from_url(cls, url: str, timeout: float = 0.05) -> "RedisCache":
        client = redis.Redis.from_url(url, socket_timeout=timeout, socket_connect_timeout=timeout)
        return cls(client)

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False

    def get(self, key: str) -> Any | None:
        try:
            raw = self.client.get(key)
        except redis.RedisError as exc:
            self.errors += 1
            log.warning("cache get failed: %s", exc)
            return None
        return json.loads(raw) if raw is not None else None

    def set(self, key: str, value: Any, ttl: int) -> None:
        try:
            self.client.set(key, json.dumps(value), ex=ttl)
        except redis.RedisError as exc:
            self.errors += 1
            log.warning("cache set failed: %s", exc)

    def delete_matching(self, pattern: str) -> int:
        try:
            keys = list(self.client.scan_iter(match=pattern, count=500))
            return int(self.client.delete(*keys)) if keys else 0
        except redis.RedisError as exc:
            self.errors += 1
            log.warning("cache purge failed: %s", exc)
            return 0


def make_cache(url: str | None) -> RedisCache | NullCache:
    """Return a working Redis cache, or a NullCache if no URL is set or Redis is unreachable."""
    if not url:
        return NullCache()
    cache = RedisCache.from_url(url)
    if cache.ping():
        return cache
    log.warning("Redis at %s is unreachable; serving without cache", url)
    return NullCache()
