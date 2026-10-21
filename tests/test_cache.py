import fakeredis
import redis

from rtrec.cache import NullCache, RedisCache, make_cache


def test_redis_round_trip():
    cache = RedisCache(fakeredis.FakeRedis())
    cache.set("k", {"items": [1, 2]}, ttl=60)
    assert cache.get("k") == {"items": [1, 2]}
    assert cache.get("missing") is None


class _BrokenClient:
    def get(self, key):
        raise redis.ConnectionError("down")

    def set(self, key, value, ex=None):
        raise redis.ConnectionError("down")


def test_redis_errors_are_counted_not_raised():
    cache = RedisCache(_BrokenClient())
    assert cache.get("k") is None
    cache.set("k", {}, ttl=1)
    assert cache.errors == 2


def test_null_cache_when_no_url_or_unreachable():
    assert isinstance(make_cache(None), NullCache)
    assert isinstance(make_cache("redis://127.0.0.1:1/0"), NullCache)
