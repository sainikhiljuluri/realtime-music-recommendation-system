
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(artifacts, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    monkeypatch.delenv("REDIS_URL", raising=False)
    from rtrec.api import app
    with TestClient(app) as test_client:
        yield test_client


def test_health_is_liveness_only(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_model_and_cache(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["users"] == 200 and body["items"] == 300
    assert body["cache_available"] is False


def test_personalized_recommendations(client):
    body = client.get("/recommendations/u5", params={"k": 4}).json()
    assert body["source"] == "personalized"
    assert len(body["items"]) == 4
    assert {"item_id", "score"} <= set(body["items"][0])


def test_cold_start_recommendations(client):
    body = client.get("/recommendations/brand-new", params={"k": 3}).json()
    assert body["source"] == "cold_start"
    assert len(body["items"]) == 3


def test_k_out_of_range_is_rejected(client):
    assert client.get("/recommendations/u5", params={"k": 0}).status_code == 422
    assert client.get("/recommendations/u5", params={"k": 51}).status_code == 422


def test_missing_artifacts_keep_health_up_but_not_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "does-not-exist"))
    monkeypatch.delenv("REDIS_URL", raising=False)
    from rtrec.api import app
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/recommendations/u1").status_code == 503


def test_api_survives_redis_outage(artifacts, monkeypatch):
    import redis

    from rtrec import api as api_module
    from rtrec.cache import RedisCache

    class Down:
        def get(self, key):
            raise redis.ConnectionError("down")

        def set(self, key, value, ex=None):
            raise redis.ConnectionError("down")

        def scan_iter(self, match, count):
            raise redis.ConnectionError("down")

    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    monkeypatch.setattr(api_module, "make_cache", lambda url: RedisCache(Down()))
    with TestClient(api_module.app) as client:
        resp = client.get("/recommendations/u2", params={"k": 3})
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 3
        # The generation read fails first, so the request skips the cache. The error is counted.
        assert client.get("/ready").json()["cache_errors"] >= 1


def test_delete_purges_user_cache_entries(artifacts, monkeypatch):
    import fakeredis

    from rtrec import api as api_module
    from rtrec.cache import RedisCache

    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(api_module, "make_cache", lambda url: RedisCache(fake))
    with TestClient(api_module.app) as client:
        client.get("/recommendations/u4", params={"k": 3})
        client.get("/recommendations/u4", params={"k": 5})
        client.get("/recommendations/u9", params={"k": 3})
        auth = {"Authorization": "Bearer test-admin-key"}
        removed = client.delete("/users/u4/recommendations", headers=auth).json()["removed"]
        assert removed == 2
        assert client.get("/recommendations/u9", params={"k": 3}).json()["cache_hit"] is True
        assert client.get("/recommendations/u4", params={"k": 3}).json()["cache_hit"] is False


def test_logs_contain_pseudonyms_not_raw_ids(artifacts, monkeypatch, caplog):
    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    monkeypatch.delenv("REDIS_URL", raising=False)
    from rtrec import api as api_module
    with caplog.at_level("INFO", logger="rtrec.api"), TestClient(api_module.app) as client:
        client.get("/recommendations/u77", params={"k": 2})
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "uid=" in text
    assert "u77" not in text
