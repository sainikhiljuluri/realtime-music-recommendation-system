import fakeredis
import pytest
from fastapi.testclient import TestClient

from rtrec import api as api_module
from rtrec.cache import RedisCache


@pytest.fixture()
def client(artifacts, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    monkeypatch.setattr(api_module, "make_cache", lambda url: RedisCache(fakeredis.FakeRedis()))
    with TestClient(api_module.app) as c:
        yield c


def test_delete_without_token_is_rejected(client):
    assert client.delete("/users/u1/recommendations").status_code == 401


def test_delete_with_wrong_token_is_forbidden(client):
    resp = client.delete("/users/u1/recommendations", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 403


def test_delete_with_correct_token_succeeds(client):
    client.get("/recommendations/u1", params={"k": 2})
    resp = client.delete("/users/u1/recommendations", headers={"Authorization": "Bearer test-admin-key"})
    assert resp.status_code == 200
    assert resp.json()["removed"] == 1


def test_delete_is_disabled_when_key_is_not_configured(artifacts, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.setattr(api_module, "make_cache", lambda url: RedisCache(fakeredis.FakeRedis()))
    with TestClient(api_module.app) as c:
        resp = c.delete("/users/u1/recommendations", headers={"Authorization": "Bearer anything"})
        assert resp.status_code == 503


def test_recommendations_remain_readable_without_token(client):
    assert client.get("/recommendations/u1", params={"k": 2}).status_code == 200


def test_wrong_scheme_is_treated_as_missing(client):
    assert client.delete("/users/u1/recommendations", headers={"Authorization": "Basic abc"}).status_code == 401
