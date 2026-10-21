import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(artifacts, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    monkeypatch.delenv("REDIS_URL", raising=False)
    from rtrec.api import app
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_model_and_cache(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
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
