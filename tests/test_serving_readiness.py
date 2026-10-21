"""Readiness behavior for the generic artifacts (see test_music_serving.py for Music4All)."""
from fastapi.testclient import TestClient

from rtrec.service import Recommender


def test_self_check_failure_keeps_api_not_ready(artifacts, monkeypatch):
    from rtrec import api as api_module

    monkeypatch.setenv("ARTIFACT_DIR", str(artifacts))
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(Recommender, "_cold_start", lambda self, k: [])
    with TestClient(api_module.app) as client:
        assert client.get("/health").status_code == 200
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["reason"] == "RuntimeError"
