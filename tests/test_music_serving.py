"""End-to-end: train a tiny Music4All-style model, export artifacts, and serve them.

These tests load only the saved files (no in-memory objects) and check that the API returns
Music4All user and track ids, excludes what each user already played, and applies the diversity
penalty selected on validation.
"""
import dataclasses
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rtrec import api as api_module
from rtrec.artifacts import CATALOG_FILE
from rtrec.cache import RedisCache
from rtrec.hybrid import score_pool, select_top_k
from rtrec.music_pipeline import MusicConfig, run
from rtrec.ranking import History
from rtrec.service import Recommender
from rtrec.telemetry import Telemetry

N_USERS, N_ITEMS, N_GENRES = 60, 80, 5
USER_OFFSET = 1_000_000        # Music4All user ids are not dense indices
CFG = MusicConfig(epochs=1, batch_size=256, candidates=30, popular_candidates=10, ranker_users=40,
                  lambdas=(0.0, 0.2), svd_factors=8)


def _track_id(i: int) -> str:
    return f"TRK{i:04d}"


@pytest.fixture(scope="module")
def music_run(tmp_path_factory):
    """Train on a tiny synthetic log, export artifacts, and return the export and result paths."""
    root = tmp_path_factory.mktemp("music")
    rng = np.random.default_rng(0)
    rows = []
    for u in range(N_USERS):
        for e in range(40):
            rows.append((u, int(rng.integers(0, N_ITEMS)), 1_400_000_000 + e * 3600 + u))
    events = pd.DataFrame(rows, columns=["user_idx", "item_idx", "ts"]).sort_values(["user_idx", "ts"])
    item_genre = rng.integers(0, N_GENRES, size=N_ITEMS)
    user_ids = np.arange(N_USERS) + USER_OFFSET
    item_ids = np.array([_track_id(i) for i in range(N_ITEMS)], dtype=object)
    export = root / "artifacts"
    run(events, item_ids, user_ids, item_genre, N_GENRES, CFG, root / "out", export_dir=export)
    return {"export": export, "out": root / "out", "events": events, "item_genre": item_genre}


@pytest.fixture()
def rec(music_run):
    return Recommender.load(music_run["export"])


def _seen_track_ids(rec, user_idx):
    return {str(rec.item_ids[i]) for i in rec.history.seen(user_idx).tolist()}


def test_export_writes_all_serving_files(music_run):
    names = {p.name for p in music_run["export"].iterdir()}
    assert {"items.faiss", "ranker.txt", "vectors.npz", "history.npz", CATALOG_FILE, "config.json",
            "two_tower.pt"} <= names


def test_config_declares_music_family_and_selected_lambda(music_run):
    config = json.loads((music_run["export"] / "config.json").read_text())
    report = json.loads((music_run["out"] / "music_metrics.json").read_text())
    assert config["model_family"] == "music4all"
    assert config["selected_lambda"] in CFG.lambdas
    assert f"hybrid_lambda_{config['selected_lambda']:g}" == report["selected_on_validation"]


def test_personalized_response_uses_track_ids_not_indices(rec):
    trained = np.flatnonzero(rec.trained_users)
    user_id = str(int(rec.user_ids[trained[0]]))
    out = rec.recommend(user_id, k=5)
    assert out["source"] == "personalized"
    assert len(out["items"]) == 5
    valid = {_track_id(i) for i in range(N_ITEMS)}
    assert all(item["item_id"] in valid for item in out["items"])


def test_served_items_exclude_tracks_the_user_already_played(rec):
    for pos in np.flatnonzero(rec.trained_users)[:20]:
        user_id = str(int(rec.user_ids[pos]))
        served = {item["item_id"] for item in rec.recommend(user_id, k=10)["items"]}
        assert not served & _seen_track_ids(rec, pos)


def test_service_matches_shared_hybrid_path(rec):
    """The served list is the list score_pool + select_top_k produce from the saved artifacts."""
    pos = int(np.flatnonzero(rec.trained_users)[0])
    items, rank_scores = score_pool(rec.index, pos, rec.user_vecs[pos], rec.history, rec.popularity_order,
                                    rec.item_vecs, rec.booster, rec.candidates, rec.popular_candidates)
    expected = [rec._item_name(i) for i, _ in select_top_k(items, rank_scores, rec.history, 8, rec.selected_lambda)]
    got = [item["item_id"] for item in rec.recommend(str(int(rec.user_ids[pos])), k=8)["items"]]
    assert got == expected


def test_unknown_user_gets_catalog_popularity(rec):
    out = rec.recommend("424242424", k=4)
    assert out["source"] == "cold_start"
    assert len(out["items"]) == 4
    assert all(item["item_id"].startswith("TRK") for item in out["items"])


def test_non_numeric_or_oversized_user_ids_are_cold_start(rec):
    for bad in ["abc", "-1", "9" * 30, "1.5", ""]:
        assert rec.recommend(bad, k=2)["source"] == "cold_start"


def test_user_without_trained_vector_is_cold_start(rec):
    trained = rec.trained_users.copy()
    pos = int(np.flatnonzero(trained)[0])
    trained[pos] = False
    masked = dataclasses.replace(rec, trained_users=trained)
    assert masked.recommend(str(int(rec.user_ids[pos])), k=3)["source"] == "cold_start"


def test_lambda_zero_serves_plain_score_order(rec):
    """With lam=0 the list is the ranker's order, as in the lambda=0 variant of the experiment."""
    pos = int(np.flatnonzero(rec.trained_users)[0])
    items, rank_scores = score_pool(rec.index, pos, rec.user_vecs[pos], rec.history, rec.popularity_order,
                                    rec.item_vecs, rec.booster, rec.candidates, rec.popular_candidates)
    plain = [int(items[j]) for j in np.argsort(-rank_scores, kind="stable")[:5]]
    got = [item["item_id"] for item in dataclasses.replace(rec, selected_lambda=0.0).recommend(str(int(rec.user_ids[pos])), k=5)["items"]]
    assert got == [rec._item_name(i) for i in plain]


def test_mismatched_catalog_is_rejected(music_run, tmp_path):
    export = tmp_path / "bad"
    shutil.copytree(music_run["export"], export)
    catalog = dict(np.load(export / CATALOG_FILE))
    catalog["item_ids"] = catalog["item_ids"][:-1]
    np.savez(export / CATALOG_FILE, **catalog)
    with pytest.raises(ValueError, match="catalog size"):
        Recommender.load(export)


def test_history_roundtrip_preserves_seen_sets(music_run):
    saved = np.load(music_run["export"] / "history.npz")
    hist = History(saved["item_genre"], saved["item_artist"], saved["item_popularity"], saved["genre_profile"],
                   saved["artist_counts"], saved["activity"], saved["seen_indptr"], saved["seen_indices"])
    assert hist.seen_indptr.shape[0] == N_USERS + 1
    assert hist.seen_indices.max() < N_ITEMS


def test_api_serves_music_artifacts(music_run, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(music_run["export"]))
    monkeypatch.delenv("REDIS_URL", raising=False)
    with TestClient(api_module.app) as client:
        ready = client.get("/ready").json()
        assert ready["status"] == "ready"
        assert ready["model_family"] == "music4all"
        assert ready["trained_users"] <= N_USERS
        body = client.get(f"/recommendations/{USER_OFFSET + 3}", params={"k": 5}).json()
        assert body["source"] == "personalized"
        assert all(item["item_id"].startswith("TRK") for item in body["items"])
        cold = client.get("/recommendations/0", params={"k": 3}).json()
        assert cold["source"] == "cold_start"


def test_readiness_fails_with_reason_when_catalog_is_missing(music_run, tmp_path, monkeypatch):
    export = tmp_path / "missing-catalog"
    shutil.copytree(music_run["export"], export)
    (export / CATALOG_FILE).unlink()
    monkeypatch.setenv("ARTIFACT_DIR", str(export))
    monkeypatch.delenv("REDIS_URL", raising=False)
    with TestClient(api_module.app) as client:
        assert client.get("/health").status_code == 200
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["reason"] == "FileNotFoundError"


def test_shutdown_releases_cache_connections(music_run, monkeypatch):
    import fakeredis

    closed = []

    class Tracking(RedisCache):
        def close(self):
            closed.append(True)
            super().close()

    monkeypatch.setenv("ARTIFACT_DIR", str(music_run["export"]))
    monkeypatch.setattr(api_module, "make_cache", lambda url: Tracking(fakeredis.FakeRedis()))
    with TestClient(api_module.app) as client:
        client.get(f"/recommendations/{USER_OFFSET + 1}", params={"k": 2})
        assert closed == []
    assert closed == [True]


def test_metrics_endpoint_counts_sources_without_user_ids(music_run, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(music_run["export"]))
    monkeypatch.delenv("REDIS_URL", raising=False)
    with TestClient(api_module.app) as client:
        client.get(f"/recommendations/{USER_OFFSET + 2}", params={"k": 2})
        client.get("/recommendations/0", params={"k": 2})
        snap = client.get("/metrics").json()
        assert snap["counters"]["requests_total"] == 2
        assert snap["counters"]["requests_cold_start"] == 1
        assert snap["model_loaded"] is True
        assert str(USER_OFFSET + 2) not in json.dumps(snap)


def test_telemetry_percentiles_and_hit_rate():
    t = Telemetry(window=100)
    for ms in range(1, 101):
        t.observe("personalized", cache_hit=ms % 4 == 0, latency_ms=float(ms))
    snap = t.snapshot()
    assert snap["counters"]["requests_total"] == 100
    assert snap["cache_hit_rate"] == 0.25
    assert snap["latency"]["p50_ms"] == pytest.approx(50.5)
    assert snap["latency"]["p99_ms"] > snap["latency"]["p95_ms"] > snap["latency"]["p50_ms"]


def test_predict_uses_single_thread_to_avoid_openmp_crash():
    from rtrec.ranking import predict

    class Spy:
        kwargs = None

        def predict(self, features, **kwargs):
            Spy.kwargs = kwargs
            return np.zeros(len(features))

    predict(Spy(), np.zeros((3, 2), dtype=np.float32))
    assert Spy.kwargs == {"num_threads": 1}
