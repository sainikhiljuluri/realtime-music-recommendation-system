"""End-to-end: train a tiny Music4All-style model, export artifacts, and serve them.

These tests load only the saved files (no in-memory objects) and check that the API returns
Music4All user and track ids, excludes what each user already played, and applies the diversity
penalty selected on validation.
"""
import dataclasses
import json
import shutil
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from rtrec import api as api_module
from rtrec.artifacts import CATALOG_FILE
from rtrec.cache import RedisCache
from rtrec.hybrid import score_pool, select_top_k
from rtrec.music_pipeline import MusicConfig, _items_frame, run
from rtrec.online import EventRejected
from rtrec.ranking import History, candidate_features
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
        snap = client.get("/stats").json()
        assert snap["requests"] == {"personalized": 1, "cold_start": 1}
        assert snap["model_loaded"] is True
        assert str(USER_OFFSET + 2) not in json.dumps(snap)


def test_telemetry_percentiles_and_hit_rate():
    t = Telemetry(window=100)
    for ms in range(1, 101):
        t.observe("personalized", cache_hit=ms % 4 == 0, latency_ms=float(ms))
    snap = t.snapshot()
    assert sum(snap["requests"].values()) == 100
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


# --- online listening features -------------------------------------------------------------



def _cutoff(music_run) -> int:
    return json.loads((music_run["export"] / "config.json").read_text())["history_cutoff_unix"]


def _listen(rec, pos, item, event_id, ts):
    return rec.ingest(str(int(rec.user_ids[pos])), _track_id(item), event_id, ts, int(time.time()))


def _recent_ts(music_run) -> int:
    return _cutoff(music_run) + 100


def test_listen_rejects_unknown_track_user_and_stale_timestamps(music_run, rec):
    pos = int(np.flatnonzero(rec.trained_users)[0])
    uid = str(int(rec.user_ids[pos]))
    ts = _recent_ts(music_run)
    with pytest.raises(EventRejected) as bad_track:
        rec.ingest(uid, "NOT-A-TRACK", "e1", ts, ts)
    assert bad_track.value.status == 422
    with pytest.raises(EventRejected) as bad_user:
        rec.ingest("424242424", _track_id(1), "e2", ts, ts)
    assert bad_user.value.status == 404
    with pytest.raises(EventRejected) as stale:
        rec.ingest(uid, _track_id(1), "e3", _cutoff(music_run) - 1, ts)
    assert stale.value.status == 422
    with pytest.raises(EventRejected) as future:
        rec.ingest(uid, _track_id(1), "e4", ts + 10_000, ts)
    assert future.value.status == 422


def test_duplicate_event_id_changes_state_once(music_run, rec):
    pos = int(np.flatnonzero(rec.trained_users)[0])
    ts = _recent_ts(music_run)
    before = rec.history.activity[pos]
    assert _listen(rec, pos, 7, "same-id", ts) == "accepted"
    assert _listen(rec, pos, 7, "same-id", ts) == "duplicate"
    assert rec.history.activity[pos] == before + 1
    assert rec.online.duplicates == 1


def test_listen_updates_user_and_item_features(music_run, rec):
    pos = int(np.flatnonzero(rec.trained_users)[0])
    item = 9
    genre = int(rec.history.item_genre[item])
    activity, pop = rec.history.activity[pos], rec.history.item_popularity[item]
    share = rec.history.genre_profile[pos, genre]
    _listen(rec, pos, item, "f1", _recent_ts(music_run))
    assert rec.history.activity[pos] == activity + 1
    assert rec.history.item_popularity[item] == pop + 1
    assert rec.history.genre_profile[pos].sum() == pytest.approx(1.0, abs=1e-5)
    assert rec.history.genre_profile[pos, genre] > share
    assert int(item) in rec.history.seen(pos).tolist()


def test_listen_changes_genre_affinity_feature_of_candidates(music_run, rec):
    pos = int(np.flatnonzero(rec.trained_users)[0])
    cands = np.arange(20)
    before = candidate_features(rec.history, pos, cands, np.zeros(20), np.arange(20)).copy()
    genre_of_first = int(rec.history.item_genre[0])
    same_genre = [int(i) for i in cands if rec.history.item_genre[i] == genre_of_first]
    _listen(rec, pos, same_genre[-1] if len(same_genre) > 1 else 0, "g1", _recent_ts(music_run))
    after = candidate_features(rec.history, pos, cands, np.zeros(20), np.arange(20))
    affinity_col = 3
    assert np.any(after[:, affinity_col] != before[:, affinity_col])


def test_listened_track_is_removed_from_recommendations(music_run, rec):
    pos = int(np.flatnonzero(rec.trained_users)[0])
    uid = str(int(rec.user_ids[pos]))
    top = rec.recommend(uid, k=5)["items"][0]["item_id"]
    assert _listen(rec, pos, int(top[3:]), "rec1", _recent_ts(music_run)) == "accepted"
    after = [item["item_id"] for item in rec.recommend(uid, k=5)["items"]]
    assert top not in after


def test_listen_invalidates_only_that_users_cache(music_run):
    import fakeredis

    from rtrec.cache import RedisCache

    cache = RedisCache(fakeredis.FakeRedis())
    rec = Recommender.load(music_run["export"], cache=cache, ttl=600)
    pos = int(np.flatnonzero(rec.trained_users)[0])
    other = int(np.flatnonzero(rec.trained_users)[1])
    uid, other_uid = str(int(rec.user_ids[pos])), str(int(rec.user_ids[other]))
    first = rec.recommend(uid, k=5)
    assert rec.recommend(uid, k=5)["cache_hit"] is True
    rec.recommend(other_uid, k=5)
    top = first["items"][0]["item_id"]
    _listen(rec, pos, int(top[3:]), "cache1", _recent_ts(music_run))
    fresh = rec.recommend(uid, k=5)
    assert fresh["cache_hit"] is False
    assert top not in [item["item_id"] for item in fresh["items"]]
    assert rec.recommend(other_uid, k=5)["cache_hit"] is True


def test_event_order_does_not_change_features(music_run):
    ts0 = _recent_ts(music_run)
    events = [(7, f"o{i}", ts0 + i) for i in range(6)]
    results = []
    for ordering in (events, list(reversed(events))):
        rec = Recommender.load(music_run["export"])
        pos = int(np.flatnonzero(rec.trained_users)[0])
        for item, eid, ts in ordering:
            _listen(rec, pos, item, eid, ts)
        results.append((rec.history.genre_profile.copy(), rec.history.activity.copy(),
                        rec.history.item_popularity.copy(), sorted(rec.history.seen(pos).tolist())))
    a, b = results
    assert np.allclose(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) and a[3] == b[3]


def test_online_features_match_offline_rebuild(music_run):
    """Applying listens online gives the same features as rebuilding History offline from all events."""
    cutoff = _cutoff(music_run)
    base = music_run["events"]
    base = base[base["ts"] < cutoff]
    rec = Recommender.load(music_run["export"])
    pos_users = np.flatnonzero(rec.trained_users)[:6]
    rng = np.random.default_rng(3)
    new_rows = []
    for n, pos in enumerate(pos_users):
        for j in range(5):
            item = int(rng.integers(0, N_ITEMS))
            ts = cutoff + 60 * (n * 10 + j) + 1
            _listen(rec, int(pos), item, f"c{n}-{j}", ts)
            new_rows.append((int(pos), item, ts))
    all_events = pd.concat([base, pd.DataFrame(new_rows, columns=["user_idx", "item_idx", "ts"])])
    items = _items_frame(N_ITEMS, music_run["item_genre"], N_GENRES)
    offline = History.from_events(all_events, items, N_USERS, N_GENRES + 1, 1)
    np.testing.assert_allclose(rec.history.genre_profile, offline.genre_profile, atol=1e-5)
    np.testing.assert_array_equal(rec.history.activity, offline.activity)
    np.testing.assert_array_equal(rec.history.item_popularity, offline.item_popularity)
    for u in range(N_USERS):
        np.testing.assert_array_equal(rec.history.seen(u), offline.seen(u))


def test_artifact_history_matches_offline_rebuild_before_any_listen(music_run):
    """Sanity check for the consistency test: the saved history is the offline rebuild of pre-cutoff events."""
    cutoff = _cutoff(music_run)
    base = music_run["events"]
    base = base[base["ts"] < cutoff]
    rec = Recommender.load(music_run["export"])
    items = _items_frame(N_ITEMS, music_run["item_genre"], N_GENRES)
    offline = History.from_events(base, items, N_USERS, N_GENRES + 1, 1)
    np.testing.assert_allclose(rec.history.genre_profile, offline.genre_profile, atol=1e-6)
    np.testing.assert_array_equal(rec.history.activity, offline.activity)


def test_api_listen_auth_and_status_codes(music_run, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(music_run["export"]))
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("INGEST_API_KEY", raising=False)
    ts = _recent_ts(music_run)
    body = {"event_id": "api-1", "track_id": _track_id(4), "timestamp": ts}
    with TestClient(api_module.app) as client:
        trained = int(np.flatnonzero(Recommender.load(music_run["export"]).trained_users)[0])
        uid = USER_OFFSET + trained
        path = f"/users/{uid}/listens"
        assert client.post(path, json=body).status_code == 503          # ingestion disabled
        monkeypatch.setenv("INGEST_API_KEY", "ingest-key")
    with TestClient(api_module.app) as client:
        assert client.post(path, json=body).status_code == 401
        auth = {"Authorization": "Bearer ingest-key"}
        assert client.post(path, json=body, headers={"Authorization": "Bearer wrong"}).status_code == 403
        first = client.post(path, json=body, headers=auth)
        assert first.status_code == 202 and first.json()["status"] == "accepted"
        assert "user_id" in first.json() and str(uid) not in first.text
        again = client.post(path, json=body, headers=auth)
        assert again.status_code == 200 and again.json()["status"] == "duplicate"
        bad_track = client.post(path, json={**body, "event_id": "api-2", "track_id": "nope"}, headers=auth)
        assert bad_track.status_code == 422
        bad_id = client.post(path, json={**body, "event_id": "has space"}, headers=auth)
        assert bad_id.status_code == 422
        unknown = client.post("/users/424242/listens", json={**body, "event_id": "api-3"}, headers=auth)
        assert unknown.status_code == 404


def test_prometheus_metrics_fixed_labels_and_cumulative_buckets(music_run, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(music_run["export"]))
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("INGEST_API_KEY", "ingest-key")
    uid = USER_OFFSET + int(np.flatnonzero(Recommender.load(music_run["export"]).trained_users)[0])
    with TestClient(api_module.app) as client:
        client.get(f"/recommendations/{uid}", params={"k": 3})
        client.post(f"/users/{uid}/listens", json={"event_id": "m1", "track_id": _track_id(2),
                                                   "timestamp": _recent_ts(music_run)},
                    headers={"Authorization": "Bearer ingest-key"})
        resp = client.get("/metrics")
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/plain")
        text = resp.text
    assert 'rtrec_recommendation_requests_total{source="personalized"} 1' in text
    assert 'rtrec_listen_events_total{outcome="accepted"} 1' in text
    assert "rtrec_model_loaded 1" in text
    assert str(uid) not in text and "TRK" not in text          # no user or track identifiers
    buckets = [int(line.rsplit(" ", 1)[1]) for line in text.splitlines()
               if line.startswith("rtrec_request_duration_seconds_bucket")]
    assert buckets == sorted(buckets) and buckets[-1] == 1


def test_forget_restores_artifact_features(music_run):
    rec = Recommender.load(music_run["export"])
    pos = int(np.flatnonzero(rec.trained_users)[0])
    before = (rec.history.genre_profile[pos].copy(), rec.history.activity[pos], rec.history.item_popularity.copy(),
              sorted(rec.history.seen(pos).tolist()))
    for n, item in enumerate([3, 5, 8]):
        _listen(rec, pos, item, f"forget-{n}", _recent_ts(music_run))
    assert rec.forget_listens(str(int(rec.user_ids[pos]))) == 3
    after = (rec.history.genre_profile[pos], rec.history.activity[pos], rec.history.item_popularity,
             sorted(rec.history.seen(pos).tolist()))
    np.testing.assert_allclose(after[0], before[0], atol=1e-6)
    assert after[1] == before[1]
    np.testing.assert_array_equal(after[2], before[2])
    assert after[3] == before[3]
    # the same event id can be applied again after deletion
    assert _listen(rec, pos, 3, "forget-0", _recent_ts(music_run)) == "accepted"
