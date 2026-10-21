import numpy as np
import pandas as pd

from rtrec.ranking import FEATURE_NAMES, History, candidate_features, predict, train_ranker


def _history():
    items = pd.DataFrame({"item_idx": range(6), "genre": [0, 0, 1, 1, 2, 2], "artist": [0, 1, 1, 2, 3, 3]})
    events = pd.DataFrame({"user_idx": [0, 0, 0, 1], "item_idx": [0, 2, 2, 4]})
    return History.from_events(events, items, n_users=2, n_genres=3, n_artists=4), items


def test_history_counts_and_seen_sets():
    h, _ = _history()
    assert h.activity.tolist() == [3.0, 1.0]
    assert sorted(h.seen(0).tolist()) == [0, 2]
    assert h.seen(1).tolist() == [4]
    np.testing.assert_allclose(h.genre_profile[0], [1 / 3, 2 / 3, 0.0], atol=1e-6)


def test_candidate_features_shape_and_ranks():
    h, _ = _history()
    feats = candidate_features(h, 0, np.array([1, 3, 5]), np.array([0.9, 0.5, 0.1], dtype=np.float32))
    assert feats.shape == (3, len(FEATURE_NAMES))
    assert feats[:, 1].tolist() == [1.0, 2.0, 3.0]


def test_ranker_learns_to_put_positive_first():
    rng = np.random.default_rng(0)
    groups, feats, labels = [], [], []
    for _ in range(60):
        x = rng.normal(size=(20, len(FEATURE_NAMES))).astype(np.float32)
        y = np.zeros(20, dtype=np.int64)
        y[0] = 1
        x[0, 0] += 3.0  # the positive has a strong retrieval score
        feats.append(x)
        labels.append(y)
        groups.append(20)
    booster = train_ranker(np.vstack(feats), np.concatenate(labels), groups, seed=0)
    probe = feats[0]
    assert np.argmax(predict(booster, probe)) == 0
