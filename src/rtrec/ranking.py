"""History features and LightGBM ranking over FAISS candidates."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "retrieval_score",
    "retrieval_rank",
    "log_item_popularity",
    "user_genre_affinity",
    "user_artist_count",
    "log_user_activity",
]


@dataclass
class History:
    """Per-user and per-item statistics computed from a set of listening events."""

    item_genre: np.ndarray        # (n_items,) int
    item_artist: np.ndarray       # (n_items,) int
    item_popularity: np.ndarray   # (n_items,) float32, event counts
    genre_profile: np.ndarray     # (n_users, n_genres) float32, sums to 1 per active user
    artist_counts: np.ndarray     # (n_users, n_artists) float32, event counts
    activity: np.ndarray          # (n_users,) float32, event counts
    seen_indptr: np.ndarray       # CSR row pointers over users into seen_indices
    seen_indices: np.ndarray      # item indices each user has already listened to

    @classmethod
    def from_events(cls, events: pd.DataFrame, items: pd.DataFrame, n_users: int, n_genres: int, n_artists: int) -> "History":
        n_items = len(items)
        item_genre = items["genre"].to_numpy(dtype=np.int64, copy=True)
        item_artist = items["artist"].to_numpy(dtype=np.int64, copy=True)
        users = events["user_idx"].to_numpy(dtype=np.int64)
        item_ids = events["item_idx"].to_numpy(dtype=np.int64)

        popularity = np.bincount(item_ids, minlength=n_items).astype(np.float32)
        activity = np.bincount(users, minlength=n_users).astype(np.float32)

        genre_counts = np.zeros((n_users, n_genres), dtype=np.float32)
        np.add.at(genre_counts, (users, item_genre[item_ids]), 1.0)
        totals = genre_counts.sum(axis=1, keepdims=True)
        genre_profile = np.divide(genre_counts, totals, out=np.zeros_like(genre_counts), where=totals > 0)

        artist_counts = np.zeros((n_users, n_artists), dtype=np.float32)
        np.add.at(artist_counts, (users, item_artist[item_ids]), 1.0)

        pairs = pd.DataFrame({"u": users, "i": item_ids}).drop_duplicates()
        pairs = pairs.sort_values(["u", "i"])
        counts = np.bincount(pairs["u"].to_numpy(), minlength=n_users)
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return cls(item_genre, item_artist, popularity, genre_profile, artist_counts, activity,
                   indptr, pairs["i"].to_numpy(dtype=np.int64))

    def seen(self, user: int) -> np.ndarray:
        return self.seen_indices[self.seen_indptr[user]:self.seen_indptr[user + 1]]

    def max_seen(self) -> int:
        return int(np.diff(self.seen_indptr).max()) if self.seen_indptr.size > 1 else 0


def candidate_features(history: History, user: int, items: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Build the ranker feature matrix for one user's candidates, in retrieval order."""
    n = len(items)
    ranks = np.arange(1, n + 1, dtype=np.float32)
    genres = history.item_genre[items]
    artists = history.item_artist[items]
    features = np.column_stack([
        scores.astype(np.float32),
        ranks,
        np.log1p(history.item_popularity[items]),
        history.genre_profile[user, genres],
        history.artist_counts[user, artists],
        np.full(n, np.log1p(history.activity[user]), dtype=np.float32),
    ])
    return features.astype(np.float32)


def train_ranker(features: np.ndarray, labels: np.ndarray, group_sizes: list[int], seed: int = 0) -> lgb.Booster:
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )
    ranker.fit(features, labels, group=group_sizes)
    return ranker.booster_


def predict(booster: lgb.Booster, features: np.ndarray) -> np.ndarray:
    return booster.predict(features)
