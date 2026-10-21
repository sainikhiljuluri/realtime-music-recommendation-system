"""History features and LightGBM ranking over FAISS candidates."""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import retrieval

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
    # Online overlay: items a user played after the artifact was built. Empty for offline use.
    extra_seen: dict = field(default_factory=dict, repr=False)

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
        base = self.seen_indices[self.seen_indptr[user]:self.seen_indptr[user + 1]]
        extra = self.extra_seen.get(user)
        if not extra:
            return base
        return np.union1d(base, np.fromiter(extra, dtype=np.int64, count=len(extra)))

    def max_seen(self) -> int:
        return int(np.diff(self.seen_indptr).max()) if self.seen_indptr.size > 1 else 0


def candidate_features(history: History, user: int, items: np.ndarray, scores: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """Build the ranker feature matrix for one user's candidates."""
    n = len(items)
    genres = history.item_genre[items]
    artists = history.item_artist[items]
    features = np.column_stack([
        scores.astype(np.float32),
        ranks.astype(np.float32),
        np.log1p(history.item_popularity[items]),
        history.genre_profile[user, genres],
        history.artist_counts[user, artists],
        np.full(n, np.log1p(history.activity[user]), dtype=np.float32),
    ])
    return features.astype(np.float32)


def generate_candidates(index, user_vec: np.ndarray, seen: np.ndarray, popularity_order: np.ndarray,
                        item_vecs: np.ndarray, n_retrieval: int, n_popular: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hybrid candidate pool: FAISS top unseen items, plus popular unseen items not already included.

    Popular items get their dot-product score and a retrieval rank of n_retrieval + 1, so the
    ranker can use them. Training and serving both call this function.
    """
    items_r, scores_r = retrieval.top_unseen(index, user_vec, seen, n_retrieval)
    ranks_r = np.arange(1, len(items_r) + 1, dtype=np.float32)
    taken = set(seen.tolist()) | set(items_r.tolist())
    popular: list[int] = []
    if n_popular > 0:
        for item in popularity_order:
            item = int(item)
            if item not in taken:
                popular.append(item)
                taken.add(item)
                if len(popular) == n_popular:
                    break
    popular_arr = np.array(popular, dtype=np.int64)
    scores_p = (item_vecs[popular_arr] @ user_vec).astype(np.float32) if len(popular_arr) else np.empty(0, np.float32)
    ranks_p = np.full(len(popular_arr), n_retrieval + 1, dtype=np.float32)
    items = np.concatenate([items_r.astype(np.int64), popular_arr])
    scores = np.concatenate([scores_r.astype(np.float32), scores_p])
    ranks = np.concatenate([ranks_r, ranks_p])
    return items, scores, ranks


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
    # num_threads=1 is required. faiss and LightGBM each carry an OpenMP runtime. With more than one
    # LightGBM thread, the API process crashed on macOS (exit 139). See scripts/openmp_probe.py.
    return booster.predict(features, num_threads=1)
