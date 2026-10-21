"""Synthetic implicit-feedback data with latent user taste and genre structure.

Every interaction is generated, so no real listening data or credentials are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Dataset:
    items: pd.DataFrame          # columns: item_idx, genre, artist
    interactions: pd.DataFrame   # columns: user_idx, item_idx, ts
    n_users: int
    n_items: int
    n_genres: int
    n_artists: int


def generate(
    n_users: int = 2000,
    n_items: int = 3000,
    n_genres: int = 12,
    n_artists: int = 400,
    latent_dim: int = 8,
    min_events: int = 15,
    max_events: int = 60,
    seed: int = 0,
) -> Dataset:
    """Sample users' listening events from a latent-factor preference model.

    Each user draws events in proportion to exp(user . item + log popularity),
    so users have consistent genre and artist preferences that a model can learn.
    """
    if min_events < 3:
        raise ValueError("min_events must be at least 3 (train, validation, test)")
    if max_events > n_items:
        raise ValueError("max_events cannot exceed n_items (events are sampled without replacement)")
    rng = np.random.default_rng(seed)
    genre = rng.integers(0, n_genres, size=n_items)
    artist = rng.integers(0, n_artists, size=n_items)
    centroids = rng.normal(size=(n_genres, latent_dim))
    item_latent = rng.normal(size=(n_items, latent_dim)) + centroids[genre]
    item_popularity = rng.gamma(shape=1.0, scale=1.0, size=n_items)
    user_latent = rng.normal(size=(n_users, latent_dim))

    rows_user, rows_item, rows_ts = [], [], []
    base_logits = np.log(item_popularity)
    for u in range(n_users):
        n_events = int(rng.integers(min_events, max_events + 1))
        logits = user_latent[u] @ item_latent.T + base_logits
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        chosen = rng.choice(n_items, size=n_events, replace=False, p=probs)
        rows_user.append(np.full(n_events, u))
        rows_item.append(chosen)
        rows_ts.append(np.arange(n_events))

    interactions = pd.DataFrame({
        "user_idx": np.concatenate(rows_user),
        "item_idx": np.concatenate(rows_item),
        "ts": np.concatenate(rows_ts),
    })
    items = pd.DataFrame({"item_idx": np.arange(n_items), "genre": genre, "artist": artist})
    return Dataset(items, interactions, n_users, n_items, n_genres, n_artists)


def leave_two_out(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each user's events into train, validation (second-to-last), and test (last).

    The two-tower model trains on train only. The ranker trains on validation positives.
    Test positives are never used for fitting anything.
    """
    ordered = interactions.sort_values(["user_idx", "ts"]).reset_index(drop=True)
    rank_from_end = ordered.groupby("user_idx").cumcount(ascending=False)
    test = ordered[rank_from_end == 0]
    validation = ordered[rank_from_end == 1]
    train = ordered[rank_from_end >= 2]
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
