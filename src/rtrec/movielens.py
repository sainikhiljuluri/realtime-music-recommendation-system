"""Loader for MovieLens 25M (GroupLens, https://grouplens.org/datasets/movielens/25m/).

The dataset is not redistributed in this repository. Download ml-25m.zip from GroupLens,
extract it, and point --dataset at the extracted directory. MovieLens has no artist
metadata, so the artist feature is a single constant bucket in this benchmark.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import Dataset


def load(root: str | Path, min_rating: float = 4.0, min_events: int = 3) -> Dataset:
    """Keep positive ratings (rating >= min_rating) as implicit feedback.

    Users with fewer than min_events positives are dropped, because leave-two-out needs
    at least three events per user. Ids are re-indexed to dense integers.
    """
    root = Path(root)
    ratings = pd.read_csv(root / "ratings.csv", usecols=["userId", "movieId", "rating", "timestamp"])
    ratings = ratings[ratings["rating"] >= min_rating]
    counts = ratings.groupby("userId").size()
    keep_users = counts[counts >= min_events].index
    ratings = ratings[ratings["userId"].isin(keep_users)]

    user_codes, user_levels = pd.factorize(ratings["userId"])
    movie_codes, movie_levels = pd.factorize(ratings["movieId"])

    movies = pd.read_csv(root / "movies.csv", usecols=["movieId", "genres"])
    movies = movies.set_index("movieId").reindex(movie_levels)
    primary = movies["genres"].fillna("(no genres listed)").str.split("|").str[0].to_numpy()
    genre_levels, genre_codes = np.unique(primary, return_inverse=True)

    items = pd.DataFrame({
        "item_idx": np.arange(len(movie_levels)),
        "genre": genre_codes.astype(np.int64),
        "artist": np.zeros(len(movie_levels), dtype=np.int64),
    })
    interactions = pd.DataFrame({
        "user_idx": user_codes.astype(np.int64),
        "item_idx": movie_codes.astype(np.int64),
        "ts": ratings["timestamp"].to_numpy(dtype=np.int64),
    })
    return Dataset(items, interactions, n_users=len(user_levels), n_items=len(movie_levels),
                   n_genres=len(genre_levels), n_artists=1)
