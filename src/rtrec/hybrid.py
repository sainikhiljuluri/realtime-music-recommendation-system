"""Hybrid candidate pool, LightGBM scoring, and diversity selection for one user.

Offline evaluation (rtrec.music_pipeline) and serving (rtrec.service) both call these
functions. A list produced offline is therefore the list the API returns for the same
artifacts and user, and the two cannot drift apart.
"""
from __future__ import annotations

import numpy as np

from .evaluation import diversity_select
from .ranking import History, candidate_features, generate_candidates, predict


def score_pool(index, user: int, user_vec: np.ndarray, history: History, popularity_order: np.ndarray,
               item_vecs: np.ndarray, booster, n_retrieval: int, n_popular: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (candidate item ids, LightGBM scores) for one user. The seen set comes from `history`."""
    items, scores, ranks = generate_candidates(index, user_vec, history.seen(user), popularity_order,
                                               item_vecs, n_retrieval, n_popular)
    if len(items) == 0:
        return items, np.empty(0, dtype=np.float32)
    return items, predict(booster, candidate_features(history, user, items, scores, ranks))


def select_top_k(items: np.ndarray, rank_scores: np.ndarray, history: History, k: int,
                 lam: float) -> list[tuple[int, float]]:
    """Top-k (item id, ranker score) after the diversity penalty lam. lam=0 is plain score order."""
    positions = diversity_select(items, rank_scores, history.item_genre, k, lam)
    return [(int(items[j]), float(rank_scores[j])) for j in positions]
