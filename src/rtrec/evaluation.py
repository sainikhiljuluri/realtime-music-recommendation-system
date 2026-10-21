"""Per-user metrics, diversity and coverage, and bootstrap confidence intervals.

Offline results are reported with bootstrap intervals over users. They are offline
estimates and do not measure real users' responses.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap confidence interval over users."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) per user, with a paired bootstrap interval. Same users in both arms."""
    return bootstrap_ci(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), n_resamples, alpha, seed)


def genre_diversity(ranked_items: list[int], item_genre: np.ndarray) -> float:
    """Distinct genres in the list divided by list length."""
    if not ranked_items:
        return 0.0
    return len({int(item_genre[i]) for i in ranked_items}) / len(ranked_items)


def coverage(recommended: list[list[int]], n_items: int) -> float:
    """Share of the catalog that appears in at least one user's list."""
    distinct = set()
    for ranked in recommended:
        distinct.update(int(i) for i in ranked)
    return len(distinct) / n_items
