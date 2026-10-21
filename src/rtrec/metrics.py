"""Ranking metrics for a single held-out positive per user."""
from __future__ import annotations

import math


def recall_at_k(ranked: list[int], target: int, k: int) -> float:
    """1.0 if the held-out item appears in the top k, else 0.0 (hit rate for one positive)."""
    return 1.0 if target in ranked[:k] else 0.0


def ndcg_at_k(ranked: list[int], target: int, k: int) -> float:
    """Normalized DCG with one relevant item: 1 / log2(rank + 1) when it is in the top k."""
    for position, item in enumerate(ranked[:k], start=1):
        if item == target:
            return 1.0 / math.log2(position + 1)
    return 0.0
