"""Deterministic experiment assignment for online A/B tests.

A user's arm depends only on (experiment name, user id), so the same user sees the same
arm on every request and across services, without storing any assignment state.
This module is not an experiment result. It is the assignment step of an online test.
"""
from __future__ import annotations

import hashlib

BUCKETS = 10_000


def bucket(experiment: str, user_id: str) -> int:
    digest = hashlib.sha256(f"{experiment}:{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % BUCKETS


def assign(experiment: str, user_id: str, treatment_share: float = 0.5) -> str:
    """Return 'treatment' for the first treatment_share of buckets, otherwise 'control'."""
    if not 0.0 <= treatment_share <= 1.0:
        raise ValueError("treatment_share must be between 0 and 1")
    return "treatment" if bucket(experiment, user_id) < treatment_share * BUCKETS else "control"
