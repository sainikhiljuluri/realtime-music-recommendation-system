"""Online listening features: validated, idempotent, incremental updates for the Music4All model.

Updated online (in memory, per process):
  genre_profile[user] and activity[user]   ranker features: genre affinity and log activity
  seen overlay[user]                       tracks just played are excluded from candidates at once
  item_popularity[item]                    ranker feature: log popularity

Frozen until the model is retrained:
  two-tower user and item vectors, so FAISS candidates come from the artifact snapshot
  popularity order used for candidate generation and cold start
  LightGBM weights

Every feature is a count (or a normalized count), so the result does not depend on the order
in which events arrive. Applying the same events offline with History.from_events gives the same
features (tests/test_online.py checks this).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from .ranking import History

MAX_CLOCK_SKEW_SECONDS = 300


class EventRejected(Exception):
    """A listen event failed validation. `status` is the HTTP status to return."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass
class OnlineFeatures:
    history: History
    cutoff_unix: int
    genre_counts: np.ndarray = field(default=None, repr=False)            # raw counts behind genre_profile
    seen_event_ids: dict = field(default_factory=dict, repr=False)        # user -> applied event ids
    applied_items: dict = field(default_factory=dict, repr=False)         # user -> items applied, for forget()
    applied: int = 0
    duplicates: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.genre_counts is None:
            # genre_profile is counts / row total, and the total equals activity. Rebuild the counts once.
            self.genre_counts = np.rint(self.history.genre_profile * self.history.activity[:, None]).astype(np.float32)

    def apply(self, user: int, item: int, event_id: str, ts: int, now: int) -> str:
        """Apply one listen event. Returns "accepted" or "duplicate". Raises EventRejected."""
        if ts < self.cutoff_unix:
            raise EventRejected(422, "event predates the history cutoff")
        if ts > now + MAX_CLOCK_SKEW_SECONDS:
            raise EventRejected(422, "timestamp is in the future")
        with self._lock:
            ids = self.seen_event_ids.setdefault(user, set())
            if event_id in ids:
                self.duplicates += 1
                return "duplicate"
            ids.add(event_id)
            h = self.history
            self.genre_counts[user, h.item_genre[item]] += 1
            h.genre_profile[user] = self.genre_counts[user] / self.genre_counts[user].sum()
            h.activity[user] += 1
            h.item_popularity[item] += 1
            h.extra_seen.setdefault(user, set()).add(int(item))
            self.applied_items.setdefault(user, []).append(int(item))
            self.applied += 1
            return "accepted"

    def forget(self, user: int) -> int:
        """Reverse every online update for one user (used by the admin delete). Returns events removed.

        Counts and profiles return to their artifact values, and the user's event ids and seen overlay
        are dropped. Other users' features change only through their own popularity contributions,
        which are removed here as well.
        """
        with self._lock:
            items = self.applied_items.pop(user, [])
            h = self.history
            for item in items:
                h.item_popularity[item] -= 1
                self.genre_counts[user, h.item_genre[item]] -= 1
            total = self.genre_counts[user].sum()
            h.genre_profile[user] = self.genre_counts[user] / total if total > 0 else 0.0
            h.activity[user] -= len(items)
            h.extra_seen.pop(user, None)
            self.seen_event_ids.pop(user, None)
            self.applied -= len(items)
            return len(items)
