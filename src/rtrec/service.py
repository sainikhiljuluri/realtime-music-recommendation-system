"""Serving: load artifacts, retrieve with FAISS, rerank with LightGBM, cache, and fall back to popularity.

Two artifact families are supported, and config.json says which one is present:

- "generic": synthetic or MovieLens artifacts from rtrec.pipeline. Users are "u<index>" and
  items are "i<index>".
- "music4all": artifacts from rtrec.music_pipeline.export_artifacts. Users are Music4All user ids
  and items are Music4All track ids. Served lists apply the diversity penalty chosen on validation.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import lightgbm as lgb
import numpy as np

from .artifacts import CATALOG_FILE, CONFIG_FILE, FAISS_FILE, HISTORY_FILE, RANKER_FILE, VECTORS_FILE
from .cache import NullCache
from .hybrid import score_pool, select_top_k
from .online import EventRejected, OnlineFeatures
from .ranking import History

log = logging.getLogger(__name__)


@dataclass
class Stats:
    requests: int = 0
    cache_hits: int = 0
    cold_starts: int = 0
    personalized: int = 0


@dataclass
class Recommender:
    index: faiss.Index
    booster: lgb.Booster
    history: History
    user_vecs: np.ndarray
    item_vecs: np.ndarray
    n_users: int
    n_items: int
    candidates: int
    popular_candidates: int
    version: str
    cache: Any = field(default_factory=NullCache)
    ttl: int = 300
    stats: Stats = field(default_factory=Stats)
    popularity_order: np.ndarray = field(default=None, repr=False)
    family: str = "generic"
    selected_lambda: float = 0.0
    user_ids: np.ndarray | None = field(default=None, repr=False)      # music4all: sorted original ids
    item_ids: np.ndarray | None = field(default=None, repr=False)      # music4all: track id per dense item index
    trained_users: np.ndarray | None = field(default=None, repr=False)  # music4all: False = cold start
    item_index: dict | None = field(default=None, repr=False)          # music4all: track id -> item index
    online: OnlineFeatures | None = field(default=None, repr=False)    # music4all: incremental listen features

    def __post_init__(self) -> None:
        if self.popularity_order is None:
            self.popularity_order = np.argsort(-self.history.item_popularity, kind="stable")
        if self.item_ids is not None and len(self.item_ids) != self.n_items:
            raise ValueError("catalog size does not match the history artifact")
        if self.user_ids is not None and len(self.user_ids) != self.n_users:
            raise ValueError("user list size does not match the history artifact")

    @classmethod
    def load(cls, artifact_dir: str | Path, cache: Any | None = None, ttl: int = 300) -> "Recommender":
        root = Path(artifact_dir)
        config_text = (root / CONFIG_FILE).read_text()
        config = json.loads(config_text)
        arrays = np.load(root / VECTORS_FILE)
        hist = np.load(root / HISTORY_FILE)
        history = History(
            item_genre=hist["item_genre"],
            item_artist=hist["item_artist"],
            item_popularity=hist["item_popularity"],
            genre_profile=hist["genre_profile"],
            artist_counts=hist["artist_counts"],
            activity=hist["activity"],
            seen_indptr=hist["seen_indptr"],
            seen_indices=hist["seen_indices"],
        )
        family = config.get("model_family", "generic")
        extra: dict[str, Any] = {}
        if family == "music4all":
            catalog = np.load(root / CATALOG_FILE)
            item_ids = catalog["item_ids"].astype(str)
            extra = {
                "user_ids": catalog["user_ids"].astype(np.int64),
                "item_ids": item_ids,
                "trained_users": catalog["trained_users"].astype(bool),
                "selected_lambda": float(config["selected_lambda"]),
                "item_index": {str(track): i for i, track in enumerate(item_ids)},
                "online": OnlineFeatures(history=history, cutoff_unix=int(config["history_cutoff_unix"])),
            }
        elif family != "generic":
            raise ValueError(f"unknown model_family: {family}")
        version = hashlib.sha1(config_text.encode()).hexdigest()[:8]
        return cls(
            index=faiss.read_index(str(root / FAISS_FILE)),
            booster=lgb.Booster(model_file=str(root / RANKER_FILE)),
            history=history,
            user_vecs=arrays["user_vecs"],
            item_vecs=arrays["item_vecs"],
            n_users=history.genre_profile.shape[0],
            n_items=history.item_genre.shape[0],
            candidates=int(config["candidates"]),
            popular_candidates=int(config.get("popular_candidates", 0)),
            version=version,
            cache=cache if cache is not None else NullCache(),
            ttl=ttl,
            family=family,
            **extra,
        )

    def self_check(self) -> None:
        """Score one cold-start request and, when a trained user exists, one personalized request.

        Runs without the cache. Raises if the model cannot produce a full list of catalog items.
        Load it before readiness is reported, so the first real request does not pay for it.
        """
        cold = self._compute("__self_check_cold__", 3)
        if cold["source"] != "cold_start" or len(cold["items"]) != 3:
            raise RuntimeError("cold-start path failed self-check")
        probe = self._probe_user_id()
        if probe is not None:
            warm = self._compute(probe, 3)
            if warm["source"] != "personalized" or len(warm["items"]) != 3:
                raise RuntimeError("personalized path failed self-check")

    def _probe_user_id(self) -> str | None:
        if self.user_ids is None:
            return "u0" if self.n_users > 0 else None
        trained = np.flatnonzero(self.trained_users)
        return str(int(self.user_ids[trained[0]])) if trained.size else None

    def _user_index(self, user_id: str) -> int | None:
        """Dense user index, or None when the user is unknown or has no trained vector."""
        if self.user_ids is None:
            user = self.parse_user(user_id)
            return user if user is not None and user < self.n_users else None
        if not user_id.isdigit() or len(user_id) > 18:
            return None
        value = int(user_id)
        pos = int(np.searchsorted(self.user_ids, value))
        if pos < len(self.user_ids) and int(self.user_ids[pos]) == value and self.trained_users[pos]:
            return pos
        return None

    def _cache_user(self, user_id: str) -> str:
        """Canonical user key for the cache, so "004" and "4" share entries and invalidation."""
        if self.user_ids is not None and user_id.isdigit() and len(user_id) <= 18:
            return str(int(user_id))
        return user_id

    @property
    def ingestion_supported(self) -> bool:
        return self.online is not None

    def ingest(self, user_id: str, track_id: str, event_id: str, ts: int, now: int) -> str:
        """Apply one listen event. Returns "accepted" or "duplicate". Raises EventRejected on invalid input.

        Only known, trained users and known tracks are accepted. A new user needs a retrain.
        """
        if self.online is None:
            raise EventRejected(501, "listen ingestion needs music4all artifacts")
        user = self._user_index(user_id)
        if user is None:
            raise EventRejected(404, "unknown or untrained user")
        item = self.item_index.get(track_id)
        if item is None:
            raise EventRejected(422, "unknown track")
        status = self.online.apply(user, item, event_id, ts, now)
        if status == "accepted":
            self.cache.bump_generation(self._cache_user(user_id))
        return status

    def _item_name(self, item: int) -> str:
        return str(self.item_ids[item]) if self.item_ids is not None else f"i{item}"

    def _popular_unseen(self, seen: np.ndarray, exclude: set[int], k: int) -> list[int]:
        blocked = set(seen.tolist()) | exclude
        out: list[int] = []
        for item in self.popularity_order:
            if int(item) not in blocked:
                out.append(int(item))
                if len(out) == k:
                    break
        return out

    def _personalized(self, user: int, k: int) -> list[dict]:
        seen = self.history.seen(user)
        items, rank_scores = score_pool(self.index, user, self.user_vecs[user], self.history,
                                        self.popularity_order, self.item_vecs, self.booster,
                                        self.candidates, self.popular_candidates)
        chosen = select_top_k(items, rank_scores, self.history, k, self.selected_lambda)
        if len(chosen) < k:  # fill from popularity when the unseen candidate pool is too small
            extra = self._popular_unseen(seen, {i for i, _ in chosen}, k - len(chosen))
            chosen += [(item, 0.0) for item in extra]
        return [{"item_id": self._item_name(item), "score": round(score, 6)} for item, score in chosen]

    def _cold_start(self, k: int) -> list[dict]:
        top = self._popular_unseen(np.empty(0, dtype=np.int64), set(), k)
        return [{"item_id": self._item_name(item), "score": 0.0} for item in top]

    @staticmethod
    def parse_user(user_id: str) -> int | None:
        if len(user_id) > 1 and user_id[0] == "u" and user_id[1:].isdigit():
            return int(user_id[1:])
        return None

    def _compute(self, user_id: str, k: int) -> dict:
        user = self._user_index(user_id)
        if user is None:
            return {"user_id": user_id, "source": "cold_start", "items": self._cold_start(k)}
        return {"user_id": user_id, "source": "personalized", "items": self._personalized(user, k)}

    def recommend(self, user_id: str, k: int = 10) -> dict:
        self.stats.requests += 1
        cache_user = self._cache_user(user_id)
        # The key carries the user's generation. A listen event bumps it, so older entries are never read.
        # When Redis is unavailable the generation is None and the cache is skipped for this request.
        generation = self.cache.generation(cache_user)
        key = None if generation is None else f"rec:{self.version}:{cache_user}:{k}:g{generation}"
        if key is not None:
            cached = self.cache.get(key)
            if cached is not None:
                self.stats.cache_hits += 1
                return {**cached, "cache_hit": True}

        result = self._compute(user_id, k)
        if result["source"] == "cold_start":
            self.stats.cold_starts += 1
        else:
            self.stats.personalized += 1
        if key is not None:
            self.cache.set(key, result, self.ttl)
        return {**result, "cache_hit": False}

    def forget_listens(self, user_id: str) -> int:
        """Reverse this user's online listen updates. Returns the number of events removed."""
        user = self._user_index(user_id)
        if self.online is None or user is None:
            return 0
        return self.online.forget(user)

    def purge_user(self, user_id: str) -> int:
        """Delete cached recommendations for one user (all k values and model versions)."""
        return self.cache.delete_matching(f"rec:*:{user_id}:*")
