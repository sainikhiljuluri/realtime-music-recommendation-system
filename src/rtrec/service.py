"""Serving: load artifacts, retrieve with FAISS, rerank with LightGBM, cache, and fall back to popularity."""
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

from . import retrieval
from .cache import NullCache, RedisCache
from .artifacts import CONFIG_FILE, FAISS_FILE, HISTORY_FILE, RANKER_FILE, VECTORS_FILE
from .ranking import History, candidate_features, predict

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
    n_users: int
    n_items: int
    candidates: int
    version: str
    cache: Any = field(default_factory=NullCache)
    ttl: int = 300
    stats: Stats = field(default_factory=Stats)

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
        version = hashlib.sha1(config_text.encode()).hexdigest()[:8]
        return cls(
            index=faiss.read_index(str(root / FAISS_FILE)),
            booster=lgb.Booster(model_file=str(root / RANKER_FILE)),
            history=history,
            user_vecs=arrays["user_vecs"],
            n_users=history.genre_profile.shape[0],
            n_items=history.item_genre.shape[0],
            candidates=int(config["candidates"]),
            version=version,
            cache=cache if cache is not None else NullCache(),
            ttl=ttl,
        )

    def _popular_unseen(self, seen: np.ndarray, exclude: set[int], k: int) -> list[int]:
        order = np.argsort(-self.history.item_popularity, kind="stable")
        blocked = set(seen.tolist()) | exclude
        out: list[int] = []
        for item in order:
            if int(item) not in blocked:
                out.append(int(item))
                if len(out) == k:
                    break
        return out

    def _personalized(self, user: int, k: int) -> list[dict]:
        seen = self.history.seen(user)
        items, scores = retrieval.top_unseen(self.index, self.user_vecs[user], seen, self.candidates)
        ranked_scores = predict(self.booster, candidate_features(self.history, user, items, scores))
        order = np.argsort(-ranked_scores, kind="stable")
        chosen = [(int(items[j]), float(ranked_scores[j])) for j in order[:k]]
        if len(chosen) < k:  # fill from popularity when retrieval has too few unseen items
            extra = self._popular_unseen(seen, {i for i, _ in chosen}, k - len(chosen))
            chosen += [(item, 0.0) for item in extra]
        return [{"item_id": f"i{item}", "score": round(score, 6)} for item, score in chosen]

    def _cold_start(self, k: int) -> list[dict]:
        top = self._popular_unseen(np.empty(0, dtype=np.int64), set(), k)
        return [{"item_id": f"i{item}", "score": 0.0} for item in top]

    @staticmethod
    def parse_user(user_id: str) -> int | None:
        if len(user_id) > 1 and user_id[0] == "u" and user_id[1:].isdigit():
            return int(user_id[1:])
        return None

    def recommend(self, user_id: str, k: int = 10) -> dict:
        self.stats.requests += 1
        key = f"rec:{self.version}:{user_id}:{k}"
        cached = self.cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return {**cached, "cache_hit": True}

        user = self.parse_user(user_id)
        if user is None or user >= self.n_users:
            source, items = "cold_start", self._cold_start(k)
            self.stats.cold_starts += 1
        else:
            source, items = "personalized", self._personalized(user, k)
            self.stats.personalized += 1
        result = {"user_id": user_id, "source": source, "items": items}
        self.cache.set(key, result, self.ttl)
        return {**result, "cache_hit": False}
