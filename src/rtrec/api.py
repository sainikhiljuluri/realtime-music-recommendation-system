"""FastAPI service. Configure with environment variables:

ARTIFACT_DIR     directory produced by `python -m rtrec.pipeline` (default: artifacts)
REDIS_URL        optional, e.g. redis://localhost:6379/0. Without it, no cache is used.
CACHE_TTL        seconds to keep a cached response (default: 300)
PSEUDONYM_KEY    secret for pseudonymous ids in logs (see rtrec/privacy.py)

Endpoints:
  GET    /health                              liveness (process is up)
  GET    /ready                               readiness (model loaded; 503 otherwise)
  GET    /recommendations/{user_id}?k=10      recommendations
  DELETE /users/{user_id}/recommendations     purge that user's cached recommendations
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .cache import make_cache
from .privacy import pseudonymize
from .service import Recommender

log = logging.getLogger("rtrec.api")


class RecommendationItem(BaseModel):
    item_id: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: str
    source: str = Field(description="'personalized' or 'cold_start'")
    cache_hit: bool
    items: list[RecommendationItem]


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_dir = os.environ.get("ARTIFACT_DIR", "artifacts")
    cache = make_cache(os.environ.get("REDIS_URL"))
    ttl = int(os.environ.get("CACHE_TTL", "300"))
    app.state.recommender = None
    try:
        app.state.recommender = Recommender.load(artifact_dir, cache=cache, ttl=ttl)
        log.info("model loaded version=%s", app.state.recommender.version)
    except Exception:  # keep serving /health; /ready reports the failure
        log.exception("failed to load artifacts from %s", artifact_dir)
    yield


app = FastAPI(title="Real-time music recommendations", version="0.1.0", lifespan=lifespan)


def _loaded() -> Recommender:
    rec = app.state.recommender
    if rec is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return rec


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    rec = app.state.recommender
    if rec is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return {
        "status": "ready",
        "model_version": rec.version,
        "users": rec.n_users,
        "items": rec.n_items,
        "cache_available": rec.cache.available,
        "cache_errors": getattr(rec.cache, "errors", 0),
    }


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def recommendations(user_id: str, request: Request, k: int = Query(10, ge=1, le=50)) -> RecommendationResponse:
    rec = _loaded()
    started = time.perf_counter()
    result = rec.recommend(user_id, k)
    log.info("recommendation uid=%s k=%d source=%s cache_hit=%s latency_ms=%.2f",
             pseudonymize(user_id), k, result["source"], result["cache_hit"],
             (time.perf_counter() - started) * 1000)
    return RecommendationResponse(**result)


@app.delete("/users/{user_id}/recommendations")
def delete_user_recommendations(user_id: str) -> dict:
    rec = _loaded()
    removed = rec.purge_user(user_id)
    log.info("purged cached recommendations uid=%s removed=%d", pseudonymize(user_id), removed)
    return {"user_id": pseudonymize(user_id), "removed": removed}
