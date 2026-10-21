"""FastAPI service. Configure with environment variables:

ARTIFACT_DIR  directory produced by `python -m rtrec.pipeline` (default: artifacts)
REDIS_URL     optional, e.g. redis://localhost:6379/0. Without it, no cache is used.
CACHE_TTL     seconds to keep a cached response (default: 300)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .cache import make_cache
from .service import Recommender


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
    app.state.recommender = Recommender.load(artifact_dir, cache=cache, ttl=ttl)
    yield


app = FastAPI(title="Real-time music recommendations", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    rec: Recommender = app.state.recommender
    cache = rec.cache
    return {
        "status": "ok",
        "model_version": rec.version,
        "users": rec.n_users,
        "items": rec.n_items,
        "cache_available": cache.available,
        "cache_errors": getattr(cache, "errors", 0),
    }


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def recommendations(user_id: str, k: int = Query(10, ge=1, le=50)) -> RecommendationResponse:
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id is required")
    result = app.state.recommender.recommend(user_id, k)
    return RecommendationResponse(**result)
