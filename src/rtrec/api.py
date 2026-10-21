"""FastAPI service. Configure with environment variables:

ARTIFACT_DIR     directory produced by the pipeline (default: artifacts). Either the synthetic or
                 MovieLens artifacts, or the Music4All artifacts. config.json says which.
REDIS_URL        optional, e.g. redis://localhost:6379/0. Without it, no cache is used.
CACHE_TTL        seconds to keep a cached response (default: 300)
PSEUDONYM_KEY    secret for pseudonymous ids in logs (see rtrec/privacy.py)
ADMIN_API_KEY    bearer token required for DELETE. If unset, DELETE returns 503 (disabled).
INGEST_API_KEY   bearer token required for POST /users/{id}/listens (the event producer). Unset: 503.

Endpoints:
  GET    /health                              liveness: the process is up
  GET    /ready                               readiness: model loaded and self-checked; 503 otherwise
  GET    /metrics                             Prometheus text format (fixed labels, no user ids)
  GET    /stats                               the same counters as JSON, with latency percentiles
  GET    /recommendations/{user_id}?k=10      recommendations
  POST   /users/{user_id}/listens             record a listen event (bearer token); 202 new, 200 duplicate
  DELETE /users/{user_id}/recommendations     purge that user's cached recommendations and listen updates (bearer token)
"""
from __future__ import annotations

import hmac
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from .cache import make_cache
from .online import EventRejected
from .privacy import pseudonymize
from .service import Recommender
from .telemetry import Telemetry

log = logging.getLogger("rtrec.api")


class RecommendationItem(BaseModel):
    item_id: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: str
    source: str = Field(description="'personalized' or 'cold_start'")
    cache_hit: bool
    items: list[RecommendationItem]


class ListenEvent(BaseModel):
    event_id: str = Field(min_length=1, max_length=64, description="client-chosen idempotency key")
    track_id: str = Field(min_length=1, max_length=64)
    timestamp: int = Field(ge=0, description="unix seconds, UTC")

    @field_validator("event_id")
    @classmethod
    def _event_id_chars(cls, value: str) -> str:
        if not all(ch.isalnum() or ch in "-_" for ch in value):
            raise ValueError("event_id may contain only letters, digits, '-' and '_'")
        return value


class ListenResponse(BaseModel):
    user_id: str
    status: str = Field(description="'accepted' or 'duplicate'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_dir = os.environ.get("ARTIFACT_DIR", "artifacts")
    cache = make_cache(os.environ.get("REDIS_URL"))
    ttl = int(os.environ.get("CACHE_TTL", "300"))
    app.state.cache = cache
    app.state.telemetry = Telemetry()
    app.state.load_error = None
    app.state.recommender = None
    app.state.load_error = None
    try:
        rec = Recommender.load(artifact_dir, cache=cache, ttl=ttl)
        rec.self_check()  # warm-up: exercises FAISS and LightGBM before readiness is reported
        app.state.recommender = rec
        log.info("model loaded family=%s version=%s", rec.family, rec.version)
    except Exception as exc:  # keep serving /health; /ready reports the failure
        app.state.load_error = type(exc).__name__
        log.exception("failed to load artifacts from %s", artifact_dir)
    yield
    # Shutdown: uvicorn has already drained in-flight requests. Release Redis connections.
    close = getattr(cache, "close", None)
    if callable(close):
        close()
    log.info("shutdown complete")


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
def ready():
    rec = app.state.recommender
    if rec is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": app.state.load_error or "not_loaded"})
    trained = int(rec.trained_users.sum()) if rec.trained_users is not None else rec.n_users
    return {
        "status": "ready",
        "model_family": rec.family,
        "model_version": rec.version,
        "users": rec.n_users,
        "trained_users": trained,
        "items": rec.n_items,
        "selected_lambda": rec.selected_lambda,
        "cache_available": rec.cache.available,
        "cache_errors": getattr(rec.cache, "errors", 0),
    }


@app.get("/stats")
def stats() -> dict:
    snap = app.state.telemetry.snapshot()
    rec = app.state.recommender
    snap["model_loaded"] = rec is not None
    snap["cache_errors"] = getattr(app.state.cache, "errors", 0)
    return snap


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    rec = app.state.recommender
    gauges = {
        "model_loaded": 1 if rec is not None else 0,
        "cache_available": 1 if app.state.cache.available else 0,
        "cache_errors_total": getattr(app.state.cache, "errors", 0),
        "online_events_applied": rec.online.applied if rec is not None and rec.online is not None else 0,
    }
    return app.state.telemetry.prometheus(gauges)


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def recommendations(user_id: str, request: Request, k: int = Query(10, ge=1, le=50)) -> RecommendationResponse:
    try:
        rec = _loaded()
    except HTTPException:
        app.state.telemetry.error(503)
        raise
    started = time.perf_counter()
    result = rec.recommend(user_id, k)
    latency_ms = (time.perf_counter() - started) * 1000
    app.state.telemetry.observe(result["source"], result["cache_hit"], latency_ms)
    log.info("recommendation uid=%s k=%d source=%s cache_hit=%s latency_ms=%.2f",
             pseudonymize(user_id), k, result["source"], result["cache_hit"], latency_ms)
    return RecommendationResponse(**result)


_bearer = HTTPBearer(auto_error=False)


def require_admin(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    """Bearer-token check for destructive operations. Compares in constant time."""
    expected = os.environ.get("ADMIN_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="deletion is disabled: ADMIN_API_KEY is not set")
    if credentials is None:
        raise HTTPException(status_code=401, detail="missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    if not hmac.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="invalid token")


def _bearer_check(credentials: HTTPAuthorizationCredentials | None, env_name: str, purpose: str) -> None:
    expected = os.environ.get(env_name)
    if not expected:
        raise HTTPException(status_code=503, detail=f"{purpose} is disabled: {env_name} is not set")
    if credentials is None:
        raise HTTPException(status_code=401, detail="missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})
    if not hmac.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="invalid token")


def require_ingest(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    _bearer_check(credentials, "INGEST_API_KEY", "listen ingestion")


@app.post("/users/{user_id}/listens", response_model=ListenResponse, dependencies=[Depends(require_ingest)])
def record_listen(user_id: str, event: ListenEvent, response: Response) -> ListenResponse:
    rec = _loaded()
    try:
        status = rec.ingest(user_id, event.track_id, event.event_id, event.timestamp, int(time.time()))
    except EventRejected as exc:
        app.state.telemetry.event("rejected")
        app.state.telemetry.error(exc.status)
        raise HTTPException(status_code=exc.status, detail=exc.reason)
    app.state.telemetry.event(status)
    response.status_code = 202 if status == "accepted" else 200
    log.info("listen uid=%s status=%s", pseudonymize(user_id), status)
    return ListenResponse(user_id=pseudonymize(user_id), status=status)


@app.delete("/users/{user_id}/recommendations", dependencies=[Depends(require_admin)])
def delete_user_recommendations(user_id: str) -> dict:
    rec = _loaded()
    removed = rec.purge_user(user_id)
    listens = rec.forget_listens(user_id)
    log.info("purged uid=%s cached=%d listens=%d", pseudonymize(user_id), removed, listens)
    return {"user_id": pseudonymize(user_id), "removed": removed, "listens_forgotten": listens}
