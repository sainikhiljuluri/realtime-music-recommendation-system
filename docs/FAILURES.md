# Failure handling

What the service does when a dependency fails, and how it recovers. Each behavior is tested unless marked otherwise.

| Failure | Behavior | Test |
|---|---|---|
| Redis is down at startup | Starts with no cache. `/ready` shows `cache_available: false`. | `test_null_cache_when_no_url_or_unreachable` |
| Redis fails during a request | The request continues without the cache. The error is counted in `cache_errors`. | `test_api_survives_redis_outage` |
| Artifacts are missing or unreadable at startup | `/health` returns 200. `/ready` and `/recommendations` return 503 with "model is not loaded". | `test_missing_artifacts_keep_health_up_but_not_ready` |
| Artifacts change while running | Not handled. The process keeps the model it loaded. Restart to load a new version. | Not tested (not implemented) |
| Request for an unknown user | Served with the popularity fallback (`source: cold_start`). Not an error. | `test_unknown_user_gets_cold_start_popularity` |
| Too few unseen candidates | Filled from popularity order. | `test_known_user_gets_personalized_k_items` |
| Invalid `k` | 422 response from validation. | `test_k_out_of_range_is_rejected` |
| Cache purge fails | Counted; returns 0 removed. | `test_redis_errors_are_counted_not_raised` |
| Music artifacts are incomplete or inconsistent (for example, the catalog size differs from the history) | Load fails with the reason. `/ready` returns 503 with `reason`. `/health` stays 200. | `test_readiness_fails_with_reason_when_catalog_is_missing`, `test_mismatched_catalog_is_rejected` |
| Model self-check fails at startup (a cold-start or personalized request does not return a full list) | Treated as a load failure. The API does not report ready, and `/ready` gives the exception type. | `test_self_check_failure_keeps_api_not_ready` |
| Unknown Music4All user, non-numeric id, or user with no trained vector | Served as cold start (catalog popularity). Not an error. | `test_unknown_user_gets_catalog_popularity`, `test_non_numeric_or_oversized_user_ids_are_cold_start`, `test_user_without_trained_vector_is_cold_start` |
| API instance is killed (SIGKILL) under load (Compose) | nginx retries on the other instance. Measured: 0 client-visible errors in 30,000 requests. | Manual, `docs/results/serving-music4all.json` |
| API instance stops gracefully (SIGTERM, Compose) | uvicorn drains in-flight requests, closes Redis connections, and exits 0 in about 1 s. | `test_shutdown_releases_cache_connections` (Redis close) and manual Compose stop |
| Listen event is malformed, duplicated, stale, from an unknown user or track, or in the future | 422, 404, or 200 `duplicate` (no state change). Each outcome is counted. | `test_listen_rejects_unknown_track_user_and_stale_timestamps`, `test_duplicate_event_id_changes_state_once`, `test_api_listen_auth_and_status_codes` |
| Redis is down during a listen | The in-memory features still update. The cache generation bump fails and is counted. Cached entries for that user may survive until TTL. | `bump_generation` error path (counted, not raised) |
| Redis is down during a recommendation | The generation read fails, so the request skips the cache and computes. Measured at 1,368 req/s with 0% hits (two instances). | `test_api_survives_redis_outage`; manual run `uncached_2` |
| Instance is killed while it holds a user's listen features | Users move to the other instance (consistent hashing). Their recent listens are gone there. Measured: no client errors; features revert to the artifact until the user listens again. | Manual, `docs/results/serving-v2.json` (kill runs). Known limitation. |
| Admin delete for a user | Cache entries are purged and the user's listen updates are reversed in memory. Features return to the artifact values. | `test_forget_restores_artifact_features` |
| Container image is built but cannot import (missing `libgomp1`) | Found by starting the container. Fixed in the Dockerfile. CI runs a container smoke test. | CI job `docker` (smoke test) |

**Health and readiness.** `/health` is liveness: it answers 200 whenever the process is running. `/ready` answers 200 only when the model is loaded, so a load balancer can stop sending traffic to a process that has no model.

**Recovery.** Restart the process to reload artifacts. redis-py reconnects on a later call once Redis is back. This is not tested separately.

**Not implemented.** Circuit breakers, retries with backoff, hot reload of artifacts, and graceful draining during deploys.

## OpenMP runtime conflict (macOS wheels)

**Symptom.** Exit code 139 (segmentation fault), or `OMP: Error #179: pthread_mutex_init failed`.

**Cause, as far as it is known.** The pip wheels for faiss, LightGBM, and (in the offline pipeline) torch and scikit-learn each ship their own `libomp.dylib`. Loading two of them into one process and running a multi-threaded LightGBM prediction crashed in every run. faiss and LightGBM alone are enough. Torch is not loaded by the API process.

**Measurements** (`scripts/openmp_probe.py`, 5 runs per cell, `OMP_NUM_THREADS=1`):

| Libraries loaded | LightGBM prediction threads | `KMP_DUPLICATE_LIB_OK=TRUE` | Flag unset |
|---|---|---|---|
| faiss + LightGBM (the API) | 4 | 5/5 crashed | 5/5 crashed |
| faiss + LightGBM (the API) | 1 | 0/5 | 0/5 |
| torch + faiss + LightGBM + scikit-learn | 4 | 5/5 crashed | 5/5 crashed |
| torch + faiss + LightGBM + scikit-learn | 1 | 0/5 | 0/5 |

Earlier, LightGBM's default thread setting crashed twice and then passed 12 of 12 reruns. I could not explain that, so I do not rely on the default setting.

**What the code does now.** Serving and evaluation call `booster.predict(..., num_threads=1)` (`rtrec/ranking.py`). faiss and torch are already single-threaded (`faiss.omp_set_num_threads(1)`, `torch.set_num_threads(1)`). The workaround flag in `rtrec/__init__.py` is still set. The single-thread probe also passed without it, but I have not run the full test suite or the API without the flag, so I have not shown it is unnecessary.

**Not done.** A durable fix is one OpenMP runtime shared by all the libraries, which means building them from source. I have not tried that. The probe does not test Linux. The Linux container runs the API under the container smoke test in CI, not under this matrix.

## Bind mounts under Colima (operational)

**Symptom.** Containers failed to start with `OSError: [Errno 35] Resource deadlock avoided` (reading `config.json`), and nginx failed with `pread() ... (35: Resource deadlock would occur)`. A later read of `ranker.txt` returned `I/O error`. The host read the same files without error.

**Cause.** Colima mounts host directories into its VM through virtiofs. Reads from containers through that mount failed intermittently, and after a fault they kept failing until Colima restarted.

**Fix.** The artifacts live in a named Docker volume, seeded from a tar stream by `scripts/seed_artifacts.sh`. The volume was checked with matching checksums for `ranker.txt` and `history.npz`. The nginx configuration is copied into a small image (`deploy/lb.Dockerfile`).

**Not verified.** I did not reproduce the fault in a controlled test, and I have not checked whether Docker Desktop's file sharing behaves the same way.
