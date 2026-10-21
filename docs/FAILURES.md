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

**Health and readiness.** `/health` is liveness: it answers 200 whenever the process is running. `/ready` answers 200 only when the model is loaded, so a load balancer can stop sending traffic to a process that has no model.

**Recovery.** Restart the process to reload artifacts. redis-py reconnects on a later call once Redis is back. This is not tested separately.

**Not implemented.** Circuit breakers, retries with backoff, hot reload of artifacts, and graceful draining during deploys.
