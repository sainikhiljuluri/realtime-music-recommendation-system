# Privacy design

This document describes what the service stores and logs, how identifiers are handled, how long data is kept, and what deletion does. It describes the design and what is implemented. It does not describe a production deployment.

## What the service handles

| Data | Where it lives | Notes |
|---|---|---|
| Request user id (`u<index>` or any string) | Request path only | Not written to logs. Used as a cache key and a model lookup. |
| Pseudonymous user id | Logs | HMAC-SHA256 of the user id under `PSEUDONYM_KEY`, truncated to 16 hex characters (`rtrec/privacy.py`). |
| Recommendation responses | Redis cache, if configured | Stored as JSON for `CACHE_TTL` seconds (default 300). |
| Listening history (training) | Model artifacts (`history.npz`, `vectors.npz`) | Per-user aggregates and seen-item lists, built from training data. |
| Latency, k, source, cache hit | Logs | No item lists are logged. |

## Pseudonymous identifiers

- Logs record `uid=<pseudonym>`. A test asserts that the raw id does not appear in logs (`tests/test_api.py::test_logs_contain_pseudonyms_not_raw_ids`).
- With `PSEUDONYM_KEY` set, the same user maps to the same pseudonym across restarts. This allows log analysis without storing raw ids.
- Without the key, each process generates a random key. Pseudonyms then cannot be linked across restarts. The service logs a warning in that case.
- The key is a secret. Anyone with the key can recompute pseudonyms for guessed ids, so it must be stored like other credentials.

## Retention

| Data | Policy in this design | Enforced by code? |
|---|---|---|
| Redis cache entries | Expire after `CACHE_TTL` (default 300 s) | Yes, Redis `EX` on write |
| Application logs | Keep for 30 days, then delete (log storage must enforce this) | No. This is a policy for the log platform. |
| Model artifacts and training aggregates | Retrained periodically; old versions removed after rollout | No. Retention is a manual operational step. |
| Synthetic or public training data | Not personal data | Not applicable |

## Deletion

`DELETE /users/{user_id}/recommendations` removes all cached recommendations for that user, for every `k` and model version (`Recommender.purge_user`). It returns the pseudonym and the number of entries removed.

**Access control.** The endpoint requires `Authorization: Bearer <ADMIN_API_KEY>`. The key is compared in constant time. A missing token returns 401, a wrong token 403, and if `ADMIN_API_KEY` is not set the endpoint is disabled and returns 503. Tests cover all four cases (`tests/test_security.py`). Users cannot yet delete their own data: that needs per-user tokens, which are not implemented.

What deletion does **not** do:
- It does not remove the user's rows from training data or model artifacts. Those contain per-user aggregates, so removing a user requires retraining without that user. This is documented, not implemented.
- It does not remove existing log lines. Log retention handles those.
- It is admin-only. Self-service deletion by the user needs per-user authentication, which is not implemented.

## Data minimization

- The service does not accept profile or contact data.
- Responses contain item ids and scores only.
- The MovieLens benchmark uses public ratings data under GroupLens terms. Its user ids are anonymous in the source files.

## Not implemented

- Per-user authentication for self-service deletion
- Differential privacy or other formal privacy guarantees for training
- Automated log expiry, data-subject access requests, and retraining after deletion


## Listen events (online features)

- **What is stored.** `POST /users/{id}/listens` keeps, per user, the set of applied `event_id`s and the list of applied track indices, in process memory. Those feed the online genre profile, activity, seen set, and popularity counts. Raw listen events (timestamps and track ids) are not written to disk or to logs. Logs carry only the HMAC pseudonym of the user id.
- **Retention.** Online state lasts until the process restarts, the user is deleted through `DELETE /users/{id}/recommendations` (which reverses the user's updates), or the instance is replaced. Nothing is persisted, so a restart removes all listen updates. The deletion path is the same for cache entries and listen features.
- **Limits.** `event_id` sets are not evicted. A process that receives many events holds that memory until restart. A user's state exists only on the instance that serves them, so the same user's data can sit on two instances after a failover, until restart.
- **Access.** Listen ingestion needs `INGEST_API_KEY`. Deletion needs `ADMIN_API_KEY`. The two tokens are separate, so the event producer cannot delete data, and deletion does not grant ingestion.
- **Metrics.** `/metrics` and `/stats` carry counts and timings only, with no user or track ids.
