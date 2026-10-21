# Real-Time Music Recommendation System

A two-stage music recommender. A PyTorch two-tower model retrieves candidate tracks with FAISS, and a LightGBM ranker reorders a hybrid candidate pool built from retrieved and popular tracks. FastAPI serves the results, Redis can cache them, and new users get a popularity list.

This repository is a research prototype. Offline evaluation runs on Music4All-Onion listening data, with separate MovieLens and synthetic benchmarks. Serving numbers are local measurements. Nothing has run on real users, and no online A/B test has been run.

## Music results: Music4All-Onion (10,000 users)

**Data.** Music4All-Onion (Moscati et al., CIKM 2022; CC-BY-4.0) provides listening events from Last.fm users. A fixed-seed sample of 10,000 users is stratified into ten equal groups by listening activity, and only users with at least 50 events are eligible. The sample holds 24.5 million events, a median of 781 events per user, and 56,051 tracks with at least one event. Events run from 2005 to 2020. Only the listening log and the track genre table are used. The dataset is not stored in this repository, and no artist metadata is used.

**Time-aware splits.** Cut points are quantiles of event time, so every stage sees only the past:

| Window | Events | Used for |
|---|---|---|
| Before the 60th percentile | 14.7M | Training the two-tower model and the history for candidates |
| 60th to 75th percentile | 3.7M | Training the LightGBM ranker (labels) |
| 75th to 85th percentile | 2.4M | Selecting the diversity penalty (validation only) |
| After the 85th percentile | 3.7M | **Test. Evaluated once.** |

Nothing from a later window reaches an earlier model or feature set. Only the validation window was used to choose among variants. The test window was evaluated once, with the configuration fixed in advance. I ran the experiment twice. The second run fixed a reporting bug (the ranker's row count) and produced identical test metrics, which confirms the run is reproducible.

**Test results** (2,300 warm users, one run; 95% percentile bootstrap intervals over users)

| Model | Recall@10 | Recall@20 | NDCG@10 | Genre diversity@10 | Novelty@10 (bits) | Coverage@10 |
|---|---|---|---|---|---|---|
| Popularity | 0.0016 [0.0013, 0.0018] | 0.0028 [0.0025, 0.0032] | 0.0558 | 0.738 | 10.39 | 0.001 |
| Matrix factorization (SVD) | 0.0015 [0.0013, 0.0017] | 0.0028 [0.0025, 0.0031] | 0.0573 | 0.641 | 12.35 | 0.025 |
| Two-tower + FAISS only | 0.0011 [0.0008, 0.0014] | 0.0020 [0.0017, 0.0024] | 0.0336 | 0.372 | 19.59 | 0.290 |
| **Hybrid + LightGBM (selected λ = 0)** | **0.0022 [0.0019, 0.0025]** | **0.0037 [0.0034, 0.0041]** | **0.0821** | 0.641 | 12.55 | 0.105 |
| Hybrid + diversity penalty λ = 0.4 | 0.0023 [0.0020, 0.0026] | 0.0037 [0.0033, 0.0041] | 0.0824 | **0.938** | 12.31 | 0.088 |

**Paired differences in Recall@20** (same users)

| Comparison | Mean | 95% interval |
|---|---|---|
| Hybrid minus popularity | +0.0009 | [+0.0005, +0.0013] |
| Hybrid minus SVD | +0.0009 | [+0.0005, +0.0013] |
| Hybrid minus two-tower retrieval only | +0.0017 | [+0.0012, +0.0021] |

**What the numbers say**
- The hybrid ranker beats popularity, SVD, and retrieval-only on Recall@20 and NDCG@10, and the intervals do not overlap. It also has the best NDCG@10 by a wide margin (0.082 against 0.056 for popularity).
- Absolute recall is low, below 0.4%. A typical test user has 230 distinct relevant tracks (median; mean 408) out of 56,051 in the catalog, so a list of 20 can recover only a small share of them even when every item is a hit. NDCG is the more informative comparison here.
- The diversity penalty trades nothing measurable in recall for much more genre variety: genre diversity rises from 0.641 to 0.938 at λ = 0.4, with Recall@20 unchanged. The validation rule chose λ = 0 because it had the highest validation Recall@20, but the λ = 0.1 and λ = 0.4 intervals overlap with λ = 0, so that choice is a near-tie.
- The two-tower model alone is weaker than popularity on this data. Its retrieval recommends tracks that are much more novel and less often correct. Reranking by LightGBM recovers accuracy and adds popularity signals.

**Caveats**
- The ranker is trained on only 4,159 users, the ones with events in the ranker window. Many users are inactive in that window.
- Only 1 cold-start user appears in the test window. Cold-start quality is therefore not measured here.
- The model is trained for 2 epochs. The configuration was not tuned.
- Music4All-Onion records listening events, not explicit skips or repeats, so the implicit signal is "was played."
- The offline setting shows how well a list matches later listening. It does not show satisfaction or engagement.

**Run cost** (macOS, 12 cores, single-threaded PyTorch, LightGBM, and FAISS): building the sample streams the 2.2 GB log twice and took 398 s with a peak of 3.5 GB. The experiment itself took 249 s with a peak of 4.8 GB.

### Independent 50,000-user run (seed 1)

A second sample: 50,000 users drawn with seed 1 (same stratification and filters), and the same configuration with seed 1 for the model. Nothing was tuned on the earlier test window, and the 10,000-user results above are unchanged. Raw metrics: `docs/results/music4all-50k-seed1.json`.

Sample: 74.7M stage-1 events, 18.7M ranker-window, 12.4M validation, 18.7M test. Test pool: 11,335 users. Ranker trained on 10,000 users (1.5M rows). Selected on validation: λ = 0.

| Model (test window) | Recall@10 | Recall@20 | NDCG@10 | Genre diversity@10 | Coverage@10 |
|---|---|---|---|---|---|
| Popularity | 0.0016 [0.0015, 0.0019] | 0.0028 | 0.0549 | 0.838 | 0.002 |
| SVD CF | 0.0018 [0.0016, 0.0019] | 0.0032 | 0.0643 | 0.594 | 0.031 |
| Two-tower + FAISS only | 0.0013 [0.0012, 0.0015] | 0.0026 | 0.0417 | 0.457 | 0.703 |
| Hybrid + LightGBM, λ = 0 (selected) | 0.0026 [0.0024, 0.0029] | 0.0044 | 0.0846 | 0.603 | 0.291 |
| Hybrid, λ = 0.1 | 0.0027 [0.0024, 0.0030] | 0.0044 | 0.0849 | 0.756 | 0.275 |
| Hybrid, λ = 0.4 | 0.0026 [0.0024, 0.0029] | 0.0043 | 0.0833 | 0.933 | 0.273 |

Paired differences in Recall@20 (hybrid λ=0 minus X), 95% bootstrap intervals: popularity +0.0017 [0.0013, 0.0020]; SVD CF +0.0013 [0.0009, 0.0016]; retrieval only +0.0018 [0.0015, 0.0022].

What this run supports: the hybrid ranker beats the popularity, SVD, and retrieval-only baselines on this sample, and the intervals don't overlap with popularity or with SVD. What it does not show: the λ grid (0, 0.1, 0.2, 0.4) differs by less than the intervals on recall and NDCG, so the choice of λ is a tie on accuracy, and diversity is the real trade-off. The 10,000-user and 50,000-user results used different samples, so compare them only as independent replications, not as one pooled estimate. Hardware: 8.8 GB peak RSS, 1,378 s total on the 12-core Mac. The model is still trained for two epochs, and I did not tune it.

## MovieLens 25M: scalability benchmark

This is a separate benchmark for scale. It is not a music-personalization result. The dataset is 12.45M positive ratings from 161,821 users over 40,858 movies, with each user's last event held out for testing (one positive per user, K = 10).

| Model | Recall@10 | NDCG@10 | Genre diversity | Coverage |
|---|---|---|---|---|
| Random unseen | 0.0003 [0.0001, 0.0006] | 0.0002 | 0.577 | 0.993 |
| Popularity | 0.0482 [0.0454, 0.0513] | 0.0242 | 0.434 | 0.002 |
| Two-tower + FAISS only | 0.0123 [0.0107, 0.0139] | 0.0057 | 0.238 | 0.505 |
| Hybrid + LightGBM | 0.0622 [0.0590, 0.0657] | 0.0302 | 0.434 | 0.101 |

**Caveat.** The first run used FAISS candidates only, and the ranker scored 0.040 there, below popularity. I then added popular candidates and reran. I made that change after seeing test results, so the hybrid numbers here are optimistic. The pre-change result is in `docs/results/movielens25m-metrics-before-hybrid.json`.

Training took 305 s for two epochs, and the full run took 403 s with a peak of 4.4 GB.

## Synthetic benchmark

With 2,000 synthetic users and 15 epochs, the hybrid ranker reaches Recall@10 of 0.114 [0.100, 0.128], against 0.015 for popularity and 0.053 for retrieval only. The synthetic data is generated from the model's own assumptions, so this only checks that the pipeline works.

## Serving the Music4All model

The API serves the Music4All model. Candidate generation, LightGBM scoring, and diversity selection run through `hybrid.py` in both the offline evaluation and the API, so the served lists are the lists the experiment evaluated.

**What updates online, and what does not.** Listen events change the features the ranker reads: the user's genre profile and activity, the seen set, and item popularity. Frozen until retraining: the two-tower user and track vectors (so FAISS candidates still come from the artifact snapshot), the popularity order used for candidate generation and cold start, and the LightGBM weights. A new listen therefore changes the ranking immediately, but it does not change which tracks are retrieved. A new user needs a retrain.

**Listen ingestion.** `POST /users/{user_id}/listens` with `{"event_id", "track_id", "timestamp"}` and a bearer token (`INGEST_API_KEY`).

| Input | Result |
|---|---|
| New valid event | 202, features updated, the user's cache entries invalidated |
| Repeated `event_id` for the same user | 200 `duplicate`, no state change (idempotent) |
| Unknown or untrained user | 404 |
| Unknown track | 422 |
| Timestamp before the history cutoff | 422 |
| Timestamp more than 5 minutes in the future | 422 |
| Malformed `event_id` | 422 (validation) |
| Missing or wrong token | 401 or 403; 503 when `INGEST_API_KEY` is unset |

**Cache invalidation.** Cache keys contain a per-user generation counter kept in Redis. A listen increments it, so every instance stops reading that user's old entries at once. The admin `DELETE` removes cached entries and also reverses that user's listen updates in memory.

**User affinity.** Nginx routes `/recommendations/{id}`, `/users/{id}/listens`, and `/users/{id}/recommendations` to one instance by user id (consistent hashing). A user's listen features live in that instance's memory, so the events and the recommendations for that user must reach the same process. If an instance dies, its users move to the other instance, which doesn't have their recent listens. Those listens are lost until a shared store exists. See Limitations.

**Readiness and metrics.** Loading runs a self-check (one cold-start and one personalized request, uncached). `/ready` returns 503 with a reason if loading fails. `GET /metrics` is Prometheus text: request counts by source, cache lookups, listen outcomes, HTTP errors by status, a latency histogram, and model state. Labels are fixed sets, so the series count does not grow with users or tracks. `GET /stats` returns the same numbers as JSON, with p50/p95/p99 latency over the last 5,000 requests.

### Measured serving performance

Hardware: a 12-core Apple Silicon Mac, with Docker under Colima (4 vCPU, 6 GB). Load clients run on the same host, so these are localhost measurements, not networked ones. Each run sends 7,700 measured requests after 300 warm-up requests, from 4 client processes with 16 concurrent requests. Raw results: `docs/results/serving-v2.json`.

The cache hit ratio depends on the user pool and the TTL, so I measured three pool sizes and report the ratio each run observed.

| Scenario (two instances, nginx, Redis) | Observed hit ratio | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Errors |
|---|---|---|---|---|---|---|
| 300-user pool, cached | 100% | 2,847 | 4.70 | 9.47 | 11.83 | 0 |
| 3,000-user pool, cached | 68% | 2,054 | 6.13 | 13.71 | 19.42 | 0 |
| 3,000-user pool, Redis down (uncached inference) | 0% | 1,368 | 10.04 | 18.93 | 23.19 | 0 |
| Redis down, one instance SIGKILLed at 3 s | 0% | 1,217 | 11.72 | 20.85 | 25.42 | 0 |
| 3,000-user pool, cached, one instance SIGKILLed at 3 s | 61% | 1,799 | 7.55 | 15.14 | 20.63 | 0 |

The failure rows show zero client errors when an instance is SIGKILLed. After the kill, requests are served by the remaining instance, so throughput falls to roughly one instance's share. Requests in flight at the kill were retried by nginx or failed over; the retry count is not logged. Memory per API container was about 250 MiB under this load.

**Listen ingestion** (`scripts/bench_ingest.py`, 4,000 events, 8 concurrent, through nginx): 1,026 events/s, p50 6.3 ms, p95 14.1 ms, p99 20.4 ms, all 202.

**Earlier worker benchmark (before online features, MovieLens and Music4All, localhost).** On the Music4All artifacts, one uvicorn process per worker count: 1 worker 2,057 req/s (p99 12.6 ms, 313 MB), 2 workers 3,067 req/s (p99 8.6 ms, 669 MB), 4 workers 3,362 req/s (p99 7.5 ms, 1,290 MB), uncached. The cache path adds one Redis read per request now, so these numbers are not directly comparable with the table above. Raw data for that worker run is in `docs/results/serving-music4all.json`, and the earlier MovieLens worker table is in `docs/results/serving-benchmark.json`.

## Running it

Requires Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check src scripts tests
pytest -q                                                  # 90 tests

# Music4All-Onion: download the Zenodo record 15394646 files
#   userid_trackid_timestamp.tsv.bz2 and id_genres_tf-idf.tsv.bz2 into <music4all dir>
python scripts/build_music_sample.py --data <music4all dir> --cache cache/music_10k.npz
python scripts/run_music.py --cache cache/music_10k.npz --out docs/results/music4all-10k --export artifacts-music4all-10k

# Serve the Music4All model
ARTIFACT_DIR=artifacts-music4all-10k PSEUDONYM_KEY=change-me uvicorn rtrec.api:app --port 8000
curl localhost:8000/ready
curl "localhost:8000/recommendations/<music4all user id>?k=10"

# Two instances behind nginx, with Redis (needs Docker; compose v2 or docker-compose)
cp .env.example .env                                       # set PSEUDONYM_KEY and ADMIN_API_KEY
scripts/seed_artifacts.sh artifacts-music4all-10k                # copy artifacts into the named volume
docker-compose up --build -d                               # http://localhost:8080

# Listen ingestion (needs INGEST_API_KEY; the stack reads it from .env)
curl -X POST -H "Authorization: Bearer $INGEST_API_KEY" -H "content-type: application/json" \
  -d '{"event_id":"e-1","track_id":"<track id>","timestamp":1760000000}' localhost:8080/users/<user id>/listens

# Load and worker benchmarks (localhost)
python scripts/sample_user_ids.py --artifacts artifacts-music4all-10k --out user_ids.txt
python scripts/bench_workers.py --artifacts artifacts-music4all-10k --user-ids-file user_ids.txt --workers 1,2,4
python scripts/bench_compose.py --users-file user_ids.txt --requests 8000 --kill rtrec-api2 --kill-after 3
python scripts/bench_ingest.py --users-file user_ids.txt --track <track id> --events 4000

# Synthetic and MovieLens benchmarks
python -m rtrec.pipeline --out artifacts --epochs 15
python -m rtrec.pipeline --dataset movielens --data ml-25m --out artifacts-ml25m --epochs 2
```

CI (`.github/workflows/ci.yml`) runs `ruff`, `pytest`, a Docker image build, and a container smoke test on every push.

## Architecture

```
offline (scripts/run_music.py)
  sample ─► chronological windows ─► two-tower (stage 1) ─► FAISS index (track vectors)
                                   ─► user vectors (stage 1)
  ranker window ─► hybrid candidates (FAISS + popular) ─► LightGBM LambdaRank
  validation    ─► choose diversity penalty lambda
  test          ─► evaluate once ─► docs/results/music4all-10k/
  export        ─► artifacts-music4all-10k/  (weights, index, ranker, vectors, history, ids, config)

online (FastAPI, one process per container)
  GET /recommendations/{user_id}   generation-keyed Redis cache ─► known user? ─► hybrid candidates (seen overlay)
                                   ─► LightGBM (online genre profile, activity, popularity) ─► diversity rerank ─► top-k
                                   └─ unknown / untrained ─► catalog popularity (cold_start)
  POST /users/{user_id}/listens    validate ─► idempotent update of in-memory features ─► bump user generation
  GET /health                      liveness
  GET /ready                       readiness: model loaded and self-checked
  GET /metrics                     Prometheus text (fixed labels, no user or track ids); /stats is JSON
  DELETE /users/{user_id}/recommendations   bearer token (ADMIN_API_KEY): cache purge and listen reversal

deployment (docker-compose.yml)
  nginx :8080 ─► api1, api2 (read-only artifacts volume, PSEUDONYM_KEY from .env) ─► redis
```

## Modules

| Module | Role |
|---|---|
| `music.py` | Music4All-Onion loader: streamed counts, fixed-seed stratified sample, chunked filtering |
| `music_pipeline.py` | Chronological splits, two-tower, FAISS, SVD baseline, LambdaRank, selection, one test evaluation, artifact export |
| `hybrid.py` | Candidate pool, LightGBM scoring, and diversity selection for one user. Shared by evaluation and serving |
| `movielens.py` | MovieLens 25M loader (scalability benchmark) |
| `pipeline.py` | Synthetic and MovieLens pipeline (leave-two-out per user) |
| `evaluation.py` | Bootstrap and paired intervals, diversity, novelty, coverage, diversity-penalized reranking |
| `metrics.py` | Recall@K and NDCG@K for one positive and for set relevance |
| `model.py`, `retrieval.py`, `ranking.py` | Two-tower model, FAISS search, history statistics, hybrid candidates, LightGBM |
| `service.py`, `api.py`, `cache.py`, `telemetry.py` | Serving for both artifact families, readiness self-check, bearer-token deletion, Redis with fallback, metrics |
| `privacy.py`, `experiments.py` | HMAC pseudonyms for logs; deterministic A/B assignment (no online test run) |

## Security and privacy

- **Deletion requires authentication.** `DELETE /users/{id}/recommendations` needs a bearer token equal to `ADMIN_API_KEY`. Tests cover a missing token (401), a wrong token (403), a correct token (200), and a disabled endpoint (503). The same call reverses that user's listen updates in memory. Self-service deletion by users is not implemented.
- **Listen ingestion requires a separate token** (`INGEST_API_KEY`), which the event producer holds. It does not grant deletion.
- **Metrics contain no user or track ids.** Prometheus labels are fixed sets: source, cache result, outcome, and status.
- **Logs contain no raw user ids.** They carry an HMAC pseudonym, and a test checks this. Every instance in the Compose stack must share one `PSEUDONYM_KEY`. Otherwise the same user gets different pseudonyms on different instances.
- **Metrics contain no user ids.** `GET /metrics` returns counts and latencies only.
- **Deletion has limits.** Purging clears cached recommendations. It does not remove a user's data from training artifacts, which needs retraining. See `docs/PRIVACY.md`.

## Engineering decisions

- **OpenMP.** FAISS and LightGBM each bundle an OpenMP runtime. Loaded together in one process, LightGBM with four threads crashed (exit 139) in 5 of 5 runs of `scripts/openmp_probe.py`, with and without `KMP_DUPLICATE_LIB_OK`. Torch is not needed to trigger it. With one thread it never crashed. The service now pins LightGBM prediction to one thread. The `KMP_DUPLICATE_LIB_OK=TRUE` setting stays. The probe passed without it in single-thread mode, but I have not run the test suite or the API without it, so I have not shown it is unnecessary. It is an unsupported workaround. The durable fix is one OpenMP runtime shared by all three, which needs source builds I have not tried. See `docs/FAILURES.md`.
- **Import order.** PyTorch is imported before FAISS.
- **Same code offline and online.** Candidate generation, LightGBM scoring, and diversity selection run through `hybrid.py` in both the evaluation and the API. The tests check that the API returns the list the shared functions produce from the saved artifacts.
- **Chronological splits, not random ones.** Random splits would let the model see the future.
- **Container image.** Artifacts are read-only, from a named volume, not baked into the image. Host bind mounts failed under Colima's virtiofs (see `docs/FAILURES.md`), so `scripts/seed_artifacts.sh` copies the artifacts into the volume. The container runs as a non-root user. LightGBM needs `libgomp1`, which the slim base image lacks. The first build passed and the container then failed at import, so the build alone does not show a container starts.

## Limitations

- **Offline only.** There are no online experiments, and no real-user outcomes.
- **One sample, one seed.** Music results come from one 10,000-user sample and one seed. Variance across seeds and samples is not measured.
- **Frozen user vectors.** Served user vectors come from the stage-1 model. Listen events update the ranker's features (genre profile, activity, seen set, popularity), but they do not update the two-tower vectors, so retrieval candidates are fixed until retraining.
- **Online state is per instance and in memory.** A user's listens are held by the instance that serves that user (affinity routing). A restart or a failover loses them, and the users moved to the other instance see the artifact features until they listen again. The fix is a shared store, which I have not built. Listen events are not persisted anywhere, so the event log is also lost.
- **Dedupe memory grows.** The set of applied `event_id`s per user is kept for the life of the process. It is not bounded or evicted.
- **Redis is one instance.** A cache outage makes every request uncached (measured above). The generation counter is read once per request, which costs one Redis round trip on the hot path.
- **Two untrained users.** Two of the 10,000 sampled users have no events before the stage-1 cut-off. The API serves them as cold start. One of them has events before the test cut-off, so the reported test pool included that user with an untrained vector. That is one of 2,300 test users. I did not change the reported results.
- **Ranker data.** The ranker is trained on 4,159 users. Cold-start test users are almost absent (one).
- **Model.** Two epochs, not tuned. No artist features (excluded by decision). No sequential modeling.
- **Localhost benchmarks only.** The client and server shared one machine. No networked, cross-host, or multi-region measurements exist.
- **Failover is not fault-tolerant at the database level.** Redis is a single instance. If it goes down, requests are served without cache, and the API keeps working.
- **Not done:** distributed tracing, multi-region deployment, automatic model rollout or rollback, self-service per-user deletion, a shared online-feature store, and durable listen logging.

## Project layout

```
src/rtrec/                  library
tests/                      90 tests (data, model, retrieval, ranking, cache, service, API,
                            security, privacy, experiments, evaluation, metrics, MovieLens,
                            music loader and pipeline, music serving end-to-end)
scripts/                    build_music_sample.py, run_music.py, sample_user_ids.py,
                            loadtest.py, bench_workers.py, bench_compose.py, bench_ingest.py, openmp_probe.py
deploy/nginx.conf           load balancer for the two-instance stack
Dockerfile, docker-compose.yml, .env.example
.github/workflows/ci.yml    lint, tests, container smoke test
docs/PRIVACY.md             privacy design, retention, deletion
docs/EXPERIMENTS.md         offline evaluation and A/B design
docs/FAILURES.md            failure handling and the OpenMP investigation
docs/PROVENANCE.md          code and dataset license audit
docs/results/               saved metrics JSON, including the music results and serving benchmarks
```

## License

MIT. See [LICENSE](LICENSE). The Music4All-Onion and MovieLens datasets are governed by their own terms and are not included.
