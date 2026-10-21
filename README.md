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

The API serves the Music4All model. The same candidate, ranking, and diversity functions run offline (during evaluation) and online (in the API), so a served list is the list the experiment evaluated.

**Artifacts.** `python scripts/run_music.py --export artifacts-music4all-10k` writes the two-tower weights, FAISS index, LightGBM ranker, user and track vectors, the history at the test cut-off, Music4All user and track ids, and a `config.json` with `"model_family": "music4all"` and the diversity penalty chosen on validation (`selected_lambda`). The directory is not committed. The API reports which family it loaded at `GET /ready`.

**Ids and unknown cases.** Requests use Music4All user ids. Responses return Music4All track ids, for example `PFYen7nm8cFFUub4`. A user id that is unknown, non-numeric, or has no trained vector gets the catalog popularity list, labeled `source: cold_start`. A user who has already played a track is never recommended it.

**Readiness.** Loading runs a self-check: one cold-start request and one personalized request, uncached. `/ready` returns 503 with a reason (for example `FileNotFoundError`) if any artifact is missing or inconsistent, and `/health` stays 200. `GET /metrics` returns request counts by source, cache hit rate, and p50/p95/p99 latency over the last 5,000 requests. It contains no user ids and covers one process only.

**OpenMP.** Serving calls LightGBM with `num_threads=1`. faiss and LightGBM loaded together crashed (exit 139) in 5 of 5 runs with four LightGBM threads, with or without the workaround flag. With one thread, there were no failures. See `docs/FAILURES.md`.

### Measured serving performance (localhost)

Hardware: a 12-core Apple Silicon Mac. Docker runs under Colima (4 vCPU, 6 GB). The load generator runs on the same machine, so these numbers are localhost measurements, not networked ones. Each run sends 8,000 requests from 4 client processes with 16 concurrent requests. Source: `docs/results/serving-music4all.json`.

| Setup | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Memory (MB) |
|---|---|---|---|---|---|
| 1 uvicorn worker, no cache | 2,057 | 6.96 | 10.72 | 12.62 | 313 |
| 2 workers, no cache | 3,067 | 4.50 | 7.13 | 8.63 | 669 |
| 4 workers, no cache | 3,362 | 4.22 | 6.14 | 7.45 | 1,290 |
| 1 worker, Redis cache (61% hits) | 2,283 | 5.62 | 10.76 | 13.35 | 314 |
| 2 workers, Redis cache | 3,335 | 3.40 | 7.61 | 9.53 | 670 |
| 4 workers, Redis cache | 3,292 | 3.35 | 8.38 | 10.38 | 1,286 |

Memory is the total RSS of the server process tree. Each worker loads its own copy of the model, so memory grows with worker count. At four workers, throughput barely rose above two, and the shared client machine is the likely limit. I did not measure that separately. The Redis cache gave a small gain here, because the workload draws from 3,000 users and the hit rate is 61%.

### Two instances behind a load balancer

`docker-compose up --build -d` starts two API containers, Redis, and an nginx load balancer on `localhost:8080`. Each API container has a readiness health check, and nginx retries a failed request on the other instance.

| Test | Result |
|---|---|
| Steady load through nginx, 7,700 measured requests | 2,174 req/s, p50 5.65 ms, p95 12.8 ms, p99 17.5 ms, 0 errors |
| Memory | about 212 MiB per API container (one worker each) |
| Failover under load: `docker kill` one instance at t=6 s, restart it at t=12 s, then stop the other gracefully at t=19 s | 30,000 requests, 0 client-visible errors. Nginx retries were not logged, so the number of retries is unknown. |
| Graceful stop under load (`docker-compose stop -t 25`) | exit code 0 in about 1 s; uvicorn logged "Application shutdown complete" |

The Compose run went through the VM network and shared 4 vCPUs between nginx, two APIs, and Redis. It is slower than the multi-process localhost run and is not a production benchmark. I have not run the stack on separate hosts, so cross-host latency and throughput are unmeasured.

### Earlier benchmark (MovieLens artifacts)

These numbers are from the MovieLens model (1,514 MB for 4 workers). They are not music results. Redis lowered throughput there because the hit rate was 2.3%, against 162,000 users. The cold-start path once re-sorted every item per request (0.912 ms). It now uses a popularity order computed at startup (0.002 ms).

| Configuration | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Memory (MB) |
|---|---|---|---|---|---|
| 1 worker | 2,589 | 5.44 | 7.79 | 9.28 | 368 |
| 2 workers | 3,184 | 3.97 | 6.33 | 7.22 | 776 |
| 4 workers | 3,890 | 3.57 | 5.05 | 6.20 | 1,514 |
| 4 workers + Redis (flushed) | 3,250 | 4.30 | 6.27 | 7.50 | 1,511 |

## Running it

Requires Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check src scripts tests
pytest -q                                                  # 78 tests

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
docker-compose up --build -d                               # http://localhost:8080

# Load and worker benchmarks (localhost)
python scripts/sample_user_ids.py --artifacts artifacts-music4all-10k --out user_ids.txt
python scripts/bench_workers.py --artifacts artifacts-music4all-10k --user-ids-file user_ids.txt --workers 1,2,4

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
  GET /recommendations/{user_id}   Redis cache ─► known user? ─► hybrid candidates ─► LightGBM ─► diversity rerank ─► top-k
                                                              └─ unknown / untrained ─► catalog popularity (cold_start)
  GET /health                      liveness
  GET /ready                       readiness: model loaded and self-checked
  GET /metrics                     counters and latency percentiles (no user ids)
  DELETE /users/{id}/recommendations   bearer token (ADMIN_API_KEY) required

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

- **Deletion requires authentication.** `DELETE /users/{id}/recommendations` needs a bearer token equal to `ADMIN_API_KEY`. Tests cover a missing token (401), a wrong token (403), a correct token (200), and a disabled endpoint (503). Self-service deletion by users is not implemented.
- **Logs contain no raw user ids.** They carry an HMAC pseudonym, and a test checks this. Every instance in the Compose stack must share one `PSEUDONYM_KEY`. Otherwise the same user gets different pseudonyms on different instances.
- **Metrics contain no user ids.** `GET /metrics` returns counts and latencies only.
- **Deletion has limits.** Purging clears cached recommendations. It does not remove a user's data from training artifacts, which needs retraining. See `docs/PRIVACY.md`.

## Engineering decisions

- **OpenMP.** FAISS and LightGBM each bundle an OpenMP runtime. Loaded together in one process, LightGBM with four threads crashed (exit 139) in 5 of 5 runs of `scripts/openmp_probe.py`, with and without `KMP_DUPLICATE_LIB_OK`. Torch is not needed to trigger it. With one thread it never crashed. The service now pins LightGBM prediction to one thread. The `KMP_DUPLICATE_LIB_OK=TRUE` setting stays. The probe passed without it in single-thread mode, but I have not run the test suite or the API without it, so I have not shown it is unnecessary. It is an unsupported workaround. The durable fix is one OpenMP runtime shared by all three, which needs source builds I have not tried. See `docs/FAILURES.md`.
- **Import order.** PyTorch is imported before FAISS.
- **Same code offline and online.** Candidate generation, LightGBM scoring, and diversity selection run through `hybrid.py` in both the evaluation and the API. The tests check that the API returns the list the shared functions produce from the saved artifacts.
- **Chronological splits, not random ones.** Random splits would let the model see the future.
- **Container image.** Artifacts are mounted read-only, not baked into the image. The container runs as a non-root user. LightGBM needs `libgomp1`, which the slim base image lacks. The first build passed and the container then failed at import, so the build alone does not show a container starts.

## Limitations

- **Offline only.** There are no online experiments, and no real-user outcomes.
- **One sample, one seed.** Music results come from one 10,000-user sample and one seed. Variance across seeds and samples is not measured.
- **Frozen user vectors.** Served user vectors come from the stage-1 model. Ranking features and the seen set use the history up to the test cut-off. Nothing updates while the service runs, and new listening needs a retrain.
- **Two untrained users.** Two of the 10,000 sampled users have no events before the stage-1 cut-off. The API serves them as cold start. One of them has events before the test cut-off, so the reported test pool included that user with an untrained vector. That is one of 2,300 test users. I did not change the reported results.
- **Ranker data.** The ranker is trained on 4,159 users. Cold-start test users are almost absent (one).
- **Model.** Two epochs, not tuned. No artist features (excluded by decision). No sequential modeling.
- **Localhost benchmarks only.** The client and server shared one machine. No networked, cross-host, or multi-region measurements exist.
- **Failover is not fault-tolerant at the database level.** Redis is a single instance. If it goes down, requests are served without cache, and the API keeps working.
- **Not done:** Prometheus export (metrics are JSON), distributed tracing, multi-region deployment, automatic model rollout or rollback, and self-service per-user deletion.

## Project layout

```
src/rtrec/                  library
tests/                      78 tests (data, model, retrieval, ranking, cache, service, API,
                            security, privacy, experiments, evaluation, metrics, MovieLens,
                            music loader and pipeline, music serving end-to-end)
scripts/                    build_music_sample.py, run_music.py, sample_user_ids.py,
                            loadtest.py, bench_workers.py, openmp_probe.py
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
