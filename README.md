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

## Serving: local measurements

12-core Mac, Python load clients (four processes, 16 concurrent requests), MovieLens artifacts, 7,600 measured requests per run, no errors in any run. These are not music results.

| Configuration | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Memory (MB) |
|---|---|---|---|---|---|
| 1 worker | 2,589 | 5.44 | 7.79 | 9.28 | 368 |
| 2 workers | 3,184 | 3.97 | 6.33 | 7.22 | 776 |
| 4 workers | **3,890** | **3.57** | **5.05** | **6.20** | 1,514 |
| 4 workers + Redis (flushed) | 3,250 | 4.30 | 6.27 | 7.50 | 1,511 |

Redis reduced throughput here. The hit rate was 2.3%, because the benchmark spreads requests across 162,000 users. The cache helps when a small set of users makes most requests.

**Optimization.** The cold-start path re-sorted every item on every request, which cost 0.912 ms per request. It now uses a popularity order computed at startup, which costs 0.002 ms. LightGBM prediction takes about 64% of personalized request time (about 0.55 ms). Forcing single-threaded prediction made no difference (0.533 → 0.561 ms), so I reverted it.

## Architecture

```
offline   build sample ─► chronological windows ─► two-tower (stage 1)
                                                 ─► FAISS index
          ranker window ─► hybrid candidates ─► LightGBM LambdaRank
          validation    ─► select diversity penalty (validation only)
          test          ─► evaluate once ─► docs/results/

online    FastAPI
          GET  /recommendations/{user_id}      cache ─► FAISS + popular candidates ─► LightGBM ─► top-k
          GET  /health                          liveness
          GET  /ready                           readiness (503 if the model is not loaded)
          DELETE /users/{user_id}/recommendations   bearer token (ADMIN_API_KEY) required
```

## Modules

| Module | Role |
|---|---|
| `music.py` | Music4All-Onion loader: streamed counts, fixed-seed stratified sample, chunked filtering |
| `music_pipeline.py` | Chronological splits, two-tower, FAISS, SVD baseline, LambdaRank, selection, one test evaluation |
| `movielens.py` | MovieLens 25M loader (scalability benchmark) |
| `pipeline.py` | Synthetic and MovieLens pipeline (leave-two-out per user) |
| `evaluation.py` | Bootstrap and paired intervals, diversity, novelty, coverage, diversity-penalized reranking |
| `metrics.py` | Recall@K and NDCG@K for one positive and for set relevance |
| `model.py`, `retrieval.py`, `ranking.py` | Two-tower model, FAISS search, history statistics, hybrid candidates, LightGBM |
| `service.py`, `api.py`, `cache.py` | Serving, bearer-token deletion, readiness, Redis with fallback |
| `privacy.py`, `experiments.py` | HMAC pseudonyms for logs; deterministic A/B assignment (no online test run) |

## Running it

Requires Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                                  # 60 tests

# Music4All-Onion: download the Zenodo record 15394646 files
#   userid_trackid_timestamp.tsv.bz2 and id_genres_tf-idf.tsv.bz2 into <music4all dir>
python scripts/build_music_sample.py --data <music4all dir> --cache cache/music_10k.npz
python scripts/run_music.py --cache cache/music_10k.npz --out docs/results/music4all-10k

# Synthetic and MovieLens benchmarks
python -m rtrec.pipeline --out artifacts --epochs 15
python -m rtrec.pipeline --dataset movielens --data ml-25m --out artifacts-ml25m --epochs 2

ARTIFACT_DIR=artifacts PSEUDONYM_KEY=change-me ADMIN_API_KEY=change-me-too uvicorn rtrec.api:app --port 8000
curl localhost:8000/recommendations/u42?k=10
curl -X DELETE -H "Authorization: Bearer change-me-too" localhost:8000/users/u42/recommendations
```

## Security and privacy

- **Deletion requires authentication.** `DELETE /users/{id}/recommendations` needs a bearer token equal to `ADMIN_API_KEY`. Tests cover a missing token (401), a wrong token (403), a correct token (200), and a disabled endpoint (503). Self-service deletion by users is not implemented.
- **Logs contain no raw user ids.** They carry an HMAC pseudonym, and a test checks this.
- **Deletion has limits.** Purging clears cached recommendations. It does not remove a user's data from training artifacts, which needs retraining. See `docs/PRIVACY.md`.

## Engineering decisions

- **OpenMP workaround.** PyTorch, FAISS, and LightGBM each load an OpenMP runtime, which crashed the process when multi-threaded on macOS. The code pins threads to one and sets `KMP_DUPLICATE_LIB_OK=TRUE`. That flag is an unsupported workaround. Removing it needs builds that share one runtime, which I have not tried.
- **Import order.** PyTorch is imported before FAISS.
- **Benchmark client.** A single Python load generator topped out near 600 req/s, so the benchmark uses four client processes.
- **Chronological splits, not random ones.** Random splits would let the model see the future.

## Limitations

- Offline only. There are no online experiments.
- Music results come from one 10,000-user sample and one seed. I have not measured variance across seeds or samples.
- The ranker sees only 4,159 users.
- The model is trained for two epochs and not tuned.
- Serving was measured on one machine, with the client and server sharing 12 cores.
- No artist features (artist metadata was excluded by decision).
- Multi-instance deployment, CI, Prometheus, failure injection, Docker verification, and multi-region operation are not done.

## Project layout

```
src/rtrec/                  library
tests/                      60 tests (data, model, retrieval, ranking, cache, service, API,
                            security, privacy, experiments, evaluation, metrics, MovieLens,
                            music loader and pipeline)
scripts/                    build_music_sample.py, run_music.py, loadtest.py, bench_workers.py
docs/PRIVACY.md             privacy design, retention, deletion
docs/EXPERIMENTS.md         offline evaluation and A/B design
docs/FAILURES.md            failure handling
docs/PROVENANCE.md          code and dataset license audit
docs/results/               saved metrics JSON, including the music results
```

## License

MIT. See [LICENSE](LICENSE). The Music4All-Onion and MovieLens datasets are governed by their own terms and are not included.
