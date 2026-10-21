# Real-Time Music Recommendation System

A two-stage recommender served over HTTP. A PyTorch two-tower model retrieves candidates with FAISS. A LightGBM ranker reorders a hybrid candidate pool using history features. FastAPI serves the results, Redis can cache them, and cold-start users get a popularity list.

It is a single-machine research prototype. Offline results are reported on MovieLens 25M and on synthetic data. The serving numbers are local measurements. Nothing here has run on production traffic.

## Results at a glance

**Offline, MovieLens 25M** (162k users, 40,858 movies, 12.4M positive ratings; 20k evaluated users; one held-out positive per user; K = 10; 95% bootstrap intervals over users)

| Model | Recall@10 | NDCG@10 | Genre diversity | Coverage |
|---|---|---|---|---|
| Random unseen | 0.0003 [0.0001, 0.0006] | 0.0002 | 0.577 | 0.993 |
| Popularity | 0.0482 [0.0454, 0.0513] | 0.0242 | 0.434 | 0.002 |
| Two-tower + FAISS only | 0.0123 [0.0107, 0.0139] | 0.0057 | 0.238 | 0.505 |
| **Hybrid candidates + LightGBM** | **0.0622 [0.0590, 0.0657]** | **0.0302** | 0.434 | 0.101 |

Paired difference in Recall@10 (same users): ranker minus popularity = +0.014 [+0.010, +0.018]; ranker minus retrieval-only = +0.050 [+0.047, +0.054].

**Caveat on how this was reached.** The first MovieLens run used FAISS candidates only. There the ranker scored 0.040, below popularity (0.048). I then added popular unseen items to the candidate pool and reran. I made that change after seeing test results. The MovieLens numbers above are therefore optimistic. The earlier run's metrics are saved as `docs/results/movielens25m-metrics-before-hybrid.json`.

**Offline, synthetic data** (2,000 users, 15 epochs): Recall@10 is 0.114 [0.100, 0.128] for the hybrid ranker, against 0.015 for popularity and 0.053 for retrieval only. Synthetic data is generated from the model's own assumptions, so these numbers show that the pipeline works. They do not show real-world quality.

**Serving, local** (12-core Mac, Python clients, 16 concurrent requests, 7,600 measured requests per run, MovieLens artifacts, no errors in any run)

| Configuration | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Memory (MB) |
|---|---|---|---|---|---|
| 1 worker | 2,589 | 5.44 | 7.79 | 9.28 | 368 |
| 2 workers | 3,184 | 3.97 | 6.33 | 7.22 | 776 |
| 4 workers | **3,890** | **3.57** | **5.05** | **6.20** | 1,514 |
| 4 workers + Redis (flushed) | 3,250 | 4.30 | 6.27 | 7.50 | 1,511 |

Redis reduced throughput here. The hit rate was 2.3%, because test users are spread across 162k users and repeat requests are rare. The cache helps when a small set of users makes most requests, which this benchmark does not model.

**Optimization** (in-process, best of 3)
- Cold-start path: 0.912 ms → 0.002 ms per request. The old code re-sorted all 40,858 items on every request. It now uses a popularity order computed once at startup.
- Personalized path: LightGBM prediction takes about 64% of request time, roughly 0.55 ms per request. Forcing single-threaded prediction made no difference (0.533 → 0.561 ms), so I reverted it. Reducing trees or batching predictions across requests would be the next steps. I haven't tried either.

**Training resources** (MovieLens 25M, 2 epochs): two-tower and ranker training took 305 s, and the full run took 403 s. Peak resident memory was 4.4 GB during the full pipeline.

## Architecture

```
offline  python -m rtrec.pipeline --dataset movielens --data ml-25m --out artifacts-ml25m
  positive ratings ─► leave-two-out split (train / validation / test per user)
  train events ─► two-tower (PyTorch, in-batch softmax) ─► user and item vectors
  item vectors ─► FAISS inner-product index
  validation events ─► hybrid candidates ─► LightGBM LambdaRank ─► ranker
  ─► artifacts: model, FAISS index, ranker, vectors, history statistics, metrics

online   FastAPI
  GET /recommendations/{user_id}
    ├─ Redis hit ─────────────────────────────────────────────► response
    ├─ unknown user ─► popularity list (cold start)
    └─ known user ─► FAISS top-100 unseen + top-50 popular unseen
                     ─► LightGBM rerank ─► top-k ─► cache ─► response
  GET /health   liveness
  GET /ready    readiness (503 if the model is not loaded)
  DELETE /users/{user_id}/recommendations   purge cached recommendations
```

## Modules

| Module | Role |
|---|---|
| `movielens.py` | Loads MovieLens 25M positives, drops users with fewer than 3 positives, re-indexes ids |
| `data.py` | Synthetic events, leave-two-out split |
| `model.py` | Two-tower model: user tower (embedding plus genre profile), item tower (item, genre, artist) |
| `retrieval.py` | FAISS index (flat or IVF), top-unseen search |
| `ranking.py` | History statistics, hybrid candidate generation, six ranker features, LightGBM LambdaRank |
| `evaluation.py` | Bootstrap and paired confidence intervals, diversity, coverage |
| `metrics.py` | Recall@K and NDCG@K for one positive per user |
| `experiments.py` | Deterministic A/B assignment (assignment only; see `docs/EXPERIMENTS.md`) |
| `privacy.py` | HMAC pseudonyms for logs |
| `cache.py` | Redis cache; errors are counted, not raised; no-op fallback |
| `service.py` | Loads artifacts, serves recommendations, purges a user's cache entries |
| `api.py` | FastAPI app: health, readiness, recommendations, deletion |
| `pipeline.py` | Training, evaluation with baselines and intervals, artifact export |
| `artifacts.py` | Artifact file names |

## Data

- **MovieLens 25M** (GroupLens): 25M ratings. I keep ratings of 4.0 or higher as implicit positives. That leaves 12.45M positives from 161,821 users (after dropping users with fewer than 3 positives) and 40,858 movies. The dataset is not in this repository. Download `ml-25m.zip` from GroupLens and extract it. MovieLens has no artist metadata, so the artist feature is constant in these results.
- **Synthetic**: 2,000 users, 3,000 items, generated from a latent-factor model with a fixed seed. It is used for tests and quick runs.

## Methodology

- **Split:** per user, the last positive is the test item, the second-to-last is the validation item, and the rest are training. The two-tower model trains only on training events. The ranker trains on validation events, with history from training events only. Test events are used only for reporting.
- **Candidates:** 100 FAISS candidates (unseen items only) plus up to 50 popular unseen items, with dot-product scores. Training and serving call the same function.
- **Ranker features:** retrieval score, rank, log popularity, the user's genre affinity, the user's play count for the item's artist, and log user activity.
- **Sampling:** 20,000 validation users for ranker training and 20,000 test users for evaluation, drawn with a fixed seed. The same users are used for every model.
- **Statistics:** 95% percentile bootstrap over users (1,000 resamples), and paired bootstrap for differences between models.

## Engineering decisions and the problems behind them

- **OpenMP conflict on macOS.** PyTorch, FAISS, and LightGBM each load an OpenMP runtime, and running them multi-threaded in one process crashed here. The code pins threads to one and sets `KMP_DUPLICATE_LIB_OK=TRUE`. That flag is an unsupported workaround. Tests check FAISS results against brute-force numpy.
- **Import order.** PyTorch is imported before FAISS. Importing FAISS first crashed PyTorch in this environment.
- **Benchmark client.** A single Python load generator topped out near 600 req/s, below what the server could serve. The benchmark runs four client processes and merges their raw latencies. Earlier single-client numbers are not comparable.
- **Cache keys include a model version.** The version is a hash of the training config, so a retrained model never serves stale results.

## Privacy

Logs contain a pseudonym (HMAC-SHA256 under `PSEUDONYM_KEY`), k, source, cache hit, and latency. They never contain raw user ids or item lists. A test checks this. Deletion purges cached recommendations for a user. It does not remove that user's rows from training data or model artifacts, which would need retraining. Full details, retention, and what is not implemented are in `docs/PRIVACY.md`.

## Failure handling

Redis down: the service runs without the cache, and `/ready` reports `cache_available: false`. Model missing or unreadable: `/health` stays up, and `/ready` and recommendations return 503. Unknown users get popularity. Invalid `k` returns 422. Each behavior has a test. Details are in `docs/FAILURES.md`.

## A/B testing

`rtrec/experiments.py` assigns users to control or treatment deterministically, by hashing the experiment name and user id. Users keep their arm across requests and servers, and different experiments are independent. No online test has been run. The plan, metrics, and guardrails are in `docs/EXPERIMENTS.md`.

## Running it

Requires Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                              # 46 tests

python -m rtrec.pipeline --out artifacts --epochs 15   # synthetic, about 7 s

# MovieLens 25M: download and extract ml-25m.zip first
python -m rtrec.pipeline --dataset movielens --data ml-25m --out artifacts-ml25m --epochs 2

ARTIFACT_DIR=artifacts-ml25m PSEUDONYM_KEY=change-me uvicorn rtrec.api:app --port 8000
curl localhost:8000/ready
curl localhost:8000/recommendations/u42?k=10
curl localhost:8000/recommendations/new-listener?k=10
curl -X DELETE localhost:8000/users/u42/recommendations

python scripts/loadtest.py --url http://127.0.0.1:8000 --users 161821 --requests 4000 --concurrency 16
python scripts/bench_workers.py --artifacts artifacts-ml25m --users 161821 --workers 1,2,4 --out bench.json
```

Redis is optional. Set `REDIS_URL=redis://127.0.0.1:6379/0` to enable it.

### Docker (not verified)

The Dockerfile and compose file are written. I could not build them here because the Docker daemon was not running. Treat them as untested.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness: model version, sizes, cache status. 503 if the model is not loaded |
| GET | `/recommendations/{user_id}?k=10` | Top-k (k from 1 to 50). `source` is `personalized` or `cold_start`. `cache_hit` shows whether Redis served it |
| DELETE | `/users/{user_id}/recommendations` | Purge that user's cached recommendations |

## Limitations

- **Offline only.** Results are from held-out ratings. They do not measure satisfaction, skips, or engagement.
- **Two training epochs** for MovieLens, chosen for run time. Hyperparameters were not tuned.
- **Design changed after seeing test results** (see the caveat above). The reported gain over popularity is therefore optimistic.
- **Coverage is low.** The ranker recommends about 10% of the catalog, against 50% for retrieval alone and 0.2% for popularity.
- **Diversity is no better than popularity.** Genre diversity is 0.434, the same as popularity.
- **Single machine, single host.** The benchmark runs on localhost, with client and server sharing 12 cores. It says nothing about network latency or multi-region deployment.
- **LightGBM per-call cost** limits the personalized path. Cutting trees or batching predictions would help, but neither is implemented.
- **Dense in-memory history.** Each worker loads the full history, about 370 MB at MovieLens scale. Memory grows with users times items for the genre and artist counts.
- **No authentication** on the deletion endpoint.
- **No retraining after deletion,** no hot model reload, and no circuit breakers.
- **Docker not verified.**

## Project layout

```
src/rtrec/         library (see Modules)
tests/             46 tests across data, model, retrieval, ranking, cache, service, API,
                   metrics, evaluation, experiments, privacy, MovieLens loader
scripts/loadtest.py       concurrent HTTP load test
scripts/bench_workers.py  1, 2, 4 worker benchmark with multiple client processes
docs/PRIVACY.md           privacy design and retention
docs/EXPERIMENTS.md       offline evaluation and A/B design
docs/FAILURES.md          failure handling and recovery
docs/results/             saved metrics and serving benchmark JSON
```

## License

MIT. See [LICENSE](LICENSE). MovieLens data is governed by its own terms and is not included.
