# Real-Time Music Recommendation System

A two-stage recommender served over HTTP. A PyTorch two-tower model retrieves candidates with FAISS. LightGBM reranks them using history features. Redis caches responses, and cold-start users get a popularity fallback.

Everything runs on **synthetic listening data** generated in this repository. No real user data or third-party datasets are used.

## Architecture

```
offline (python -m rtrec.pipeline)
  synthetic events ─► leave-two-out split (train / validation / test)
  train events ─► two-tower model (PyTorch, in-batch softmax) ─► user + item vectors
  item vectors ─► FAISS inner-product index
  validation events ─► candidates from FAISS ─► LightGBM LambdaRank ranker
  artifacts/ ─► model, FAISS index, ranker, vectors, history statistics

online (FastAPI)
  GET /recommendations/{user_id}
    ├─ Redis cache hit ──────────────────────────────► response
    ├─ unknown user ─► popularity fallback (cold start)
    └─ known user ─► FAISS top-100 unseen ─► LightGBM rerank ─► top-k ─► cache ─► response
```

## Evaluation (offline, synthetic data)

Setup: 2,000 users, 3,000 items, 12 genres, 400 artists. Each user has 15 to 60 events. The last event is the test positive, the second-to-last is the validation positive, and the rest are training. Metrics use one held-out positive per user, K = 10, 100 retrieval candidates, and seed 0.

| Method | Recall@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.015 | 0.0079 |
| Two-tower + FAISS retrieval only | 0.053 | 0.0253 |
| Two-tower + FAISS + LightGBM ranker | **0.117** | **0.0584** |

Numbers come from `artifacts/metrics.json`. With 2,000 users and one positive each, the standard error of Recall@10 is about 0.007, so differences below roughly 0.015 are not reliable.

**Caveat on epoch choice:** I trained for 5 and 15 epochs and selected 15 after seeing test metrics for both. No separate validation split was used for that choice, so the reported test numbers are slightly optimistic.

## Serving latency (measured locally)

Setup: one uvicorn worker on the same machine as the load generator, 2,000 users, 3,000 requests per run, K = 10, 5% cold-start ids, seed 1 or 2. Latency is measured at the client.

| Configuration | Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|---|
| No cache | 1 | 795 | 1.23 | 1.37 | 1.48 |
| No cache | 8 | 1,346 | 5.03 | 10.09 | 14.24 |
| No cache | 32 | 541 | 41.8 | 154.4 | 240.4 |
| Redis, cold cache per run | 1 | 785 | 1.42 | 1.64 | 1.80 |
| Redis, cold cache per run | 8 | 1,221 | 5.46 | 11.14 | 15.32 |
| Redis, cold cache per run | 32 | 534 | 43.1 | 158.5 | 234.6 |

Redis reached a 45% hit rate on these runs, but it did not lower latency. The model path already takes about 1 ms, so a Redis round trip and JSON encoding cost about as much as they save. Redis is useful only when the model path is expensive.

At concurrency 32, throughput drops because a single Python worker is saturated. More workers would help, but I did not measure that.

## Running it

Requires Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m rtrec.pipeline --out artifacts --epochs 15   # ~7 s on a laptop CPU
pytest -q

uvicorn rtrec.api:app --port 8000
curl 'http://127.0.0.1:8000/health'
curl 'http://127.0.0.1:8000/recommendations/u42?k=10'
curl 'http://127.0.0.1:8000/recommendations/new-listener?k=10'   # cold start
```

To use Redis, start one and set `REDIS_URL`:

```bash
redis-server --port 6379 &
REDIS_URL=redis://127.0.0.1:6379/0 uvicorn rtrec.api:app --port 8000
```

If Redis is down, the service still answers and reports `cache_available: false` on `/health`.

Load test (against a running server):

```bash
python scripts/loadtest.py --url http://127.0.0.1:8000 --users 2000 --requests 3000 --concurrency 8
```

### Docker (not yet verified)

```bash
python -m rtrec.pipeline --out artifacts --epochs 15
docker compose up --build
```

The Dockerfile and compose file are written, but I could not build them here because the Docker daemon was not running. Treat them as untested until you have built them.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Model version, sizes, cache availability and error count |
| GET | `/recommendations/{user_id}?k=10` | Top-k items (k from 1 to 50). `source` is `personalized` or `cold_start`. `cache_hit` shows whether Redis served the response. |

User ids have the form `u<index>`. Any other id is treated as a cold-start user.

## Design decisions

- **Leave-two-out evaluation.** The two-tower model trains only on training events. The ranker trains on validation events, so its labels are never seen by the retrieval model. Test events are used only for reporting.
- **Single-threaded execution.** On macOS, the pip wheels for PyTorch, FAISS, and LightGBM each load an OpenMP runtime. Running them multi-threaded in one process crashed the process here, so `rtrec/__init__.py` sets `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE`. The second flag is an unsupported workaround. The tests check FAISS results against brute-force numpy to confirm the results are correct.
- **Import order.** `rtrec.pipeline` imports torch before faiss. Importing faiss first crashed torch in this environment.
- **Exact search.** The default index is `IndexFlatIP`. An IVF option exists and is tested for exact results when all lists are probed. I did not benchmark IVF recall or speed.
- **Cache keys include a model version.** The version is a hash of the training config, so a retrained model never serves stale results.

## Limitations

- Synthetic data only. The model has not been evaluated on real listening behavior.
- Single-positive evaluation with 2,000 users. The metrics are indicative, not precise.
- The ranker is trained on one validation positive per user, which is a small training set.
- No online experiment, A/B test, or production traffic.
- The load test ran on one machine with a single uvicorn worker.
- Dense FAISS flat search is exact but scales linearly with the catalog size.
- User histories and genre and artist counts are dense in memory, so they grow with users times items.
- Redis is not required, and the cache did not improve latency in these measurements.
- The Docker image is untested.

## Project layout

```
src/rtrec/
  data.py        synthetic events, leave-two-out split
  model.py       two-tower model (PyTorch)
  retrieval.py   FAISS index and top-unseen search
  ranking.py     history statistics, candidate features, LightGBM ranker
  cache.py       Redis cache with no-op fallback
  service.py     loads artifacts, serves recommendations
  api.py         FastAPI app
  pipeline.py    training, evaluation, artifact export
  metrics.py     Recall@K, NDCG@K
  artifacts.py   artifact file names
tests/           27 tests: data, model, retrieval, ranking, cache, service, API, metrics
scripts/loadtest.py   concurrent HTTP load test
```

## License

MIT. See [LICENSE](LICENSE).
