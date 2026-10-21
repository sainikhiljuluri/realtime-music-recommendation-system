"""Concurrent HTTP load test for the recommendations endpoint.

Example:
  python scripts/loadtest.py --url http://127.0.0.1:8000 --users 2000 --requests 3000 --concurrency 8

Reports latency percentiles measured at the client, throughput, errors, and cache hit rate.
Cold-start requests use ids that the model has never seen (--cold-fraction).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p / 100 * (len(ordered) - 1)))))
    return ordered[idx]


async def run(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    ids = []
    for _ in range(args.requests):
        if rng.random() < args.cold_fraction:
            ids.append(f"cold-{rng.randrange(10**9)}")
        else:
            ids.append(f"u{rng.randrange(args.users)}")

    latencies: list[float] = []
    cache_hits = 0
    errors = 0
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(base_url=args.url, timeout=10.0) as client:
        async def one(user_id: str) -> None:
            nonlocal cache_hits, errors
            async with sem:
                started = time.perf_counter()
                try:
                    resp = await client.get(f"/recommendations/{user_id}", params={"k": args.k})
                    elapsed = (time.perf_counter() - started) * 1000
                    if resp.status_code != 200:
                        errors += 1
                        return
                    latencies.append(elapsed)
                    cache_hits += bool(resp.json().get("cache_hit"))
                except httpx.HTTPError:
                    errors += 1

        # Warm-up requests are not measured.
        for user_id in ids[: args.warmup]:
            await client.get(f"/recommendations/{user_id}", params={"k": args.k})

        latencies.clear()
        cache_hits = 0
        errors = 0
        started = time.perf_counter()
        await asyncio.gather(*(one(u) for u in ids[args.warmup:]))
        wall = time.perf_counter() - started

    ok = len(latencies)
    raw = {"_latencies": list(latencies), "_wall": wall, "_cache_hits": cache_hits}
    return {**raw, 
        "url": args.url,
        "concurrency": args.concurrency,
        "requests": ok + errors,
        "errors": errors,
        "throughput_rps": round(ok / wall, 1) if wall > 0 else float("nan"),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else None,
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "p99": round(percentile(latencies, 99), 2),
        },
        "cache_hit_rate": round(cache_hits / ok, 3) if ok else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=2000)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--cold-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-out", default=None, help="write raw latencies and wall time as JSON")
    args = parser.parse_args()
    report = asyncio.run(run(args))
    if args.raw_out:
        Path(args.raw_out).write_text(json.dumps({"latencies_ms": report.pop("_latencies"),
                                                  "wall_seconds": report.pop("_wall"),
                                                  "errors": report["errors"],
                                                  "cache_hits": report.pop("_cache_hits")}))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
