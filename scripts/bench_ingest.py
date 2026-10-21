"""Measure listen-event ingestion through the load balancer: throughput, latency, and outcomes.

Each event gets a fresh event_id, so every request is applied. Events go to users from a user-id
file, and each user's events stay on one instance (user-affinity routing in deploy/nginx.conf).

  INGEST_API_KEY=... python scripts/bench_ingest.py --url http://127.0.0.1:8080 --users-file ids.txt --events 2000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
import uuid
from pathlib import Path

import httpx


async def main_async(args: argparse.Namespace) -> dict:
    users = Path(args.users_file).read_text().split()
    token = os.environ["INGEST_API_KEY"]
    rng = random.Random(args.seed)
    now = int(time.time())
    latencies: list[float] = []
    statuses: dict[int, int] = {}
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, timeout=10.0,
                                 headers={"Authorization": f"Bearer {token}"}) as client:
        async def one(i: int) -> None:
            user = rng.choice(users)
            body = {"event_id": f"bench-{uuid.uuid4().hex}", "track_id": args.track,
                    "timestamp": now - rng.randrange(0, 3600)}
            async with sem:
                started = time.perf_counter()
                resp = await client.post(f"/users/{user}/listens", json=body)
                latencies.append((time.perf_counter() - started) * 1000)
                statuses[resp.status_code] = statuses.get(resp.status_code, 0) + 1

        started = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(args.events)))
        wall = time.perf_counter() - started
    lat = sorted(latencies)

    def pct(p: float) -> float:
        return round(lat[min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))], 2)

    return {"events": args.events, "concurrency": args.concurrency, "status_counts": statuses,
            "throughput_eps": round(args.events / wall, 1),
            "latency_ms": {"mean": round(statistics.fmean(lat), 2), "p50": pct(50), "p95": pct(95), "p99": pct(99)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--users-file", required=True)
    parser.add_argument("--track", required=True, help="a catalog track id to listen to")
    parser.add_argument("--events", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(main_async(args)), indent=2))


if __name__ == "__main__":
    main()
