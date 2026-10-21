"""In-process serving metrics: request counters and latency percentiles over a rolling window.

Exposed as JSON at GET /metrics. The output holds counts and timings only, with no user ids.
Each process keeps its own numbers. Behind a load balancer, read every instance.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque

import numpy as np


class Telemetry:
    def __init__(self, window: int = 5000) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._latency_ms: deque[float] = deque(maxlen=window)
        self._started = time.time()

    def observe(self, source: str, cache_hit: bool, latency_ms: float) -> None:
        with self._lock:
            self._counts["requests_total"] += 1
            self._counts[f"requests_{source}"] += 1
            self._counts["cache_hits" if cache_hit else "cache_misses"] += 1
            self._latency_ms.append(latency_ms)

    def error(self, status: int) -> None:
        with self._lock:
            self._counts[f"errors_{status}"] += 1

    def snapshot(self) -> dict:
        with self._lock:
            counts = dict(self._counts)
            latencies = np.asarray(self._latency_ms, dtype=np.float64)
        latency = {"window": int(latencies.size)}
        if latencies.size:
            p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
            latency |= {"p50_ms": round(float(p50), 3), "p95_ms": round(float(p95), 3),
                        "p99_ms": round(float(p99), 3)}
        total = counts.get("requests_total", 0)
        hits = counts.get("cache_hits", 0)
        return {
            "uptime_seconds": round(time.time() - self._started, 1),
            "counters": counts,
            "cache_hit_rate": round(hits / total, 4) if total else None,
            "latency": latency,
        }
