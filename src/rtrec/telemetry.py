"""In-process serving metrics, exposed in Prometheus text format at GET /metrics.

Labels are fixed sets (source, cache result, event outcome, HTTP status). Nothing is labeled
with a user or track id, so the number of series does not grow with traffic. Each process
exports its own counters. Prometheus should scrape every instance and aggregate with sum().
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque

import numpy as np

LATENCY_BUCKETS_SECONDS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


class Telemetry:
    def __init__(self, window: int = 5000) -> None:
        self._lock = threading.Lock()
        self._requests: Counter[str] = Counter()          # by source
        self._cache: Counter[str] = Counter()             # hit / miss
        self._events: Counter[str] = Counter()            # accepted / duplicate / rejected
        self._errors: Counter[str] = Counter()            # by HTTP status
        self._bucket_counts = [0] * (len(LATENCY_BUCKETS_SECONDS) + 1)
        self._latency_sum = 0.0
        self._latency_count = 0
        self._latency_ms: deque[float] = deque(maxlen=window)
        self._started = time.time()

    def observe(self, source: str, cache_hit: bool, latency_ms: float) -> None:
        with self._lock:
            self._requests[source] += 1
            self._cache["hit" if cache_hit else "miss"] += 1
            seconds = latency_ms / 1000
            self._latency_sum += seconds
            self._latency_count += 1
            self._bucket_counts[int(np.searchsorted(LATENCY_BUCKETS_SECONDS, seconds, side="left"))] += 1
            self._latency_ms.append(latency_ms)

    def event(self, outcome: str) -> None:
        with self._lock:
            self._events[outcome] += 1

    def error(self, status: int) -> None:
        with self._lock:
            self._errors[str(status)] += 1

    def snapshot(self) -> dict:
        """JSON view for /stats. Counts and timings only, no ids."""
        with self._lock:
            requests = dict(self._requests)
            cache = dict(self._cache)
            events = dict(self._events)
            latencies = np.asarray(self._latency_ms, dtype=np.float64)
        latency = {"window": int(latencies.size)}
        if latencies.size:
            p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
            latency |= {"p50_ms": round(float(p50), 3), "p95_ms": round(float(p95), 3),
                        "p99_ms": round(float(p99), 3)}
        hits, misses = cache.get("hit", 0), cache.get("miss", 0)
        total = hits + misses
        return {
            "uptime_seconds": round(time.time() - self._started, 1),
            "requests": requests,
            "cache": cache,
            "cache_hit_rate": round(hits / total, 4) if total else None,
            "listen_events": events,
            "latency": latency,
        }

    def prometheus(self, gauges: dict[str, float]) -> str:
        """Prometheus exposition text. `gauges` holds process state such as model_loaded."""
        with self._lock:
            requests = dict(self._requests)
            cache = dict(self._cache)
            events = dict(self._events)
            errors = dict(self._errors)
            buckets = list(self._bucket_counts)
            latency_sum, latency_count = self._latency_sum, self._latency_count
        lines: list[str] = []

        def counter(name: str, help_text: str, label: str, values: dict[str, int]) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            for key in sorted(values):
                lines.append(f'{name}{{{label}="{key}"}} {values[key]}')

        counter("rtrec_recommendation_requests_total", "Recommendation requests by source.", "source", requests)
        counter("rtrec_recommendation_cache_lookups_total", "Recommendation cache lookups by result.", "result", cache)
        counter("rtrec_listen_events_total", "Listen events by outcome.", "outcome", events)
        counter("rtrec_http_errors_total", "Error responses by HTTP status.", "status", errors)

        lines.append("# HELP rtrec_request_duration_seconds Recommendation request latency, measured in the handler.")
        lines.append("# TYPE rtrec_request_duration_seconds histogram")
        cumulative = 0
        for bound, count in zip(LATENCY_BUCKETS_SECONDS, buckets):
            cumulative += count
            lines.append(f'rtrec_request_duration_seconds_bucket{{le="{bound}"}} {cumulative}')
        cumulative += buckets[-1]
        lines.append(f'rtrec_request_duration_seconds_bucket{{le="+Inf"}} {cumulative}')
        lines.append(f"rtrec_request_duration_seconds_sum {latency_sum:.6f}")
        lines.append(f"rtrec_request_duration_seconds_count {latency_count}")

        for name, value in sorted(gauges.items()):
            lines.append(f"# TYPE rtrec_{name} gauge")
            lines.append(f"rtrec_{name} {value}")
        return "\n".join(lines) + "\n"
