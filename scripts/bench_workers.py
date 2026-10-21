"""Benchmark the API at 1, 2, and 4 uvicorn workers under identical load.

Each configuration starts a fresh server, waits for /ready, runs the same seeded load test,
and records resident memory of the server process tree. Results are written as JSON.

Example:
  python scripts/bench_workers.py --artifacts artifacts-ml25m --requests 4000 --concurrency 16 --out bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rss_mb(pid: int) -> float:
    """Resident memory of a process and its children, in MB."""
    pids = [pid]
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    pids += [int(p) for p in out]
    total_kb = 0
    for p in pids:
        r = subprocess.run(["ps", "-o", "rss=", "-p", str(p)], capture_output=True, text=True).stdout.strip()
        total_kb += int(r) if r else 0
    return round(total_kb / 1024, 1)


def wait_ready(url: str, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/ready", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("server did not become ready")


def run_clients(url: str, args: argparse.Namespace, workers: int) -> dict:
    """Run several load-generator processes at once and merge their raw latencies.

    One Python client tops out near 600 req/s, below what a 2- or 4-worker server can serve,
    so a single client would hide server scaling. Merged percentiles use all raw latencies.
    """
    procs = max(1, args.clients)
    per_proc_req = args.requests // procs
    per_proc_conc = max(1, args.concurrency // procs)
    raw_files, children = [], []
    for i in range(procs):
        raw = ROOT / f".bench-raw-{workers}-{i}.json"
        raw_files.append(raw)
        children.append(subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "loadtest.py"), "--url", url,
             "--users", str(args.users), "--requests", str(per_proc_req),
             "--warmup", str(args.warmup if i == 0 else 0),
             "--concurrency", str(per_proc_conc), "--seed", str(7 + i), "--raw-out", str(raw)]
            + (["--user-ids-file", args.user_ids_file] if args.user_ids_file else []),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
    for child in children:
        child.wait()
    latencies, wall, errors, hits = [], 0.0, 0, 0
    for raw in raw_files:
        data = json.loads(raw.read_text())
        latencies += data["latencies_ms"]
        wall = max(wall, data["wall_seconds"])
        errors += data["errors"]
        hits += data["cache_hits"]
        raw.unlink()
    lat = sorted(latencies)

    def pct(p: float) -> float:
        return round(lat[min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))], 2) if lat else float("nan")

    ok = len(lat)
    return {
        "clients": procs,
        "requests": ok + errors,
        "errors": errors,
        "error_rate": round(errors / max(1, ok + errors), 4),
        "throughput_rps": round(ok / wall, 1) if wall > 0 else float("nan"),
        "latency_ms": {"mean": round(sum(lat) / ok, 2) if ok else None,
                       "p50": pct(50), "p95": pct(95), "p99": pct(99)},
        "cache_hit_rate": round(hits / ok, 3) if ok else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--workers", default="1,2,4")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--requests", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--users", type=int, default=0, help="synthetic u<n> ids when no id file is given")
    parser.add_argument("--user-ids-file", default=None, help="whitespace-separated real user ids to request")
    parser.add_argument("--clients", type=int, default=4, help="load-generator processes")
    parser.add_argument("--redis-url", default=None, help="enable the Redis cache; flushed before each run")
    parser.add_argument("--out", default="bench.json")
    args = parser.parse_args()

    env = {**os.environ, "ARTIFACT_DIR": args.artifacts, "PYTHONPATH": str(ROOT / "src"),
           "PSEUDONYM_KEY": "benchmark-key", "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}
    env.pop("REDIS_URL", None)
    if args.redis_url:
        env["REDIS_URL"] = args.redis_url
    results = []
    for workers in [int(w) for w in args.workers.split(",")]:
        port = args.port + workers
        if args.redis_url:
            import redis as redis_lib
            redis_lib.Redis.from_url(args.redis_url).flushall()
        url = f"http://127.0.0.1:{port}"
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "rtrec.api:app", "--port", str(port),
             "--workers", str(workers), "--log-level", "warning"],
            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            wait_ready(url)
            time.sleep(2)
            report = run_clients(url, args, workers)
            report["workers"] = workers
            report["rss_mb_total"] = rss_mb(server.pid)
            results.append(report)
            print(f"workers={workers} rps={report['throughput_rps']} p99={report['latency_ms']['p99']} "
                  f"errors={report['errors']} rss_mb={report['rss_mb_total']}", flush=True)
        finally:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
