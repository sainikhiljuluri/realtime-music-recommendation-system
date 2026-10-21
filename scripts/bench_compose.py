"""Benchmark the Compose stack through the load balancer.

Each run uses a user-id file (see scripts/sample_user_ids.py). A smaller pool gives a higher cache
hit ratio, and the script reports the hit ratio it actually observed. Optionally it stops one API
container partway through the run (SIGKILL) to measure the failure case.

  python scripts/bench_compose.py --url http://127.0.0.1:8080 --users-file ids.txt --requests 8000
  python scripts/bench_compose.py ... --kill rtrec-api2 --kill-after 4

Clients run on the same host as the stack, so these are localhost measurements.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_workers  # noqa: E402


def docker_stats() -> list[str]:
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}} {{.CPUPerc}}"],
                         capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.startswith("rtrec-")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--users-file", required=True)
    parser.add_argument("--requests", type=int, default=8000)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--kill", default=None, help="container to SIGKILL during the run")
    parser.add_argument("--kill-after", type=float, default=4.0, help="seconds into the run")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ns = argparse.Namespace(clients=args.clients, requests=args.requests, warmup=args.warmup,
                            concurrency=args.concurrency, users=0, user_ids_file=args.users_file)
    events: list[dict] = []
    if args.kill:
        def killer() -> None:
            time.sleep(args.kill_after)
            subprocess.run(["docker", "kill", args.kill], capture_output=True)
            events.append({"t": args.kill_after, "event": f"killed {args.kill}"})
        threading.Thread(target=killer, daemon=True).start()

    started = time.time()
    result = bench_workers.run_clients(args.url, ns, workers=0)
    result = {
        "label": args.label,
        "user_pool": sum(1 for _ in Path(args.users_file).read_text().split()),
        **{k: result[k] for k in ["requests", "errors", "error_rate", "throughput_rps", "latency_ms", "cache_hit_rate"]},
        "wall_seconds": round(time.time() - started, 1),
        "events": events,
        "docker_stats_after": docker_stats(),
    }
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
