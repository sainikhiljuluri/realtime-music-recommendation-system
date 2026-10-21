"""Stream the Music4All-Onion log and save a stratified 10,000-user sample to a cache file."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rtrec.music import load_sample, save_sample, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="directory with the Music4All-Onion files")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--users", type=int, default=10_000)
    parser.add_argument("--min-events", type=int, default=50)
    parser.add_argument("--strata", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    sample = load_sample(Path(args.data), n_users=args.users, min_events=args.min_events,
                         n_strata=args.strata, seed=args.seed)
    save_sample(sample, Path(args.cache))
    info = summary(sample)
    info["load_seconds"] = round(time.time() - started, 1)
    Path(args.cache).with_suffix(".summary.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
