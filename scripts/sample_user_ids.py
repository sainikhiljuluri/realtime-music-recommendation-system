"""Write a seeded sample of trained Music4All user ids for load tests (one id per line).

  python scripts/sample_user_ids.py --artifacts artifacts-music4all-10k --count 2000 --out <file>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rtrec.artifacts import CATALOG_FILE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    catalog = np.load(Path(args.artifacts) / CATALOG_FILE)
    trained = catalog["user_ids"][catalog["trained_users"]]
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(trained, size=min(args.count, len(trained)), replace=False)
    Path(args.out).write_text("\n".join(str(int(u)) for u in chosen) + "\n")
    print(f"wrote {len(chosen)} trained user ids to {args.out}")


if __name__ == "__main__":
    main()
