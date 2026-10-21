"""Run the Music4All-Onion experiment on a saved sample.

Build the sample first (streams the 2.2 GB log, not stored in the repository):
  python scripts/build_music_sample.py --data <music4all dir> --cache <cache file>

Then:
  python scripts/run_music.py --cache <cache file> --out docs/results/music4all-10k --export artifacts-music4all-10k

--export writes the serving artifacts the API loads (ARTIFACT_DIR=artifacts-music4all-10k).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtrec.music import load_saved, summary
from rtrec.music_pipeline import MusicConfig, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", default="docs/results/music4all-10k")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export", default=None, help="directory for serving artifacts (not committed)")
    args = parser.parse_args()
    sample = load_saved(Path(args.cache))
    cfg = MusicConfig(epochs=args.epochs, seed=args.seed)
    result = run(sample.events, sample.item_ids, sample.user_ids, sample.item_genre, sample.n_genres,
                 cfg, Path(args.out), export_dir=Path(args.export) if args.export else None)
    result["dataset"] = summary(sample)
    Path(args.out, "music_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"selected": result["selected_on_validation"],
                      "test": {m: round(v["recall@20"]["mean"], 4) for m, v in result["test"]["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
