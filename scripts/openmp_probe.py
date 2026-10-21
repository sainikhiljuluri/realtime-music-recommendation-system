"""Reproduce the OpenMP conflict between native libraries that share one process (macOS wheels).

Each pip wheel ships its own OpenMP runtime (libomp.dylib). Loading faiss and LightGBM into one
process, as the API does, is enough to trigger the conflict. The probe loads the libraries, then
runs LightGBM prediction with a chosen thread count.

  --libs serving   faiss + lightgbm (what the API process loads, torch is not imported)
  --libs all       torch + faiss + lightgbm + scikit-learn (what the offline pipeline loads)
  --threads N      LightGBM prediction threads. Crashes (exit 139) in our runs at N=4.

  python scripts/openmp_probe.py --libs serving --threads 4    # crashed 5/5
  python scripts/openmp_probe.py --libs serving --threads 1    # 0 failures
Compare with KMP_DUPLICATE_LIB_OK set and unset. See docs/FAILURES.md.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libs", choices=["serving", "all"], required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    if args.libs == "all":
        import torch  # noqa: F401
    import faiss
    import lightgbm as lgb
    import numpy as np
    if args.libs == "all":
        from sklearn.decomposition import TruncatedSVD
    rng = np.random.default_rng(0)
    x = rng.random((400, 6), dtype=np.float32)
    faiss.IndexFlatIP(6).add(x)
    if args.libs == "all":
        TruncatedSVD(2).fit_transform(x)
    booster = lgb.LGBMRanker(n_estimators=20, verbose=-1, n_jobs=1).fit(x, np.arange(400) % 2, group=[200, 200]).booster_
    booster.predict(x, num_threads=args.threads)
    print(f"ok libs={args.libs} threads={args.threads}")


if __name__ == "__main__":
    main()
