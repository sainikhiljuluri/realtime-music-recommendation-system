"""Music4All-Onion experiment: chronological splits, two-tower + FAISS + LightGBM, baselines,
validation-only selection, and one test evaluation.

Time layout (cut points are quantiles of event time over the sample):

  [0, t60)            stage 1: two-tower training events and history for candidates
  [t60, t75)          ranker-training window: labels for LambdaRank, history = stage 1
  [t75, t85)          validation window: selection only, history = stage 1 + ranker window
  [t85, end]          test window: reported once, history = everything before t85

Nothing from a later window reaches an earlier stage's model or features.
"""
from __future__ import annotations

import json
import platform
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch  # imported before faiss (see rtrec/pipeline.py)
import faiss
import lightgbm as lgb  # noqa: F401  (ensures lightgbm loads with the same runtime settings)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from . import retrieval
from .evaluation import bootstrap_ci, coverage, diversity_rerank, genre_diversity, novelty, paired_diff_ci, self_information
from .metrics import ndcg_at_k_set, recall_at_k_set
from .pipeline import Config as _TrainCfg, encode_all, train_two_tower
from .ranking import History, candidate_features, generate_candidates, predict, train_ranker

torch.set_num_threads(1)


@dataclass(frozen=True)
class MusicConfig:
    seed: int = 0
    cut_quantiles: tuple = (0.60, 0.75, 0.85)
    epochs: int = 2
    batch_size: int = 2048
    learning_rate: float = 1e-3
    candidates: int = 100
    popular_candidates: int = 50
    ranker_users: int = 10_000
    lambdas: tuple = (0.0, 0.1, 0.2, 0.4)   # diversity penalty grid, chosen on validation
    svd_factors: int = 64
    k_values: tuple = (10, 20)


def _items_frame(n_items: int, item_genre: np.ndarray, n_genres: int) -> pd.DataFrame:
    genre = np.where(item_genre < 0, n_genres, item_genre).astype(np.int64)  # bucket n_genres = unknown
    return pd.DataFrame({"item_idx": np.arange(n_items), "genre": genre, "artist": np.zeros(n_items, dtype=np.int64)})


def _sets(df: pd.DataFrame) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for u, i in zip(df["user_idx"].to_numpy(), df["item_idx"].to_numpy()):
        out.setdefault(int(u), set()).add(int(i))
    return out


def _popularity_ranked(order: np.ndarray, seen: np.ndarray, k: int) -> list[int]:
    blocked = set(seen.tolist())
    out: list[int] = []
    for i in order:
        if int(i) not in blocked:
            out.append(int(i))
            if len(out) == k:
                break
    return out


def run(sample_events: pd.DataFrame, item_ids: np.ndarray, user_ids: np.ndarray, item_genre: np.ndarray,
        n_genres: int, cfg: MusicConfig, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # Restrict the catalog to tracks with at least one event, so coverage is not diluted by unreachable tracks.
    used, inverse = np.unique(sample_events["item_idx"].to_numpy(), return_inverse=True)
    item_ids, item_genre = item_ids[used], item_genre[used]
    ev = sample_events.assign(item_idx=inverse.astype(np.int64))
    n_users, n_items = len(user_ids), len(item_ids)
    t60, t75, t85 = (int(np.quantile(ev["ts"].to_numpy(), q)) for q in cfg.cut_quantiles)
    stage1 = ev[ev["ts"] < t60]
    rank_win = ev[(ev["ts"] >= t60) & (ev["ts"] < t75)]
    val = ev[(ev["ts"] >= t75) & (ev["ts"] < t85)]
    test = ev[ev["ts"] >= t85]
    items_df = _items_frame(n_items, item_genre, n_genres)
    genre_of = items_df["genre"].to_numpy()
    n_genre_buckets = n_genres + 1

    h60 = History.from_events(stage1, items_df, n_users, n_genre_buckets, 1)
    h75 = History.from_events(ev[ev["ts"] < t75], items_df, n_users, n_genre_buckets, 1)
    h85 = History.from_events(ev[ev["ts"] < t85], items_df, n_users, n_genre_buckets, 1)

    # --- stage 1: two-tower on stage-1 events only
    tt_cfg = _TrainCfg(epochs=cfg.epochs, batch_size=cfg.batch_size, learning_rate=cfg.learning_rate, seed=cfg.seed)
    model = train_two_tower(tt_cfg, stage1, h60, n_users, n_items)
    user_vecs, item_vecs = encode_all(model, h60)
    index = retrieval.build_index(item_vecs)

    # --- collaborative filtering baseline: truncated SVD on stage-1 play counts
    counts = csr_matrix((np.ones(len(stage1), dtype=np.float32), (stage1["user_idx"], stage1["item_idx"])),
                        shape=(n_users, n_items))
    counts.sum_duplicates()
    svd = TruncatedSVD(n_components=cfg.svd_factors, random_state=cfg.seed)
    svd_user = svd.fit_transform(counts)
    svd_item = svd.components_.T

    # --- stage 2: LambdaRank on ranker-training window
    rng = np.random.default_rng(cfg.seed)
    rank_targets = _sets(rank_win)
    stage1_users = np.unique(stage1["user_idx"].to_numpy())
    eligible = np.array([u for u in stage1_users if u in rank_targets])
    rank_users = np.sort(rng.choice(eligible, size=min(cfg.ranker_users, len(eligible)), replace=False))
    order60 = np.argsort(-h60.item_popularity, kind="stable")
    feats, labels, groups = [], [], []
    for u in rank_users:
        items, scores, ranks = generate_candidates(index, user_vecs[u], h60.seen(u), order60, item_vecs,
                                                   cfg.candidates, cfg.popular_candidates)
        if len(items) == 0:
            continue
        feats.append(candidate_features(h60, u, items, scores, ranks))
        labels.append(np.isin(items, list(rank_targets[u])).astype(np.int64))
        groups.append(len(items))
    booster = train_ranker(np.vstack(feats), np.concatenate(labels), groups, seed=cfg.seed)
    train_seconds = round(time.time() - started, 2)

    self_info = self_information(h60.item_popularity)

    def score_user(u: int, hist: History, order: np.ndarray, k_max: int) -> dict[str, list[int]]:
        seen = hist.seen(u)
        pop = _popularity_ranked(order, seen, k_max)
        retr_items, _ = retrieval.top_unseen(index, user_vecs[u], seen, cfg.candidates)
        items, scores, ranks = generate_candidates(index, user_vecs[u], seen, order, item_vecs,
                                                   cfg.candidates, cfg.popular_candidates)
        rank_scores = predict(booster, candidate_features(hist, u, items, scores, ranks))
        svd_scores = svd_item @ svd_user[u]
        svd_scores[seen] = -np.inf
        svd_top = [int(i) for i in np.argsort(-svd_scores, kind="stable")[:k_max]]
        out = {
            "popularity": pop,
            "svd_cf": svd_top,
            "retrieval": [int(i) for i in retr_items[:k_max]],
        }
        for lam in cfg.lambdas:
            out[f"hybrid_lambda_{lam:g}"] = diversity_rerank(items, rank_scores, genre_of, k_max, lam)
        return out

    def evaluate(window: pd.DataFrame, hist: History, order: np.ndarray, users_pool: np.ndarray,
                 extra_cold: bool) -> dict:
        targets = _sets(window)
        users = np.array(sorted(u for u in users_pool if u in targets))
        k_max = max(cfg.k_values)
        per = {}
        lists = {}
        for u in users:
            ranked_all = score_user(int(u), hist, order, k_max)
            for name, ranked in ranked_all.items():
                per.setdefault(name, {f"recall@{k}": [] for k in cfg.k_values} | {"ndcg@10": [], "genre_diversity@10": []})
                lists.setdefault(name, [])
                for k in cfg.k_values:
                    per[name][f"recall@{k}"].append(recall_at_k_set(ranked, targets[int(u)], k))
                per[name]["ndcg@10"].append(ndcg_at_k_set(ranked, targets[int(u)], 10))
                per[name]["genre_diversity@10"].append(genre_diversity(ranked[:10], genre_of))
                lists[name].append(ranked[:10])
        report = {}
        for name in per:
            report[name] = {
                "recall@10": _ci(per[name]["recall@10"], cfg.seed),
                "recall@20": _ci(per[name]["recall@20"], cfg.seed),
                "ndcg@10": _ci(per[name]["ndcg@10"], cfg.seed),
                "genre_diversity@10": float(np.mean(per[name]["genre_diversity@10"])),
                "novelty@10": float(np.mean([novelty(l, self_info) for l in lists[name]])),
                "coverage@10": coverage(lists[name], n_items),
            }
        return {"users_evaluated": int(len(users)), "models": report,
                "per_user": {name: {"recall@10": per[name]["recall@10"], "recall@20": per[name]["recall@20"],
                                    "ndcg@10": per[name]["ndcg@10"]} for name in per}, "cold_users": extra_cold}

    warm_val_pool = stage1_users
    val_result = evaluate(val, h75, np.argsort(-h75.item_popularity, kind="stable"), warm_val_pool, False)

    # --- selection on validation only: the diversity penalty with the best validation Recall@20
    lam_names = [f"hybrid_lambda_{lam:g}" for lam in cfg.lambdas]
    selected = max(lam_names, key=lambda n: val_result["models"][n]["recall@20"]["mean"])

    # --- one test evaluation with the selected configuration
    test_warm_pool = np.unique(ev[ev["ts"] < t85]["user_idx"].to_numpy())
    test_result = evaluate(test, h85, np.argsort(-h85.item_popularity, kind="stable"), test_warm_pool, False)
    all_test_users = set(np.unique(test["user_idx"].to_numpy()).tolist())
    cold_test_users = sorted(all_test_users - set(test_warm_pool.tolist()))
    cold_result = None
    if cold_test_users:
        targets = _sets(test)
        pop_top = _popularity_ranked(np.argsort(-h85.item_popularity, kind="stable"), np.empty(0, np.int64), 20)
        cold_result = {
            "users": len(cold_test_users),
            "popularity_recall@20": float(np.mean([recall_at_k_set(pop_top, targets[u], 20) for u in cold_test_users])),
        }

    paired = {}
    pu = test_result["per_user"]
    for a, b in ((selected, "popularity"), (selected, "svd_cf"), (selected, "retrieval"), (selected, "hybrid_lambda_0")):
        mean, lo, hi = paired_diff_ci(np.array(pu[a]["recall@20"]), np.array(pu[b]["recall@20"]), seed=cfg.seed)
        paired[f"recall@20_{a}_minus_{b}"] = {"mean": mean, "ci95": [lo, hi]}

    result = {
        "config": asdict(cfg),
        "time_cutoffs_unix": {"t60": t60, "t75": t75, "t85": t85},
        "events": {"stage1": int(len(stage1)), "ranker_window": int(len(rank_win)),
                   "validation": int(len(val)), "test": int(len(test))},
        "ranker_training": {"users": int(len(groups)), "rows": int(sum(groups))},
        "validation": {k: v for k, v in val_result.items() if k != "per_user"},
        "selected_on_validation": selected,
        "test": {k: v for k, v in test_result.items() if k != "per_user"},
        "test_paired_recall@20": paired,
        "cold_start_test": cold_result,
        "timing": {
            "train_and_rank_seconds": train_seconds,
            "total_seconds": round(time.time() - started, 2),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
                                 if platform.system() == "Darwin" else
                                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        },
        "platform": platform.platform(),
    }
    (out_dir / "music_metrics.json").write_text(json.dumps(result, indent=2))
    return result


def _ci(values: list[float], seed: int) -> dict:
    mean, lo, hi = bootstrap_ci(np.asarray(values, dtype=np.float64), seed=seed)
    return {"mean": mean, "ci95": [lo, hi]}
