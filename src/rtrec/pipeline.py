"""Offline pipeline: train the two-tower model, build the FAISS index, train the LightGBM
ranker, evaluate on held-out events, and save artifacts for serving.

Run:
  python -m rtrec.pipeline --out artifacts                           # synthetic data
  python -m rtrec.pipeline --dataset movielens --data ml-25m --out artifacts-ml25m
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch  # imported before faiss: importing faiss first crashes torch on macOS (OpenMP)

from . import data, movielens, retrieval
from .artifacts import CONFIG_FILE, FAISS_FILE, HISTORY_FILE, METRICS_FILE, MODEL_FILE, RANKER_FILE, VECTORS_FILE
from .evaluation import bootstrap_ci, coverage, genre_diversity, paired_diff_ci
from .metrics import ndcg_at_k, recall_at_k
from .model import TwoTower
from .ranking import History, candidate_features, generate_candidates, predict, train_ranker

torch.set_num_threads(1)  # see rtrec/__init__.py


@dataclass(frozen=True)
class Config:
    dataset: str = "synthetic"          # "synthetic" or "movielens"
    data_path: str | None = None        # MovieLens directory when dataset == "movielens"
    min_rating: float = 4.0
    n_users: int = 2000                 # synthetic only
    n_items: int = 3000                 # synthetic only
    n_genres: int = 12                  # synthetic only
    n_artists: int = 400                # synthetic only
    epochs: int = 15
    batch_size: int = 512
    learning_rate: float = 1e-3
    candidates: int = 100               # FAISS candidates per user
    popular_candidates: int = 50        # popular unseen items added to the pool
    k: int = 10
    ranker_users: int = 20000           # users sampled for ranker training
    eval_users: int = 20000             # users sampled for evaluation
    seed: int = 0


def train_two_tower(cfg: Config, train: pd.DataFrame, history: History, n_users: int, n_items: int) -> TwoTower:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = TwoTower(n_users, n_items, history.genre_profile.shape[1], history.artist_counts.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    users = train["user_idx"].to_numpy()
    items = train["item_idx"].to_numpy()
    genre_profile = torch.as_tensor(history.genre_profile)
    item_genre = torch.as_tensor(history.item_genre)
    item_artist = torch.as_tensor(history.item_artist)
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(users))
        total, batches = 0.0, 0
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start + cfg.batch_size]
            if len(idx) < 2:
                continue
            u = torch.as_tensor(users[idx])
            i = torch.as_tensor(items[idx])
            user_vec = model.encode_user(u, genre_profile[u])
            item_vec = model.encode_item(i, item_genre[i], item_artist[i])
            loss = model.in_batch_softmax_loss(user_vec, item_vec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        print(f"epoch {epoch + 1}/{cfg.epochs} loss={total / max(batches, 1):.4f}", flush=True)
    return model


def encode_all(model: TwoTower, history: History) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        users = model.encode_user(torch.arange(history.genre_profile.shape[0]),
                                  torch.as_tensor(history.genre_profile))
        items = model.encode_item(torch.arange(history.item_genre.shape[0]),
                                  torch.as_tensor(history.item_genre),
                                  torch.as_tensor(history.item_artist))
    return users.numpy().astype(np.float32), items.numpy().astype(np.float32)


def build_ranker_training(index, user_vecs, item_vecs, history: History, validation: pd.DataFrame,
                          users: np.ndarray, cfg: Config):
    """Training rows come from validation events; history is train-only, so nothing leaks."""
    target_of = validation.drop_duplicates("user_idx").set_index("user_idx")["item_idx"]
    order = np.argsort(-history.item_popularity, kind="stable")
    features, labels, groups = [], [], []
    for user in users:
        target = int(target_of.loc[user])
        items, scores, ranks = generate_candidates(index, user_vecs[user], history.seen(user), order,
                                                   item_vecs, cfg.candidates, cfg.popular_candidates)
        if len(items) == 0:
            continue
        features.append(candidate_features(history, user, items, scores, ranks))
        labels.append((items == target).astype(np.int64))
        groups.append(len(items))
    return np.vstack(features), np.concatenate(labels), groups


def evaluate(index, user_vecs, item_vecs, history: History, booster, test: pd.DataFrame, users: np.ndarray,
             n_items: int, cfg: Config) -> dict:
    """One held-out positive per user. History covers train and validation events.

    Models: random unseen items, popularity, FAISS retrieval only, and the full pipeline
    (hybrid candidates reranked by LightGBM).
    """
    k = cfg.k
    target_of = test.drop_duplicates("user_idx").set_index("user_idx")["item_idx"]
    popularity_order = np.argsort(-history.item_popularity, kind="stable")
    rng = np.random.default_rng(cfg.seed)
    per_user = {name: {"recall": [], "ndcg": [], "diversity": []} for name in ("random", "popularity", "retrieval", "ranker")}
    lists = {name: [] for name in per_user}
    for user in users:
        target = int(target_of.loc[user])
        seen = history.seen(user)
        seen_set = set(seen.tolist())
        unseen_all = np.setdiff1d(np.arange(n_items), seen, assume_unique=True)
        retr_items, _ = retrieval.top_unseen(index, user_vecs[user], seen, cfg.candidates)
        items, scores, ranks = generate_candidates(index, user_vecs[user], seen, popularity_order,
                                                   item_vecs, cfg.candidates, cfg.popular_candidates)
        rank_scores = predict(booster, candidate_features(history, user, items, scores, ranks))
        ranked = {
            "random": [int(i) for i in rng.choice(unseen_all, size=k, replace=False)],
            "popularity": [int(i) for i in popularity_order if int(i) not in seen_set][:k],
            "retrieval": [int(i) for i in retr_items[:k]],
            "ranker": [int(i) for i in items[np.argsort(-rank_scores, kind="stable")][:k]],
        }
        for name, lst in ranked.items():
            per_user[name]["recall"].append(recall_at_k(lst, target, k))
            per_user[name]["ndcg"].append(ndcg_at_k(lst, target, k))
            per_user[name]["diversity"].append(genre_diversity(lst, history.item_genre))
            lists[name].append(lst)
    out = {}
    for name in per_user:
        recall_mean, r_lo, r_hi = bootstrap_ci(np.array(per_user[name]["recall"]), seed=cfg.seed)
        ndcg_mean, n_lo, n_hi = bootstrap_ci(np.array(per_user[name]["ndcg"]), seed=cfg.seed)
        out[name] = {
            "recall_at_k": {"mean": recall_mean, "ci95": [r_lo, r_hi]},
            "ndcg_at_k": {"mean": ndcg_mean, "ci95": [n_lo, n_hi]},
            "diversity": float(np.mean(per_user[name]["diversity"])),
            "coverage": coverage(lists[name], n_items),
        }
    paired = {}
    for a, b in (("ranker", "retrieval"), ("ranker", "popularity"), ("retrieval", "popularity")):
        mean, lo, hi = paired_diff_ci(np.array(per_user[a]["recall"]), np.array(per_user[b]["recall"]), seed=cfg.seed)
        paired[f"recall_{a}_minus_{b}"] = {"mean": mean, "ci95": [lo, hi]}
    return {"k": k, "candidates": cfg.candidates, "users_evaluated": len(users),
            "models": out, "paired_recall_differences": paired}


def load_dataset(cfg: Config):
    if cfg.dataset == "movielens":
        if not cfg.data_path:
            raise ValueError("--data is required for the movielens dataset")
        return movielens.load(cfg.data_path, min_rating=cfg.min_rating)
    if cfg.dataset == "synthetic":
        return data.generate(n_users=cfg.n_users, n_items=cfg.n_items, n_genres=cfg.n_genres,
                             n_artists=cfg.n_artists, seed=cfg.seed)
    raise ValueError(f"unknown dataset: {cfg.dataset}")


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024, 1)


def run(cfg: Config, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ds = load_dataset(cfg)
    train, validation, test = data.leave_two_out(ds.interactions)
    history_train = History.from_events(train, ds.items, ds.n_users, ds.n_genres, ds.n_artists)
    history_full = History.from_events(pd.concat([train, validation]), ds.items,
                                       ds.n_users, ds.n_genres, ds.n_artists)
    print(f"dataset={cfg.dataset} users={ds.n_users} items={ds.n_items} "
          f"events={len(ds.interactions)} train={len(train)} val={len(validation)} test={len(test)}", flush=True)

    t0 = time.time()
    model = train_two_tower(cfg, train, history_train, ds.n_users, ds.n_items)
    user_vecs, item_vecs = encode_all(model, history_train)
    index = retrieval.build_index(item_vecs)
    rng = np.random.default_rng(cfg.seed)
    all_users = np.sort(validation["user_idx"].unique())
    rank_users = rng.choice(all_users, size=min(cfg.ranker_users, len(all_users)), replace=False)
    feats, labels, groups = build_ranker_training(index, user_vecs, item_vecs, history_train, validation, rank_users, cfg)
    booster = train_ranker(feats, labels, groups, seed=cfg.seed)
    train_seconds = round(time.time() - t0, 2)

    eval_pool = np.sort(test["user_idx"].unique())
    eval_users = rng.choice(eval_pool, size=min(cfg.eval_users, len(eval_pool)), replace=False)
    metrics = evaluate(index, user_vecs, item_vecs, history_full, booster, test, eval_users, ds.n_items, cfg)
    metrics["dataset"] = {
        "name": cfg.dataset,
        "users": ds.n_users,
        "items": ds.n_items,
        "events_total": int(len(ds.interactions)),
        "events_train": int(len(train)),
        "events_validation": int(len(validation)),
        "events_test": int(len(test)),
    }
    metrics["training"] = {
        "two_tower_and_ranker_seconds": train_seconds,
        "ranker_training_users": int(len(rank_users)),
        "ranker_training_rows": int(len(labels)),
        "peak_rss_mb": _peak_rss_mb(),
        "total_seconds": round(time.time() - started, 2),
        "platform": platform.platform(),
    }

    torch.save(model.state_dict(), out_dir / MODEL_FILE)
    faiss.write_index(index, str(out_dir / FAISS_FILE))
    booster.save_model(str(out_dir / RANKER_FILE))
    np.savez(out_dir / VECTORS_FILE, user_vecs=user_vecs, item_vecs=item_vecs)
    np.savez(out_dir / HISTORY_FILE,
             item_genre=history_full.item_genre,
             item_artist=history_full.item_artist,
             item_popularity=history_full.item_popularity,
             genre_profile=history_full.genre_profile,
             artist_counts=history_full.artist_counts,
             activity=history_full.activity,
             seen_indptr=history_full.seen_indptr,
             seen_indices=history_full.seen_indices)
    (out_dir / CONFIG_FILE).write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the recommender.")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "movielens"])
    parser.add_argument("--data", default=None, help="MovieLens 25M directory")
    parser.add_argument("--users", type=int, default=2000)
    parser.add_argument("--items", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--ranker-users", type=int, default=20000)
    parser.add_argument("--popular-candidates", type=int, default=50)
    parser.add_argument("--eval-users", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = Config(dataset=args.dataset, data_path=args.data, n_users=args.users, n_items=args.items,
                 epochs=args.epochs, ranker_users=args.ranker_users, eval_users=args.eval_users,
                 popular_candidates=args.popular_candidates, seed=args.seed)
    print(json.dumps(run(cfg, Path(args.out)), indent=2))


if __name__ == "__main__":
    main()
