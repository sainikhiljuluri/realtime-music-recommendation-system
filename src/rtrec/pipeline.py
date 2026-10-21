"""Offline pipeline: train the two-tower model, build the FAISS index, train the
LightGBM ranker, evaluate on held-out events, and save artifacts for serving.

Run: python -m rtrec.pipeline --out artifacts
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch  # imported before faiss: importing faiss first crashes torch on macOS (OpenMP runtimes)
import faiss
import numpy as np
import pandas as pd

from . import data, retrieval
from .artifacts import CONFIG_FILE, FAISS_FILE, HISTORY_FILE, METRICS_FILE, MODEL_FILE, RANKER_FILE, VECTORS_FILE
from .metrics import ndcg_at_k, recall_at_k
from .model import TwoTower
from .ranking import History, candidate_features, predict, train_ranker

@dataclass(frozen=True)
class Config:
    n_users: int = 2000
    n_items: int = 3000
    n_genres: int = 12
    n_artists: int = 400
    epochs: int = 5
    batch_size: int = 512
    learning_rate: float = 1e-3
    candidates: int = 100
    k: int = 10
    seed: int = 0


torch.set_num_threads(1)  # see rtrec/__init__.py


def train_two_tower(cfg: Config, train: pd.DataFrame, history: History) -> TwoTower:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = TwoTower(cfg.n_users, cfg.n_items, cfg.n_genres, cfg.n_artists)
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


def build_ranker_training(index, user_vecs, history: History, validation: pd.DataFrame, n: int):
    """Training rows come from validation events; history is train-only, so nothing leaks."""
    features, labels, groups = [], [], []
    for user, group in validation.groupby("user_idx"):
        target = int(group["item_idx"].iloc[0])
        items, scores = retrieval.top_unseen(index, user_vecs[user], history.seen(user), n)
        if len(items) == 0:
            continue
        features.append(candidate_features(history, user, items, scores))
        labels.append((items == target).astype(np.int64))
        groups.append(len(items))
    return np.vstack(features), np.concatenate(labels), groups


def evaluate(index, user_vecs, history: History, booster, test: pd.DataFrame, popularity: np.ndarray, cfg: Config) -> dict:
    """Held-out test positive per user. History covers train and validation events."""
    k = cfg.k
    order_by_popularity = np.argsort(-popularity, kind="stable")
    rows = {"popularity": ([], []), "retrieval": ([], []), "ranker": ([], [])}
    for user, target in zip(test["user_idx"].to_numpy(), test["item_idx"].to_numpy()):
        seen = history.seen(user)
        seen_set = set(seen.tolist())
        pop_top = [int(i) for i in order_by_popularity if int(i) not in seen_set][:k]
        items, scores = retrieval.top_unseen(index, user_vecs[user], seen, cfg.candidates)
        ret_top = [int(i) for i in items[:k]]
        rank_scores = predict(booster, candidate_features(history, user, items, scores))
        rank_top = [int(i) for i in items[np.argsort(-rank_scores, kind="stable")][:k]]
        for name, ranked in (("popularity", pop_top), ("retrieval", ret_top), ("ranker", rank_top)):
            rows[name][0].append(recall_at_k(ranked, int(target), k))
            rows[name][1].append(ndcg_at_k(ranked, int(target), k))
    return {
        "users_evaluated": len(test),
        "k": k,
        "candidates": cfg.candidates,
        "recall_at_k": {name: float(np.mean(v[0])) for name, v in rows.items()},
        "ndcg_at_k": {name: float(np.mean(v[1])) for name, v in rows.items()},
    }


def run(cfg: Config, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ds = data.generate(n_users=cfg.n_users, n_items=cfg.n_items, n_genres=cfg.n_genres,
                       n_artists=cfg.n_artists, seed=cfg.seed)
    train, validation, test = data.leave_two_out(ds.interactions)
    history_train = History.from_events(train, ds.items, ds.n_users, ds.n_genres, ds.n_artists)
    history_full = History.from_events(pd.concat([train, validation]), ds.items,
                                       ds.n_users, ds.n_genres, ds.n_artists)

    model = train_two_tower(cfg, train, history_train)
    user_vecs, item_vecs = encode_all(model, history_train)
    index = retrieval.build_index(item_vecs)

    feats, labels, groups = build_ranker_training(index, user_vecs, history_train, validation, cfg.candidates)
    booster = train_ranker(feats, labels, groups, seed=cfg.seed)

    metrics = evaluate(index, user_vecs, history_full, booster, test, history_full.item_popularity, cfg)
    metrics["ranker_training_rows"] = int(len(labels))
    metrics["build_seconds"] = round(time.time() - started, 2)

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
    parser.add_argument("--users", type=int, default=2000)
    parser.add_argument("--items", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = Config(n_users=args.users, n_items=args.items, epochs=args.epochs, seed=args.seed)
    print(json.dumps(run(cfg, Path(args.out)), indent=2))


if __name__ == "__main__":
    main()
