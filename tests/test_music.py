import bz2

import numpy as np
import pandas as pd
import pytest

from rtrec import music
from rtrec.evaluation import diversity_rerank, novelty, self_information
from rtrec.metrics import ndcg_at_k_set, recall_at_k_set
from rtrec.music import MusicSample, load_sample, stratified_user_sample
from rtrec.music_pipeline import MusicConfig, run


def _write_fixture(root):
    tracks = [f"t{i:03d}" for i in range(12)]
    genres = pd.DataFrame(np.zeros((12, 4)), index=pd.Index(tracks, name="id"), columns=["g0", "g1", "g2", "g3"])
    for i, t in enumerate(tracks):
        genres.loc[t, f"g{i % 4}"] = 1.0 + i * 0.01
    with bz2.open(root / "id_genres_tf-idf.tsv.bz2", "wt") as f:
        genres.to_csv(f, sep="\t")
    rng = np.random.default_rng(0)
    rows = ["user_id\ttrack_id\ttimestamp"]
    for user in range(1, 31):
        n = 20 + (user % 7) * 5
        base = pd.Timestamp("2014-01-01")
        for e in range(n):
            t = tracks[int(rng.integers(0, 12))]
            ts = (base + pd.Timedelta(hours=int(rng.integers(0, 24 * 300)))).strftime("%Y-%m-%d %H:%M:%S")
            rows.append(f"{user}\t{t}\t{ts}")
    with bz2.open(root / "userid_trackid_timestamp.tsv.bz2", "wt") as f:
        f.write("\n".join(rows) + "\n")


def test_stratified_sample_is_deterministic_and_sized():
    counts = pd.Series(np.arange(100, 1100), index=np.arange(1000))
    a = stratified_user_sample(counts, n_users=50, min_events=0, n_strata=5, seed=3)
    b = stratified_user_sample(counts, n_users=50, min_events=0, n_strata=5, seed=3)
    assert len(a) == 50 and np.array_equal(a, b)


def test_stratified_sample_covers_activity_range():
    counts = pd.Series(np.arange(100, 1100), index=np.arange(1000))
    chosen = stratified_user_sample(counts, n_users=100, min_events=0, n_strata=10, seed=0)
    chosen_counts = counts.loc[chosen]
    assert chosen_counts.min() < 200 and chosen_counts.max() > 1000 - 100


def test_loader_reads_bz2_and_keeps_chronological_order(tmp_path):
    _write_fixture(tmp_path)
    sample = load_sample(tmp_path, n_users=10, min_events=10, n_strata=5, seed=0)
    assert len(sample.user_ids) == 10
    assert sample.n_genres == 4
    ev = sample.events
    assert ev.groupby("user_idx")["ts"].apply(lambda s: s.is_monotonic_increasing).all()
    assert ev["item_idx"].max() < len(sample.item_ids)


def test_metrics_for_set_relevance():
    assert recall_at_k_set([1, 2, 3, 4], {2, 9}, k=2) == 0.5
    assert recall_at_k_set([1, 2, 3, 4], {2, 9}, k=4) == 0.5
    assert ndcg_at_k_set([2, 9, 1], {2, 9}, k=3) == pytest.approx((1 + 1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert ndcg_at_k_set([5, 6], {2}, k=2) == 0.0


def test_novelty_prefers_unpopular_items():
    counts = np.array([1000.0, 10.0, 0.0])
    si = self_information(counts)
    assert si[2] > si[1] > si[0]
    assert novelty([2], si) > novelty([0], si)


def test_diversity_rerank_breaks_up_repeated_genres():
    items = np.array([0, 1, 2, 3])
    scores = np.array([0.9, 0.85, 0.8, 0.5])
    genres = np.array([0, 0, 0, 1])          # items 0, 1, 2 share genre 0
    assert diversity_rerank(items, scores, genres, k=2, lam=0.0) == [0, 1]
    assert diversity_rerank(items, scores, genres, k=2, lam=0.4) == [0, 3]


def test_music_pipeline_end_to_end_on_tiny_sample(tmp_path):
    rng = np.random.default_rng(0)
    n_users, n_items = 60, 80
    rows = []
    for u in range(n_users):
        for e in range(40):
            rows.append((u, int(rng.integers(0, n_items)), 1_400_000_000 + e * 3600 + u))
    events = pd.DataFrame(rows, columns=["user_idx", "item_idx", "ts"]).sort_values(["user_idx", "ts"])
    item_genre = rng.integers(0, 5, size=n_items)
    cfg = MusicConfig(epochs=1, batch_size=256, candidates=30, popular_candidates=10, ranker_users=40,
                      lambdas=(0.0, 0.2), svd_factors=8)
    result = run(events, np.array([f"x{i}" for i in range(n_items)], dtype=object),
                 np.arange(n_users), item_genre, 5, cfg, tmp_path)
    assert result["selected_on_validation"] in {"hybrid_lambda_0", "hybrid_lambda_0.2"}
    assert result["events"]["test"] > 0
    assert (tmp_path / "music_metrics.json").exists()


def test_split_windows_do_not_overlap_in_time(tmp_path):
    rng = np.random.default_rng(1)
    events = pd.DataFrame({"user_idx": rng.integers(0, 20, 2000), "item_idx": rng.integers(0, 30, 2000),
                           "ts": rng.integers(0, 10**6, 2000)}).sort_values(["user_idx", "ts"])
    result = run(events, np.array([f"x{i}" for i in range(30)], dtype=object), np.arange(20),
                 rng.integers(0, 3, 30), 3,
                 MusicConfig(epochs=1, batch_size=128, candidates=20, popular_candidates=5, ranker_users=10,
                             lambdas=(0.0,), svd_factors=4), tmp_path)
    cuts = result["time_cutoffs_unix"]
    assert cuts["t60"] < cuts["t75"] < cuts["t85"]
    assert sum(result["events"].values()) == len(events)
