import numpy as np

from rtrec import retrieval


def _vectors(n=400, d=16, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_flat_index_matches_brute_force():
    vectors = _vectors()
    queries = _vectors(n=5, seed=1)
    index = retrieval.build_index(vectors)
    scores, ids = retrieval.search(index, queries, k=10)
    brute = np.argsort(-(queries @ vectors.T), axis=1)[:, :10]
    assert np.array_equal(ids, brute)


def test_ivf_with_all_lists_probed_is_exact():
    vectors = _vectors()
    queries = _vectors(n=5, seed=2)
    index = retrieval.build_index(vectors, n_lists=8)
    _, ids = retrieval.search(index, queries, k=10, nprobe=8)
    brute = np.argsort(-(queries @ vectors.T), axis=1)[:, :10]
    assert np.array_equal(ids, brute)


def test_top_unseen_excludes_seen_and_returns_n():
    vectors = _vectors()
    index = retrieval.build_index(vectors)
    seen = np.arange(0, 50)
    items, scores = retrieval.top_unseen(index, vectors[0], seen, n=20)
    assert len(items) == 20
    assert not np.isin(items, seen).any()
    assert np.all(np.diff(scores) <= 1e-6)
