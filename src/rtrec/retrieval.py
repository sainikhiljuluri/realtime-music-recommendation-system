"""FAISS candidate retrieval over item embeddings (inner product on normalized vectors)."""
from __future__ import annotations

import faiss
import numpy as np

faiss.omp_set_num_threads(1)  # see rtrec/__init__.py


def build_index(item_vectors: np.ndarray, n_lists: int = 0) -> faiss.Index:
    """Exact inner-product index by default. Set n_lists > 0 for an IVF index."""
    vectors = np.ascontiguousarray(item_vectors, dtype=np.float32)
    dim = vectors.shape[1]
    if n_lists <= 0:
        index = faiss.IndexFlatIP(dim)
    else:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_lists, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
    index.add(vectors)
    return index


def search(index: faiss.Index, query_vectors: np.ndarray, k: int, nprobe: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, item_ids) of shape (n_queries, k)."""
    if isinstance(index, faiss.IndexIVFFlat):
        index.nprobe = nprobe
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
    scores, ids = index.search(queries, k)
    return scores, ids


def top_unseen(index: faiss.Index, user_vec: np.ndarray, seen: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-n unseen candidates. Searching n + len(seen) guarantees n unseen results when available."""
    k = min(index.ntotal, n + len(seen))
    scores, ids = search(index, user_vec[None, :], k)
    mask = ~np.isin(ids[0], seen)
    return ids[0][mask][:n], scores[0][mask][:n]
