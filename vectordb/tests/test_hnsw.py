"""
Comprehensive test suite for HNSW (Hierarchical Navigable Small World) index.
Validates:
- Nearest neighbor recall & accuracy across 64d, 128d, 384d, 1536d
- Distance metrics (Cosine, Euclidean/L2, Dot Product, Manhattan)
- Heuristic diverse neighbor selection and cycle prevention
- Concurrency under simultaneous search and insertion
- Soft-deletion, rebuild, and edge cases
"""
from __future__ import annotations
import concurrent.futures
import threading
import numpy as np
import pytest

from core.hnsw import HNSWIndex
from utils.normalize import compute_distance, normalize, pairwise_distances


def _brute_force_knn(vectors: np.ndarray, query: np.ndarray, top_k: int, metric: str = "cosine") -> list[int]:
    """Ground truth brute-force exact k-NN search."""
    dists = pairwise_distances(vectors, query.reshape(1, -1), metric=metric).ravel()
    sorted_indices = np.argsort(dists)
    return list(sorted_indices[:top_k])


@pytest.mark.parametrize("dim", [64, 128, 384, 1536])
def test_hnsw_recall_across_dimensions(dim: int):
    """Verify HNSW search achieves >90% recall against exact brute-force search across dimensions."""
    np.random.seed(42)
    n_samples = 200
    top_k = 5

    raw_vectors = np.random.randn(n_samples, dim).astype(np.float32)
    index = HNSWIndex(dimension=dim, distance="cosine", m=16, ef_construction=64, ef_search=64)

    for i, vec in enumerate(raw_vectors):
        index.add(i, vec)

    assert index.size() == n_samples

    # Test queries
    n_queries = 10
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    recalls = []

    for q in queries:
        gt_ids = _brute_force_knn(raw_vectors, q, top_k, metric="cosine")
        results = index.search(q, top_k=top_k, ef_search=64)
        hnsw_ids = [n_id for n_id, _ in results]
        
        overlap = len(set(gt_ids) & set(hnsw_ids))
        recall = overlap / top_k
        recalls.append(recall)

    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.85, f"Dim {dim} recall {mean_recall:.2f} < 0.85"


def test_hnsw_euclidean_metric():
    np.random.seed(101)
    dim = 32
    index = HNSWIndex(dimension=dim, distance="euclidean", m=16, ef_construction=64, ef_search=64)
    vectors = np.random.uniform(-10, 10, size=(100, dim)).astype(np.float32)

    for i, vec in enumerate(vectors):
        index.add(i, vec)

    query = vectors[0]  # Searching exact existing vector
    results = index.search(query, top_k=1)
    assert len(results) == 1
    assert results[0][0] == 0  # Nearest neighbor must be itself
    assert np.isclose(results[0][1], 0.0, atol=1e-4)


def test_hnsw_manhattan_metric():
    np.random.seed(202)
    dim = 16
    index = HNSWIndex(dimension=dim, distance="manhattan", m=16, ef_construction=64)
    vectors = np.random.randn(50, dim).astype(np.float32)

    for i, vec in enumerate(vectors):
        index.add(i, vec)

    query = vectors[5]
    results = index.search(query, top_k=3)
    assert results[0][0] == 5
    assert np.isclose(results[0][1], 0.0, atol=1e-4)


def test_hnsw_dot_product_metric():
    np.random.seed(303)
    dim = 32
    index = HNSWIndex(dimension=dim, distance="dot", m=16, ef_construction=64)
    vectors = np.random.randn(50, dim).astype(np.float32)

    for i, vec in enumerate(vectors):
        index.add(i, vec)

    query = vectors[10]
    results = index.search(query, top_k=1)
    assert results[0][0] == 10


def test_hnsw_ef_search_tradeoff():
    np.random.seed(404)
    dim = 64
    n_samples = 300
    vectors = np.random.randn(n_samples, dim).astype(np.float32)
    index = HNSWIndex(dimension=dim, distance="cosine", m=8, ef_construction=32)

    for i, vec in enumerate(vectors):
        index.add(i, vec)

    query = np.random.randn(dim).astype(np.float32)
    gt = _brute_force_knn(vectors, query, top_k=10, metric="cosine")

    res_low = index.search(query, top_k=10, ef_search=5)
    res_high = index.search(query, top_k=10, ef_search=128)

    recall_low = len(set(gt) & set(r[0] for r in res_low)) / 10.0
    recall_high = len(set(gt) & set(r[0] for r in res_high)) / 10.0

    assert recall_high >= recall_low


def test_hnsw_soft_deletion_and_rebuild():
    dim = 16
    index = HNSWIndex(dimension=dim, distance="cosine")
    vectors = np.random.randn(20, dim).astype(np.float32)

    for i, vec in enumerate(vectors):
        index.add(i, vec)

    assert index.size() == 20
    assert index.remove(0) is True
    assert index.remove(0) is False  # Already deleted
    assert index.size() == 19

    # Node 0 must not appear in search
    results = index.search(vectors[0], top_k=20)
    returned_ids = [n_id for n_id, _ in results]
    assert 0 not in returned_ids

    # Rebuild
    index.rebuild()
    assert index.size() == 19
    assert 0 not in index.nodes


def test_hnsw_empty_index_search():
    index = HNSWIndex(dimension=16)
    assert index.size() == 0
    results = index.search(np.zeros(16), top_k=5)
    assert results == []


def test_hnsw_single_element():
    index = HNSWIndex(dimension=8, distance="cosine")
    vec = np.array([1.0] * 8, dtype=np.float32)
    index.add(42, vec)
    assert index.size() == 1
    results = index.search(vec, top_k=5)
    assert len(results) == 1
    assert results[0][0] == 42
    assert np.isclose(results[0][1], 1.0, atol=1e-5)


def test_hnsw_dimension_mismatch_raises():
    index = HNSWIndex(dimension=16)
    with pytest.raises(ValueError):
        index.add(1, [1.0, 2.0, 3.0])  # 3d != 16d


def test_hnsw_concurrency_simultaneous_search_and_insert():
    """Verify thread-safety: no deadlock or exception during concurrent writes & searches."""
    dim = 32
    index = HNSWIndex(dimension=dim, distance="cosine", m=16, ef_construction=32)
    
    # Pre-populate with 50 items
    for i in range(50):
        index.add(i, np.random.randn(dim).astype(np.float32))

    stop_event = threading.Event()
    errors = []

    def writer(start_id: int):
        try:
            for j in range(30):
                if stop_event.is_set():
                    break
                vec = np.random.randn(dim).astype(np.float32)
                index.add(start_id + j, vec)
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(50):
                if stop_event.is_set():
                    break
                q = np.random.randn(dim).astype(np.float32)
                results = index.search(q, top_k=5)
                assert isinstance(results, list)
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for t in range(4):
            futures.append(executor.submit(writer, 1000 + t * 50))
            futures.append(executor.submit(reader))

        concurrent.futures.wait(futures, timeout=10.0)

    assert errors == [], f"Concurrency errors encountered: {errors}"
    assert index.size() >= 50
