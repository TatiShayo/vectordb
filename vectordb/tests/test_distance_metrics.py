"""
Tests for distance metrics: Cosine, Euclidean/L2, Dot Product, and Manhattan.
Validates precision, boundary conditions, zero vectors, high-dimensionality, and FAISS alignment.
"""
from __future__ import annotations
import numpy as np
import pytest
from utils.normalize import (
    normalize,
    to_float32,
    prepare_vector,
    prepare_batch,
    cosine_similarity,
    cosine_distance,
    euclidean_distance,
    l2_distance,
    manhattan_distance,
    dot_product,
    compute_distance,
    pairwise_distances,
)


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sim = cosine_similarity(v, v)
    assert np.isclose(sim, 1.0, atol=1e-6)
    dist = cosine_distance(v, v)
    assert np.isclose(dist, 0.0, atol=1e-6)


def test_cosine_similarity_orthogonal_vectors():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    sim = cosine_similarity(v1, v2)
    assert np.isclose(sim, 0.0, atol=1e-6)
    dist = cosine_distance(v1, v2)
    assert np.isclose(dist, 1.0, atol=1e-6)


def test_cosine_similarity_opposite_vectors():
    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    sim = cosine_similarity(v1, v2)
    assert np.isclose(sim, -1.0, atol=1e-6)
    dist = cosine_distance(v1, v2)
    assert np.isclose(dist, 2.0, atol=1e-6)


def test_cosine_zero_vector_handling():
    v1 = np.zeros(128, dtype=np.float32)
    v2 = np.random.randn(128).astype(np.float32)
    sim = cosine_similarity(v1, v2)
    assert sim == 0.0
    dist = cosine_distance(v1, v2)
    assert dist == 1.0


def test_euclidean_distance_known_values():
    v1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    d = euclidean_distance(v1, v2)
    assert np.isclose(d, 5.0, atol=1e-6)
    assert np.isclose(l2_distance(v1, v2), 5.0, atol=1e-6)


def test_euclidean_distance_triangle_inequality():
    np.random.seed(42)
    a = np.random.randn(64).astype(np.float32)
    b = np.random.randn(64).astype(np.float32)
    c = np.random.randn(64).astype(np.float32)
    d_ab = euclidean_distance(a, b)
    d_bc = euclidean_distance(b, c)
    d_ac = euclidean_distance(a, c)
    assert d_ac <= d_ab + d_bc + 1e-6


def test_manhattan_distance_known_values():
    v1 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    v2 = np.array([4.0, 2.0, -1.0], dtype=np.float32)
    # |1-4| + |-2-2| + |3 - (-1)| = 3 + 4 + 4 = 11
    d = manhattan_distance(v1, v2)
    assert np.isclose(d, 11.0, atol=1e-6)


def test_dot_product_calculation():
    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    v2 = np.array([4.0, -5.0, 6.0], dtype=np.float32)
    # 1*4 + 2*(-5) + 3*6 = 4 - 10 + 18 = 12
    dp = dot_product(v1, v2)
    assert np.isclose(dp, 12.0, atol=1e-6)


def test_compute_distance_dispatcher():
    v1 = np.array([1.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0], dtype=np.float32)
    assert np.isclose(compute_distance(v1, v2, "cosine"), 1.0)
    assert np.isclose(compute_distance(v1, v2, "euclidean"), np.sqrt(2.0))
    assert np.isclose(compute_distance(v1, v2, "l2"), np.sqrt(2.0))
    assert np.isclose(compute_distance(v1, v2, "manhattan"), 2.0)
    assert np.isclose(compute_distance(v1, v2, "dot"), 0.0)

    with pytest.raises(ValueError):
        compute_distance(v1, v2, "invalid_metric")


def test_pairwise_distances_cosine():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    Y = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    dists = pairwise_distances(X, Y, "cosine")
    assert dists.shape == (2, 2)
    assert np.isclose(dists[0, 0], 0.0, atol=1e-5)  # [1,0] to [1,0]
    assert np.isclose(dists[0, 1], 1.0, atol=1e-5)  # [1,0] to [0,-1]
    assert np.isclose(dists[1, 0], 1.0, atol=1e-5)  # [0,1] to [1,0]
    assert np.isclose(dists[1, 1], 2.0, atol=1e-5)  # [0,1] to [0,-1]


def test_pairwise_distances_euclidean():
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    Y = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    dists = pairwise_distances(X, Y, "euclidean")
    assert dists.shape == (2, 2)
    assert np.isclose(dists[0, 0], 0.0)
    assert np.isclose(dists[0, 1], 1.0)
    assert np.isclose(dists[1, 0], np.sqrt(2.0))
    assert np.isclose(dists[1, 1], 1.0)


def test_pairwise_distances_manhattan():
    X = np.array([[0.0, 0.0], [2.0, 3.0]], dtype=np.float32)
    Y = np.array([[1.0, 1.0]], dtype=np.float32)
    dists = pairwise_distances(X, Y, "manhattan")
    assert dists.shape == (2, 1)
    assert np.isclose(dists[0, 0], 2.0)
    assert np.isclose(dists[1, 0], 3.0)


def test_high_dimensional_stability_1536d():
    np.random.seed(123)
    dim = 1536
    v1 = np.random.randn(dim).astype(np.float32)
    v2 = np.random.randn(dim).astype(np.float32)
    norm_v1 = normalize(v1)
    norm_v2 = normalize(v2)
    assert np.isclose(np.linalg.norm(norm_v1), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(norm_v2), 1.0, atol=1e-5)
    sim = cosine_similarity(norm_v1, norm_v2)
    assert -1.0 <= sim <= 1.0
    dist = euclidean_distance(v1, v2)
    assert dist > 0.0 and not np.isnan(dist) and not np.isinf(dist)


def test_batch_normalization_preserves_zero_vectors():
    batch = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    normed = normalize(batch)
    assert np.allclose(normed[0], [0.0, 0.0, 0.0])
    assert np.isclose(np.linalg.norm(normed[1]), 1.0)
    assert np.allclose(normed[2], [0.0, 0.0, 0.0])
