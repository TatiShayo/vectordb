"""
Test suite for Vector Quantization — Scalar (int8, uint8, binary) & Product Quantization (PQ).
Validates reconstruction loss bounds, bit-packing, ADC distance lookup, and rescoring.
"""
from __future__ import annotations
import numpy as np
import pytest

from core.quantizer import Quantizer, ProductQuantizer
from utils.normalize import normalize, cosine_similarity


def test_scalar_quantizer_int8_precision_bounds():
    np.random.seed(42)
    dim = 128
    vectors = normalize(np.random.randn(200, dim).astype(np.float32))

    sq = Quantizer(mode="int8", dimension=dim)
    sq.fit(vectors)

    encoded = sq.encode(vectors)
    assert encoded.dtype == np.int8
    assert encoded.shape == (200, dim)
    assert np.all(encoded >= -127) and np.all(encoded <= 127)

    decoded = sq.decode(encoded)
    assert decoded.dtype == np.float32
    assert decoded.shape == (200, dim)

    error_metrics = sq.reconstruction_error(vectors)
    assert error_metrics["mse"] < 0.0001, f"MSE {error_metrics['mse']} exceeds bound"
    assert error_metrics["mean_cosine_similarity"] > 0.995


def test_scalar_quantizer_uint8_affine():
    np.random.seed(42)
    dim = 64
    # Arbitrary non-centered range [-50, 150]
    vectors = np.random.uniform(-50, 150, size=(100, dim)).astype(np.float32)

    sq = Quantizer(mode="uint8", dimension=dim)
    sq.fit(vectors)

    encoded = sq.encode(vectors)
    assert encoded.dtype == np.uint8
    assert np.all(encoded >= 0) and np.all(encoded <= 255)

    decoded = sq.decode(encoded)
    error_metrics = sq.reconstruction_error(vectors)
    # Relative reconstruction error < 1%
    rel_error = np.mean(np.abs(vectors - decoded) / 200.0)
    assert rel_error < 0.01


def test_scalar_quantizer_binary_compression():
    np.random.seed(42)
    dim = 64  # 64 floats (256 bytes) -> 8 bytes in binary
    vectors = normalize(np.random.randn(50, dim).astype(np.float32))

    sq = Quantizer(mode="binary", dimension=dim)
    encoded = sq.encode(vectors)
    assert encoded.shape == (50, 8)  # 64 / 8 = 8 bytes
    assert encoded.dtype == np.uint8

    decoded = sq.decode(encoded)
    assert decoded.shape == (50, dim)
    # Decoded values are in {-1.0, +1.0}
    assert set(np.unique(decoded)).issubset({-1.0, 1.0})


def test_scalar_quantizer_serialization():
    dim = 32
    vectors = np.random.randn(50, dim).astype(np.float32)
    sq1 = Quantizer(mode="int8", dimension=dim).fit(vectors)
    data = sq1.to_dict()

    sq2 = Quantizer.from_dict(data)
    assert sq2.mode == "int8"
    assert sq2.dimension == dim
    assert np.allclose(sq1._scale, sq2._scale)

    # Encode gives identical results
    test_vec = vectors[:5]
    assert np.array_equal(sq1.encode(test_vec), sq2.encode(test_vec))


def test_scalar_quantizer_rescore():
    np.random.seed(123)
    dim = 32
    candidates = normalize(np.random.randn(20, dim).astype(np.float32))
    query = normalize(np.random.randn(dim).astype(np.float32))
    candidate_ids = [f"c{i}" for i in range(20)]

    top_ids, top_scores = Quantizer.rescore(query, candidates, candidate_ids, top_k=5)
    assert len(top_ids) == 5
    assert len(top_scores) == 5
    # Scores must be in descending order
    assert all(top_scores[i] >= top_scores[i + 1] for i in range(4))


def test_product_quantizer_fit_and_encode():
    np.random.seed(42)
    dim = 64
    m = 8  # 8 subspaces of 8 dim each
    nbits = 8  # 256 centroids per subspace

    vectors = normalize(np.random.randn(300, dim).astype(np.float32))
    pq = ProductQuantizer(dimension=dim, m=m, nbits=nbits)
    pq.fit(vectors, n_iter=5)

    assert pq._fitted is True
    assert pq.codebook.shape == (m, 256, 8)

    codes = pq.encode(vectors)
    assert codes.shape == (300, m)
    assert codes.dtype == np.uint8

    decoded = pq.decode(codes)
    assert decoded.shape == (300, dim)
    assert decoded.dtype == np.float32

    metrics = pq.reconstruction_error(vectors)
    assert metrics["mse"] < 0.05
    assert metrics["mean_cosine_similarity"] > 0.85


def test_product_quantizer_asymmetric_distance_computation():
    np.random.seed(42)
    dim = 32
    m = 4
    nbits = 8

    vectors = normalize(np.random.randn(100, dim).astype(np.float32))
    pq = ProductQuantizer(dimension=dim, m=m, nbits=nbits).fit(vectors, n_iter=5)

    query = normalize(np.random.randn(dim).astype(np.float32))
    codes = pq.encode(vectors)

    adc_dists = pq.compute_asymmetric_distances(query, codes)
    assert adc_dists.shape == (100,)
    assert np.all(adc_dists >= 0.0)

    # Reconstructed exact L2 distances should correlate strongly with ADC distances
    decoded = pq.decode(codes)
    exact_l2_sq = np.sum((decoded - query[np.newaxis, :]) ** 2, axis=1)
    assert np.allclose(adc_dists, exact_l2_sq, atol=1e-4)


def test_product_quantizer_serialization():
    dim = 32
    m = 4
    nbits = 4  # 16 centroids
    vectors = np.random.randn(50, dim).astype(np.float32)
    pq1 = ProductQuantizer(dimension=dim, m=m, nbits=nbits).fit(vectors, n_iter=3)

    data = pq1.to_dict()
    pq2 = ProductQuantizer.from_dict(data)

    assert pq2.dimension == dim
    assert pq2.m == m
    assert pq2.nbits == nbits
    assert np.allclose(pq1.codebook, pq2.codebook)

    test_codes = pq1.encode(vectors[:5])
    test_codes2 = pq2.encode(vectors[:5])
    assert np.array_equal(test_codes, test_codes2)


def test_product_quantizer_invalid_params():
    with pytest.raises(ValueError):
        ProductQuantizer(dimension=64, m=7)  # 64 not divisible by 7

    with pytest.raises(ValueError):
        ProductQuantizer(dimension=64, m=8, nbits=5)  # nbits must be 4 or 8
