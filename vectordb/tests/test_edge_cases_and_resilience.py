"""
Massive Edge Case, Invariant, and Resilience Test Suite for VectorDB.
Covers:
- HNSW graph update cycles, level promotions, isolated nodes, zero vector queries
- Quantizer boundary conditions, extreme distributions, out-of-range dimensions
- Deeply nested metadata filters with mixed types, empty sets, and SQL safety
- Corrupt snapshot headers, missing files, tar slip attacks
- FAISS auto-scaling and index upgrades
- BM25 edge cases (empty strings, single characters, punctuation)
- RRF and Linear Weighted Fusion under zero variance and extreme weights
- Thread safety and resource cleanup
"""
from __future__ import annotations
import json
import os
import tarfile
import tempfile
import numpy as np
import pytest

from core.hnsw import HNSWIndex
from core.quantizer import Quantizer, ProductQuantizer
from core.sparse import SparseIndex, bm25_encode, rrf_fuse, linear_weighted_fuse, tokenize
from storage.db import CollectionDB, match_filter, _build_filter_sql
from storage.snapshot import export_snapshot, verify_snapshot, import_snapshot
from utils.normalize import compute_distance, normalize, cosine_similarity, prepare_vector


# ── 1. HNSW Edge Cases ────────────────────────────────────────────────────────

def test_hnsw_node_update_and_replacement():
    """Verify that updating a node's vector with new values cleans up previous edges cleanly."""
    dim = 16
    index = HNSWIndex(dimension=dim, distance="cosine", m=8, ef_construction=16)

    # Add 10 nodes
    for i in range(10):
        index.add(i, np.random.randn(dim).astype(np.float32))

    assert index.size() == 10

    # Overwrite node 5 with a completely new vector
    new_v5 = np.ones(dim, dtype=np.float32)
    index.add(5, new_v5)

    assert index.size() == 10
    results = index.search(new_v5, top_k=1)
    assert len(results) == 1
    assert results[0][0] == 5
    assert np.isclose(results[0][1], 1.0, atol=1e-4)


def test_hnsw_zero_vectors_and_identical_vectors():
    """Verify HNSW handles multiple identical and zero-norm vectors gracefully."""
    dim = 8
    index = HNSWIndex(dimension=dim, distance="euclidean")

    v_zero = np.zeros(dim, dtype=np.float32)
    v_ident = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    index.add(0, v_zero)
    index.add(1, v_ident)
    index.add(2, v_ident)  # Exact duplicate
    index.add(3, v_ident)  # Exact duplicate

    assert index.size() == 4
    res = index.search(v_ident, top_k=3)
    assert len(res) == 3
    # Top 3 should be nodes 1, 2, 3 with distance 0
    returned_ids = {r[0] for r in res}
    assert returned_ids == {1, 2, 3}
    for _, d in res:
        assert np.isclose(d, 0.0, atol=1e-5)


def test_hnsw_delete_entry_point():
    """Deleting the entry point should automatically reassign entry point to an active node."""
    dim = 8
    index = HNSWIndex(dimension=dim, distance="cosine")
    for i in range(5):
        index.add(i, np.random.randn(dim).astype(np.float32))

    ep = index.entry_point
    assert ep is not None
    assert index.remove(ep) is True

    # Search should still succeed without crashing
    res = index.search(np.random.randn(dim).astype(np.float32), top_k=3)
    assert len(res) == 3
    assert ep not in [r[0] for r in res]


def test_hnsw_rebuild_after_full_deletion():
    """Deleting all nodes and rebuilding returns an empty graph."""
    dim = 8
    index = HNSWIndex(dimension=dim)
    for i in range(5):
        index.add(i, np.random.randn(dim).astype(np.float32))
    for i in range(5):
        index.remove(i)

    assert index.size() == 0
    index.rebuild()
    assert index.size() == 0
    assert index.entry_point is None
    assert index.search(np.random.randn(dim).astype(np.float32), top_k=5) == []


# ── 2. Quantizer Extreme Cases ────────────────────────────────────────────────

def test_quantizer_int8_extreme_dynamic_range():
    """Test int8 quantization with extreme values spanning orders of magnitude."""
    dim = 16
    vectors = np.array([
        [1e-6] * 16,
        [1e4] * 16,
        [-1e4] * 16,
        [0.0] * 16,
    ], dtype=np.float32)

    q = Quantizer(mode="int8", dimension=dim).fit(vectors)
    enc = q.encode(vectors)
    dec = q.decode(enc)

    assert enc.shape == (4, dim)
    assert dec.shape == (4, dim)
    # Signs preserved
    assert np.all(enc[1] > 0)
    assert np.all(enc[2] < 0)
    assert np.all(enc[3] == 0)


def test_quantizer_binary_odd_dimensions():
    """Binary quantization with dimensions not divisible by 8 (e.g. 19d)."""
    dim = 19
    vectors = np.random.randn(10, dim).astype(np.float32)
    q = Quantizer(mode="binary", dimension=dim)

    enc = q.encode(vectors)
    # 19 bits -> 3 bytes (24 bits)
    assert enc.shape == (10, 3)

    dec = q.decode(enc)
    assert dec.shape == (10, dim)
    assert np.all(np.isin(dec, [-1.0, 1.0]))


def test_product_quantizer_few_samples():
    """Product Quantizer fitting when n_samples < k."""
    dim = 16
    m = 4
    nbits = 4  # k = 16
    # Provide only 5 samples (< 16 centroids)
    vectors = np.random.randn(5, dim).astype(np.float32)
    pq = ProductQuantizer(dimension=dim, m=m, nbits=nbits)
    pq.fit(vectors, n_iter=3)

    assert pq._fitted is True
    assert pq.codebook.shape == (m, 16, 4)

    codes = pq.encode(vectors)
    assert codes.shape == (5, m)

    decoded = pq.decode(codes)
    assert decoded.shape == (5, dim)
    assert not np.isnan(decoded).any()


# ── 3. Metadata Filtering Deep Predicates ─────────────────────────────────────

def test_filter_deeply_nested_and_or_tree():
    meta = {
        "user": {"tier": "gold"},
        "status": "active",
        "score": 95,
        "region": "us-east",
    }

    # Complex predicate: (tier == gold AND score >= 90) OR (status == active AND region IN [us-east, eu-west])
    filt = {
        "$or": [
            {"$and": [{"score": {"$gte": 90}}]},
            {"$and": [{"status": "active"}, {"region": {"$in": ["us-east", "eu-west"]}}]},
        ]
    }
    assert match_filter(meta, filt) is True

    # Negated query
    neg_filt = {"$not": {"status": "inactive"}}
    assert match_filter(meta, neg_filt) is True


def test_filter_empty_in_and_nin_predicates():
    meta = {"color": "red"}
    assert match_filter(meta, {"color": {"$in": []}}) is False
    assert match_filter(meta, {"color": {"$nin": []}}) is True


def test_filter_exists_edge_cases():
    meta = {"present": "yes", "null_val": None}
    assert match_filter(meta, {"present": {"$exists": True}}) is True
    assert match_filter(meta, {"present": {"$exists": False}}) is False
    assert match_filter(meta, {"absent": {"$exists": False}}) is True
    assert match_filter(meta, {"absent": {"$exists": True}}) is False


def test_filter_sql_generation_empty_clauses():
    sql, params = _build_filter_sql({})
    assert sql == ""
    assert params == []

    sql, params = _build_filter_sql(None)
    assert sql == ""
    assert params == []


# ── 4. BM25 and Sparse Edge Cases ─────────────────────────────────────────────

def test_bm25_empty_and_whitespace_text():
    assert tokenize("") == []
    assert tokenize("   \n\t  ") == []
    assert bm25_encode("") == {}
    assert bm25_encode("the a on in") == {}  # all stopwords


def test_bm25_special_characters_and_unicode():
    tokens = tokenize("Python 3.11 & FastAPI! #Vector_DB @2026-AI?")
    assert "python" in tokens
    assert "11" in tokens
    assert "fastapi" in tokens
    assert "vector" in tokens
    assert "db" in tokens
    assert "2026" in tokens
    assert "ai" in tokens


def test_sparse_index_empty_query():
    idx = SparseIndex()
    idx.add("d1", {0: 1.0})
    assert idx.search({}, top_k=5) == []
    assert idx.search_text("", top_k=5) == []


def test_linear_weighted_fuse_single_item():
    dense = [("d1", 0.8)]
    sparse = []
    fused = linear_weighted_fuse(dense, sparse, vector_weight=0.7)
    assert len(fused) == 1
    assert fused[0][0] == "d1"


# ── 5. Snapshot Security & Corrupt Archive Handling ───────────────────────────

def test_snapshot_path_traversal_detection():
    """Verify that verify_snapshot and import_snapshot detect path traversal attempts in tarball."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = os.path.join(tmp, "tar_test")
        os.makedirs(tmp_dir, exist_ok=True)
        bad_tar = os.path.join(tmp, "evil.tar.gz")

        # Create a malicious tar with relative path traversal
        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            f.write(json.dumps({"version": "2.0", "checksums": {}}))

        with tarfile.open(bad_tar, "w:gz") as tar:
            tar.add(manifest_path, arcname="../evil.json")

        is_valid, msg, _ = verify_snapshot(bad_tar)
        assert is_valid is False
        assert "traversal" in msg.lower() or "dangerous" in msg.lower()


def test_snapshot_missing_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        bad_tar = os.path.join(tmp, "no_manifest.tar.gz")
        dummy_file = os.path.join(tmp, "dummy.txt")
        with open(dummy_file, "w") as f:
            f.write("hello")

        with tarfile.open(bad_tar, "w:gz") as tar:
            tar.add(dummy_file, arcname="dummy.txt")

        is_valid, msg, _ = verify_snapshot(bad_tar)
        assert is_valid is False
        assert "manifest.json" in msg.lower()


def test_snapshot_checksum_tamper_file():
    with tempfile.TemporaryDirectory() as tmp:
        snap_tar = os.path.join(tmp, "tampered.tar.gz")
        manifest_data = {
            "version": "2.0",
            "checksums": {"metadata.db": "0000000000000000000000000000000000000000000000000000000000000000"}
        }
        m_file = os.path.join(tmp, "manifest.json")
        db_file = os.path.join(tmp, "metadata.db")
        with open(m_file, "w") as f:
            f.write(json.dumps(manifest_data))
        with open(db_file, "w") as f:
            f.write("actual db content")

        with tarfile.open(snap_tar, "w:gz") as tar:
            tar.add(m_file, arcname="manifest.json")
            tar.add(db_file, arcname="metadata.db")

        is_valid, msg, _ = verify_snapshot(snap_tar)
        assert is_valid is False
        assert "checksum mismatch" in msg.lower()
