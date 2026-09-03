"""
Test suite for BM25 text tokenization, sparse inverted index, and hybrid search fusion.
Validates:
- Tokenization correctness and stopword filtering
- BM25 TF-IDF encoding and corpus IDF weighting
- Sparse inverted index insertion, deletion, and query ranking
- Reciprocal Rank Fusion (RRF) and Linear Weighted Score Fusion
"""
from __future__ import annotations
import pytest
from core.sparse import (
    SparseIndex,
    bm25_encode,
    linear_weighted_fuse,
    rrf_fuse,
    tokenize,
)


def test_tokenize_clean_and_stopwords():
    text = "The Quick brown Fox jumps OVER the Lazy Dog! 123 in a box."
    tokens = tokenize(text)
    # Stopwords like 'the', 'over', 'in', 'a' removed; 'quick', 'brown', 'fox', 'jumps', 'lazy', 'dog', '123', 'box' kept
    assert "the" not in tokens
    assert "in" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "jumps" in tokens
    assert "lazy" in tokens
    assert "dog" in tokens
    assert "123" in tokens
    assert "box" in tokens


def test_bm25_encode_tf_saturation():
    text = "vector database vector vector index search"
    vocab = {}
    vec = bm25_encode(text, vocab=vocab, k1=1.5, b=0.75, avg_doc_len=10.0, doc_len=6)
    assert len(vec) == 4  # vector, database, index, search

    vid = vocab["vector"]
    did = vocab["database"]
    # TF for 'vector' is 3, TF for 'database' is 1. Weight of 'vector' should be higher but sub-linear (saturated)
    assert vec[vid] > vec[did]
    assert vec[vid] < 3.0 * vec[did]  # Saturation check


def test_sparse_index_add_and_search():
    index = SparseIndex()
    # Doc 1: term 0 (weight 1.0), term 1 (weight 2.0)
    index.add("doc1", {0: 1.0, 1: 2.0})
    # Doc 2: term 1 (weight 1.0), term 2 (weight 3.0)
    index.add("doc2", {1: 1.0, 2: 3.0})
    # Doc 3: term 3 (weight 5.0)
    index.add("doc3", {3: 5.0})

    assert index.size() == 3

    # Query for term 1
    results = index.search({1: 1.0}, top_k=5)
    matched_ids = [d_id for d_id, _ in results]
    assert "doc1" in matched_ids
    assert "doc2" in matched_ids
    assert "doc3" not in matched_ids

    # Query for term 3
    results3 = index.search({3: 1.0}, top_k=5)
    assert len(results3) == 1
    assert results3[0][0] == "doc3"


def test_sparse_index_text_search_with_idf():
    index = SparseIndex()
    index.add_text("doc1", "machine learning neural networks deep learning")
    index.add_text("doc2", "relational database SQL queries tables")
    index.add_text("doc3", "deep learning models and neural representations")

    results = index.search_text("deep learning", top_k=2)
    assert len(results) >= 2
    matched_ids = [d_id for d_id, _ in results]
    assert "doc1" in matched_ids or "doc3" in matched_ids
    assert "doc2" not in matched_ids


def test_sparse_index_remove_and_clear():
    index = SparseIndex()
    index.add("doc1", {1: 2.0, 2: 3.0})
    index.add("doc2", {1: 1.0, 3: 4.0})
    assert index.size() == 2

    index.remove("doc1")
    assert index.size() == 1
    results = index.search({1: 1.0}, top_k=5)
    assert len(results) == 1
    assert results[0][0] == "doc2"

    index.clear()
    assert index.size() == 0
    assert index.search({1: 1.0}, top_k=5) == []


def test_rrf_fuse_ranking():
    # List 1: A (rank 1), B (rank 2), C (rank 3)
    list1 = [("docA", 0.9), ("docB", 0.8), ("docC", 0.7)]
    # List 2: B (rank 1), A (rank 2), D (rank 3)
    list2 = [("docB", 10.0), ("docA", 5.0), ("docD", 2.0)]

    fused = rrf_fuse([list1, list2], k=60)
    doc_ids = [d_id for d_id, _ in fused]

    # Both docA and docB are in top 2 of both lists, so they must be at the top of fused list
    assert doc_ids[0] in ("docA", "docB")
    assert doc_ids[1] in ("docA", "docB")
    assert "docC" in doc_ids
    assert "docD" in doc_ids


def test_rrf_fuse_custom_weights():
    list1 = [("docA", 0.9)]
    list2 = [("docB", 0.9)]

    # Heavy weight on list2
    fused = rrf_fuse([list1, list2], k=60, weights=[0.1, 0.9])
    assert fused[0][0] == "docB"
    assert fused[1][0] == "docA"


def test_linear_weighted_fuse():
    dense_results = [("doc1", 0.95), ("doc2", 0.60), ("doc3", 0.40)]
    sparse_results = [("doc2", 15.0), ("doc3", 10.0), ("doc4", 5.0)]

    fused = linear_weighted_fuse(dense_results, sparse_results, vector_weight=0.5)
    fused_dict = dict(fused)

    # All 4 documents present
    assert set(fused_dict.keys()) == {"doc1", "doc2", "doc3", "doc4"}
    # Scores are ordered descending
    scores = [s for _, s in fused]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_linear_weighted_fuse_boundary_weights():
    dense = [("doc1", 1.0), ("doc2", 0.0)]
    sparse = [("doc2", 10.0), ("doc1", 0.0)]

    # Pure dense (weight = 1.0) -> doc1 is first
    fused_dense = linear_weighted_fuse(dense, sparse, vector_weight=1.0)
    assert fused_dense[0][0] == "doc1"

    # Pure sparse (weight = 0.0) -> doc2 is first
    fused_sparse = linear_weighted_fuse(dense, sparse, vector_weight=0.0)
    assert fused_sparse[0][0] == "doc2"
