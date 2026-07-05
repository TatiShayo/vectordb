"""
Sparse vector support — B02/B05.

Implements a lightweight inverted index for sparse vectors (BM25-style).
Sparse vectors are {term_id: weight} dicts. We store them in SQLite and
build an in-memory inverted index for fast dot-product search.

This enables true hybrid search:
  1. Dense FAISS search → dense candidates with cosine scores
  2. Sparse inverted-index search → keyword candidates with BM25 scores
  3. RRF fusion → combined ranking
"""
from __future__ import annotations

import heapq
import json
import logging
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SparseIndex:
    """In-memory inverted index for sparse vectors."""

    def __init__(self):
        # term_id → {doc_id: weight}
        self._index: Dict[int, Dict[str, float]] = defaultdict(dict)
        self._doc_norms: Dict[str, float] = {}
        self._lock = threading.RLock()

    def add(self, doc_id: str, sparse_vec: Dict[int, float]) -> None:
        if not sparse_vec:
            return
        norm = np.sqrt(sum(w * w for w in sparse_vec.values()))
        with self._lock:
            self._doc_norms[doc_id] = norm
            for term_id, weight in sparse_vec.items():
                self._index[term_id][doc_id] = weight

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self._doc_norms.pop(doc_id, None)
            for posting in self._index.values():
                posting.pop(doc_id, None)

    def search(
        self, query_sparse: Dict[int, float], top_k: int
    ) -> List[Tuple[str, float]]:
        """Dot-product sparse search. Returns [(doc_id, score)] sorted desc."""
        if not query_sparse:
            return []
        scores: Dict[str, float] = defaultdict(float)
        q_norm = np.sqrt(sum(w * w for w in query_sparse.values()))
        if q_norm < 1e-9:
            return []

        with self._lock:
            for term_id, q_weight in query_sparse.items():
                if term_id not in self._index:
                    continue
                for doc_id, d_weight in self._index[term_id].items():
                    scores[doc_id] += q_weight * d_weight

        # Normalize
        results = []
        for doc_id, score in scores.items():
            d_norm = self._doc_norms.get(doc_id, 1.0)
            norm_score = score / (q_norm * d_norm + 1e-9)
            results.append((doc_id, norm_score))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def size(self) -> int:
        return len(self._doc_norms)

    def clear(self) -> None:
        with self._lock:
            self._index.clear()
            self._doc_norms.clear()


def bm25_encode(
    text: str,
    vocab: Optional[Dict[str, int]] = None,
    k1: float = 1.5,
    b: float = 0.75,
    avg_doc_len: float = 100.0,
    doc_len: Optional[int] = None,
) -> Dict[int, float]:
    """
    Simple BM25 term frequency encoder.
    Returns {term_id: bm25_weight} for all non-stopword terms.
    vocab maps term → int ID (auto-assigns if None passed as mutable dict).
    """
    import re
    STOPWORDS = {
        "the","a","an","in","on","at","is","it","its","of","for","to",
        "and","or","but","not","with","as","by","from","that","this",
        "was","are","be","been","have","has","had","do","did","will",
    }
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    if not tokens:
        return {}

    dl = doc_len or len(tokens)
    tf_map: Dict[str, int] = defaultdict(int)
    for t in tokens:
        tf_map[t] += 1

    result: Dict[int, float] = {}
    if vocab is None:
        vocab = {}

    for term, tf in tf_map.items():
        if term not in vocab:
            vocab[term] = len(vocab)
        term_id = vocab[term]
        # BM25 TF component (IDF=1 here; real IDF needs corpus stats)
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_doc_len))
        result[term_id] = bm25_tf

    return result


def rrf_fuse(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion — B01.
    Fuses multiple ranked lists into one using rank positions (not scores).
    k=60 is standard; lower k emphasises top ranks more aggressively.
    weights: per-list multipliers (default: equal weights).
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = defaultdict(float)
    for ranked, w in zip(ranked_lists, weights):
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            scores[doc_id] += w / (k + rank)

    result = sorted(scores.items(), key=lambda x: -x[1])
    return result
