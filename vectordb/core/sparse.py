"""
Sparse vector index, BM25 text tokenization & hybrid search fusion.

Includes:
- BM25 corpus statistics & inverted index with true IDF computation
- Reciprocal Rank Fusion (RRF)
- Normalized Linear Weighted Scoring
"""
from __future__ import annotations
import heapq
import json
import logging
import math
import re
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

STOPWORDS: Set[str] = {
    "the", "a", "an", "in", "on", "at", "is", "it", "its", "of", "for", "to",
    "and", "or", "but", "not", "with", "as", "by", "from", "that", "this",
    "was", "are", "be", "been", "have", "has", "had", "do", "did", "will",
    "would", "shall", "should", "may", "might", "can", "could", "there",
    "their", "they", "them", "which", "who", "whom", "whose", "where", "when",
}


def tokenize(text: str) -> List[str]:
    """Tokenize string into lowercase alphanumeric terms, filtering stopwords and short tokens."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def bm25_encode(
    text: str,
    vocab: Optional[Dict[str, int]] = None,
    k1: float = 1.5,
    b: float = 0.75,
    avg_doc_len: float = 100.0,
    doc_len: Optional[int] = None,
    idf_map: Optional[Dict[str, float]] = None,
) -> Dict[int, float]:
    """
    BM25 term frequency encoder.
    Returns {term_id: weight} for terms in text.
    If vocab is provided, maps term -> term_id. If vocab is a mutable dict, auto-assigns IDs.
    """
    tokens = tokenize(text)
    if not tokens:
        return {}

    dl = doc_len or len(tokens)
    tf_map: Dict[str, int] = defaultdict(int)
    for t in tokens:
        tf_map[t] += 1

    if vocab is None:
        vocab = {}

    result: Dict[int, float] = {}
    for term, tf in tf_map.items():
        if term not in vocab:
            vocab[term] = len(vocab)
        term_id = vocab[term]
        idf = idf_map.get(term, 1.0) if idf_map else 1.0
        # Standard BM25 TF component
        tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / max(avg_doc_len, 1.0))))
        result[term_id] = float(tf_norm * idf)

    return result


class SparseIndex:
    """In-memory inverted index for sparse vectors and BM25 search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # term_id -> {doc_id: weight}
        self._index: Dict[int, Dict[str, float]] = defaultdict(dict)
        self._doc_norms: Dict[str, float] = {}
        self._doc_lens: Dict[str, int] = {}
        self._vocab: Dict[str, int] = {}
        self._rev_vocab: Dict[int, str] = {}
        self._k1 = k1
        self._b = b
        self._lock = threading.RLock()

    def add(self, doc_id: str, sparse_vec: Dict[int, float], doc_len: Optional[int] = None) -> None:
        """Add a sparse vector representation for doc_id."""
        if not sparse_vec:
            return
        norm = float(np.sqrt(sum(w * w for w in sparse_vec.values())))
        with self._lock:
            self._doc_norms[doc_id] = norm if norm > 1e-9 else 1.0
            if doc_len is not None:
                self._doc_lens[doc_id] = doc_len
            for term_id, weight in sparse_vec.items():
                self._index[term_id][doc_id] = float(weight)

    def add_text(self, doc_id: str, text: str) -> None:
        """Tokenize and add text directly using BM25 weights."""
        tokens = tokenize(text)
        if not tokens:
            return
        with self._lock:
            for t in tokens:
                if t not in self._vocab:
                    tid = len(self._vocab)
                    self._vocab[t] = tid
                    self._rev_vocab[tid] = t

            avg_dl = sum(self._doc_lens.values()) / max(len(self._doc_lens), 1) if self._doc_lens else len(tokens)
            vec = bm25_encode(text, vocab=self._vocab, k1=self._k1, b=self._b, avg_doc_len=avg_dl, doc_len=len(tokens))
            self.add(doc_id, vec, doc_len=len(tokens))

    def remove(self, doc_id: str) -> None:
        """Remove a document from the inverted index."""
        with self._lock:
            self._doc_norms.pop(doc_id, None)
            self._doc_lens.pop(doc_id, None)
            for posting in list(self._index.values()):
                posting.pop(doc_id, None)

    def search(
        self, query_sparse: Dict[int, float], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Dot-product sparse search. Returns [(doc_id, score)] sorted descending."""
        if not query_sparse:
            return []

        q_norm = float(np.sqrt(sum(w * w for w in query_sparse.values())))
        if q_norm < 1e-9:
            return []

        scores: Dict[str, float] = defaultdict(float)
        with self._lock:
            for term_id, q_weight in query_sparse.items():
                if term_id not in self._index:
                    continue
                for doc_id, d_weight in self._index[term_id].items():
                    scores[doc_id] += q_weight * d_weight

        # Normalize by vector norms (cosine in sparse space)
        results = []
        for doc_id, score in scores.items():
            d_norm = self._doc_norms.get(doc_id, 1.0)
            norm_score = score / (q_norm * d_norm + 1e-9)
            results.append((doc_id, float(norm_score)))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def search_text(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search by query text using BM25 IDF weights."""
        tokens = tokenize(text)
        if not tokens:
            return []

        with self._lock:
            n_docs = len(self._doc_norms)
            if n_docs == 0:
                return []

            # Compute IDF for query terms
            idf_map: Dict[str, float] = {}
            for t in tokens:
                tid = self._vocab.get(t)
                df = len(self._index.get(tid, {})) if tid is not None else 0
                # BM25 IDF
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                idf_map[t] = max(0.0, idf)

            avg_dl = sum(self._doc_lens.values()) / max(len(self._doc_lens), 1)
            q_vec = bm25_encode(text, vocab=self._vocab, k1=self._k1, b=self._b, avg_doc_len=avg_dl, idf_map=idf_map)

        return self.search(q_vec, top_k)

    def size(self) -> int:
        with self._lock:
            return len(self._doc_norms)

    def clear(self) -> None:
        with self._lock:
            self._index.clear()
            self._doc_norms.clear()
            self._doc_lens.clear()
            self._vocab.clear()
            self._rev_vocab.clear()


def rrf_fuse(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF).
    Fuses multiple ranked lists using rank positions:
      RRF_score(d) = sum_m ( w_m / (k + rank_m(d)) )
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = defaultdict(float)
    for ranked, w in zip(ranked_lists, weights):
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            scores[doc_id] += float(w) / (k + rank)

    result = sorted(scores.items(), key=lambda x: -x[1])
    return result


def linear_weighted_fuse(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    vector_weight: float = 0.7,
) -> List[Tuple[str, float]]:
    """
    Linear Weighted Score Fusion with min-max normalization.
      Fused_score(d) = w * norm_dense(d) + (1 - w) * norm_sparse(d)
    """
    vector_weight = float(np.clip(vector_weight, 0.0, 1.0))
    sparse_weight = 1.0 - vector_weight

    # Normalize dense scores
    dense_norm: Dict[str, float] = {}
    if dense_results:
        d_scores = [s for _, s in dense_results]
        d_min, d_max = min(d_scores), max(d_scores)
        d_range = d_max - d_min
        for doc_id, s in dense_results:
            dense_norm[doc_id] = (s - d_min) / d_range if d_range > 1e-9 else 1.0

    # Normalize sparse scores
    sparse_norm: Dict[str, float] = {}
    if sparse_results:
        s_scores = [s for _, s in sparse_results]
        s_min, s_max = min(s_scores), max(s_scores)
        s_range = s_max - s_min
        for doc_id, s in sparse_results:
            sparse_norm[doc_id] = (s - s_min) / s_range if s_range > 1e-9 else 1.0

    all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
    combined: Dict[str, float] = {}

    for doc_id in all_ids:
        d_val = dense_norm.get(doc_id, 0.0)
        s_val = sparse_norm.get(doc_id, 0.0)
        combined[doc_id] = vector_weight * d_val + sparse_weight * s_val

    return sorted(combined.items(), key=lambda x: -x[1])
