"""
Reranking — B03/B04/B14.

B03: Cross-encoder reranker (sentence-transformers CrossEncoder)
B04: Maximal Marginal Relevance (MMR) — diversity-aware result dedup
B14: Score normalisation — consistent 0-1 range in result set
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── B03 Cross-encoder reranker ────────────────────────────────────────────────

import threading

_ce_model = None
_ce_model_name: str | None = None
_ce_lock = threading.Lock()


def _load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _ce_model, _ce_model_name
    if _ce_model is not None and _ce_model_name == model_name:
        return _ce_model
    with _ce_lock:
        if _ce_model is not None and _ce_model_name == model_name:
            return _ce_model
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {model_name}")
            _ce_model = CrossEncoder(model_name)
            _ce_model_name = model_name
            logger.info("Cross-encoder loaded.")
        except ImportError:
            raise RuntimeError("sentence-transformers required: pip install sentence-transformers")
    return _ce_model


def cross_encoder_rerank(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    text_field: str = "text",
) -> List[Dict[str, Any]]:
    """
    Rerank results using a cross-encoder model.
    Each result must have metadata[text_field] with the document text.
    Returns top_k results with updated 'score' from cross-encoder.
    """
    if not results:
        return results

    ce = _load_cross_encoder(model_name)
    pairs = [
        (query, r["metadata"].get(text_field, str(r["metadata"])))
        for r in results
    ]
    scores = ce.predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)
        result["original_score"] = result["score"]
        result["score"] = float(score)

    results.sort(key=lambda r: -r["score"])
    return results[:top_k]


def is_reranker_available(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> bool:
    try:
        _load_cross_encoder(model_name)
        return True
    except Exception:
        return False


# ── B04 Maximal Marginal Relevance (MMR) ─────────────────────────────────────

def mmr(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_ids: List[str],
    candidate_scores: np.ndarray,
    top_k: int,
    lambda_mult: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    MMR selects results that are both relevant AND diverse.

    lambda_mult = 1.0 → pure relevance (same as vanilla search)
    lambda_mult = 0.0 → pure diversity
    lambda_mult = 0.5 → balanced (default, recommended)

    Algorithm:
      For each step, pick the candidate that maximises:
        λ * sim(candidate, query) - (1-λ) * max(sim(candidate, already_selected))
    """
    if len(candidate_ids) == 0:
        return []

    selected_indices: List[int] = []
    remaining_indices = list(range(len(candidate_ids)))

    # Precompute pairwise similarities among candidates
    norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normalised = candidate_vecs / norms

    # Query similarities (already in candidate_scores)
    query_sims = candidate_scores

    for _ in range(min(top_k, len(candidate_ids))):
        if not remaining_indices:
            break

        if not selected_indices:
            # First pick: highest relevance
            best = max(remaining_indices, key=lambda i: query_sims[i])
        else:
            selected_vecs = normalised[selected_indices]
            best_mmr = -np.inf
            best = remaining_indices[0]

            for i in remaining_indices:
                rel = lambda_mult * query_sims[i]
                # Redundancy: max similarity with any already-selected
                redundancy = (normalised[i] @ selected_vecs.T).max()
                diversity = (1 - lambda_mult) * redundancy
                mmr_score = rel - diversity
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best = i

        selected_indices.append(best)
        remaining_indices.remove(best)

    return [(candidate_ids[i], float(query_sims[i])) for i in selected_indices]


# ── B14 Score normalisation ───────────────────────────────────────────────────

def normalise_scores(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Min-max normalise scores in the result set to [0, 1]."""
    if not results:
        return results
    scores = np.array([r["score"] for r in results], dtype=np.float32)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        for r in results:
            r["score"] = 1.0
    else:
        for r, s in zip(results, scores):
            r["score"] = float((s - lo) / (hi - lo))
    return results
