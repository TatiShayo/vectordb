"""Search router — all search modes with cache, MMR, reranking, RRF."""
from __future__ import annotations
import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from api.auth import require_auth
from core.engine import get_engine
from models.schemas import (
    BatchSearchRequest, BatchSearchResponse, HybridSearchRequest,
    IDSearchRequest, SearchRequest, SearchResponse, SearchResult,
    TextSearchRequest, SparseSearchRequest,
)
from utils.metrics import metrics
from utils.prometheus import prom

router = APIRouter(prefix="/collections/{collection_name}", tags=["search"])

def _col(name):
    try: return get_engine().get(name)
    except KeyError: raise HTTPException(404, f"Collection '{name}' not found")

def _to_sr(results, ms, cached=False) -> SearchResponse:
    return SearchResponse(
        results=[SearchResult(id=r["id"], score=r["score"],
                              metadata=r["metadata"], vector=r.get("vector"),
                              rerank_score=r.get("rerank_score"))
                 for r in results],
        total_returned=len(results),
        query_time_ms=round(ms, 2),
        cached=cached,
    )

@router.post("/search", response_model=SearchResponse)
def search(collection_name: str, body: SearchRequest,
                 key: str = Depends(require_auth)):
    col = _col(collection_name)
    t0 = time.perf_counter()
    try:
        results, cached = col.search(
            body.vector, body.top_k, body.filter, body.include_vector,
            body.score_threshold, body.ef_search, body.use_mmr, body.mmr_lambda,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))

    # Optional cross-encoder reranking B03
    if body.rerank and body.rerank_query and results:
        try:
            from core.reranker import cross_encoder_rerank
            from config import RERANKER_MODEL
            results = cross_encoder_rerank(body.rerank_query, results,
                                           body.top_k, RERANKER_MODEL)
        except Exception as exc:
            pass  # Reranker optional — degrade gracefully

    ms = (time.perf_counter() - t0) * 1000
    metrics.inc_search(ms)
    prom.search_total.inc(); prom.search_latency.observe(ms/1000)
    return _to_sr(results, ms, cached)

@router.post("/search/by-text", response_model=SearchResponse)
def search_by_text(collection_name: str, body: TextSearchRequest,
                         key: str = Depends(require_auth)):
    from utils.embedder import embed, is_available
    from config import EMBEDDING_MODEL
    if not is_available(EMBEDDING_MODEL):
        raise HTTPException(503, "sentence-transformers not installed")
    col = _col(collection_name)
    t0 = time.perf_counter()
    vector = embed(body.text, EMBEDDING_MODEL).tolist()
    results, cached = col.search(vector, body.top_k, body.filter,
                                 body.include_vector, body.score_threshold)
    if body.rerank and results:
        try:
            from core.reranker import cross_encoder_rerank
            from config import RERANKER_MODEL
            results = cross_encoder_rerank(body.text, results, body.top_k, RERANKER_MODEL)
        except Exception:
            pass
    ms = (time.perf_counter() - t0) * 1000
    metrics.inc_search(ms); prom.search_total.inc()
    return _to_sr(results, ms, cached)

@router.post("/search/by-id", response_model=SearchResponse)
def search_by_id(collection_name: str, body: IDSearchRequest,
                       key: str = Depends(require_auth)):
    col = _col(collection_name)
    t0 = time.perf_counter()
    try:
        results, cached = col.search_by_id(body.id, body.top_k,
                                            body.filter, body.include_vector)
    except KeyError as e:
        raise HTTPException(404, str(e))
    ms = (time.perf_counter() - t0) * 1000
    metrics.inc_search(ms); prom.search_total.inc()
    return _to_sr(results, ms, cached)

@router.post("/search/hybrid", response_model=SearchResponse)
def hybrid_search(collection_name: str, body: HybridSearchRequest,
                        key: str = Depends(require_auth)):
    col = _col(collection_name)
    t0 = time.perf_counter()
    try:
        results = col.hybrid_search(body.vector, body.text, body.top_k,
                                    body.vector_weight, body.filter,
                                    body.include_vector, body.fusion)
    except ValueError as e:
        raise HTTPException(422, str(e))
    ms = (time.perf_counter() - t0) * 1000
    metrics.inc_search(ms); prom.search_total.inc()
    return _to_sr(results, ms)

@router.post("/search/sparse", response_model=SearchResponse)
def sparse_search(collection_name: str, body: SparseSearchRequest,
                        key: str = Depends(require_auth)):
    """B05 — search using a sparse vector (inverted index)."""
    col = _col(collection_name)
    t0 = time.perf_counter()
    sv_dict = dict(zip(body.sparse_vector.indices, body.sparse_vector.values))
    results = col.search_sparse(sv_dict, body.top_k, body.filter)
    ms = (time.perf_counter() - t0) * 1000
    metrics.inc_search(ms); prom.search_total.inc()
    return _to_sr(results, ms)

@router.post("/search/batch", response_model=BatchSearchResponse)
def batch_search(collection_name: str, body: BatchSearchRequest,
                       key: str = Depends(require_auth)):
    col = _col(collection_name)
    t0_total = time.perf_counter()
    responses: List[SearchResponse] = []
    for q in body.queries:
        t0 = time.perf_counter()
        try:
            results, cached = col.search(
                q.vector, q.top_k, q.filter,
                body.include_vector, q.score_threshold,
            )
        except ValueError:
            results, cached = [], False
        ms = (time.perf_counter() - t0) * 1000
        metrics.inc_search(ms); prom.search_total.inc()
        responses.append(_to_sr(results, ms, cached))
    total_ms = (time.perf_counter() - t0_total) * 1000
    return BatchSearchResponse(responses=responses,
                               total_time_ms=round(total_ms, 2))
