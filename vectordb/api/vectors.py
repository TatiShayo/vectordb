"""Vectors router — upsert, batch, get, delete, scroll, patch, count, facets."""
from __future__ import annotations
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from api.auth import require_auth, require_admin
from core.engine import get_engine
from models.schemas import (
    BatchUpsertRequest, BatchUpsertResponse, CountResponse,
    FacetsResponse, MetadataPatch, ScrollResponse,
    UpsertResponse, VectorRecord, VectorUpsert,
)
from utils.metrics import metrics
from utils.prometheus import prom

router = APIRouter(prefix="/collections/{collection_name}/vectors", tags=["vectors"])

def _col(name):
    try: return get_engine().get(name)
    except KeyError: raise HTTPException(404, f"Collection '{name}' not found")

def _to_record(r: dict) -> VectorRecord:
    return VectorRecord(id=r["id"], vector=r.get("vector"),
                        metadata=r["metadata"], created_at=r["created_at"],
                        updated_at=r["updated_at"], expires_at=r.get("expires_at"))

@router.post("", response_model=UpsertResponse)
def upsert(collection_name: str, body: VectorUpsert,
                 key: str = Depends(require_auth)):
    col = _col(collection_name)
    if body.vector is None:
        raise HTTPException(422, "vector is required")
    sv = None
    if body.sparse_vector:
        sv = {"indices": body.sparse_vector.indices,
              "values": body.sparse_vector.values}
    try:
        status = col.upsert(body.id, body.vector, body.metadata,
                            body.ttl_seconds, sv)
    except ValueError as e:
        raise HTTPException(422, str(e))
    metrics.inc_upsert(); prom.upsert_total.inc()
    return UpsertResponse(id=body.id, status=status)

@router.post("/batch", response_model=BatchUpsertResponse)
def upsert_batch(collection_name: str, body: BatchUpsertRequest,
                       key: str = Depends(require_auth)):
    col = _col(collection_name)
    errors, valid_items = [], []
    for item in body.vectors:
        if item.vector is None:
            errors.append({"id": item.id, "error": "vector required"})
            continue
        if len(item.vector) != col.dimension:
            errors.append({"id": item.id,
                           "error": f"dim {len(item.vector)} ≠ {col.dimension}"})
            continue
        sv = None
        if item.sparse_vector:
            sv = {"indices": item.sparse_vector.indices,
                  "values": item.sparse_vector.values}
        valid_items.append({"id": item.id, "vector": item.vector,
                            "metadata": item.metadata,
                            "ttl_seconds": item.ttl_seconds,
                            "sparse_vector": sv})
    inserted = updated = 0
    if valid_items:
        try:
            inserted, updated = col.upsert_batch(valid_items)
        except Exception as e:
            raise HTTPException(500, str(e))
    metrics.inc_upsert(inserted + updated)
    prom.upsert_total.inc(inserted + updated)
    return BatchUpsertResponse(inserted=inserted, updated=updated, errors=errors)

@router.get("/scroll", response_model=ScrollResponse)
def scroll(collection_name: str,
                 limit: int = Query(100, ge=1, le=1000),
                 offset: int = Query(0, ge=0),
                 include_vector: bool = Query(False),
                 key: str = Depends(require_auth)):
    col = _col(collection_name)
    records, total = col.scroll(limit, offset, None, include_vector)
    return ScrollResponse(vectors=[_to_record(r) for r in records],
                          total=total, offset=offset,
                          has_more=(offset + limit) < total)

@router.get("/count", response_model=CountResponse)
def count(collection_name: str, key: str = Depends(require_auth)):
    """D07 — vector count."""
    col = _col(collection_name)
    return CountResponse(count=col.count())

@router.get("/facets/{field}", response_model=FacetsResponse)
def facets(collection_name: str, field: str,
                 limit: int = Query(100, ge=1, le=1000),
                 key: str = Depends(require_auth)):
    """D08 — distinct value counts for a metadata field."""
    col = _col(collection_name)
    values = col.facets(field, limit)
    return FacetsResponse(field=field, values=values)

@router.get("/{vector_id}", response_model=VectorRecord)
def get_vector(collection_name: str, vector_id: str,
                     include_vector: bool = Query(False),
                     key: str = Depends(require_auth)):
    col = _col(collection_name)
    record = col.get(vector_id, include_vector)
    if record is None:
        raise HTTPException(404, f"Vector '{vector_id}' not found")
    return _to_record(record)

@router.patch("/{vector_id}", response_model=UpsertResponse)
def patch_metadata(collection_name: str, vector_id: str,
                         body: MetadataPatch, key: str = Depends(require_auth)):
    """D10 — update metadata without re-indexing."""
    col = _col(collection_name)
    ok = col.patch_metadata(vector_id, body.metadata)
    if not ok:
        raise HTTPException(404, f"Vector '{vector_id}' not found")
    return UpsertResponse(id=vector_id, status="patched")

@router.delete("/by-filter")
def delete_by_filter(collection_name: str, body: dict,
                            key: str = Depends(require_auth)):
    """D06 — bulk delete by metadata filter."""
    col = _col(collection_name)
    n = col.delete_by_filter(body.get("filter", {}))
    metrics.inc_delete(); prom.delete_total.inc(n)
    return {"deleted": n}

@router.delete("/{vector_id}")
def delete_vector(collection_name: str, vector_id: str,
                        key: str = Depends(require_auth)):
    col = _col(collection_name)
    if not col.delete(vector_id):
        raise HTTPException(404, f"Vector '{vector_id}' not found")
    metrics.inc_delete(); prom.delete_total.inc()
    return {"id": vector_id, "status": "deleted"}
