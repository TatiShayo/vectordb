"""Collections router."""
from __future__ import annotations
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timezone
from api.auth import require_auth, require_admin
from core.engine import get_engine
from models.schemas import (
    CollectionCreate, CollectionInfo, CollectionUpdate, DistanceMetric,
)

router = APIRouter(prefix="/collections", tags=["collections"])


def _info(col) -> CollectionInfo:
    return CollectionInfo(
        name=col.name, dimension=col.dimension,
        distance=DistanceMetric(col.distance),
        index_type=col.index_type.value,
        quant_mode=col.quant_mode,
        description=col.description,
        vector_count=col.vector_count,
        disk_size_bytes=col.disk_size_bytes,
        created_at=col.created_at, updated_at=col.updated_at,
        ivfpq_m=col.ivfpq_m,
        ivfpq_nbits=col.ivfpq_nbits,
    )


@router.get("", response_model=List[CollectionInfo])
def list_collections(key: str = Depends(require_auth)):
    return [_info(c) for c in get_engine().list()]


@router.post("", response_model=CollectionInfo, status_code=201)
def create_collection(body: CollectionCreate,
                            key: str = Depends(require_admin)):
    try:
        col = get_engine().create(
            name=body.name, dimension=body.dimension,
            distance=body.distance.value,
            index_type=body.index_type.value if body.index_type else None,
            quant_mode=body.quant_mode.value,
            description=body.description,
            hnsw_m=body.hnsw_m, hnsw_ef_construction=body.hnsw_ef_construction,
            ivfpq_m=body.ivfpq_m, ivfpq_nbits=body.ivfpq_nbits,
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return _info(col)


@router.get("/{name}", response_model=CollectionInfo)
def get_collection(name: str, key: str = Depends(require_auth)):
    try:
        return _info(get_engine().get(name))
    except KeyError:
        raise HTTPException(404, f"Collection '{name}' not found")


@router.patch("/{name}", response_model=CollectionInfo)
def update_collection(name: str, body: CollectionUpdate,
                            key: str = Depends(require_admin)):
    try:
        col = get_engine().get(name)
    except KeyError:
        raise HTTPException(404, f"Collection '{name}' not found")
    if body.description is not None:
        col.description = body.description
        col.updated_at = datetime.now(timezone.utc).isoformat()
        get_engine()._save_registry()
    return _info(col)


@router.delete("/{name}", status_code=204)
def delete_collection(name: str, key: str = Depends(require_admin)):
    if not get_engine().delete(name):
        raise HTTPException(404, f"Collection '{name}' not found")
