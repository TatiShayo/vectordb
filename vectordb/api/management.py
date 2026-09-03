"""Management router — health, metrics, rebuild, snapshot, tasks, cache."""
from __future__ import annotations
import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse
from api.auth import require_admin, require_auth
from core.engine import get_engine
from models.schemas import HealthResponse, MetricsResponse, SnapshotResponse, TaskResponse
from utils.metrics import metrics
from utils.prometheus import prom
from utils.tasks import tasks
from utils.cache import get_cache

router = APIRouter(prefix="/admin", tags=["admin"])
VERSION = "2.0.0"


@router.get("/health", response_model=HealthResponse)
def health():
    engine = get_engine()
    return HealthResponse(status="ok", collections=len(engine.list()),
                          total_vectors=engine.total_vectors(), version=VERSION)

@router.get("/health/ready", response_model=HealthResponse)
def ready():
    """G04 — readiness (indexes loaded)."""
    engine = get_engine()
    cols = engine.list()
    return HealthResponse(status="ready" if cols is not None else "loading",
                          collections=len(cols), total_vectors=engine.total_vectors(),
                          version=VERSION)

@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(key: str = Depends(require_auth)):
    engine = get_engine()
    snap = metrics.snapshot()
    cache_stats = get_cache().stats()
    # Update prometheus gauges
    prom.vector_count.set(engine.total_vectors())
    prom.collection_count.set(len(engine.list()))
    prom.cache_hits._v = cache_stats["hits"]
    prom.cache_misses._v = cache_stats["misses"]
    return MetricsResponse(**snap, cache=cache_stats,
                           collections=engine.per_collection_stats())

@router.get("/metrics/prometheus", response_class=PlainTextResponse)
def prometheus_metrics():
    """G01 — Prometheus scrape endpoint (no auth required for scraper)."""
    engine = get_engine()
    prom.vector_count.set(engine.total_vectors())
    prom.collection_count.set(len(engine.list()))
    return PlainTextResponse(prom.text(),
                             media_type="text/plain; version=0.0.4")

@router.post("/save")
def force_save(key: str = Depends(require_admin)):
    get_engine().save_all()
    return {"status": "saved"}

@router.post("/cache/clear")
def clear_cache(key: str = Depends(require_admin)):
    get_cache().clear()
    return {"status": "cleared"}

@router.get("/cache/stats")
def cache_stats(key: str = Depends(require_auth)):
    return get_cache().stats()

# ── Per-collection ops ────────────────────────────────────────────────────────

@router.post("/collections/{collection_name}/rebuild")
def rebuild_index(collection_name: str, key: str = Depends(require_admin)):
    try: col = get_engine().get(collection_name)
    except KeyError: raise HTTPException(404, f"Collection '{collection_name}' not found")

    def _do(): return col.rebuild_index()
    task_id = tasks.submit(f"rebuild:{collection_name}", _do)
    return {"task_id": task_id, "status": "started",
            "collection": collection_name}

@router.post("/collections/{collection_name}/reap")
def reap_expired(collection_name: str, key: str = Depends(require_admin)):
    try: col = get_engine().get(collection_name)
    except KeyError: raise HTTPException(404, f"Collection '{collection_name}' not found")
    n = col.reap_expired()
    return {"reaped": n}

# ── Snapshots ─────────────────────────────────────────────────────────────────

@router.post("/collections/{collection_name}/snapshot",
             response_model=SnapshotResponse)
def create_snapshot(collection_name: str, key: str = Depends(require_admin)):
    """C02 — export collection to a .tar.gz snapshot."""
    try: col = get_engine().get(collection_name)
    except KeyError: raise HTTPException(404, f"Collection '{collection_name}' not found")

    from config import SNAPSHOT_DIR
    from datetime import datetime, timezone
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = str(SNAPSHOT_DIR / f"{collection_name}_{ts}.tar.gz")

    from storage.snapshot import export_snapshot
    manifest = export_snapshot(col, out_path)
    return SnapshotResponse(
        collection=collection_name,
        path=out_path,
        size_bytes=manifest["size_bytes"],
        vector_count=manifest["vector_count"],
        created_at=manifest["created_at"],
        export_seconds=manifest.get("export_seconds", 0),
    )

@router.get("/collections/{collection_name}/snapshot/download")
def download_snapshot(collection_name: str, key: str = Depends(require_admin)):
    """Stream the latest snapshot file as a download."""
    from config import SNAPSHOT_DIR
    snaps = sorted(SNAPSHOT_DIR.glob(f"{collection_name}_*.tar.gz"), reverse=True)
    if not snaps:
        raise HTTPException(404, "No snapshot found — create one first")
    return FileResponse(str(snaps[0]),
                        media_type="application/gzip",
                        filename=snaps[0].name)

@router.post("/collections/{collection_name}/restore")
def restore_snapshot(collection_name: str, snapshot_path: str,
                           key: str = Depends(require_admin)):
    """C03 — import snapshot into a new or existing collection."""
    from config import SNAPSHOT_DIR, COLLECTIONS_DIR
    try:
        resolved_path = (SNAPSHOT_DIR / Path(snapshot_path).name).resolve()
        if not resolved_path.is_relative_to(SNAPSHOT_DIR.resolve()):
            raise HTTPException(400, "Invalid snapshot path")
    except Exception:
        raise HTTPException(400, "Invalid snapshot path")
    
    snapshot_path = str(resolved_path)
    if not resolved_path.exists():
        raise HTTPException(404, f"Snapshot not found: {snapshot_path}")
    from storage.snapshot import import_snapshot
    manifest = import_snapshot(snapshot_path, str(COLLECTIONS_DIR), collection_name)
    # Re-register collection in engine from restored files
    engine = get_engine()
    col_settings = manifest.get("collection", {})
    try:
        col = engine.get(collection_name)
        from storage.checkpointer import load_or_rebuild
        load_or_rebuild(col._index, col._index_path, col._db)
        col._load_sparse()
    except KeyError:
        engine.create(
            name=collection_name,
            dimension=col_settings.get("dimension", 384),
            distance=col_settings.get("distance", "cosine"),
            index_type=col_settings.get("index_type"),
            quant_mode=col_settings.get("quant_mode", "float32"),
            description=col_settings.get("description", "Restored from snapshot"),
            hnsw_m=col_settings.get("hnsw_m"),
            hnsw_ef_construction=col_settings.get("hnsw_ef_construction"),
            ivfpq_m=col_settings.get("ivfpq_m"),
            ivfpq_nbits=col_settings.get("ivfpq_nbits"),
        )
    return {"status": "restored", "collection": collection_name,
            "vector_count": manifest.get("vector_count", 0)}


# ── Tasks ─────────────────────────────────────────────────────────────────────

@router.get("/tasks", response_model=list)
def list_tasks(key: str = Depends(require_auth)):
    """G09 — list background tasks."""
    return tasks.list()

@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, key: str = Depends(require_auth)):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return TaskResponse(**task, message=task.get("message",""))

# ── Audit log stub ────────────────────────────────────────────────────────────

@router.get("/audit")
def get_audit_log(key: str = Depends(require_admin)):
    """F05 — last 500 audit events."""
    from utils.audit import get_audit_log as _log
    return _log()
