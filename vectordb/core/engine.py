"""Engine — manages all collections with registry, reaper, autosave."""
from __future__ import annotations
import json, logging, os, re, shutil, threading, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    COLLECTIONS_DIR, DATA_DIR, REGISTRY_FILE,
    REAPER_INTERVAL_SECONDS, AUTOSAVE_INTERVAL_SEC,
)
from core.collection import Collection

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self):
        self._collections: Dict[str, Collection] = {}
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        COLLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_registry()
        self._start_background()

    # ── Registry ──────────────────────────────────────────────────────────────
    def _load_registry(self):
        if not REGISTRY_FILE.exists():
            return
        try:
            entries = json.loads(REGISTRY_FILE.read_text())
            for e in entries:
                col = Collection(
                    name=e["name"],
                    data_dir=str(COLLECTIONS_DIR),
                    dimension=e.get("dimension", 384),
                    distance=e.get("distance", "cosine"),
                    index_type=e.get("index_type"),
                    quant_mode=e.get("quant_mode", "float32"),
                    description=e.get("description", ""),
                    hnsw_m=e.get("hnsw_m"),
                    hnsw_ef_construction=e.get("hnsw_ef_construction"),
                    created_at=e.get("created_at"),
                    updated_at=e.get("updated_at"),
                    ivfpq_m=e.get("ivfpq_m"),
                    ivfpq_nbits=e.get("ivfpq_nbits"),
                )
                self._collections[e["name"]] = col
                logger.info(f"Loaded '{e['name']}' "
                            f"({col.vector_count} vecs, {col.index_type})")
        except Exception as exc:
            logger.error(f"Registry load failed: {exc}")

    def _save_registry(self):
        with self._lock:
            entries = [c.to_dict() for c in self._collections.values()]
            tmp = str(REGISTRY_FILE) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(entries, f, indent=2)
            os.replace(tmp, str(REGISTRY_FILE))

    # ── CRUD ──────────────────────────────────────────────────────────────────
    def create(self, name, dimension=384, distance="cosine",
               index_type=None, quant_mode="float32", description="",
               hnsw_m=None, hnsw_ef_construction=None,
               ivfpq_m=None, ivfpq_nbits=None) -> Collection:
        with self._lock:
            if name in self._collections:
                raise ValueError(f"Collection '{name}' already exists")
            col = Collection(
                name=name, data_dir=str(COLLECTIONS_DIR),
                dimension=dimension, distance=distance,
                index_type=index_type, quant_mode=quant_mode,
                description=description, hnsw_m=hnsw_m,
                hnsw_ef_construction=hnsw_ef_construction,
                ivfpq_m=ivfpq_m, ivfpq_nbits=ivfpq_nbits,
            )
            self._collections[name] = col
            self._save_registry()
            logger.info(f"Created '{name}' (dim={dimension}, dist={distance})")
            return col

    def get(self, name: str) -> Collection:
        with self._lock:
            col = self._collections.get(name)
            if col is None:
                raise KeyError(f"Collection '{name}' not found")
            return col

    def list(self) -> List[Collection]:
        with self._lock:
            return list(self._collections.values())

    def delete(self, name: str) -> bool:
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            return False
        with self._lock:
            col = self._collections.pop(name, None)
            if col is not None:
                try:
                    col.close()
                except Exception as exc:
                    logger.warning(f"Could not close collection '{name}' before deletion: {exc}")
            
            folder = (COLLECTIONS_DIR / name).resolve()
            try:
                if not folder.is_relative_to(COLLECTIONS_DIR.resolve()):
                    if col is not None:
                        self._collections[name] = col
                    return False
            except Exception:
                if col is not None:
                    self._collections[name] = col
                return False

            folder_exists = folder.exists()
            if col is not None or folder_exists:
                if folder_exists:
                    try:
                        shutil.rmtree(folder)
                    except Exception as exc:
                        logger.warning(f"Could not remove files for '{name}': {exc}")
                self._save_registry()
                logger.info(f"Deleted collection '{name}'")
                return True
            return False

    def exists(self, name: str) -> bool:
        with self._lock:
            return name in self._collections

    # ── Stats ──────────────────────────────────────────────────────────────────
    def total_vectors(self) -> int:
        with self._lock:
            return sum(c.vector_count for c in self._collections.values())

    def per_collection_stats(self) -> Dict:
        with self._lock:
            return {
                name: {
                    "vector_count": col.vector_count,
                    "index_type": col.index_type.value,
                    "dimension": col.dimension,
                    "disk_size_bytes": col.disk_size_bytes,
                }
                for name, col in self._collections.items()
            }

    def save_all(self):
        with self._lock:
            cols = list(self._collections.values())
        for col in cols:
            try:
                col.force_save()
            except Exception as exc:
                logger.error(f"Save failed for '{col.name}': {exc}")
        self._save_registry()

    def _start_background(self):
        self._reaper_thread = threading.Thread(target=self._reaper_loop,
                                               daemon=True, name="vdb-reaper")
        self._autosave_thread = threading.Thread(target=self._autosave_loop,
                                                 daemon=True, name="vdb-autosave")
        self._reaper_thread.start()
        self._autosave_thread.start()

    def shutdown(self):
        self._shutdown_event.set()
        if hasattr(self, "_reaper_thread"):
            self._reaper_thread.join(timeout=5.0)
        if hasattr(self, "_autosave_thread"):
            self._autosave_thread.join(timeout=5.0)
        with self._lock:
            cols = list(self._collections.values())
        for col in cols:
            try:
                col.close()
            except Exception as exc:
                logger.error(f"Error closing collection '{col.name}': {exc}")

    def _reaper_loop(self):
        while not self._shutdown_event.is_set():
            if self._shutdown_event.wait(REAPER_INTERVAL_SECONDS):
                break
            with self._lock:
                cols = list(self._collections.values())
            for col in cols:
                try:
                    n = col.reap_expired()
                    if n:
                        logger.info(f"Reaper: removed {n} expired from '{col.name}'")
                except Exception as exc:
                    logger.error(f"Reaper error '{col.name}': {exc}")

    def _autosave_loop(self):
        while not self._shutdown_event.is_set():
            if self._shutdown_event.wait(AUTOSAVE_INTERVAL_SEC):
                break
            try:
                self.save_all()
                logger.debug("Auto-save complete")
            except Exception as exc:
                logger.error(f"Auto-save error: {exc}")


_engine: Optional[Engine] = None
_engine_lock = threading.Lock()


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = Engine()
    return _engine
