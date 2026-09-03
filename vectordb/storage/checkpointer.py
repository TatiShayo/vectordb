"""
Atomic persistence and crash recovery for FAISS indexes.

Strategy:
  1. Write to a .tmp file in the same directory.
  2. os.replace() — atomic on POSIX, near-atomic on Windows (same FS).
  3. SQLite is crash-safe in WAL mode.

On startup / load:
  - If the .faiss file exists, attempt to load it.
  - If loading fails (file corrupt, truncated, invalid header) or vector count differs,
    rebuild the index cleanly from the SQLite metadata.db database.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from core.indexer import FAISSIndex, IndexType

logger = logging.getLogger(__name__)


def save(index: FAISSIndex, index_path: str) -> None:
    """Save FAISS index atomically."""
    tmp = index_path + ".tmp"
    index.save(tmp)
    os.replace(tmp, index_path)
    logger.debug(f"Saved FAISS index ({index.size} vectors) -> {index_path}")


def load_or_rebuild(index: FAISSIndex, index_path: str, db) -> None:
    """
    Load FAISS from disk; automatically recover and rebuild from SQLite
    if the index file is missing, stale, or corrupted.
    """
    if Path(index_path).exists():
        try:
            index.load(index_path)
            db_count = db.count()
            if abs(index.size - db_count) > 0:
                logger.warning(
                    f"FAISS has {index.size} vectors but SQLite has {db_count}. "
                    "Rebuilding index from SQLite."
                )
                _rebuild_from_db(index, db)
            else:
                logger.info(f"Loaded FAISS index: {index.size} vectors ({index.index_type})")
            return
        except Exception as exc:
            logger.warning(f"Could not load FAISS index ({exc}). Initiating automatic recovery rebuild.")

    _rebuild_from_db(index, db)


def _rebuild_from_db(index: FAISSIndex, db) -> None:
    vectors, faiss_ids = db.all_vectors_for_rebuild()
    n = len(faiss_ids)
    if n == 0:
        logger.info("Empty collection — starting with a fresh Flat index.")
        return

    from core.indexer import _auto_type

    best_type = _auto_type(n)
    logger.info(f"Rebuilding index ({best_type}) from {n} SQLite vectors ...")
    index.rebuild(vectors, faiss_ids, new_type=best_type)
    logger.info("Rebuild complete.")
