"""
Snapshot API — C02/C03.
Point-in-time export of a collection to a .tar.gz archive.
Archives contain:
  - index.faiss   (FAISS binary)
  - metadata.db   (SQLite WAL-checkpointed)
  - manifest.json (version, timestamp, collection settings)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.collection import Collection


def export_snapshot(col: "Collection", output_path: str) -> Dict:
    """
    Export collection to a .tar.gz snapshot.
    Returns manifest dict with metadata about the snapshot.
    """
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 1. Force save FAISS index
        col.force_save()

        # 2. Checkpoint SQLite WAL → main file before copying
        db_path = col._db.db_path
        _checkpoint_sqlite(db_path)

        # 3. Copy files into temp dir
        index_src = Path(col._index_path)
        if index_src.exists():
            shutil.copy2(index_src, tmp_dir / "index.faiss")

        shutil.copy2(db_path, tmp_dir / "metadata.db")

        # 4. Write manifest
        manifest = {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection": col.to_dict(),
            "vector_count": col.vector_count,
            "faiss_index_exists": index_src.exists(),
        }
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # 5. Pack into tar.gz
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(tmp_dir, arcname="")

    elapsed = time.time() - t0
    manifest["export_seconds"] = round(elapsed, 2)
    manifest["output_path"] = output_path
    manifest["size_bytes"] = os.path.getsize(output_path)
    logger.info(
        f"Snapshot created: {output_path} "
        f"({manifest['vector_count']} vecs, {manifest['size_bytes']} bytes, {elapsed:.2f}s)"
    )
    return manifest


def import_snapshot(
    snapshot_path: str,
    target_data_dir: str,
    collection_name: str,
) -> Dict:
    """
    Import a .tar.gz snapshot into target_data_dir/collection_name/.
    Returns the manifest from the archive.
    """
    dest = Path(target_data_dir) / collection_name
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with tarfile.open(snapshot_path, "r:gz") as tar:
            for member in tar.getmembers():
                mpath = Path(member.name)
                if mpath.is_absolute() or member.name.startswith("/") or member.name.startswith("\\"):
                    raise ValueError(f"Dangerous path in snapshot: {member.name}")
                if ".." in mpath.parts or ".." in member.name.replace("\\", "/").split("/"):
                    raise ValueError(f"Dangerous path in snapshot: {member.name}")
            tar.extractall(tmp_dir)

        manifest_file = tmp_dir / "manifest.json"
        if not manifest_file.exists():
            raise ValueError("Invalid snapshot: missing manifest.json")

        manifest = json.loads(manifest_file.read_text())

        # Copy files to destination
        for fname in ("index.faiss", "metadata.db"):
            src = tmp_dir / fname
            if src.exists():
                shutil.copy2(src, dest / fname)

    logger.info(f"Snapshot imported: {collection_name} from {snapshot_path}")
    return manifest


def _checkpoint_sqlite(db_path: str) -> None:
    """Force WAL checkpoint so the main .db file is fully up-to-date."""
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA wal_checkpoint(FULL)")
        conn.close()
    except Exception as exc:
        logger.warning(f"WAL checkpoint failed (non-fatal): {exc}")
