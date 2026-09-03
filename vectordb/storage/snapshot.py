"""
Snapshot API — Point-in-time export/import with SHA256 checksum integrity verification
and corrupt snapshot recovery.
"""
from __future__ import annotations
import hashlib
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
from typing import TYPE_CHECKING, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.collection import Collection


def _file_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_snapshot(col: "Collection", output_path: str) -> Dict:
    """
    Export collection to a .tar.gz snapshot with SHA256 checksums.
    Returns manifest dict with snapshot metadata.
    """
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 1. Force save FAISS index
        col.force_save()

        # 2. Checkpoint SQLite WAL -> main file
        db_path = col._db.db_path
        _checkpoint_sqlite(db_path)

        # 3. Copy files into temp dir and compute checksums
        checksums: Dict[str, str] = {}
        index_src = Path(col._index_path)
        if index_src.exists():
            dest_index = tmp_dir / "index.faiss"
            shutil.copy2(index_src, dest_index)
            checksums["index.faiss"] = _file_sha256(dest_index)

        dest_db = tmp_dir / "metadata.db"
        shutil.copy2(db_path, dest_db)
        checksums["metadata.db"] = _file_sha256(dest_db)

        centroids_src = Path(col._centroids_path)
        if centroids_src.exists():
            dest_cent = tmp_dir / "centroids.npy"
            shutil.copy2(centroids_src, dest_cent)
            checksums["centroids.npy"] = _file_sha256(dest_cent)

        # 4. Write manifest with checksums
        manifest = {
            "version": "2.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection": col.to_dict(),
            "vector_count": col.vector_count,
            "faiss_index_exists": index_src.exists(),
            "checksums": checksums,
        }
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # 5. Pack into tar.gz
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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


def verify_snapshot(snapshot_path: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    Verifies snapshot archive integrity and SHA256 checksums without modifying system state.
    Returns (is_valid, message, manifest_dict).
    """
    if not os.path.exists(snapshot_path):
        return False, f"Snapshot file not found: {snapshot_path}", None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            with tarfile.open(snapshot_path, "r:gz") as tar:
                for member in tar.getmembers():
                    mpath = Path(member.name)
                    if mpath.is_absolute() or member.name.startswith("/") or member.name.startswith("\\"):
                        return False, f"Dangerous path in snapshot: {member.name}", None
                    if ".." in mpath.parts or ".." in member.name.replace("\\", "/").split("/"):
                        return False, f"Path traversal detected in snapshot: {member.name}", None
                tar.extractall(tmp_dir)

            manifest_file = tmp_dir / "manifest.json"
            if not manifest_file.exists():
                return False, "Missing manifest.json in snapshot archive", None

            manifest = json.loads(manifest_file.read_text())
            checksums = manifest.get("checksums", {})

            # Verify checksums of extracted files
            for fname, expected_hash in checksums.items():
                target_file = tmp_dir / fname
                if not target_file.exists():
                    return False, f"Manifest file missing from archive: {fname}", manifest
                actual_hash = _file_sha256(target_file)
                if actual_hash != expected_hash:
                    return False, f"Checksum mismatch for {fname}: expected {expected_hash}, got {actual_hash}", manifest

            return True, "Snapshot verified successfully", manifest

    except Exception as exc:
        return False, f"Snapshot verification error: {exc}", None


def import_snapshot(
    snapshot_path: str,
    target_data_dir: str,
    collection_name: str,
    verify_checksums: bool = True,
) -> Dict:
    """
    Import a .tar.gz snapshot into target_data_dir/collection_name/.
    Validates manifest, checks integrity, and handles corrupted files gracefully.
    """
    if verify_checksums:
        is_valid, msg, _ = verify_snapshot(snapshot_path)
        if not is_valid:
            raise ValueError(f"Corrupt or invalid snapshot: {msg}")

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

        # Copy database and index files to destination
        for fname in ("metadata.db", "index.faiss", "centroids.npy"):
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
