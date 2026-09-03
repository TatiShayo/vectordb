"""
Test suite for Snapshot Persistence, Checkpointer, Integrity Checks, and Corrupt File Recovery.
"""
from __future__ import annotations
import json
import os
import shutil
import tempfile
import numpy as np
import pytest

from core.engine import Engine
from core.indexer import FAISSIndex, IndexType
from storage.checkpointer import save, load_or_rebuild
from storage.snapshot import export_snapshot, import_snapshot, verify_snapshot


@pytest.fixture
def test_env():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.environ["VDB_DATA_DIR"] = data_dir
        engine = Engine()
        yield engine, data_dir
        engine.shutdown()


def test_atomic_checkpointer_save_and_load(test_env):
    engine, data_dir = test_env
    col = engine.create("col_atomic", dimension=8, distance="cosine")
    
    # Add vectors
    vecs = np.random.randn(20, 8).astype(np.float32)
    items = [{"id": f"v{i}", "vector": v.tolist(), "metadata": {"i": i}} for i, v in enumerate(vecs)]
    col.upsert_batch(items)
    col.force_save()

    index_path = col._index_path
    assert os.path.exists(index_path)

    # Verify atomic saving: temp file should not persist
    assert not os.path.exists(index_path + ".tmp")


def test_corrupt_index_recovery(test_env):
    """Corrupt index.faiss on disk and verify load_or_rebuild automatically recovers from SQLite."""
    engine, data_dir = test_env
    col = engine.create("col_corrupt", dimension=8, distance="cosine")
    
    vecs = np.random.randn(25, 8).astype(np.float32)
    items = [{"id": f"v{i}", "vector": v.tolist(), "metadata": {"i": i}} for i, v in enumerate(vecs)]
    col.upsert_batch(items)
    col.force_save()

    index_path = col._index_path
    assert os.path.exists(index_path)

    # Intentionally corrupt the index file with garbage bytes
    with open(index_path, "wb") as f:
        f.write(b"CORRUPTED_GARBAGE_BYTES_HEADER_INVALID_1234567890")

    # Re-initialize index object and load
    new_index = FAISSIndex(dimension=8)
    load_or_rebuild(new_index, index_path, col._db)

    # Should recover all 25 vectors cleanly from SQLite
    assert new_index.size == 25
    q = vecs[0].reshape(1, -1)
    ids, scores = new_index.search(q, top_k=5)
    assert len(ids) == 5


def test_count_mismatch_triggers_rebuild(test_env):
    """When index size != db count (e.g. crash mid-operation), rebuild occurs."""
    engine, data_dir = test_env
    col = engine.create("col_mismatch", dimension=8, distance="cosine")
    
    vecs = np.random.randn(10, 8).astype(np.float32)
    items = [{"id": f"v{i}", "vector": v.tolist(), "metadata": {"i": i}} for i, v in enumerate(vecs)]
    col.upsert_batch(items)
    col.force_save()

    # Add 5 more vectors directly to DB without updating FAISS file
    extra = np.random.randn(5, 8).astype(np.float32)
    extra_fids = col._db.next_faiss_ids(5)
    col._db.upsert_batch([(f"extra_{i}", fid, v, {}, None, None) for i, (fid, v) in enumerate(zip(extra_fids, extra))])

    assert col._db.count() == 15

    new_index = FAISSIndex(dimension=8)
    load_or_rebuild(new_index, col._index_path, col._db)

    # Rebuilt to match the DB count of 15
    assert new_index.size == 15


def test_snapshot_export_verify_and_import(test_env):
    engine, data_dir = test_env
    col = engine.create("col_snap", dimension=8, distance="cosine", description="Snapshot Test")

    vecs = np.random.randn(30, 8).astype(np.float32)
    items = [{"id": f"v{i}", "vector": v.tolist(), "metadata": {"tag": "snap"}} for i, v in enumerate(vecs)]
    col.upsert_batch(items)

    snap_path = os.path.join(data_dir, "snapshots", "col_snap.tar.gz")
    manifest = export_snapshot(col, snap_path)

    assert os.path.exists(snap_path)
    assert manifest["vector_count"] == 30
    assert "checksums" in manifest
    assert "metadata.db" in manifest["checksums"]

    # 1. Verify snapshot integrity
    is_valid, msg, loaded_manifest = verify_snapshot(snap_path)
    assert is_valid is True, msg
    assert loaded_manifest["vector_count"] == 30

    # 2. Import into a restored collection
    restore_target = os.path.join(data_dir, "restored_cols")
    restored_manifest = import_snapshot(snap_path, restore_target, "col_snap_restored")
    assert restored_manifest["vector_count"] == 30
    assert os.path.exists(os.path.join(restore_target, "col_snap_restored", "metadata.db"))


def test_snapshot_tamper_detection(test_env):
    """Tampering with snapshot bytes fails verify_snapshot."""
    engine, data_dir = test_env
    col = engine.create("col_tamper", dimension=8, distance="cosine")
    col.upsert("v1", [1.0] * 8, {"k": "v"})

    snap_path = os.path.join(data_dir, "snapshots", "tamper.tar.gz")
    export_snapshot(col, snap_path)

    # Corrupt the tar.gz file
    with open(snap_path, "r+b") as f:
        f.seek(100)
        f.write(b"\xFF\xFF\xFF\xFF\xFF")

    is_valid, msg, _ = verify_snapshot(snap_path)
    assert is_valid is False
    assert "error" in msg.lower() or "mismatch" in msg.lower() or "tar" in msg.lower()
