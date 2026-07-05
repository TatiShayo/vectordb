import os
import shutil
import sys
import uuid
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["VDB_DATA_DIR"] = "/tmp/vdb_test_v2"

from core.engine import Engine
from core.indexer import IndexType
from models.schemas import IndexType as SchemaIndexType

def test_ivfpq_and_centroids():
    engine = Engine()
    
    # Generate a unique collection name to avoid file locking conflicts
    col_name = f"test_pq_col_{uuid.uuid4().hex[:8]}"
    if engine.exists(col_name):
        engine.delete(col_name)

    # Create IVFPQ collection
    col = engine.create(
        name=col_name,
        dimension=128,
        distance="cosine",
        index_type=SchemaIndexType.IVFPQ,
        ivfpq_m=8,
        ivfpq_nbits=6,
        description="IVF-PQ Caching Test"
    )

    # 1. Verify params are stored
    assert col.ivfpq_m == 8
    assert col.ivfpq_nbits == 6
    assert col.index_type == IndexType.IVFPQ

    # 2. Check registry saving
    registry_data = col.to_dict()
    assert registry_data["ivfpq_m"] == 8
    assert registry_data["ivfpq_nbits"] == 6

    # 3. Add vectors and trigger training/rebuild
    np.random.seed(42)
    vectors = np.random.randn(300, 128).astype(np.float32)
    # Normalize for cosine similarity
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    items = []
    for i, vec in enumerate(vectors):
        items.append({
            "id": f"v{i}",
            "vector": vec.tolist(),
            "metadata": {"index": i}
        })
    col.upsert_batch(items)
    
    # Rebuild to force training/indexing
    col.rebuild_index()

    # Verify that centroids.npy was saved
    assert col._centroids_path.exists()
    
    # Load centroids and verify shape
    centroids = np.load(str(col._centroids_path))
    # nlist defaults to max(IVF_NLIST, _auto_nlist(0)) which is 100
    assert centroids.shape == (100, 128)

    # Verify search is functional
    q = np.random.randn(128).astype(np.float32)
    q = q / np.linalg.norm(q)
    results, cached = col.search(q.tolist(), top_k=5)
    assert len(results) <= 5

    # 4. Graceful stop and restart
    engine.shutdown()

    # Re-initialize engine to verify loading centroids from disk
    engine2 = Engine()
    col2 = engine2.get(col_name)

    assert col2.ivfpq_m == 8
    assert col2.ivfpq_nbits == 6
    assert col2.index_type == IndexType.IVFPQ
    
    # Check that it loads centroids from index
    assert col2._index.centroids is not None
    assert col2._index.centroids.shape == (100, 128)

    # Ensure search still works on loaded collection
    results2, cached2 = col2.search(q.tolist(), top_k=5)
    assert len(results2) <= 5

    # Clean up
    engine2.delete(col_name)
    engine2.shutdown()
