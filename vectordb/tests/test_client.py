"""
Comprehensive test suite for VectorDB Python Client (both sync and async).
Tests all CRUD endpoints, search modes, metadata filters, admin operations, and error handling.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["VDB_DATA_DIR"] = "/tmp/vdb_client_tests"
os.environ["VDB_RATE_LIMIT"] = "false"

from main import app
from client.client import VectorDBClient, AsyncVectorDBClient


@pytest.fixture(scope="module")
def client_env():
    # Use FastAPI TestClient as transport or mock
    with TestClient(app) as tc:
        yield tc


def test_sync_client_collection_lifecycle(monkeypatch):
    """Test VectorDBClient collection operations using test client."""
    from main import app as _app
    tc = TestClient(_app, headers={"X-API-Key": "admin-secret"})
    
    # Mock httpx in client to use TestClient
    client = VectorDBClient(api_key="admin-secret")
    client._client = tc

    # 1. Create collection
    col_name = "test_sync_col"
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    created = client.create_collection(col_name, dimension=8, distance="cosine", description="Sync Test")
    assert created["name"] == col_name
    assert created["dimension"] == 8

    # 2. List collections
    cols = client.list_collections()
    assert col_name in [c["name"] for c in cols]

    # 3. Get collection info
    info = client.get_collection(col_name)
    assert info["name"] == col_name

    # 4. Upsert single vector
    vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    res = client.upsert(col_name, "vec_1", vec, metadata={"tag": "demo", "price": 29.99}, ttl_seconds=3600)
    assert res["status"] in ("inserted", "updated")

    # 5. Get vector
    rec = client.get(col_name, "vec_1", include_vector=True)
    assert rec["id"] == "vec_1"
    assert len(rec["vector"]) == 8
    assert rec["metadata"]["tag"] == "demo"

    # 6. Count and facets
    cnt = client.count(col_name)
    assert cnt["count"] >= 1

    fac = client.facets(col_name, "tag")
    assert "demo" in fac["values"]

    # 7. Patch metadata
    patched = client.patch_metadata(col_name, "vec_1", {"tag": "updated_demo", "stock": 50})
    assert patched["status"] == "patched"

    # 8. Search
    results = client.search(col_name, vec, top_k=5)
    assert len(results) >= 1
    assert results[0]["id"] == "vec_1"

    # 9. Search by ID
    id_results = client.search_by_id(col_name, "vec_1", top_k=3)
    assert len(id_results) >= 1
    assert id_results[0]["id"] == "vec_1"

    # 10. Scroll
    scroll_res = client.scroll(col_name, limit=10)
    assert scroll_res["total"] >= 1

    # 11. Delete vector
    del_res = client.delete(col_name, "vec_1")
    assert del_res["status"] == "deleted"

    # 12. Delete collection
    client.delete_collection(col_name)


def test_sync_client_batch_operations():
    """Test batch upsert and batch search."""
    from main import app as _app
    tc = TestClient(_app, headers={"X-API-Key": "admin-secret"})
    client = VectorDBClient(api_key="admin-secret")
    client._client = tc

    col_name = "test_sync_batch_col"
    try:
        client.delete_collection(col_name)
    except Exception:
        pass
    client.create_collection(col_name, dimension=4, distance="cosine")

    vectors = [
        {"id": f"batch_{i}", "vector": [float(i), float(i+1), float(i+2), float(i+3)], "metadata": {"idx": i}}
        for i in range(10)
    ]
    batch_res = client.upsert_batch(col_name, vectors)
    assert batch_res["inserted"] == 10

    # Batch search
    queries = [
        {"vector": [0.0, 1.0, 2.0, 3.0], "top_k": 3},
        {"vector": [9.0, 10.0, 11.0, 12.0], "top_k": 3},
    ]
    bsearch_res = client.batch_search(col_name, queries)
    assert len(bsearch_res) == 2
    assert len(bsearch_res[0]) <= 3

    # Delete by filter
    client.delete_by_filter(col_name, {"idx": {"$lt": 5}})
    cnt = client.count(col_name)
    assert cnt["count"] == 5

    client.delete_collection(col_name)


def test_sync_client_admin_endpoints():
    """Test health, metrics, force_save, cache stats, and task management."""
    from main import app as _app
    tc = TestClient(_app, headers={"X-API-Key": "admin-secret"})
    client = VectorDBClient(api_key="admin-secret")
    client._client = tc

    # Health
    health = client.health()
    assert health["status"] == "ok"

    # Metrics
    metrics = client.metrics()
    assert "uptime_seconds" in metrics

    # Force save
    saved = client.force_save()
    assert saved["status"] == "saved"

    # Cache
    cstats = client.cache_stats()
    assert "size" in cstats

    cleared = client.clear_cache()
    assert cleared["status"] == "cleared"

    # Tasks
    task_list = client.list_tasks()
    assert isinstance(task_list, list)


@pytest.mark.asyncio
async def test_async_client_lifecycle():
    """Test AsyncVectorDBClient methods."""
    from httpx import AsyncClient, ASGITransport
    from main import app as _app

    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test", headers={"X-API-Key": "admin-secret"}) as ac:
        client = AsyncVectorDBClient(api_key="admin-secret")
        client._client = ac

        col_name = "test_async_col"
        try:
            await client.delete_collection(col_name)
        except Exception:
            pass

        created = await client.create_collection(col_name, dimension=4, distance="cosine")
        assert created["name"] == col_name

        # Upsert
        up = await client.upsert(col_name, "av1", [1.0, 0.0, 0.0, 0.0], {"type": "async"})
        assert up["status"] in ("inserted", "updated")

        # Get
        rec = await client.get(col_name, "av1", include_vector=True)
        assert rec["id"] == "av1"
        assert rec["metadata"]["type"] == "async"

        # Search
        results = await client.search(col_name, [1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 1
        assert results[0]["id"] == "av1"

        # Count
        cnt = await client.count(col_name)
        assert cnt["count"] == 1

        # Delete
        await client.delete(col_name, "av1")
        await client.delete_collection(col_name)
