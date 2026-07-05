"""VectorDB v2 test suite — 35 tests."""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["VDB_DATA_DIR"] = "/tmp/vdb_test_v2"
os.environ["VDB_RATE_LIMIT"] = "false"

from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def app():
    from main import app as _app
    return _app

@pytest.fixture(scope="session")
def client(app):
    return TestClient(app, headers={"X-API-Key": "user-secret"})

@pytest.fixture(scope="session")
def admin(app):
    return TestClient(app, headers={"X-API-Key": "admin-secret"})

@pytest.fixture(scope="session")
def col(admin):
    admin.delete("/collections/tc")
    r = admin.post("/collections", json={
        "name":"tc","dimension":8,"distance":"cosine","description":"Test"
    })
    assert r.status_code == 201, r.text
    yield "tc"
    admin.delete("/collections/tc")

def rv(dim=8):
    v = np.random.randn(dim).astype(np.float32)
    return (v/np.linalg.norm(v)).tolist()


# ── Auth ──────────────────────────────────────────────────────────────────────
def test_no_key_401(app):
    assert TestClient(app).post("/collections",json={"name":"x","dimension":4}).status_code==401

def test_bad_key_403(app):
    assert TestClient(app,headers={"X-API-Key":"bad"}).get("/collections").status_code==403

def test_health_public(app):
    r = TestClient(app).get("/admin/health")
    assert r.status_code==200 and r.json()["status"]=="ok"

def test_health_ready(app):
    r = TestClient(app).get("/admin/health/ready")
    assert r.status_code==200

# ── Collections ───────────────────────────────────────────────────────────────
def test_create_list(admin, col):
    r = admin.get("/collections")
    assert r.status_code==200
    assert col in [c["name"] for c in r.json()]

def test_create_duplicate_409(admin, col):
    assert admin.post("/collections",json={"name":col,"dimension":8}).status_code==409

def test_get_collection(client, col):
    r = client.get(f"/collections/{col}")
    assert r.status_code==200 and r.json()["dimension"]==8

def test_get_missing_404(client):
    assert client.get("/collections/nope").status_code==404

def test_update_description(admin, col):
    r = admin.patch(f"/collections/{col}", json={"description":"Updated"})
    assert r.status_code==200 and r.json()["description"]=="Updated"

# ── Upsert ────────────────────────────────────────────────────────────────────
def test_insert(client, col):
    r = client.post(f"/collections/{col}/vectors",
                    json={"id":"v1","vector":rv(),"metadata":{"cat":"a"}})
    assert r.status_code==200 and r.json()["status"]=="inserted"

def test_update(client, col):
    r = client.post(f"/collections/{col}/vectors",
                    json={"id":"v1","vector":rv(),"metadata":{"cat":"b"}})
    assert r.status_code==200 and r.json()["status"]=="updated"

def test_wrong_dim_422(client, col):
    assert client.post(f"/collections/{col}/vectors",
                       json={"id":"bad","vector":[.1,.2],"metadata":{}}).status_code==422

def test_batch_upsert(client, col):
    vecs = [{"id":f"b{i}","vector":rv(),"metadata":{"i":i,"cat":"batch"}} for i in range(20)]
    r = client.post(f"/collections/{col}/vectors/batch", json={"vectors":vecs})
    assert r.status_code==200
    d = r.json()
    assert d["inserted"]==20 and d["errors"]==[]

def test_patch_metadata(client, col):
    r = client.patch(f"/collections/{col}/vectors/v1", json={"metadata":{"cat":"patched"}})
    assert r.status_code==200 and r.json()["status"]=="patched"
    assert client.get(f"/collections/{col}/vectors/v1").json()["metadata"]["cat"]=="patched"

# ── Read ──────────────────────────────────────────────────────────────────────
def test_get_vector(client, col):
    r = client.get(f"/collections/{col}/vectors/v1")
    assert r.status_code==200 and r.json()["id"]=="v1"

def test_get_with_vector(client, col):
    r = client.get(f"/collections/{col}/vectors/v1?include_vector=true")
    assert r.status_code==200 and len(r.json()["vector"])==8

def test_get_missing_404(client, col):
    assert client.get(f"/collections/{col}/vectors/ghost").status_code==404

def test_count(client, col):
    r = client.get(f"/collections/{col}/vectors/count")
    assert r.status_code==200 and r.json()["count"]>=1

def test_facets(client, col):
    r = client.get(f"/collections/{col}/vectors/facets/cat")
    assert r.status_code==200
    vals = r.json()["values"]
    assert "batch" in vals

def test_scroll(client, col):
    r = client.get(f"/collections/{col}/vectors/scroll?limit=5&offset=0")
    assert r.status_code==200
    d = r.json()
    assert len(d["vectors"])<=5 and d["total"]>0

# ── Delete ────────────────────────────────────────────────────────────────────
def test_delete_vector(client, col):
    client.post(f"/collections/{col}/vectors",json={"id":"del_me","vector":rv(),"metadata":{}})
    assert client.delete(f"/collections/{col}/vectors/del_me").status_code==200
    assert client.get(f"/collections/{col}/vectors/del_me").status_code==404

def test_delete_missing_404(client, col):
    assert client.delete(f"/collections/{col}/vectors/nope").status_code==404

# ── Search ────────────────────────────────────────────────────────────────────
def test_search(client, col):
    r = client.post(f"/collections/{col}/search",json={"vector":rv(),"top_k":5})
    assert r.status_code==200
    d = r.json()
    assert len(d["results"])<=5 and d["total_returned"]==len(d["results"])

def test_search_by_id_self(client, col):
    r = client.post(f"/collections/{col}/search/by-id",json={"id":"v1","top_k":3})
    assert r.status_code==200
    results = r.json()["results"]
    assert results and results[0]["id"]=="v1"

def test_search_filter(client, col):
    r = client.post(f"/collections/{col}/search",
                    json={"vector":rv(),"top_k":10,"filter":{"cat":"batch"}})
    assert r.status_code==200
    for res in r.json()["results"]:
        assert res["metadata"].get("cat")=="batch"

def test_search_score_threshold(client, col):
    r = client.post(f"/collections/{col}/search",
                    json={"vector":rv(),"top_k":10,"score_threshold":0.99})
    for res in r.json()["results"]:
        assert res["score"]>=0.99

def test_search_mmr(client, col):
    r = client.post(f"/collections/{col}/search",
                    json={"vector":rv(),"top_k":5,"use_mmr":True,"mmr_lambda":0.5})
    assert r.status_code==200

def test_batch_search(client, col):
    qs = [{"vector":rv(),"top_k":3} for _ in range(4)]
    r = client.post(f"/collections/{col}/search/batch",json={"queries":qs})
    assert r.status_code==200 and len(r.json()["responses"])==4

def test_hybrid_search(client, col):
    r = client.post(f"/collections/{col}/search/hybrid",
                    json={"vector":rv(),"text":"batch demo","top_k":3})
    assert r.status_code==200

def test_cache_hit(client, col):
    q = rv()
    r1 = client.post(f"/collections/{col}/search",json={"vector":q,"top_k":3})
    r2 = client.post(f"/collections/{col}/search",json={"vector":q,"top_k":3})
    assert r2.status_code==200
    assert r2.json().get("cached") is True  # second hit should be cached

# ── TTL ───────────────────────────────────────────────────────────────────────
def test_ttl_sets_expires_at(client, col):
    client.post(f"/collections/{col}/vectors",
                json={"id":"ttl1","vector":rv(),"metadata":{},"ttl_seconds":3600})
    r = client.get(f"/collections/{col}/vectors/ttl1")
    assert r.status_code==200 and r.json()["expires_at"] is not None

# ── Admin ─────────────────────────────────────────────────────────────────────
def test_metrics(client):
    r = client.get("/admin/metrics")
    assert r.status_code==200
    m = r.json()
    assert "uptime_seconds" in m and m["upserts"]>0

def test_prometheus(app):
    r = TestClient(app).get("/admin/metrics/prometheus")
    assert r.status_code==200 and "vdb_" in r.text

def test_cache_stats(client):
    r = client.get("/admin/cache/stats")
    assert r.status_code==200 and "hits" in r.json()

def test_force_save(admin):
    assert admin.post("/admin/save").status_code==200

def test_rebuild_starts_task(admin, col):
    r = admin.post(f"/admin/collections/{col}/rebuild")
    assert r.status_code==200 and "task_id" in r.json()

def test_tasks_list(client):
    r = client.get("/admin/tasks")
    assert r.status_code==200

def test_persistence_files(admin, col):
    admin.post("/admin/save")
    from config import COLLECTIONS_DIR, REGISTRY_FILE
    assert (COLLECTIONS_DIR/col/"index.faiss").exists()
    assert (COLLECTIONS_DIR/col/"metadata.db").exists()
    assert REGISTRY_FILE.exists()
    reg = json.loads(REGISTRY_FILE.read_text())
    assert col in [e["name"] for e in reg]
