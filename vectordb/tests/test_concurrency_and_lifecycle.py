"""
Test suite for Concurrency, Cache, Rate Limiting, Tasks, Metrics, and System Lifecycle.
"""
from __future__ import annotations
import concurrent.futures
import time
import numpy as np
import pytest

from utils.cache import QueryCache
from utils.ratelimit import RateLimiter, TokenBucket
from utils.metrics import Metrics
from utils.prometheus import PrometheusRegistry
from utils.tasks import TaskStore
from utils.audit import record, get_audit_log
from core.collection import Collection


def test_token_bucket_rate_limiter():
    # 60 RPM = 1 token/sec, capacity 2
    tb = TokenBucket(capacity=2.0, rate=1.0)
    assert tb.consume(1.0) is True
    assert tb.consume(1.0) is True
    assert tb.consume(1.0) is False  # Exhausted

    limiter = RateLimiter(rpm=60, burst_multiplier=2.0)
    assert limiter.check("user-1", cost=1.0) is True
    assert limiter.remaining("user-1") >= 0.0


def test_query_cache_lru_and_ttl():
    cache = QueryCache(max_size=3, ttl_seconds=0.1)
    q1 = np.array([1.0, 0.0], dtype=np.float32)
    q2 = np.array([0.0, 1.0], dtype=np.float32)
    q3 = np.array([1.0, 1.0], dtype=np.float32)
    q4 = np.array([2.0, 2.0], dtype=np.float32)

    cache.put("col", q1, 5, [{"id": "res1"}])
    cache.put("col", q2, 5, [{"id": "res2"}])
    cache.put("col", q3, 5, [{"id": "res3"}])

    # Cache hit
    assert cache.get("col", q1, 5) is not None

    # Adding 4th element evicts LRU (which is q2 because q1 was accessed)
    cache.put("col", q4, 5, [{"id": "res4"}])
    assert cache.get("col", q2, 5) is None
    assert cache.get("col", q4, 5) is not None

    # TTL expiry
    time.sleep(0.15)
    assert cache.get("col", q1, 5) is None


def test_query_cache_invalidation_on_write():
    cache = QueryCache(max_size=10, ttl_seconds=60.0)
    q = np.array([1.0, 0.0], dtype=np.float32)
    cache.put("col_a", q, 5, ["res_a"])
    cache.put("col_b", q, 5, ["res_b"])

    assert cache.get("col_a", q, 5) is not None
    assert cache.get("col_b", q, 5) is not None

    invalidated = cache.invalidate_collection("col_a")
    assert invalidated >= 1
    assert cache.get("col_a", q, 5) is None
    assert cache.get("col_b", q, 5) is not None


def test_metrics_rolling_average():
    m = Metrics()
    m.inc_request()
    m.inc_upsert(10)
    m.inc_delete(2)
    m.inc_search(15.0)
    m.inc_search(25.0)

    snap = m.snapshot()
    assert snap["total_requests"] == 1
    assert snap["upserts"] == 10
    assert snap["deletes"] == 2
    assert snap["searches"] == 2
    assert np.isclose(snap["avg_search_ms"], 20.0)


def test_prometheus_registry_rendering():
    prom = PrometheusRegistry()
    prom.vector_count.set(1500)
    prom.collection_count.set(5)
    prom.search_total.inc(10)
    prom.search_latency.observe(0.015)

    text = prom.text()
    assert "vdb_vectors_total 1500" in text
    assert "vdb_collections_total 5" in text
    assert 'vdb_requests_total{op="search"} 10' in text
    assert "vdb_search_seconds_bucket" in text


def test_task_store_execution():
    store = TaskStore()

    def successful_job(x, y):
        return x + y

    task_id = store.submit("add_job", successful_job, 10, 20)
    time.sleep(0.1)

    task = store.get(task_id)
    assert task["status"] == "done"
    assert task["result"] == 30

    def failing_job():
        raise ValueError("Job crashed intentionally")

    fail_id = store.submit("fail_job", failing_job)
    time.sleep(0.1)
    fail_task = store.get(fail_id)
    assert fail_task["status"] == "error"
    assert "Job crashed" in fail_task["message"]


def test_audit_log_ring_buffer():
    for i in range(10):
        record(f"action_{i}", "secret_key_12345", f"/resource/{i}")

    logs = get_audit_log()
    assert len(logs) >= 10
    # API key masked
    assert logs[0]["key"].endswith("…")


def test_collection_concurrent_reads_and_writes(tmp_path):
    """Stress test: concurrent threads reading and writing to the same collection simultaneously."""
    col = Collection("stress_col", str(tmp_path), dimension=8, distance="cosine")
    
    # Pre-populate
    for i in range(20):
        col.upsert(f"init_{i}", np.random.randn(8).tolist(), {"idx": i})

    errors = []

    def writer_thread(worker_id: int):
        try:
            for j in range(20):
                vid = f"w{worker_id}_{j}"
                col.upsert(vid, np.random.randn(8).tolist(), {"worker": worker_id, "j": j})
        except Exception as e:
            errors.append(e)

    def reader_thread():
        try:
            for _ in range(30):
                q = np.random.randn(8).tolist()
                results, _ = col.search(q, top_k=5)
                assert isinstance(results, list)
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(writer_thread, 1),
            executor.submit(writer_thread, 2),
            executor.submit(reader_thread),
            executor.submit(reader_thread),
        ]
        concurrent.futures.wait(futures, timeout=10.0)

    assert errors == [], f"Concurrent collection stress failed: {errors}"
    assert col.vector_count >= 60
    col.close()
