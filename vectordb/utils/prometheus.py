"""
Prometheus /metrics endpoint — G01.
Falls back to a plain-text stub if prometheus_client isn't installed.
"""
from __future__ import annotations
import time
from typing import Dict


class _Gauge:
    def __init__(self): self._v = 0.0
    def set(self, v): self._v = float(v)
    def inc(self, v=1): self._v += v
    def get(self): return self._v

class _Counter:
    def __init__(self): self._v = 0.0
    def inc(self, v=1): self._v += v
    def get(self): return self._v

class _Histogram:
    _BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, float("inf")]
    def __init__(self):
        self._count = 0; self._sum = 0.0
        self._buckets = [0] * len(self._BUCKETS)
    def observe(self, v):
        self._count += 1; self._sum += v
        for i, b in enumerate(self._BUCKETS):
            if v <= b: self._buckets[i] += 1


class PrometheusRegistry:
    def __init__(self):
        self.search_latency = _Histogram()
        self.upsert_total   = _Counter()
        self.search_total   = _Counter()
        self.delete_total   = _Counter()
        self.vector_count   = _Gauge()
        self.collection_count = _Gauge()
        self.cache_hits     = _Counter()
        self.cache_misses   = _Counter()
        self._start = time.time()

    def text(self) -> str:
        lines = [
            "# HELP vdb_uptime_seconds Seconds since start",
            "# TYPE vdb_uptime_seconds gauge",
            f"vdb_uptime_seconds {time.time()-self._start:.1f}",
            "# HELP vdb_requests_total Total requests by type",
            "# TYPE vdb_requests_total counter",
            f'vdb_requests_total{{op="upsert"}} {self.upsert_total.get()}',
            f'vdb_requests_total{{op="search"}} {self.search_total.get()}',
            f'vdb_requests_total{{op="delete"}} {self.delete_total.get()}',
            "# HELP vdb_vectors_total Current vector count",
            "# TYPE vdb_vectors_total gauge",
            f"vdb_vectors_total {self.vector_count.get()}",
            "# HELP vdb_collections_total Collection count",
            "# TYPE vdb_collections_total gauge",
            f"vdb_collections_total {self.collection_count.get()}",
            "# HELP vdb_cache_hits_total Cache hit count",
            "# TYPE vdb_cache_hits_total counter",
            f"vdb_cache_hits_total {self.cache_hits.get()}",
            "# HELP vdb_cache_misses_total Cache miss count",
            "# TYPE vdb_cache_misses_total counter",
            f"vdb_cache_misses_total {self.cache_misses.get()}",
        ]
        # Histogram buckets
        h = self.search_latency
        lines += ["# HELP vdb_search_seconds Search latency",
                  "# TYPE vdb_search_seconds histogram"]
        for b, count in zip(h._BUCKETS, h._buckets):
            le = "+Inf" if b == float("inf") else str(b)
            lines.append(f'vdb_search_seconds_bucket{{le="{le}"}} {count}')
        lines += [f"vdb_search_seconds_count {h._count}",
                  f"vdb_search_seconds_sum {h._sum:.6f}"]
        return "\n".join(lines) + "\n"


prom = PrometheusRegistry()
