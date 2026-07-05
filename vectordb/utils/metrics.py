"""In-memory rolling metrics."""
from __future__ import annotations
import time, threading
from collections import deque
from typing import Deque, Dict


class _Window:
    def __init__(self, n=1000):
        self._s: Deque[float] = deque(maxlen=n)
        self._l = threading.Lock()
    def record(self, v):
        with self._l: self._s.append(v)
    def avg(self):
        with self._l:
            return sum(self._s)/len(self._s) if self._s else 0.0


class Metrics:
    def __init__(self):
        self._start = time.time()
        self._lock = threading.Lock()
        self.requests = 0
        self.searches = 0
        self.upserts  = 0
        self.deletes  = 0
        self._slat = _Window()

    def inc_request(self):
        with self._lock: self.requests += 1

    def inc_search(self, ms=0.0):
        with self._lock: self.searches += 1
        self._slat.record(ms)

    def inc_upsert(self, n=1):
        with self._lock: self.upserts += n

    def inc_delete(self, n=1):
        with self._lock: self.deletes += n

    def snapshot(self) -> Dict:
        with self._lock:
            return {
                "uptime_seconds": round(time.time()-self._start, 1),
                "total_requests": self.requests,
                "searches": self.searches,
                "upserts": self.upserts,
                "deletes": self.deletes,
                "avg_search_ms": round(self._slat.avg(), 2),
            }


metrics = Metrics()
