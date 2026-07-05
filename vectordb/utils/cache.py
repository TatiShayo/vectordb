"""
LRU query cache — E01.

Caches (collection_name, vector_hash, top_k, filter_hash) → results.
Thread-safe. Configurable max size and TTL per entry.
"""
from __future__ import annotations

import hashlib
import json
import logging
import struct
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _hash_vector(v: np.ndarray) -> str:
    """Fast hash of a float32 vector via its bytes."""
    return hashlib.md5(v.astype(np.float32).tobytes()).hexdigest()


def _hash_filter(f: Optional[Dict]) -> str:
    if not f:
        return "none"
    return hashlib.md5(
        json.dumps(f, sort_keys=True).encode()
    ).hexdigest()[:8]


class QueryCache:
    """Thread-safe LRU cache for vector search results."""

    def __init__(self, max_size: int = 1024, ttl_seconds: float = 60.0):
        self._max = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _key(
        self,
        collection: str,
        vector: np.ndarray,
        top_k: int,
        filter_dict: Optional[Dict],
        mode: str = "search",
    ) -> str:
        vh = _hash_vector(vector)
        fh = _hash_filter(filter_dict)
        return f"{mode}:{collection}:{vh}:{top_k}:{fh}"

    def get(
        self,
        collection: str,
        vector: np.ndarray,
        top_k: int,
        filter_dict: Optional[Dict] = None,
        mode: str = "search",
    ) -> Optional[Any]:
        key = self._key(collection, vector, top_k, filter_dict, mode)
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            ts, value = self._store[key]
            if time.time() - ts > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(
        self,
        collection: str,
        vector: np.ndarray,
        top_k: int,
        value: Any,
        filter_dict: Optional[Dict] = None,
        mode: str = "search",
    ) -> None:
        key = self._key(collection, vector, top_k, filter_dict, mode)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.time(), value)
            while len(self._store) > self._max:
                self._store.popitem(last=False)  # evict LRU

    def invalidate_collection(self, collection: str) -> int:
        """Remove all cache entries for a collection (call on write)."""
        with self._lock:
            keys = []
            for k in self._store:
                parts = k.split(":", 2)
                if len(parts) >= 2 and parts[1] == collection:
                    keys.append(k)
            for k in keys:
                del self._store[k]
            return len(keys)

    def stats(self) -> Dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_size": self._max,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0


# Singleton
_cache: Optional[QueryCache] = None


def get_cache() -> QueryCache:
    global _cache
    if _cache is None:
        from config import CACHE_MAX_SIZE, CACHE_TTL_SECONDS
        _cache = QueryCache(CACHE_MAX_SIZE, CACHE_TTL_SECONDS)
    return _cache
