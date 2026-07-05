"""
Token bucket rate limiter — F02.

Per API-key limits. Each key gets its own bucket refilled at `rate` tokens/sec.
Requests cost 1 token by default; search costs `search_cost` tokens.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional


class TokenBucket:
    def __init__(self, capacity: float, rate: float):
        self.capacity = capacity   # max tokens
        self.rate = rate           # tokens per second
        self._tokens = capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def remaining(self) -> float:
        with self._lock:
            now = time.monotonic()
            return min(self.capacity, self._tokens + (now - self._last) * self.rate)


class RateLimiter:
    """Per-key token bucket store."""

    def __init__(self, rpm: int = 1000, burst_multiplier: float = 2.0):
        self._rate = rpm / 60.0          # tokens/sec
        self._capacity = rpm / 60.0 * burst_multiplier
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _get(self, key: str) -> TokenBucket:
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(self._capacity, self._rate)
            return self._buckets[key]

    def check(self, key: str, cost: float = 1.0) -> bool:
        """Returns True if allowed, False if rate-limited."""
        return self._get(key).consume(cost)

    def remaining(self, key: str) -> float:
        return self._get(key).remaining()

    def stats(self) -> Dict:
        return {
            "rate_rpm": self._rate * 60,
            "burst_capacity": self._capacity,
            "tracked_keys": len(self._buckets),
        }


_limiter: Optional[RateLimiter] = None


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        from config import RATE_LIMIT_RPM
        _limiter = RateLimiter(rpm=RATE_LIMIT_RPM)
    return _limiter
