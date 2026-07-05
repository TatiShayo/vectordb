"""
HTTP middleware stack:
- F02: Per-key rate limiting
- G02: Request ID + structured logging
- G03: Basic timing header
"""
from __future__ import annotations
import time, uuid, logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("vectordb.http")


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        from config import RATE_LIMIT_ENABLED, API_KEY_HEADER, SEARCH_COST
        if not RATE_LIMIT_ENABLED:
            return await call_next(request)
        key = request.headers.get(API_KEY_HEADER, "anonymous")
        cost = SEARCH_COST if "/search" in request.url.path else 1.0
        from utils.ratelimit import get_limiter
        limiter = get_limiter()
        if not limiter.check(key, cost):
            remaining = limiter.remaining(key)
            return Response(
                content='{"detail":"Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={"X-RateLimit-Remaining": str(int(remaining)),
                         "Retry-After": "1"},
            )
        resp = await call_next(request)
        resp.headers["X-RateLimit-Remaining"] = str(int(limiter.remaining(key)))
        return resp


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time"] = f"{ms:.1f}ms"
        if response.status_code >= 400:
            logger.warning(f"{request.method} {request.url.path} "
                           f"→ {response.status_code} ({ms:.1f}ms) [{request_id}]")
        else:
            logger.debug(f"{request.method} {request.url.path} "
                         f"→ {response.status_code} ({ms:.1f}ms) [{request_id}]")
        from utils.metrics import metrics
        metrics.inc_request()
        return response
