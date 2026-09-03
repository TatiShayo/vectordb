"""
API key authentication.

Header: X-API-Key
Roles:
  admin — full access (create/delete collections, rebuild, force-save, get vectors with vectors)
  user  — read/write vectors + search, no destructive collection ops
"""
from __future__ import annotations

import hmac
from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from config import ADMIN_KEYS, ALL_KEYS, API_KEY_HEADER

_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def _matches_any(candidate: str, allowed_keys: set[str]) -> bool:
    candidate_bytes = candidate.encode("utf-8")
    for valid_key in allowed_keys:
        valid_bytes = valid_key.encode("utf-8")
        if len(candidate_bytes) == len(valid_bytes) and hmac.compare_digest(candidate_bytes, valid_bytes):
            return True
    return False


def _resolve(key: str | None) -> str:
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if not _matches_any(key, ALL_KEYS):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


async def require_auth(request: Request, api_key: str | None = Security(_scheme)) -> str:
    """Dependency: any valid key."""
    return _resolve(api_key)


async def require_admin(request: Request, api_key: str | None = Security(_scheme)) -> str:
    """Dependency: admin key only."""
    key = _resolve(api_key)
    if not _matches_any(key, ADMIN_KEYS):
        raise HTTPException(status_code=403, detail="Admin access required")
    return key


def is_admin(api_key: str) -> bool:
    return _matches_any(api_key, ADMIN_KEYS)


async def require_auth_with_audit(request: Request, api_key: str | None = Security(_scheme)) -> str:
    """Dependency: any valid key, with every authenticated call recorded to the audit log."""
    key = _resolve(api_key)
    from utils.audit import record
    record("request", key, str(request.url.path))
    return key
