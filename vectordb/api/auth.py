"""
API key authentication.

Header: X-API-Key
Roles:
  admin — full access (create/delete collections, rebuild, force-save, get vectors with vectors)
  user  — read/write vectors + search, no destructive collection ops
"""
from __future__ import annotations

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from config import ADMIN_KEYS, ALL_KEYS, API_KEY_HEADER

_scheme = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def _resolve(key: str | None) -> str:
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if key not in ALL_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


async def require_auth(request: Request, api_key: str | None = Security(_scheme)) -> str:
    """Dependency: any valid key."""
    return _resolve(api_key)


async def require_admin(request: Request, api_key: str | None = Security(_scheme)) -> str:
    """Dependency: admin key only."""
    key = _resolve(api_key)
    if key not in ADMIN_KEYS:
        raise HTTPException(status_code=403, detail="Admin access required")
    return key


def is_admin(api_key: str) -> bool:
    return api_key in ADMIN_KEYS


async def require_auth_with_audit(request: Request, api_key: str | None = Security(_scheme)) -> str:
    """Dependency: any valid key, with every authenticated call recorded to the audit log."""
    key = _resolve(api_key)
    from utils.audit import record
    record("request", key, str(request.url.path))
    return key
