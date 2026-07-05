"""F05 — append-only audit log (in-memory ring buffer, 500 events)."""
from __future__ import annotations
import threading, time
from collections import deque
from typing import Any, Dict, List, Optional

_log: deque = deque(maxlen=500)
_lock = threading.Lock()

def record(action: str, key: str, resource: str, detail: Optional[str] = None):
    with _lock:
        _log.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": action, "key": key[:8] + "…",
            "resource": resource, "detail": detail,
        })

def get_audit_log() -> List[Dict]:
    with _lock:
        return list(reversed(_log))
