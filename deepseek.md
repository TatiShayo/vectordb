# vectordb — DeepSeek Audit

**Date:** 2026-07-13
**Path:** `C:\Users\TATI\Desktop\DEV\vectordb\`
**Stack:** Python / FastAPI + SQLite + FAISS
**Tier:** 1 — Critical
**Dependencies:** Partial (lower-bound only, no lockfile)

---

## 🔴 Security Vulnerabilities

### SQL Injection — Critical

| Severity | File | Line(s) | Vulnerability | Exact Fix |
|----------|------|---------|---------------|-----------|
| 🔴 CRITICAL | `vectordb/storage/db.py` | 285-287 | `_filter_clause()` interpolates filter_dict keys directly into SQL via f-string: `f"json_extract(metadata,'$.{k}')=?"`. If filter_dict keys come from user input, this is direct SQL injection. | Add allowlist validation of field names before constructing clause: `ALLOWED_FILTER_FIELDS = {"category", "type", "status", ...}` and reject unknown keys. |
| 🔴 CRITICAL | `vectordb/storage/db.py` | 169-170, 236, 240, 249 | Same `where` variable built by `_filter_clause()` injected into multiple query strings via f-string: `f"SELECT COUNT(*) FROM vectors {where}"` | After fixing `_filter_clause()`, all callers are safe. Also sanitize `ORDER BY` direction at line 170 — currently user-controlled `order_dir` is validated but could be stricter. |
| 🟠 HIGH | `vectordb/storage/db.py` | 142, 163 | f-string with `{ph}` placeholder list in `IN` clause — while `ph` is internally derived from `len(rows)`, this pattern is dangerous if `len(rows)` ever becomes attacker-influenced. | Replace with `?` placeholder list built by the query executor itself rather than pre-building. |
| 🟡 MEDIUM | `vectordb/client/client.py` | 41 | Hardcoded default API key `"user-secret"` in constructor. Anyone reading the code knows the default. | Change to `api_key: Optional[str] = None` and require explicit key or read from env: `os.environ.get("VECTORDB_API_KEY")`. |
| 🟡 MEDIUM | `vectordb/client/client.py` | 228 | Documentation example uses hardcoded `api_key="user-secret"`. | Update docstring to show env var pattern: `api_key=os.environ["VECTORDB_API_KEY"]`. |

### Auth

- ✅ RBAC with admin/user roles
- ✅ Per-key rate limiting (token bucket)
- ✅ Audit logging
- ⚠️ No key rotation mechanism — keys stored in config only, no lifecycle management

---

## 🟠 Performance Issues

| Severity | File | Line(s) | Issue | Exact Fix |
|----------|------|---------|-------|-----------|
| 🟠 HIGH | `vectordb/storage/db.py` | 36 | Single SQLite connection (`check_same_thread=False`) with one `threading.RLock()` — under concurrent FastAPI requests, this becomes a bottleneck. | Use a connection pool (2-4 connections) with `Queue`-based checkout, or switch to `aiosqlite` for async. |
| 🟡 MEDIUM | `vectordb/storage/db.py` | 163-170 | SELECT with filter but no LIMIT — full table scans on every filtered search. | Add pagination via `LIMIT ? OFFSET ?` for all SELECT queries with filters. |

---

## 🟡 UI/UX

This is an API/backend project. No frontend to audit. The admin dashboard (if any exists) was not found in the explored files.

---

## 🔧 Session: 2026-07-14 — Multi-Agent Deep Audit Sweep (Round 1)

**Status:** Not audited in this round. Previously fixed (July 5): `require_auth_with_audit` rewritten as real FastAPI dependency (was broken dead code). Sweep Round 2 will cover Tier 3.

| Category | Package | Issue | Fix |
|----------|---------|-------|-----|
| 🔴 CRITICAL | ALL packages | `requirements.txt` uses `>=` lower-bound only — e.g., `fastapi>=0.111.0`, `pydantic>=2.7.0`, `numpy>=1.26.0`. A new major release will silently break everything. | Add upper bounds: `fastapi>=0.111.0,<1.0.0`, `pydantic>=2.7.0,<3.0.0`, `numpy>=1.26.0,<2.0.0`. Then generate a `requirements-lock.txt` with `pip freeze`. |
| 🟡 MEDIUM | `sentence-transformers` | Listed as optional but pulls torch + transformers (~3GB). Not separated into extras. | Move to `[optional]` extras group in `pyproject.toml` or add comment in requirements. |
| 🟡 MEDIUM | `faiss-cpu>=1.8.0` | Lower-bound only — `faiss` API changes between minor versions. | Pin to `faiss-cpu>=1.8.0,<2.0.0`. |

### Missing Dev Tooling
- No `pytest-cov` — no test coverage tracking
- No `ruff` or `mypy` — no linting or static type checking
- No `.python-version` file — Python version not specified
- No `requirements-lock.txt` — no reproducible builds

---

## 📋 Priority Fix Queue

1. **[CRITICAL — SQL Injection]** `vectordb/storage/db.py:285-287` — Add field name allowlist to `_filter_clause()`: validate keys against `ALLOWED_FILTER_FIELDS` set, reject unknown keys with 400 error.
2. **[CRITICAL — SQL Injection]** `vectordb/storage/db.py:142,163` — Replace f-string `IN ({ph})` pattern with parameterized query building that doesn't use f-strings for SQL structure.
3. **[HIGH — Hardcoded Secret]** `vectordb/client/client.py:41` — Change `api_key: str = "user-secret"` to `api_key: Optional[str] = None` and read from `os.environ.get("VECTORDB_API_KEY", "user-secret")`.
4. **[HIGH — Performance]** `vectordb/storage/db.py:36` — Implement connection pooling (3 connections) instead of single shared connection.
5. **[MEDIUM — Dependencies]** `requirements.txt` — Add upper bounds (`<1.0.0`, `<3.0.0`, `<2.0.0`) to all packages. Split `sentence-transformers` to extras.
6. **[MEDIUM — Docs]** `vectordb/client/client.py:228` — Update docstring example to use env var pattern.

---

## Cross-Cutting Fixes Applied

- [ ] `.python-version` added (should be `3.11`)
- [ ] `verify` script added
- [ ] `requirements-lock.txt` tracked
- [ ] `ruff` and `mypy` added as dev deps
