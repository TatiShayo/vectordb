"""
Per-collection SQLite storage layer — with rich metadata predicate filtering,
FTS5 full-text search, TTL expiration, and transaction safety.
"""
from __future__ import annotations
import json
import logging
import os
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=10000;
CREATE TABLE IF NOT EXISTS vectors (
    id            TEXT    PRIMARY KEY,
    faiss_id      INTEGER UNIQUE NOT NULL,
    raw_vector    BLOB    NOT NULL,
    metadata      TEXT    NOT NULL DEFAULT '{}',
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL,
    expires_at    TEXT,
    sparse_vector TEXT
);
CREATE INDEX IF NOT EXISTS idx_faiss_id   ON vectors(faiss_id);
CREATE INDEX IF NOT EXISTS idx_expires_at ON vectors(expires_at) WHERE expires_at IS NOT NULL;
CREATE TABLE IF NOT EXISTS _counters (key TEXT PRIMARY KEY, value INTEGER NOT NULL DEFAULT 0);
INSERT OR IGNORE INTO _counters VALUES ('faiss_id', 0);
CREATE VIRTUAL TABLE IF NOT EXISTS vectors_fts USING fts5(id UNINDEXED, content, tokenize='porter ascii');
"""

# Regex for safe metadata keys (prevents SQL injection)
_SAFE_KEY_REGEX = re.compile(r"^[a-zA-Z0-9_.-]+$")


def _sqlite_regexp(expr: str, item: Optional[str]) -> bool:
    """SQLite custom function for REGEXP operator."""
    if item is None or expr is None:
        return False
    try:
        return re.search(expr, str(item)) is not None
    except Exception:
        return False


def match_filter(metadata: Dict[str, Any], filter_dict: Optional[Dict[str, Any]]) -> bool:
    """
    In-memory evaluation of metadata against filter queries.
    Supports: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $regex, $exists, $and, $or, $not.
    """
    if not filter_dict:
        return True
    if not isinstance(metadata, dict):
        return False

    for key, condition in filter_dict.items():
        if key == "$and":
            if not isinstance(condition, list):
                return False
            if not all(match_filter(metadata, sub) for sub in condition):
                return False
        elif key == "$or":
            if not isinstance(condition, list):
                return False
            if not any(match_filter(metadata, sub) for sub in condition):
                return False
        elif key == "$not":
            if not isinstance(condition, dict):
                return False
            if match_filter(metadata, condition):
                return False
        else:
            val = metadata.get(key)
            if isinstance(condition, dict):
                for op, target in condition.items():
                    if op == "$eq":
                        if val != target:
                            return False
                    elif op == "$ne":
                        if val == target:
                            return False
                    elif op == "$gt":
                        if val is None or not (val > target):
                            return False
                    elif op == "$gte":
                        if val is None or not (val >= target):
                            return False
                    elif op == "$lt":
                        if val is None or not (val < target):
                            return False
                    elif op == "$lte":
                        if val is None or not (val <= target):
                            return False
                    elif op == "$in":
                        if not isinstance(target, (list, set, tuple)) or val not in target:
                            return False
                    elif op == "$nin":
                        if isinstance(target, (list, set, tuple)) and val in target:
                            return False
                    elif op == "$regex":
                        if val is None or not re.search(str(target), str(val)):
                            return False
                    elif op == "$exists":
                        exists = key in metadata
                        if exists != bool(target):
                            return False
                    else:
                        raise ValueError(f"Unknown filter operator: {op}")
            else:
                # Direct equality match
                if val != condition:
                    return False

    return True


def _build_filter_sql(filter_dict: Optional[Dict[str, Any]]) -> Tuple[str, List[Any]]:
    """
    Compiles a filter dictionary into parameterized SQLite conditions.
    Returns (sql_expression, params_list).
    """
    if not filter_dict:
        return "", []

    def _parse_item(key: str, val: Any) -> Tuple[str, List[Any]]:
        if key == "$and":
            if not isinstance(val, list) or not val:
                return "1=1", []
            clauses, params = [], []
            for sub in val:
                c, p = _build_filter_sql(sub)
                if c:
                    clauses.append(f"({c})")
                    params.extend(p)
            return " AND ".join(clauses) if clauses else "1=1", params

        elif key == "$or":
            if not isinstance(val, list) or not val:
                return "1=1", []
            clauses, params = [], []
            for sub in val:
                c, p = _build_filter_sql(sub)
                if c:
                    clauses.append(f"({c})")
                    params.extend(p)
            return " OR ".join(clauses) if clauses else "1=1", params

        elif key == "$not":
            if not isinstance(val, dict) or not val:
                return "1=1", []
            c, p = _build_filter_sql(val)
            return f"NOT ({c})", p

        else:
            if not _SAFE_KEY_REGEX.match(key):
                raise ValueError(f"Invalid metadata key: {key!r}")

            json_field = f"json_extract(metadata, '$.{key}')"

            if isinstance(val, dict):
                op_clauses, op_params = [], []
                for op, target in val.items():
                    if op == "$eq":
                        op_clauses.append(f"{json_field} = ?")
                        op_params.append(target)
                    elif op == "$ne":
                        op_clauses.append(f"({json_field} IS NULL OR {json_field} != ?)")
                        op_params.append(target)
                    elif op == "$gt":
                        op_clauses.append(f"{json_field} > ?")
                        op_params.append(target)
                    elif op == "$gte":
                        op_clauses.append(f"{json_field} >= ?")
                        op_params.append(target)
                    elif op == "$lt":
                        op_clauses.append(f"{json_field} < ?")
                        op_params.append(target)
                    elif op == "$lte":
                        op_clauses.append(f"{json_field} <= ?")
                        op_params.append(target)
                    elif op == "$in":
                        if not isinstance(target, (list, tuple, set)):
                            raise ValueError("$in requires a list/tuple")
                        if not target:
                            op_clauses.append("0=1")
                        else:
                            ph = ",".join("?" * len(target))
                            op_clauses.append(f"{json_field} IN ({ph})")
                            op_params.extend(target)
                    elif op == "$nin":
                        if not isinstance(target, (list, tuple, set)):
                            raise ValueError("$nin requires a list/tuple")
                        if target:
                            ph = ",".join("?" * len(target))
                            op_clauses.append(f"({json_field} IS NULL OR {json_field} NOT IN ({ph}))")
                            op_params.extend(target)
                    elif op == "$regex":
                        op_clauses.append(f"{json_field} REGEXP ?")
                        op_params.append(str(target))
                    elif op == "$exists":
                        if target:
                            op_clauses.append(f"{json_field} IS NOT NULL")
                        else:
                            op_clauses.append(f"{json_field} IS NULL")
                    else:
                        raise ValueError(f"Unknown filter operator: {op}")
                return " AND ".join(op_clauses), op_params
            else:
                return f"{json_field} = ?", [val]

    clauses: List[str] = []
    params: List[Any] = []
    for k, v in filter_dict.items():
        c, p = _parse_item(k, v)
        if c:
            clauses.append(c)
            params.extend(p)

    if not clauses:
        return "", []
    return " AND ".join(clauses), params


class CollectionDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._c = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
        self._c.row_factory = sqlite3.Row
        self._c.create_function("REGEXP", 2, _sqlite_regexp)
        with self._lock:
            self._c.executescript(_SCHEMA)
            self._c.commit()

    # ── Counter ───────────────────────────────────────────────────────────────
    def next_faiss_ids(self, n: int = 1) -> List[int]:
        with self._lock:
            self._c.execute("UPDATE _counters SET value=value+? WHERE key='faiss_id'", (n,))
            top = self._c.execute("SELECT value FROM _counters WHERE key='faiss_id'").fetchone()[0]
            self._c.commit()
        return list(range(top - n + 1, top + 1))

    # ── Single upsert ─────────────────────────────────────────────────────────
    def upsert(
        self,
        vec_id: str,
        faiss_id: int,
        raw_vector: np.ndarray,
        metadata: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
        sparse_vector: Optional[Dict] = None,
    ) -> str:
        now = _now()
        blob = raw_vector.astype(np.float32).tobytes()
        meta_json = json.dumps(metadata)
        expires_at = _expires(ttl_seconds)
        sv_json = json.dumps(sparse_vector) if sparse_vector else None

        with self._lock:
            row = self._c.execute("SELECT rowid FROM vectors WHERE id=?", (vec_id,)).fetchone()
            if row:
                old_rowid = row[0]
                self._c.execute(
                    "UPDATE vectors SET faiss_id=?,raw_vector=?,metadata=?,"
                    "updated_at=?,expires_at=?,sparse_vector=? WHERE id=?",
                    (faiss_id, blob, meta_json, now, expires_at, sv_json, vec_id),
                )
                new_rowid = self._c.execute("SELECT rowid FROM vectors WHERE id=?", (vec_id,)).fetchone()[0]
                self._c.execute("DELETE FROM vectors_fts WHERE rowid=?", (old_rowid,))
                self._c.execute("INSERT INTO vectors_fts(rowid,id,content) VALUES(?,?,?)", (new_rowid, vec_id, meta_json))
                self._c.commit()
                return "updated"
            else:
                self._c.execute(
                    "INSERT INTO vectors(id,faiss_id,raw_vector,metadata,created_at,updated_at,expires_at,sparse_vector)"
                    " VALUES(?,?,?,?,?,?,?,?)",
                    (vec_id, faiss_id, blob, meta_json, now, now, expires_at, sv_json),
                )
                new_rowid = self._c.execute("SELECT rowid FROM vectors WHERE id=?", (vec_id,)).fetchone()[0]
                self._c.execute("INSERT INTO vectors_fts(rowid,id,content) VALUES(?,?,?)", (new_rowid, vec_id, meta_json))
                self._c.commit()
                return "inserted"

    # ── Batch upsert ──────────────────────────────────────────────────────────
    def upsert_batch(self, items: List[Tuple]) -> Tuple[int, int]:
        """items: [(vec_id, faiss_id, raw_vec, metadata, ttl, sparse_vector)]"""
        now = _now()
        inserted = updated = 0
        with self._lock:
            for vec_id, faiss_id, raw_vec, metadata, ttl, sv in items:
                blob = raw_vec.astype(np.float32).tobytes()
                meta_json = json.dumps(metadata)
                expires_at = _expires(ttl)
                sv_json = json.dumps(sv) if sv else None
                row = self._c.execute("SELECT rowid FROM vectors WHERE id=?", (vec_id,)).fetchone()
                if row:
                    old_rowid = row[0]
                    self._c.execute(
                        "UPDATE vectors SET faiss_id=?,raw_vector=?,metadata=?,"
                        "updated_at=?,expires_at=?,sparse_vector=? WHERE id=?",
                        (faiss_id, blob, meta_json, now, expires_at, sv_json, vec_id),
                    )
                    nr = self._c.execute("SELECT rowid FROM vectors WHERE id=?", (vec_id,)).fetchone()[0]
                    self._c.execute("DELETE FROM vectors_fts WHERE rowid=?", (old_rowid,))
                    self._c.execute("INSERT INTO vectors_fts(rowid,id,content) VALUES(?,?,?)", (nr, vec_id, meta_json))
                    updated += 1
                else:
                    self._c.execute(
                        "INSERT INTO vectors(id,faiss_id,raw_vector,metadata,created_at,updated_at,expires_at,sparse_vector)"
                        " VALUES(?,?,?,?,?,?,?,?)",
                        (vec_id, faiss_id, blob, meta_json, now, now, expires_at, sv_json),
                    )
                    nr = self._c.execute("SELECT rowid FROM vectors WHERE id=?", (vec_id,)).fetchone()[0]
                    self._c.execute("INSERT INTO vectors_fts(rowid,id,content) VALUES(?,?,?)", (nr, vec_id, meta_json))
                    inserted += 1
            self._c.commit()
        return inserted, updated

    # ── Delete ────────────────────────────────────────────────────────────────
    def delete(self, vec_id: str) -> Optional[int]:
        with self._lock:
            row = self._c.execute("SELECT faiss_id,rowid FROM vectors WHERE id=?", (vec_id,)).fetchone()
            if not row:
                return None
            faiss_id, rowid = row[0], row[1]
            self._c.execute("DELETE FROM vectors WHERE id=?", (vec_id,))
            self._c.execute("DELETE FROM vectors_fts WHERE rowid=?", (rowid,))
            self._c.commit()
        return faiss_id

    def delete_expired(self) -> List[int]:
        now = _now()
        with self._lock:
            rows = self._c.execute(
                "SELECT faiss_id,rowid FROM vectors WHERE expires_at IS NOT NULL AND expires_at<?", (now,)
            ).fetchall()
            if not rows:
                return []
            fids = [r[0] for r in rows]
            ph = ",".join("?" * len(rows))
            self._c.execute(f"DELETE FROM vectors_fts WHERE rowid IN ({ph})", [r[1] for r in rows])
            self._c.execute("DELETE FROM vectors WHERE expires_at IS NOT NULL AND expires_at<?", (now,))
            self._c.commit()
        return fids

    # ── Reads ─────────────────────────────────────────────────────────────────
    def get(self, vec_id: str, include_vector: bool = False) -> Optional[Dict]:
        with self._lock:
            row = self._c.execute("SELECT * FROM vectors WHERE id=?", (vec_id,)).fetchone()
        return _row(row, include_vector) if row else None

    def get_faiss_id(self, vec_id: str) -> Optional[int]:
        with self._lock:
            row = self._c.execute("SELECT faiss_id FROM vectors WHERE id=?", (vec_id,)).fetchone()
        return row[0] if row else None

    def get_by_faiss_ids(self, faiss_ids: List[int], include_vector: bool = False) -> Dict[int, Dict]:
        if not faiss_ids:
            return {}
        ph = ",".join("?" * len(faiss_ids))
        with self._lock:
            rows = self._c.execute(f"SELECT * FROM vectors WHERE faiss_id IN ({ph})", faiss_ids).fetchall()
        return {r["faiss_id"]: _row(r, include_vector) for r in rows}

    def scroll(
        self, limit: int = 100, offset: int = 0, filter_dict: Optional[Dict] = None, include_vector: bool = False
    ) -> Tuple[List[Dict], int]:
        where_clause, params = _build_filter_sql(filter_dict)
        where_sql = f"WHERE {where_clause}" if where_clause else ""
        with self._lock:
            total = self._c.execute(f"SELECT COUNT(*) FROM vectors {where_sql}", params).fetchone()[0]
            rows = self._c.execute(
                f"SELECT * FROM vectors {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
        return [_row(r, include_vector) for r in rows], total

    def filter_faiss_ids(self, faiss_ids: List[int], filter_dict: Optional[Dict]) -> List[int]:
        if not filter_dict or not faiss_ids:
            return faiss_ids
        cond_sql, cond_params = _build_filter_sql(filter_dict)
        if not cond_sql:
            return faiss_ids
        ph = ",".join("?" * len(faiss_ids))
        with self._lock:
            rows = self._c.execute(
                f"SELECT faiss_id FROM vectors WHERE faiss_id IN ({ph}) AND ({cond_sql})",
                faiss_ids + cond_params,
            ).fetchall()
        return [r[0] for r in rows]

    def keyword_search(self, query: str, limit: int = 500) -> List[str]:
        try:
            safe_q = query.replace('"', '""')
            with self._lock:
                rows = self._c.execute(
                    "SELECT v.id FROM vectors v JOIN vectors_fts f ON v.rowid=f.rowid "
                    "WHERE vectors_fts MATCH ? ORDER BY rank LIMIT ?",
                    (safe_q, limit),
                ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.OperationalError as exc:
            logger.warning(f"FTS5 error: {exc}")
            return []

    def all_vectors_for_rebuild(self) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            rows = self._c.execute("SELECT faiss_id,raw_vector FROM vectors").fetchall()
        if not rows:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int64)
        faiss_ids = np.array([r[0] for r in rows], dtype=np.int64)
        vecs = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        return vecs, faiss_ids

    def patch_metadata(self, vec_id: str, metadata: Dict[str, Any]) -> bool:
        now = _now()
        meta_json = json.dumps(metadata)
        with self._lock:
            row = self._c.execute("SELECT rowid FROM vectors WHERE id=?", (vec_id,)).fetchone()
            if not row:
                return False
            old_rowid = row[0]
            self._c.execute("UPDATE vectors SET metadata=?,updated_at=? WHERE id=?", (meta_json, now, vec_id))
            nr = self._c.execute("SELECT rowid FROM vectors WHERE id=?", (vec_id,)).fetchone()[0]
            self._c.execute("DELETE FROM vectors_fts WHERE rowid=?", (old_rowid,))
            self._c.execute("INSERT INTO vectors_fts(rowid,id,content) VALUES(?,?,?)", (nr, vec_id, meta_json))
            self._c.commit()
        return True

    def count(self) -> int:
        with self._lock:
            return self._c.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def count_filtered(self, filter_dict: Optional[Dict] = None) -> int:
        where_clause, params = _build_filter_sql(filter_dict)
        where_sql = f"WHERE {where_clause}" if where_clause else ""
        with self._lock:
            return self._c.execute(f"SELECT COUNT(*) FROM vectors {where_sql}", params).fetchone()[0]

    def facets(self, field: str, limit: int = 100) -> Dict[str, int]:
        if not _SAFE_KEY_REGEX.match(field):
            raise ValueError(f"Invalid field name: {field!r}")
        with self._lock:
            rows = self._c.execute(
                f"SELECT json_extract(metadata,'$.{field}') as v, COUNT(*) as c "
                "FROM vectors WHERE v IS NOT NULL GROUP BY v ORDER BY c DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {str(r[0]): r[1] for r in rows}

    def get_all_sparse(self) -> List[Tuple[str, Optional[Dict]]]:
        with self._lock:
            rows = self._c.execute(
                "SELECT id,sparse_vector FROM vectors WHERE sparse_vector IS NOT NULL"
            ).fetchall()
        result = []
        for r in rows:
            try:
                result.append((r[0], json.loads(r[1])))
            except Exception:
                pass
        return result

    def size_bytes(self) -> int:
        try:
            return os.path.getsize(self.db_path)
        except OSError:
            return 0

    def close(self):
        with self._lock:
            self._c.close()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expires(s: Optional[int]) -> Optional[str]:
    if s is None:
        return None
    return (datetime.now(timezone.utc) + timedelta(seconds=s)).isoformat()


def _row(row: sqlite3.Row, include_vector: bool) -> Dict:
    d = dict(row)
    d["metadata"] = json.loads(d["metadata"])
    if include_vector and d.get("raw_vector"):
        d["vector"] = np.frombuffer(d["raw_vector"], dtype=np.float32).tolist()
    d.pop("raw_vector", None)
    d.pop("sparse_vector", None)
    return d
