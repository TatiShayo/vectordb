"""Per-collection SQLite — v4 final (all methods in class, sparse support, correct bindings)."""
from __future__ import annotations
import json, logging, os, sqlite3, threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=10000;
CREATE TABLE IF NOT EXISTS vectors (
    id           TEXT    PRIMARY KEY,
    faiss_id     INTEGER UNIQUE NOT NULL,
    raw_vector   BLOB    NOT NULL,
    metadata     TEXT    NOT NULL DEFAULT '{}',
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL,
    expires_at   TEXT,
    sparse_vector TEXT
);
CREATE INDEX IF NOT EXISTS idx_faiss_id   ON vectors(faiss_id);
CREATE INDEX IF NOT EXISTS idx_expires_at ON vectors(expires_at) WHERE expires_at IS NOT NULL;
CREATE TABLE IF NOT EXISTS _counters (key TEXT PRIMARY KEY, value INTEGER NOT NULL DEFAULT 0);
INSERT OR IGNORE INTO _counters VALUES ('faiss_id', 0);
CREATE VIRTUAL TABLE IF NOT EXISTS vectors_fts USING fts5(id UNINDEXED, content, tokenize='porter ascii');
"""


class CollectionDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._c = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
        self._c.row_factory = sqlite3.Row
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
    def upsert(self, vec_id: str, faiss_id: int, raw_vector: np.ndarray,
               metadata: Dict[str, Any], ttl_seconds: Optional[int] = None,
               sparse_vector: Optional[Dict] = None) -> str:
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
                    (faiss_id, blob, meta_json, now, expires_at, sv_json, vec_id)
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
                    (vec_id, faiss_id, blob, meta_json, now, now, expires_at, sv_json)
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
                        (faiss_id, blob, meta_json, now, expires_at, sv_json, vec_id)
                    )
                    nr = self._c.execute("SELECT rowid FROM vectors WHERE id=?", (vec_id,)).fetchone()[0]
                    self._c.execute("DELETE FROM vectors_fts WHERE rowid=?", (old_rowid,))
                    self._c.execute("INSERT INTO vectors_fts(rowid,id,content) VALUES(?,?,?)", (nr, vec_id, meta_json))
                    updated += 1
                else:
                    self._c.execute(
                        "INSERT INTO vectors(id,faiss_id,raw_vector,metadata,created_at,updated_at,expires_at,sparse_vector)"
                        " VALUES(?,?,?,?,?,?,?,?)",
                        (vec_id, faiss_id, blob, meta_json, now, now, expires_at, sv_json)
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

    def scroll(self, limit=100, offset=0, filter_dict=None, include_vector=False) -> Tuple[List[Dict], int]:
        where, params = _filter_clause(filter_dict)
        with self._lock:
            total = self._c.execute(f"SELECT COUNT(*) FROM vectors {where}", params).fetchone()[0]
            rows = self._c.execute(
                f"SELECT * FROM vectors {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset]
            ).fetchall()
        return [_row(r, include_vector) for r in rows], total

    def filter_faiss_ids(self, faiss_ids: List[int], filter_dict: Optional[Dict]) -> List[int]:
        if not filter_dict or not faiss_ids:
            return faiss_ids
        ph = ",".join("?" * len(faiss_ids))
        conds, cparams = [], []
        for k, v in filter_dict.items():
            conds.append(f"json_extract(metadata,'$.{k}')=?")
            cparams.append(v)
        with self._lock:
            rows = self._c.execute(
                f"SELECT faiss_id FROM vectors WHERE faiss_id IN ({ph}) AND {' AND '.join(conds)}",
                faiss_ids + cparams
            ).fetchall()
        return [r[0] for r in rows]

    def keyword_search(self, query: str, limit: int = 500) -> List[str]:
        try:
            safe_q = query.replace('"', '""')
            with self._lock:
                rows = self._c.execute(
                    "SELECT v.id FROM vectors v JOIN vectors_fts f ON v.rowid=f.rowid "
                    "WHERE vectors_fts MATCH ? ORDER BY rank LIMIT ?",
                    (safe_q, limit)
                ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.OperationalError as exc:
            logger.warning(f"FTS5 error: {exc}")
            return []

    def all_vectors_for_rebuild(self) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            rows = self._c.execute("SELECT faiss_id,raw_vector FROM vectors").fetchall()
        if not rows:
            return np.empty((0,0), dtype=np.float32), np.empty(0, dtype=np.int64)
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

    def count_filtered(self, filter_dict=None) -> int:
        where, params = _filter_clause(filter_dict)
        with self._lock:
            return self._c.execute(f"SELECT COUNT(*) FROM vectors {where}", params).fetchone()[0]

    def facets(self, field: str, limit: int = 100) -> Dict[str, int]:
        with self._lock:
            rows = self._c.execute(
                f"SELECT json_extract(metadata,'$.{field}') as v, COUNT(*) as c "
                "FROM vectors WHERE v IS NOT NULL GROUP BY v ORDER BY c DESC LIMIT ?",
                (limit,)
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

def _filter_clause(filter_dict: Optional[Dict]) -> Tuple[str, List]:
    if not filter_dict:
        return "", []
    conds, params = [], []
    for k, v in filter_dict.items():
        conds.append(f"json_extract(metadata,'$.{k}')=?")
        params.append(v)
    return "WHERE " + " AND ".join(conds), params
