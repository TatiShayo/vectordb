"""
Collection — coordinates FAISS + SQLite + sparse index + cache + rich filtering.
"""
from __future__ import annotations
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from core.indexer import FAISSIndex, IndexType, _auto_type
from core.sparse import SparseIndex, linear_weighted_fuse, rrf_fuse
from storage.db import CollectionDB, match_filter
from storage.checkpointer import save as _save, load_or_rebuild
from utils.normalize import prepare_vector, prepare_batch
from utils.cache import get_cache
from config import (
    MAX_DELETED_RATIO, OVERFETCH_FACTOR, MIN_OVERFETCH,
    IVF_NPROBE, CACHE_ENABLED, HNSW_M, HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH, AUTO_SAVE_INTERVAL, SLOW_QUERY_MS,
)

logger = logging.getLogger(__name__)


class Collection:
    def __init__(
        self,
        name: str,
        data_dir: str,
        dimension: int = 384,
        distance: str = "cosine",
        index_type: Optional[str] = None,
        quant_mode: str = "float32",
        description: str = "",
        hnsw_m: Optional[int] = None,
        hnsw_ef_construction: Optional[int] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        ivfpq_m: Optional[int] = None,
        ivfpq_nbits: Optional[int] = None,
    ):
        self.name = name
        self.dimension = dimension
        self.distance = distance.lower()
        self.quant_mode = quant_mode
        self.description = description
        self.hnsw_m = hnsw_m or HNSW_M
        self.hnsw_ef_construction = hnsw_ef_construction or HNSW_EF_CONSTRUCTION
        self.ivfpq_m = ivfpq_m
        self.ivfpq_nbits = ivfpq_nbits
        self.created_at = created_at or _now()
        self.updated_at = updated_at or _now()
        self._desired_itype = IndexType(index_type) if index_type else None

        self._dir = Path(data_dir) / name
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = str(self._dir / "index.faiss")
        self._db_path = str(self._dir / "metadata.db")
        self._centroids_path = self._dir / "centroids.npy"

        centroids = None
        if self._centroids_path.exists():
            try:
                centroids = np.load(str(self._centroids_path), allow_pickle=False)
            except Exception as exc:
                logger.error(f"Failed to load centroids from {self._centroids_path}: {exc}")

        self._index = FAISSIndex(
            dimension,
            self._desired_itype or IndexType.FLAT,
            distance=self.distance,
            hnsw_m=self.hnsw_m,
            hnsw_ef_construction=self.hnsw_ef_construction,
            ivfpq_m=self.ivfpq_m,
            ivfpq_nbits=self.ivfpq_nbits,
            centroids=centroids,
        )
        self._db = CollectionDB(self._db_path)
        self._sparse = SparseIndex()
        self._op_count = 0
        self._deleted_count = 0
        self._save_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._closed = False

        load_or_rebuild(self._index, self._index_path, self._db)
        if self._index.index_type != IndexType.FLAT:
            self._index.set_nprobe(IVF_NPROBE)
        self._index.set_ef_search(HNSW_EF_SEARCH)
        self._load_sparse()

    # ── Upsert ────────────────────────────────────────────────────────────────
    def upsert(
        self,
        vec_id: str,
        raw_vector: Union[Sequence[float], np.ndarray],
        metadata: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
        sparse_vector: Optional[Dict] = None,
    ) -> str:
        if self._closed:
            raise RuntimeError("Collection is closed")
        with self._write_lock:
            normalize = (self.distance == "cosine")
            vec = prepare_vector(raw_vector, normalize)
            if len(vec) != self.dimension:
                raise ValueError(f"Vector dim {len(vec)} != {self.dimension}")

            old_fid = self._db.get_faiss_id(vec_id)
            if old_fid is not None:
                self._index.remove([old_fid])
                self._sparse.remove(vec_id)

            [faiss_id] = self._db.next_faiss_ids(1)
            status = self._db.upsert(vec_id, faiss_id, vec, metadata, ttl_seconds, sparse_vector)
            self._index.add(vec.reshape(1, -1), np.array([faiss_id], dtype=np.int64))

            if sparse_vector:
                sv = dict(zip(sparse_vector.get("indices", []), sparse_vector.get("values", [])))
                self._sparse.add(vec_id, sv)

            if CACHE_ENABLED:
                get_cache().invalidate_collection(self.name)
            self._on_write()
            return status

    def upsert_batch(self, items: List[Dict[str, Any]]) -> Tuple[int, int]:
        if self._closed:
            raise RuntimeError("Collection is closed")
        with self._write_lock:
            normalize = (self.distance == "cosine")
            ids = [it["id"] for it in items]
            existing_fids = [self._db.get_faiss_id(i) for i in ids]
            old_fids = [f for f in existing_fids if f is not None]
            if old_fids:
                self._index.remove(old_fids)
                for i, fid in zip(ids, existing_fids):
                    if fid is not None:
                        self._sparse.remove(i)

            faiss_ids = self._db.next_faiss_ids(len(items))
            db_items = []
            vectors_list = []

            for item, fid in zip(items, faiss_ids):
                raw = item.get("vector") or []
                if raw:
                    vec = prepare_vector(raw, normalize)
                    if len(vec) != self.dimension:
                        raise ValueError(f"Vector '{item['id']}' dim mismatch")
                else:
                    vec = np.zeros(self.dimension, dtype=np.float32)
                vectors_list.append(vec)
                sv = item.get("sparse_vector")
                db_items.append(
                    (item["id"], fid, vec, item.get("metadata", {}), item.get("ttl_seconds"), sv)
                )

            inserted, updated = self._db.upsert_batch(db_items)
            mat = np.vstack(vectors_list).astype(np.float32)
            ids_arr = np.array(faiss_ids, dtype=np.int64)
            self._index.add(mat, ids_arr)

            for item, fid in zip(items, faiss_ids):
                sv = item.get("sparse_vector")
                if sv:
                    self._sparse.add(item["id"], dict(zip(sv.get("indices", []), sv.get("values", []))))

            if CACHE_ENABLED:
                get_cache().invalidate_collection(self.name)
            self._on_write(len(items))
            return inserted, updated

    # ── Search ────────────────────────────────────────────────────────────────
    def search(
        self,
        query_vector: Union[Sequence[float], np.ndarray],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_vector: bool = False,
        score_threshold: Optional[float] = None,
        ef_search: Optional[int] = None,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        import time as _time

        t0 = _time.perf_counter()

        normalize = (self.distance == "cosine")
        q = prepare_vector(query_vector, normalize)
        if len(q) != self.dimension:
            raise ValueError(f"Query dim {len(q)} != {self.dimension}")

        # Cache check
        if CACHE_ENABLED and not use_mmr:
            cached = get_cache().get(self.name, q, top_k, filter_dict)
            if cached is not None:
                from utils.prometheus import prom

                prom.cache_hits.inc()
                return cached, True

        if ef_search:
            self._index.set_ef_search(ef_search)

        overfetch = max(top_k * OVERFETCH_FACTOR, MIN_OVERFETCH)
        faiss_ids, scores = self._index.search(q, overfetch)

        results = self._assemble(faiss_ids, scores, top_k, filter_dict, include_vector, score_threshold)

        # MMR post-processing
        if use_mmr and results:
            results = self._apply_mmr(q, results, top_k, mmr_lambda)

        ms = (_time.perf_counter() - t0) * 1000
        if ms > SLOW_QUERY_MS:
            logger.warning(f"Slow query on '{self.name}': {ms:.1f}ms top_k={top_k} filter={filter_dict}")

        if ef_search:
            self._index.set_ef_search(HNSW_EF_SEARCH)

        if CACHE_ENABLED and not use_mmr:
            get_cache().put(self.name, q, top_k, results, filter_dict)
            from utils.prometheus import prom

            prom.cache_misses.inc()

        return results, False

    def search_by_id(
        self, vec_id: str, top_k: int = 10, filter_dict: Optional[Dict] = None, include_vector: bool = False
    ) -> Tuple[List[Dict], bool]:
        record = self._db.get(vec_id, include_vector=True)
        if record is None:
            raise KeyError(f"Vector '{vec_id}' not found")
        return self.search(record["vector"], top_k, filter_dict, include_vector)

    def hybrid_search(
        self,
        query_vector: Union[Sequence[float], np.ndarray],
        text: str,
        top_k: int = 10,
        vector_weight: float = 0.7,
        filter_dict: Optional[Dict] = None,
        include_vector: bool = False,
        fusion: str = "rrf",
    ) -> List[Dict[str, Any]]:
        normalize = (self.distance == "cosine")
        q = prepare_vector(query_vector, normalize)
        overfetch = max(top_k * OVERFETCH_FACTOR, MIN_OVERFETCH)
        faiss_ids, vec_scores = self._index.search(q, overfetch)

        # Dense candidates
        dense_records = self._db.get_by_faiss_ids(faiss_ids.tolist(), False)
        dense_ranked = []
        for fid, sc in zip(faiss_ids, vec_scores):
            rec = dense_records.get(int(fid))
            if rec:
                dense_ranked.append((rec["id"], float(sc)))

        # Keyword candidates
        kw_ids = self._db.keyword_search(text, limit=500)
        kw_ranked = [(vid, 1.0 / (i + 1)) for i, vid in enumerate(kw_ids)]

        if fusion == "rrf":
            fused = rrf_fuse([dense_ranked, kw_ranked], weights=[vector_weight, 1.0 - vector_weight])
        else:
            # Linear weighted fusion with normalization
            fused = linear_weighted_fuse(dense_ranked, kw_ranked, vector_weight=vector_weight)

        # Fetch records and apply filter
        results = []
        for vid, score in fused[: top_k * OVERFETCH_FACTOR]:
            rec = self._db.get(vid, include_vector)
            if rec and (not filter_dict or match_filter(rec["metadata"], filter_dict)):
                rec["score"] = float(score)
                results.append(rec)
                if len(results) >= top_k:
                    break
        return results

    def search_sparse(
        self, sparse_vector_dict: Dict[int, float], top_k: int = 10, filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search using sparse vector (inverted index)."""
        raw_results = self._sparse.search(sparse_vector_dict, top_k * 5)
        results = []
        for vid, score in raw_results:
            rec = self._db.get(vid, False)
            if rec and (not filter_dict or match_filter(rec["metadata"], filter_dict)):
                rec["score"] = float(score)
                results.append(rec)
                if len(results) >= top_k:
                    break
        return results

    def batch_search(self, queries: List[Dict[str, Any]], include_vector: bool = False) -> List[List[Dict]]:
        return [
            self.search(
                q["vector"],
                q.get("top_k", 10),
                q.get("filter"),
                include_vector,
                q.get("score_threshold"),
            )[0]
            for q in queries
        ]

    # ── CRUD ──────────────────────────────────────────────────────────────────
    def get(self, vec_id: str, include_vector: bool = False) -> Optional[Dict]:
        if self._closed:
            raise RuntimeError("Collection is closed")
        return self._db.get(vec_id, include_vector)

    def delete(self, vec_id: str) -> bool:
        if self._closed:
            raise RuntimeError("Collection is closed")
        with self._write_lock:
            faiss_id = self._db.delete(vec_id)
            if faiss_id is None:
                return False
            self._index.remove([faiss_id])
            self._sparse.remove(vec_id)
            self._deleted_count += 1
            if CACHE_ENABLED:
                get_cache().invalidate_collection(self.name)
            return True

    def close(self):
        self._closed = True
        self._db.close()

    def delete_by_filter(self, filter_dict: Dict) -> int:
        """Bulk delete by metadata filter."""
        if self._closed:
            raise RuntimeError("Collection is closed")
        with self._write_lock:
            records, total = self._db.scroll(limit=10000, filter_dict=filter_dict)
            deleted = 0
            for rec in records:
                if self.delete(rec["id"]):
                    deleted += 1
            return deleted

    def patch_metadata(self, vec_id: str, metadata: Dict) -> bool:
        """Update only metadata, no re-indexing."""
        if self._closed:
            raise RuntimeError("Collection is closed")
        return self._db.patch_metadata(vec_id, metadata)

    def count(self, filter_dict: Optional[Dict] = None) -> int:
        if self._closed:
            raise RuntimeError("Collection is closed")
        return self._db.count_filtered(filter_dict)

    def facets(self, field: str, limit: int = 100) -> Dict[str, int]:
        if self._closed:
            raise RuntimeError("Collection is closed")
        return self._db.facets(field, limit)

    def scroll(
        self, limit: int = 100, offset: int = 0, filter_dict: Optional[Dict] = None, include_vector: bool = False
    ) -> Tuple[List[Dict], int]:
        if self._closed:
            raise RuntimeError("Collection is closed")
        return self._db.scroll(limit, offset, filter_dict, include_vector)

    # ── Maintenance ───────────────────────────────────────────────────────────
    def reap_expired(self) -> int:
        if self._closed:
            raise RuntimeError("Collection is closed")
        with self._write_lock:
            fids = self._db.delete_expired()
            if fids:
                self._index.remove(fids)
                self._deleted_count += len(fids)
                self._maybe_rebuild()
            return len(fids)

    def rebuild_index(self) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("Collection is closed")
        with self._write_lock:
            vectors, faiss_ids = self._db.all_vectors_for_rebuild()
            desired = self._desired_itype or _auto_type(len(faiss_ids))
            logger.info(f"[{self.name}] Rebuilding -> {desired} ({len(faiss_ids)} vecs)")

            centroids = None
            if self._centroids_path.exists():
                try:
                    centroids = np.load(str(self._centroids_path), allow_pickle=False)
                except Exception as exc:
                    logger.error(f"Failed to load centroids from {self._centroids_path}: {exc}")

            self._index.rebuild(
                vectors,
                faiss_ids,
                desired,
                ivfpq_m=self.ivfpq_m,
                ivfpq_nbits=self.ivfpq_nbits,
                centroids=centroids,
                distance=self.distance,
            )
            self._deleted_count = 0
            self._save_centroids_if_needed()
            self._save()
            return {"index_type": desired.value, "vector_count": len(faiss_ids)}

    def force_save(self):
        self._save()

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def vector_count(self) -> int:
        return self._db.count()

    @property
    def disk_size_bytes(self) -> int:
        total = self._db.size_bytes()
        try:
            total += os.path.getsize(self._index_path)
        except OSError:
            pass
        return total

    @property
    def index_type(self) -> IndexType:
        return self._index.index_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "distance": self.distance,
            "quant_mode": self.quant_mode,
            "index_type": self.index_type.value,
            "description": self.description,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "ivfpq_m": self.ivfpq_m,
            "ivfpq_nbits": self.ivfpq_nbits,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    # ── Internals ─────────────────────────────────────────────────────────────
    def _assemble(
        self,
        faiss_ids: np.ndarray,
        scores: np.ndarray,
        top_k: int,
        filter_dict: Optional[Dict],
        include_vector: bool,
        score_threshold: Optional[float],
    ) -> List[Dict[str, Any]]:
        if not len(faiss_ids):
            return []
        if score_threshold is not None:
            mask = scores >= score_threshold
            faiss_ids, scores = faiss_ids[mask], scores[mask]
        if filter_dict:
            kept = set(self._db.filter_faiss_ids(faiss_ids.tolist(), filter_dict))
            mask = np.array([fid in kept for fid in faiss_ids])
            faiss_ids, scores = faiss_ids[mask], scores[mask]
        faiss_ids, scores = faiss_ids[:top_k], scores[:top_k]
        records = self._db.get_by_faiss_ids(faiss_ids.tolist(), include_vector)
        out = []
        for fid, sc in zip(faiss_ids, scores):
            rec = records.get(int(fid))
            if rec:
                rec["score"] = float(sc)
                out.append(rec)
        return out

    def _apply_mmr(self, q: np.ndarray, results: List[Dict], top_k: int, lam: float) -> List[Dict]:
        from core.reranker import mmr

        if not results:
            return results
        vecs = []
        for r in results:
            rv = r.get("vector") or self._db.get(r["id"], True).get("vector", [])
            vecs.append(np.array(rv, dtype=np.float32) if rv else np.zeros(self.dimension))
        mat = np.vstack(vecs)
        ids = [r["id"] for r in results]
        scores_arr = np.array([r["score"] for r in results])
        pairs = mmr(q, mat, ids, scores_arr, top_k, lam)
        id_to_rec = {r["id"]: r for r in results}
        return [id_to_rec[vid] | {"score": sc} for vid, sc in pairs]

    def _load_sparse(self):
        try:
            rows = self._db.get_all_sparse()
            for vec_id, sv in rows:
                if sv:
                    self._sparse.add(vec_id, sv)
        except Exception as exc:
            logger.warning(f"Could not load sparse index: {exc}")

    def _on_write(self, n: int = 1):
        self._op_count += n
        if self._op_count % AUTO_SAVE_INTERVAL == 0:
            self._save()
        if self._index.needs_upgrade:
            self.rebuild_index()
        self._save_centroids_if_needed()

    def _maybe_rebuild(self):
        n = self.vector_count
        if n > 0 and self._deleted_count / n >= MAX_DELETED_RATIO:
            self.rebuild_index()

    def _save_centroids_if_needed(self):
        with self._write_lock:
            if not self._centroids_path.exists():
                centroids = self._index.get_centroids()
                if centroids is not None:
                    try:
                        tmp_path = str(self._centroids_path).replace(".npy", ".tmp.npy")
                        np.save(tmp_path, centroids)
                        os.replace(tmp_path, str(self._centroids_path))
                        logger.info(f"Saved extracted centroids to {self._centroids_path}")
                    except Exception as exc:
                        logger.error(f"Failed to save centroids to {self._centroids_path}: {exc}")

    def _save(self):
        with self._save_lock:
            _save(self._index, self._index_path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
