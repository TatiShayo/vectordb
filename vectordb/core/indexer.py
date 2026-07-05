"""
Thread-safe FAISS index wrapper — enhanced with:
- A08: per-query efSearch tuning
- A13: auto-compute nlist
- A14: M/efConstruction tunable at init
- Proper HNSW ef_search setter
"""
from __future__ import annotations
import logging, threading
from enum import Enum
from typing import List, Set, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    FLAT="Flat"; IVF="IVF"; HNSW="HNSW"; IVFPQ="IVFPQ"


def _auto_type(n: int) -> IndexType:
    from config import FLAT_TO_IVF_THRESHOLD, IVF_TO_HNSW_THRESHOLD
    if n < FLAT_TO_IVF_THRESHOLD: return IndexType.FLAT
    if n < IVF_TO_HNSW_THRESHOLD: return IndexType.IVF
    return IndexType.HNSW


def _auto_nlist(n: int) -> int:
    """A13: nlist = 4*sqrt(N), clamped 4–4096."""
    import math
    return max(4, min(4096, int(4 * math.sqrt(max(n, 1)))))


def _build_index(dim, itype, hnsw_m=32, hnsw_ef_construction=64, ivfpq_m=None, ivfpq_nbits=None, centroids=None):
    import faiss
    from config import IVF_NLIST
    if itype == IndexType.FLAT:
        base = faiss.IndexFlatIP(dim)
    elif itype == IndexType.IVF:
        nlist = max(IVF_NLIST, _auto_nlist(0))
        q = faiss.IndexFlatIP(dim)
        if centroids is not None:
            q.add(centroids.astype(np.float32))
        base = faiss.IndexIVFFlat(q, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    elif itype == IndexType.IVFPQ:
        nlist = max(IVF_NLIST, _auto_nlist(0))
        q = faiss.IndexFlatIP(dim)
        if centroids is not None:
            q.add(centroids.astype(np.float32))
        m = ivfpq_m if ivfpq_m is not None else 8
        nbits = ivfpq_nbits if ivfpq_nbits is not None else 8
        base = faiss.IndexIVFPQ(q, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    elif itype == IndexType.HNSW:
        base = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = hnsw_ef_construction
    else:
        raise ValueError(f"Unknown index type: {itype}")
    return faiss.IndexIDMap2(base)


class FAISSIndex:
    def __init__(self, dimension, index_type=IndexType.FLAT,
                 hnsw_m=32, hnsw_ef_construction=64,
                 ivfpq_m=None, ivfpq_nbits=None, centroids=None):
        import faiss
        self.dimension = dimension
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.ivfpq_m = ivfpq_m
        self.ivfpq_nbits = ivfpq_nbits
        self.centroids = centroids
        self._lock = threading.RLock()
        self._index = _build_index(dimension, index_type, hnsw_m, hnsw_ef_construction,
                                   ivfpq_m=ivfpq_m, ivfpq_nbits=ivfpq_nbits, centroids=centroids)
        self._trained = (index_type not in (IndexType.IVF, IndexType.IVFPQ))
        self._soft_deleted: Set[int] = set()
        self._pending_upgrade: Optional[IndexType] = None

    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        assert vectors.dtype == np.float32 and ids.dtype == np.int64
        with self._lock:
            if self.index_type in (IndexType.IVF, IndexType.IVFPQ) and not self._trained:
                self._train_ivf(vectors)
            if self.index_type not in (IndexType.IVF, IndexType.IVFPQ) or self._trained:
                self._index.add_with_ids(vectors, ids)
            if self._pending_upgrade is None:
                desired = _auto_type(self._index.ntotal)
                if desired != self.index_type:
                    self._pending_upgrade = desired
                    logger.info(f"Auto-upgrade queued: {self.index_type}→{desired} at {self._index.ntotal} vecs")

    def search(self, query: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if query.ndim == 1: query = query.reshape(1,-1)
        assert query.dtype == np.float32
        with self._lock:
            n = self._index.ntotal
            if n == 0:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            k = min(top_k, n)
            scores, ids = self._index.search(query, k)
        raw_ids, raw_scores = ids[0], scores[0]
        mask = raw_ids != -1
        if self._soft_deleted:
            mask &= ~np.isin(raw_ids, list(self._soft_deleted))
        return raw_ids[mask], raw_scores[mask]

    def remove(self, faiss_ids: List[int]) -> int:
        if not faiss_ids: return 0
        import faiss
        with self._lock:
            if self.index_type == IndexType.HNSW:
                before = len(self._soft_deleted)
                self._soft_deleted.update(faiss_ids)
                return len(self._soft_deleted) - before
            arr = np.array(faiss_ids, dtype=np.int64)
            sel = faiss.IDSelectorBatch(arr.size, faiss.swig_ptr(arr))
            return int(self._index.remove_ids(sel))

    def rebuild(self, vectors, ids, new_type=None, ivfpq_m=None, ivfpq_nbits=None, centroids=None):
        if new_type is None:
            new_type = _auto_type(len(ids)) if len(ids) > 0 else IndexType.FLAT
        with self._lock:
            from config import IVF_NLIST
            self._soft_deleted.clear()
            self._pending_upgrade = None
            self.index_type = new_type
            self.ivfpq_m = ivfpq_m
            self.ivfpq_nbits = ivfpq_nbits
            self.centroids = centroids
            self._index = _build_index(self.dimension, new_type,
                                       self.hnsw_m, self.hnsw_ef_construction,
                                       ivfpq_m=self.ivfpq_m, ivfpq_nbits=self.ivfpq_nbits,
                                       centroids=self.centroids)
            if new_type in (IndexType.IVF, IndexType.IVFPQ):
                nlist = max(IVF_NLIST, _auto_nlist(len(ids)))
                if centroids is not None or len(vectors) >= nlist:
                    self._train_ivf(vectors, _nlist=nlist, _skip_lock=True)
                else:
                    self.index_type = IndexType.FLAT
                    self._index = _build_index(self.dimension, IndexType.FLAT)
                    self._trained = True
            if (self.index_type not in (IndexType.IVF, IndexType.IVFPQ) or self._trained) and len(ids):
                self._index.add_with_ids(vectors, ids)

    def save(self, path):
        import faiss
        with self._lock:
            faiss.write_index(self._index, path)

    def load(self, path):
        import faiss
        with self._lock:
            self._index = faiss.read_index(path)
            self.dimension = self._index.d
            self._soft_deleted.clear()
            self._infer_type()

    def set_nprobe(self, nprobe: int):
        import faiss
        with self._lock:
            if self.index_type in (IndexType.IVF, IndexType.IVFPQ):
                try: faiss.downcast_index(self._index.index).nprobe = nprobe
                except Exception: pass

    def set_ef_search(self, ef: int):
        """A08: tune efSearch per query."""
        import faiss
        with self._lock:
            if self.index_type == IndexType.HNSW:
                try: faiss.downcast_index(self._index.index).hnsw.efSearch = ef
                except Exception: pass

    @property
    def size(self):
        return self._index.ntotal - len(self._soft_deleted)

    @property
    def needs_upgrade(self):
        return self._pending_upgrade is not None

    @property
    def pending_upgrade_type(self):
        return self._pending_upgrade

    def get_centroids(self) -> Optional[np.ndarray]:
        import faiss
        with self._lock:
            if self.index_type not in (IndexType.IVF, IndexType.IVFPQ):
                return None
            try:
                inner = faiss.downcast_index(self._index.index)
                nlist = inner.nlist
                quantizer = faiss.downcast_index(inner.quantizer)
                if quantizer.ntotal == 0:
                    return None
                return quantizer.reconstruct_n(0, nlist)
            except Exception as exc:
                logger.error(f"Failed to get centroids: {exc}")
                return None

    def _train_ivf(self, vectors, _nlist=None, _skip_lock=False):
        import faiss
        from config import IVF_NLIST
        nlist = _nlist or IVF_NLIST
        
        if self.centroids is not None:
            try:
                inner = faiss.downcast_index(self._index.index)
                if inner.quantizer.ntotal == 0:
                    inner.quantizer.add(self.centroids.astype(np.float32))
                if self.index_type == IndexType.IVFPQ:
                    if not inner.is_trained:
                        if len(vectors) > 0:
                            inner.train(vectors)
                else:
                    inner.is_trained = True
                self._trained = True
                logger.info(f"IVF trained with pre-loaded centroids ({nlist} centroids)")
            except Exception as exc:
                logger.error(f"IVF training with pre-loaded centroids failed: {exc}")
            return

        if len(vectors) < nlist:
            logger.warning(f"Need ≥{nlist} vectors to train IVF, have {len(vectors)}")
            return
        try:
            inner = faiss.downcast_index(self._index.index)
            inner.train(vectors)
            self._trained = True
            logger.info(f"IVF trained: {len(vectors)} vecs, {nlist} centroids")
        except Exception as exc:
            logger.error(f"IVF training failed: {exc}")

    def _infer_type(self):
        import faiss
        try:
            inner = faiss.downcast_index(self._index.index)
            if isinstance(inner, faiss.IndexIVFFlat):
                self.index_type = IndexType.IVF; self._trained = inner.is_trained
            elif isinstance(inner, faiss.IndexIVFPQ):
                self.index_type = IndexType.IVFPQ; self._trained = inner.is_trained
            elif isinstance(inner, faiss.IndexHNSWFlat):
                self.index_type = IndexType.HNSW; self._trained = True
            else:
                self.index_type = IndexType.FLAT; self._trained = True
        except Exception:
            self.index_type = IndexType.FLAT; self._trained = True
