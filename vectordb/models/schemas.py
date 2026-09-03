"""Pydantic v2 schemas — expanded for all new features."""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
import re
from pydantic import BaseModel, Field, field_validator
from config import DEFAULT_TOP_K, MAX_TOP_K

class IndexType(str, Enum):
    FLAT="Flat"; IVF="IVF"; HNSW="HNSW"; IVFPQ="IVFPQ"

class DistanceMetric(str, Enum):
    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    L2 = "l2"
    MANHATTAN = "manhattan"


class QuantMode(str, Enum):
    FLOAT32="float32"; INT8="int8"; BINARY="binary"

# ── Collections ───────────────────────────────────────────────────────────────
class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    dimension: int = Field(384, ge=1, le=65536)
    distance: DistanceMetric = DistanceMetric.COSINE
    index_type: Optional[IndexType] = None
    quant_mode: QuantMode = QuantMode.FLOAT32
    description: str = ""
    hnsw_m: Optional[int] = Field(None, ge=4, le=128)
    hnsw_ef_construction: Optional[int] = Field(None, ge=8, le=512)
    ivfpq_m: Optional[int] = Field(None, ge=1, le=256)
    ivfpq_nbits: Optional[int] = Field(None, ge=1, le=16)
    @field_validator("name")
    @classmethod
    def _valid_name(cls, v):
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("name must be alphanumeric with _ or -")
        return v

class CollectionUpdate(BaseModel):
    description: Optional[str] = None

class CollectionInfo(BaseModel):
    name: str; dimension: int; distance: DistanceMetric
    index_type: IndexType; quant_mode: QuantMode; description: str
    vector_count: int; disk_size_bytes: int
    created_at: str; updated_at: str
    ivfpq_m: Optional[int] = None
    ivfpq_nbits: Optional[int] = None

# ── Vectors ───────────────────────────────────────────────────────────────────
class SparseVector(BaseModel):
    """Sparse vector: {term_id_str: weight}"""
    indices: List[int]
    values: List[float]

class VectorUpsert(BaseModel):
    id: str = Field(..., min_length=1, max_length=256)
    vector: Optional[List[float]] = None
    sparse_vector: Optional[SparseVector] = None
    metadata: Dict[str, Any] = {}
    ttl_seconds: Optional[int] = Field(None, ge=1)

class BatchUpsertRequest(BaseModel):
    vectors: List[VectorUpsert] = Field(..., min_length=1, max_length=1000)

class BatchUpsertResponse(BaseModel):
    inserted: int; updated: int; errors: List[Dict[str, str]] = []

class VectorRecord(BaseModel):
    id: str; vector: Optional[List[float]] = None
    metadata: Dict[str, Any]
    created_at: str; updated_at: str; expires_at: Optional[str] = None

class UpsertResponse(BaseModel):
    id: str; status: str

# ── Search ────────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    vector: List[float]
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    filter: Optional[Dict[str, Any]] = None
    include_vector: bool = False
    score_threshold: Optional[float] = None
    ef_search: Optional[int] = Field(None, ge=8, le=2048)  # A08
    use_mmr: bool = False                                    # B04
    mmr_lambda: float = Field(0.5, ge=0.0, le=1.0)         # B04
    rerank: bool = False                                     # B03
    rerank_query: Optional[str] = None                      # B03

class TextSearchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1024)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    filter: Optional[Dict[str, Any]] = None
    include_vector: bool = False
    score_threshold: Optional[float] = None
    rerank: bool = False

class IDSearchRequest(BaseModel):
    id: str; top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    filter: Optional[Dict[str, Any]] = None
    include_vector: bool = False

class HybridSearchRequest(BaseModel):
    vector: List[float]; text: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    vector_weight: float = Field(0.7, ge=0.0, le=1.0)
    filter: Optional[Dict[str, Any]] = None
    include_vector: bool = False
    fusion: str = Field("rrf", pattern="^(rrf|weighted)$")  # B01

class BatchSearchRequest(BaseModel):
    queries: List[SearchRequest] = Field(..., min_length=1, max_length=100)
    include_vector: bool = False

class SparseSearchRequest(BaseModel):
    sparse_vector: SparseVector
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
    filter: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    id: str; score: float; metadata: Dict[str, Any]
    vector: Optional[List[float]] = None
    rerank_score: Optional[float] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]; total_returned: int; query_time_ms: float
    cached: bool = False

class BatchSearchResponse(BaseModel):
    responses: List[SearchResponse]; total_time_ms: float

# ── Scroll ────────────────────────────────────────────────────────────────────
class ScrollResponse(BaseModel):
    vectors: List[VectorRecord]; total: int; offset: int; has_more: bool

# ── Metadata ops ─────────────────────────────────────────────────────────────
class MetadataPatch(BaseModel):
    metadata: Dict[str, Any]

class CountResponse(BaseModel):
    count: int; filter: Optional[Dict[str, Any]] = None

class FacetsResponse(BaseModel):
    field: str; values: Dict[str, int]  # value → count

# ── Admin ─────────────────────────────────────────────────────────────────────
class MetricsResponse(BaseModel):
    uptime_seconds: float; total_requests: int
    searches: int; upserts: int; deletes: int
    avg_search_ms: float; collections: Dict[str, Dict[str, Any]]
    cache: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str; collections: int; total_vectors: int; version: str

class SnapshotResponse(BaseModel):
    collection: str; path: str; size_bytes: int; vector_count: int
    created_at: str; export_seconds: float

class TaskResponse(BaseModel):
    task_id: str; status: str; message: str
    started_at: str; finished_at: Optional[str] = None
    result: Optional[Dict] = None
