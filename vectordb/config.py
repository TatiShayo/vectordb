"""VectorDB — all configuration, 100% env-overridable."""
import os
from pathlib import Path

# ── Data storage ──────────────────────────────────────────────────────────────
DATA_DIR         = Path(os.environ.get("VDB_DATA_DIR", "data"))
COLLECTIONS_DIR  = DATA_DIR / "collections"
REGISTRY_FILE    = DATA_DIR / "registry.json"
SNAPSHOTS_DIR    = DATA_DIR / "snapshots"

# ── Index ─────────────────────────────────────────────────────────────────────
DEFAULT_DIMENSION       = int(os.environ.get("VDB_DIMENSION", 384))
AUTO_SAVE_INTERVAL      = int(os.environ.get("VDB_AUTO_SAVE", 100))
AUTO_SCALE_INDEX        = os.environ.get("VDB_AUTO_SCALE", "true").lower() == "true"
FLAT_TO_IVF_THRESHOLD   = int(os.environ.get("VDB_FLAT_THRESHOLD", 10_000))
IVF_TO_HNSW_THRESHOLD   = int(os.environ.get("VDB_HNSW_THRESHOLD", 100_000))
IVF_NLIST               = int(os.environ.get("VDB_IVF_NLIST", 100))
IVF_NPROBE              = int(os.environ.get("VDB_IVF_NPROBE", 10))
HNSW_M                  = int(os.environ.get("VDB_HNSW_M", 32))
HNSW_EF_CONSTRUCTION    = int(os.environ.get("VDB_HNSW_EF", 64))
HNSW_EF_SEARCH          = int(os.environ.get("VDB_HNSW_EF_SEARCH", 64))
MAX_DELETED_RATIO       = float(os.environ.get("VDB_REBUILD_RATIO", 0.3))

# ── Auth ──────────────────────────────────────────────────────────────────────
API_KEY_HEADER  = "X-API-Key"
ADMIN_KEY       = os.environ.get("VDB_ADMIN_KEY", "admin-secret")
USER_KEYS       = set(k.strip() for k in os.environ.get("VDB_USER_KEYS", "user-secret").split(",") if k.strip())
ALL_KEYS        = {ADMIN_KEY} | USER_KEYS
ADMIN_KEYS      = {ADMIN_KEY}

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_RPM      = int(os.environ.get("VDB_RATE_LIMIT_RPM", 1000))
RATE_LIMIT_ENABLED  = os.environ.get("VDB_RATE_LIMIT", "true").lower() == "true"

# ── Search ────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K   = 10
MAX_TOP_K       = 1000
OVERFETCH_FACTOR= 10
MIN_OVERFETCH   = 100
SEARCH_COST     = 5.0   # token cost for search vs plain request

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_ENABLED       = os.environ.get("VDB_CACHE", "true").lower() == "true"
CACHE_MAX_SIZE      = int(os.environ.get("VDB_CACHE_SIZE", 1024))
CACHE_TTL_SECONDS   = float(os.environ.get("VDB_CACHE_TTL", 60.0))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL         = os.environ.get("VDB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL          = os.environ.get("VDB_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_ENABLED        = os.environ.get("VDB_RERANKER", "false").lower() == "true"

# ── Background tasks ──────────────────────────────────────────────────────────
REAPER_INTERVAL_SECONDS = int(os.environ.get("VDB_REAPER_INTERVAL", 60))
AUTOSAVE_INTERVAL_SEC   = int(os.environ.get("VDB_AUTOSAVE_SEC", 300))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL       = os.environ.get("VDB_LOG_LEVEL", "INFO")
LOG_FORMAT      = os.environ.get("VDB_LOG_FORMAT", "text")    # text | json
SLOW_QUERY_MS   = float(os.environ.get("VDB_SLOW_QUERY_MS", 100.0))

# ── Snapshot ──────────────────────────────────────────────────────────────────
SNAPSHOT_DIR    = DATA_DIR / "snapshots"
