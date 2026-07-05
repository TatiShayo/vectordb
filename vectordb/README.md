# ⚡ VectorDB

A **lightweight, production-ready vector database** built from scratch in Python.  
Single-binary, zero-cost dependencies, demo-able in 60 seconds.

```
POST /collections/products/search  →  top-k similar vectors in <2ms (Flat, 10K vecs)
```

---

## What makes this different

| Feature | This project | ChromaDB | Pinecone |
|---|---|---|---|
| **Storage** | Per-collection SQLite + FAISS | DuckDB/SQLite + HNSW | Cloud-only |
| **Raw vector retrieval** | ✅ (search/by-id works) | ❌ | ✅ |
| **TTL / auto-expiry** | ✅ background reaper | ❌ | ❌ |
| **Hybrid search** | ✅ vector + FTS5 keyword | ❌ | ❌ |
| **Auto index upgrade** | ✅ Flat→IVF→HNSW | manual | managed |
| **Batch search** | ✅ 100 queries/call | ❌ | ✅ |
| **Admin UI** | ✅ built-in | ❌ | cloud UI |
| **Cost** | $0 | $0 | $$$ |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
│  /collections  /vectors  /search  /admin                │
└──────────────┬──────────────────────────────────────────┘
               │
      ┌────────▼────────┐
      │     Engine      │  ← manages all collections
      │  (singleton)    │  ← loads registry on startup
      │                 │  ← runs TTL reaper thread
      └────────┬────────┘
               │  one per collection
      ┌────────▼────────────────────────────────┐
      │              Collection                  │
      │                                          │
      │  ┌──────────────┐   ┌─────────────────┐ │
      │  │  FAISSIndex  │   │  CollectionDB   │ │
      │  │              │   │  (SQLite WAL)   │ │
      │  │ Flat/IVF/HNSW│   │                 │ │
      │  │ auto-upgrade │   │ raw_vector BLOB │ │
      │  │ thread-safe  │   │ FTS5 full-text  │ │
      │  │   RLock      │   │ metadata JSON   │ │
      │  └──────────────┘   │ TTL expires_at  │ │
      │                     └─────────────────┘ │
      └─────────────────────────────────────────┘
               │
      ┌────────▼────────┐
      │  Checkpointer   │  ← atomic save (tmp → rename)
      │  (crash-safe)   │  ← rebuild from SQLite on mismatch
      └─────────────────┘
```

### Key design decisions

**1. Per-collection FAISS index**  
Each collection owns a separate FAISS instance. Namespace isolation without any cross-collection locking. Scales to many small collections easily.

**2. Raw vectors stored in SQLite**  
Unlike most FAISS wrappers, we persist the float32 bytes as BLOBs alongside metadata. This enables:
- `search/by-id` — find similar items to a stored vector without re-sending it  
- Full index rebuild after crash (no data loss)  
- Correct delete/update behaviour (remove old vector, insert new)

**3. Auto index upgrade (Flat → IVF → HNSW)**  
- `< 10K vectors` → **IndexFlatIP** — exact, zero config, ~0.3ms query  
- `10K–100K vectors` → **IndexIVFFlat** — ~10× faster, ~95% recall with nprobe=10  
- `> 100K vectors` → **IndexHNSWFlat** — graph-based, ~99% recall, highest throughput  
Transitions happen automatically on write; `POST /admin/collections/{name}/rebuild` triggers manually.

**4. HNSW soft delete**  
FAISS HNSW doesn't support true removal. We maintain a soft-delete set and exclude those IDs post-search. When `delete_ratio ≥ 30%`, `rebuild_index()` compacts everything cleanly.

**5. Atomic persistence**  
Write FAISS index → `.faiss.tmp` → `os.replace()`. On Linux this is a single syscall (rename), guaranteed atomic. SQLite uses WAL mode — readers never block writers.

**6. Hybrid search (FTS5 + vector)**  
Combines cosine similarity score (weighted 0–1) with FTS5 BM25 keyword rank. No extra dependencies — SQLite FTS5 is bundled in the standard library.

---

## Quick start

### Option A — Docker (recommended)

```bash
docker compose up -d
# API: http://localhost:8000
# UI:  http://localhost:8000/ui
# Docs: http://localhost:8000/docs
```

### Option B — Local Python

```bash
cd vectordb
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Seed demo data

```bash
python seed_data.py --n 500 --dim 384
```

---

## API Reference

All requests require `X-API-Key` header.  
Default keys: `admin-secret` (full access), `user-secret` (read/write vectors).

### Collections

```http
GET    /collections                    # list all
POST   /collections                    # create (admin)
GET    /collections/{name}             # get info
PATCH  /collections/{name}             # update description (admin)
DELETE /collections/{name}             # delete (admin)
```

**Create collection:**
```json
POST /collections
{
  "name": "products",
  "dimension": 384,
  "distance": "cosine",
  "index_type": null,        // null = auto-select
  "description": "optional"
}
```

### Vectors

```http
POST   /collections/{name}/vectors          # upsert one
POST   /collections/{name}/vectors/batch    # upsert up to 1000
GET    /collections/{name}/vectors/{id}     # get by ID
DELETE /collections/{name}/vectors/{id}     # delete
DELETE /collections/{name}/vectors          # batch delete (body: ["id1","id2"])
GET    /collections/{name}/vectors/scroll   # paginate all vectors
```

**Upsert with TTL:**
```json
POST /collections/sessions/vectors
{
  "id": "sess_abc123",
  "vector": [0.1, 0.2, ...],
  "metadata": {"user_id": "42", "page": "/checkout"},
  "ttl_seconds": 86400
}
```

### Search

```http
POST /collections/{name}/search           # by vector
POST /collections/{name}/search/by-id     # by stored ID
POST /collections/{name}/search/by-text   # embed text → search
POST /collections/{name}/search/hybrid    # vector + keyword
POST /collections/{name}/search/batch     # up to 100 queries/call
```

**Vector search with filter:**
```json
POST /collections/products/search
{
  "vector": [0.1, -0.3, ...],
  "top_k": 10,
  "filter": {"category": "electronics", "in_stock": true},
  "score_threshold": 0.7,
  "include_vector": false
}
```

**Hybrid search:**
```json
POST /collections/products/search/hybrid
{
  "vector": [0.1, -0.3, ...],
  "text": "wireless headphones noise cancelling",
  "top_k": 10,
  "vector_weight": 0.7,
  "filter": {"in_stock": true}
}
```

**Batch search (one round-trip):**
```json
POST /collections/products/search/batch
{
  "queries": [
    {"vector": [...], "top_k": 5},
    {"vector": [...], "top_k": 3, "filter": {"category": "kitchen"}}
  ],
  "include_vector": false
}
```

### Admin

```http
GET  /admin/health        # public, no auth
GET  /admin/metrics       # uptime, ops counts, per-collection stats
POST /admin/save          # flush all FAISS to disk (admin)
POST /admin/collections/{name}/rebuild    # force index rebuild (admin)
POST /admin/collections/{name}/reap       # force TTL cleanup (admin)
```

---

## Python Client

```python
from client.client import VectorDBClient

db = VectorDBClient("http://localhost:8000", api_key="user-secret")

# Create a collection
db.create_collection("docs", dimension=384)

# Upsert
db.upsert("docs", "doc_1", vector=[...], metadata={"title": "Hello World"})

# Batch upsert
db.upsert_batch("docs", [
    {"id": "doc_2", "vector": [...], "metadata": {"title": "Article Two"}},
    {"id": "doc_3", "vector": [...], "metadata": {"title": "Article Three"}},
])

# Search
results = db.search("docs", vector=[...], top_k=5)
for r in results:
    print(r["id"], r["score"], r["metadata"])

# Search by stored ID (no need to re-send the vector)
similar = db.search_by_id("docs", id="doc_1", top_k=5)

# Hybrid search
results = db.hybrid_search("docs", vector=[...], text="hello world", top_k=5)

# Scroll all vectors
page = db.scroll("docs", limit=100, offset=0)

# TTL vector (expires in 1 hour)
db.upsert("sessions", "sess_abc", vector=[...], metadata={}, ttl_seconds=3600)
```

### Async client

```python
import asyncio
from client.client import AsyncVectorDBClient

async def main():
    async with AsyncVectorDBClient("http://localhost:8000") as db:
        results = await db.search("products", vector=[...], top_k=10)
        print(results)

asyncio.run(main())
```

---

## FAISS index type guide

| Size | Index | Exact? | Memory | Query time |
|---|---|---|---|---|
| < 10K vecs | **Flat** | ✅ 100% recall | 1× | ~0.2ms |
| 10K–100K | **IVF** | ~95% (nprobe=10) | 1.1× | ~0.5ms |
| > 100K | **HNSW** | ~99% (ef=64) | 2× | ~1ms |
| > 1M | **IVF+PQ** (future) | ~90% | 0.1× | ~2ms |

**Memory for float32 vectors:**
| Dimension | 100K vecs | 1M vecs |
|---|---|---|
| 384 | 147 MB | 1.47 GB |
| 768 | 294 MB | 2.94 GB |
| 1536 | 589 MB | 5.89 GB |

---

## Running tests

```bash
pip install pytest httpx
pytest tests/ -v
```

Expected: 30+ tests covering auth, CRUD, all search modes, TTL, persistence, and rebuild.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `VDB_DATA_DIR` | `data` | Where collections are stored |
| `VDB_ADMIN_KEY` | `admin-secret` | Admin API key |
| `VDB_USER_KEYS` | `user-secret` | Comma-separated user keys |
| `VDB_DIMENSION` | `384` | Default dimension |
| `VDB_AUTO_SCALE` | `true` | Enable auto index upgrade |
| `VDB_FLAT_THRESHOLD` | `10000` | Flat→IVF threshold |
| `VDB_HNSW_THRESHOLD` | `100000` | IVF→HNSW threshold |
| `VDB_AUTO_SAVE` | `100` | Save every N write ops |
| `VDB_REAPER_INTERVAL` | `60` | TTL check interval (seconds) |

---

## Project structure

```
vectordb/
├── main.py              # FastAPI app, lifespan, routers
├── config.py            # All settings, env-overridable
├── seed_data.py         # Demo data seeder
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── core/
│   ├── indexer.py       # Thread-safe FAISS wrapper (auto-upgrade, soft delete)
│   ├── collection.py    # Per-collection coordinator (FAISS + SQLite in sync)
│   └── engine.py        # Global manager, registry, TTL reaper
│
├── storage/
│   ├── db.py            # Per-collection SQLite (WAL, FTS5, BLOB vectors)
│   └── checkpointer.py  # Atomic save/load, crash recovery via rebuild
│
├── api/
│   ├── auth.py          # API key middleware (admin/user roles)
│   ├── collections.py   # Collection CRUD
│   ├── vectors.py       # Vector upsert, get, delete, scroll, batch
│   ├── search.py        # All search endpoints
│   └── management.py    # Health, metrics, rebuild, force-save
│
├── models/
│   └── schemas.py       # All Pydantic models
│
├── utils/
│   ├── normalize.py     # L2 normalization
│   ├── embedder.py      # Lazy sentence-transformer loader
│   └── metrics.py       # In-memory rolling metrics
│
├── client/
│   └── client.py        # Sync + async Python clients
│
├── web/
│   └── index.html       # Admin UI (single-file, no build step)
│
└── tests/
    └── test_api.py      # Full test suite (30+ tests)
```
