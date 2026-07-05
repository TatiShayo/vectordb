# Lightweight Vector Database — Phase Plan
**Project:** vectordb  
**Budget:** $0 (all free/open-source)  
**Builder:** Gemini CLI  
**Status:** Planned, awaiting execution after contentrec  

---

## What it is

A lightweight vector database service. Think mini-Pinecone or ChromaDB but built from scratch to demonstrate system design skills. It stores embeddings, searches by similarity, and exposes a clean API.

## Why it impresses interviewers

- **System design**: ANN indexing, persistence, scaling strategies
- **Data structures**: IVF, HNSW, product quantization
- **API design**: Clean CRUD for vectors
- **Performance optimization**: FAISS, batch operations
- **Client library**: pip-installable Python client

---

## Architecture

```
Client Apps → FastAPI Server → FAISS Index → Disk Storage (JSON/SQLite)
                ↓
         Client Library (pip package)
```

---

## Phases

### Phase 1: Core Vector Service
**Files created:** ~10  
**Gemini sessions:** 2

**What builds:**
```
vectordb/
├── main.py                    # FastAPI entry
├── config.py                  # Config constants
├── requirements.txt           # Dependencies
├── index/
│   ├── __init__.py
│   └── faiss_manager.py       # FAISS index wrapper (add, search, delete, save, load)
├── storage/
│   ├── __init__.py
│   ├── metadata.py            # SQLite for vector metadata (id, namespace, tags, payload)
│   └── persistence.py         # Save/load FAISS index to disk
├── api/
│   ├── __init__.py
│   ├── vectors.py             # POST /vectors (upsert), GET /vectors/{id}, DELETE /vectors/{id}
│   └── search.py              # POST /search (query vector, top-k)
├── models/
│   ├── __init__.py
│   └── schemas.py             # Pydantic models
└── seed_data.py               # Sample data to test with
```

**API endpoints:**
```
POST   /vectors          → {"id": str, "vector": [float], "metadata": {...}}
POST   /vectors/batch    → [{"id": str, "vector": [float], ...}]
GET    /vectors/{id}     → {"id": str, "vector": [float], "metadata": {...}}
DELETE /vectors/{id}     → {"status": "deleted"}
POST   /search           → {"vector": [float], "top_k": 10} → [{"id": str, "score": float}, ...]
POST   /search/by-id     → {"id": str, "top_k": 10} → vector search by existing ID
GET    /stats            → {"total_vectors": N, "dimension": D, "index_type": "Flat"}
```

**Key design:**
- FAISS IndexFlatIP (inner product) for exact search — accurate, fast for <100K vectors
- Namespaces for multi-tenancy (different apps share the same service)
- SQLite for metadata — vectors in FAISS, metadata in SQLite (separation of concerns)
- Save/load: `faiss.write_index()` + `faiss.read_index()` for persistence
- Auto-save timer (every N operations)

**Dependencies:** fastapi, uvicorn, faiss-cpu, numpy, pydantic

---

### Phase 2: Advanced Indexing
**Files added:** ~4  
**Gemini sessions:** 1

- IVF (Inverted File Index) for faster search on large datasets
- Index rebuilding (re-train IVF centroids)
- Auto-detect optimal index type based on dataset size
- Benchmarks: exact vs approximate recall comparison

---

### Phase 3: Python Client Library
**Files added:** ~5  
**Gemini sessions:** 1

```
client/
├── __init__.py
├── client.py            # VectorDBClient class (connect, upsert, search, delete)
├── sync.py              # Bulk sync helper
├── setup.py             # pip installable package
└── README.md
```

The client makes it pip-installable: `pip install vectordb-client` from local.

---

### Phase 4: Docker + Docs + Demo
**Files added:** ~4

- Dockerfile, docker-compose.yml
- HTML demo page (add vectors, search, see results)
- README with architecture diagram
- Postman collection

---

## PROMPT_SESSION.md for Research Phase

Prompt for Gemini CLI research:

```
You are a research agent. Study vector database architecture.

Research these systems:
1. Pinecone — architecture, indexing, API design
2. ChromaDB — open-source, how it works under the hood
3. FAISS — Meta's library, index types (Flat, IVF, HNSW, PQ), tradeoffs
4. Qdrant — Rust-based, HNSW indexing, filtering
5. Milvus — distributed vector database architecture

For each:
- Storage architecture (where do vectors live?)
- Index types supported and tradeoffs (speed vs accuracy)
- Filtering strategy (metadata pre-filter vs post-filter)
- Scaling approach (single node vs distributed)

Also research practical FAISS knowledge:
- What is IndexFlatIP vs IndexIVFFlat vs IndexHNSWFlat
- How to save/load FAISS indices
- How to delete vectors from FAISS (hint: IDMap wrappers)
- Memory requirements per million vectors at different dimensions

Deliverable: Save report to:
C:\Users\TATI\Desktop\Clients\vectordb\research_phase0.md
```

---

## Build Prompt for Gemini (Phase 1)

```
You are a backend engineer. Build a lightweight vector database service.

## Rules
- Use only free, open-source libraries
- No paid APIs
- All computation on CPU

## Tech Stack
- FastAPI + Uvicorn
- FAISS (faiss-cpu)
- SQLite (Python stdlib)
- NumPy

## What to build

### 1. index/faiss_manager.py
Class FAISSManager:
- __init__(dimension=384, index_type="Flat")
- add(vectors: np.ndarray, ids: list[int]) — add vectors to index
- search(query_vector, top_k=10) → list of (id, score)
- remove(ids: list[int]) — remove vectors by ID (use IDMap wrapper)
- save(path) — save index to disk
- load(path) — load index from disk
- get_size() — number of vectors in index
- Use faiss.IndexIDMap2 around the base index for ID support

### 2. storage/metadata.py
Class MetadataStore (SQLite):
- init_db() — create vectors table (id TEXT PK, namespace TEXT, metadata JSON, created_at, updated_at)
- upsert_metadata(vector_id, namespace, metadata)
- get_metadata(vector_id) → dict
- delete_metadata(vector_id)
- list_by_namespace(namespace) → list of dicts
- get_count() → int

### 3. storage/persistence.py
- save_checkpoint(faiss_manager, metadata_store, path) — save state
- load_checkpoint(faiss_manager, metadata_store, path) — restore state
- Handles graceful shutdown and restart

### 4. api/vectors.py
- POST /vectors — upsert: {id, vector, namespace, metadata={}}
- POST /vectors/batch — upsert multiple: [{id, vector, metadata}]
- GET /vectors/{id} — get vector + metadata
- DELETE /vectors/{id} — remove
- All operations sync FAISS + SQLite

### 5. api/search.py
- POST /search — {vector, top_k=10, namespace=None, filters={}} → [{id, score, metadata}]
- POST /search/by-id — {id, top_k=10} → search using stored vector
- POST /search/by-text — {text, top_k=10} → embed text with sentence-transformers, then search
- Support metadata filtering: pre-filter or post-filter

### 6. models/schemas.py
Pydantic models for all request/response schemas.

### 7. main.py
FastAPI app, mount routers, startup (load state), shutdown (save state),
/health, /stats endpoints, CORS enabled.

### 8. config.py
- DIMENSION = 384
- INDEX_TYPE = "Flat"
- AUTO_SAVE_INTERVAL = 100
- DB_PATH = "data/vectordb.sqlite"
- INDEX_PATH = "data/faiss.index"

### 9. seed_data.py
Generate 1000 random 384-dim vectors with fake metadata (title, description, tags).
Demonstrates all API endpoints.

### 10. requirements.txt
fastapi, uvicorn, faiss-cpu, numpy, pydantic, sentence-transformers

## Testing
Verify: server starts, can add vectors, search returns results,
persistence works across restart, delete works, batch operations work.
```
