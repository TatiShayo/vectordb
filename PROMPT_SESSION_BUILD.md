You are a backend engineer. Build a lightweight vector database service.

## Rules
- Use only free, open-source libraries
- No paid APIs
- All computation on CPU
- Keep dependencies minimal
- IMPORTANT: FAISS is NOT thread-safe. All FAISS operations MUST be protected by a threading.Lock

## Tech Stack
- FastAPI + Uvicorn for web server
- FAISS (faiss-cpu) for vector indexing
- SQLite3 with thread-safe access (run_in_threadpool or aiosqlite)
- NumPy for vector operations
- sentence-transformers (optional, for text-to-vector search)
- threading.Lock for all FAISS operations

## What to build

### 1. index/faiss_manager.py
Class `FAISSManager`:
- `__init__(self, dimension=384, index_type="Flat")` — initialize FAISS index
  - For "Flat": use IndexIDMap2(IndexFlatIP(dimension))
  - For "IVF": use IndexIDMap2(IndexIVFFlat(...)) with 100 centroids
  - For "HNSW": use IndexIDMap2(IndexHNSWFlat(dimension, 32))
  - Initialize self._lock = threading.Lock()
- `add(self, vectors: np.ndarray, ids: np.ndarray)` — add vectors with integer IDs
  - Vectors should be float32 numpy array of shape (N, dimension)
  - Normalize vectors to unit length (L2 normalization) for cosine similarity via inner product
  - Acquire self._lock before any FAISS operation
- `search(self, query_vector: np.ndarray, top_k: int = 10)` → list of (id, score) tuples
  - Normalize query vector before search
  - Acquire self._lock before FAISS search
- `remove(self, ids: list)` — remove vectors by ID (FAISS IDMap supports this)
  - Acquire self._lock before FAISS remove
- `save(self, path: str)` — write index to disk using faiss.write_index
  - Acquire self._lock before save
- `load(self, path: str)` — load index from disk
- `get_size(self)` → int — number of vectors
- `get_dimension(self)` → int — vector dimension

### 2. storage/metadata.py
Class `MetadataStore`:
- `__init__(self, db_path: str)`
- `init_db(self)` — create SQLite table:
  ```sql
  CREATE TABLE IF NOT EXISTS vectors (
    id TEXT PRIMARY KEY,
    faiss_id INTEGER UNIQUE,
    namespace TEXT DEFAULT 'default',
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  CREATE INDEX IF NOT EXISTS idx_namespace ON vectors(namespace);
  ```
- `upsert(self, vector_id: str, faiss_id: int, namespace: str = "default", metadata: dict = {})`
- `get(self, vector_id: str)` → dict or None
- `delete(self, vector_id: str)` → bool
- `get_by_namespace(self, namespace: str)` → list of dicts
- `get_count(self)` → int
- `get_next_faiss_id(self)` → int (auto-incrementing counter)
- `search_metadata(self, filters: dict)` → list of vector_ids matching filter conditions

### 3. storage/persistence.py
- `save_state(faiss_manager, metadata_store, base_path)` — ATOMIC save
  - Save FAISS index to a temp file (.faiss.tmp), then rename to .faiss (atomic on same filesystem)
  - Save SQLite checkpoint
  - This prevents inconsistency if crash happens mid-save
- `load_state(faiss_manager, metadata_store, base_path)` — restore from disk
- Creates/reads from directory: base_path/index.faiss + base_path/metadata.db

### 4. api/vectors.py
FastAPI router with:
- `POST /vectors` — upsert a single vector
  - Body: `{"id": str, "vector": [float], "namespace": str, "metadata": {}}`
  - Returns: `{"id": str, "status": "upserted"}`
  - Normalize vector, add to FAISS, store metadata in SQLite

- `POST /vectors/batch` — upsert multiple vectors
  - Body: `{"vectors": [{"id": str, "vector": [float], "namespace": str, "metadata": {}}]}`
  - Returns: `{"status": "ok", "count": int}`

- `GET /vectors/{vector_id}` — get vector metadata (NOT the vector itself for performance)
  - Returns: `{"id": str, "namespace": str, "metadata": {}}`

- `DELETE /vectors/{vector_id}` — remove vector
  - Removes from FAISS index AND metadata store
  - Returns: `{"status": "deleted"}`

### 5. api/search.py
FastAPI router with:
- `POST /search` — search by vector
  - Body: `{"vector": [float], "top_k": int (default 10), "namespace": str (optional)}`
  - Returns: `{"results": [{"id": str, "score": float, "metadata": {}}]}`
  - If namespace provided: OVER-FETCH from FAISS (fetch top_k * 10, or at least 100), then filter results by namespace in SQLite. This guarantees top_k accurate results after filtering.
  - Acquire FAISS lock during search

- `POST /search/by-id` — search using a stored vector
  - Body: `{"id": str, "top_k": int (default 10)}`
  - Fetches the stored vector, runs search, returns results

- `POST /search/by-text` — search using natural language (requires sentence-transformers)
  - Body: `{"text": str, "top_k": int (default 10)}`
  - Embeds text using all-MiniLM-L6-v2, then searches
  - Returns: `{"text": str, "results": [...]}`

### 6. api/management.py
FastAPI router with:
- `GET /stats` — return system stats
  - `{"total_vectors": int, "dimension": int, "index_type": str, "namespaces": [str]}`
- `POST /rebuild` — rebuild IVF index (trains centroids on current data)
  - Only works for IVF indexes
  - Returns: `{"status": "rebuilding", "total_vectors": int}`
- `POST /save` — force save state to disk
- `GET /health` — health check

### 7. models/schemas.py
Pydantic models:
- VectorUpsert (id, vector, namespace="default", metadata={})
- VectorUpsertBatch (vectors: list[VectorUpsert])
- SearchRequest (vector, top_k=10, namespace=None)
- SearchByIdRequest (id, top_k=10)
- SearchByTextRequest (text, top_k=10)
- SearchResult (id, score, metadata)
- SearchResponse (results: list[SearchResult])
- StatsResponse (total_vectors, dimension, index_type, namespaces)

### 8. main.py
- Create FastAPI app with title="VectorDB", version="0.1.0"
- On startup: init FAISSManager, MetadataStore, load state
- On shutdown: save state
- Include all routers
- Add CORS middleware (allow all origins)
- Add exception handlers
- Create directories for data if they don't exist

### 9. config.py
```python
DIMENSION = 384
INDEX_TYPE = "Flat"
AUTO_SAVE_INTERVAL = 100  # save every N operations
DATA_DIR = "data"
INDEX_PATH = "data/index.faiss"
DB_PATH = "data/metadata.db"
```

### 10. seed_data.py
Script that inserts 100 random vectors with fake metadata (title, description, tags) to demonstrate functionality.

### 11. requirements.txt
```
fastapi
uvicorn
faiss-cpu
numpy
pydantic
sentence-transformers
```

## Verification
After building, verify:
1. Server starts without errors
2. Can upsert a vector and get 200 response
3. Can search and get results back
4. Can upsert batch of 10 vectors
5. Can delete a vector
6. Search returns correct items
7. Server restart preserves data (persistence works)
8. /stats returns correct counts
