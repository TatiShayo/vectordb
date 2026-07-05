You are a research agent. Study vector database architecture for a project called "vectordb" — a lightweight vector database service.

## Research targets

Study these systems in detail:

1. **Pinecone** — architecture, API design, how they handle indexing at scale
2. **ChromaDB** — open-source, how it works under the hood, storage design
3. **FAISS (Meta)** — all index types (Flat, IVF, HNSW, PQ), tradeoffs between speed and accuracy
4. **Qdrant** — Rust-based, HNSW indexing, filtering strategy
5. **Milvus** — distributed vector database architecture

For each, answer:
- Storage architecture (where do vectors physically live?)
- Index types supported and the speed-vs-accuracy tradeoffs
- Filtering strategy (metadata pre-filter vs post-filter vs hybrid)
- How they handle delete/update of vectors
- Scaling model (single node vs distributed)

## Practical FAISS research

You need to understand FAISS deeply:

- What is IndexFlatIP vs IndexIVFFlat vs IndexHNSWFlat?
- How does the IDMap wrapper work for supporting deletions?
- How to save/load FAISS indices to disk?
- Memory requirements per million vectors at different dimensions (384, 768, 1536)
- When IVF outperforms Flat (the dataset size threshold)
- What faiss-cpu can do vs faiss-gpu

## Deliverable

Save a comprehensive research report to:
C:\Users\TATI\Desktop\Clients\vectordb\research_phase0.md

Cover: all 5 systems compared, FAISS deep-dive with benchmarks, architectural decisions for our vectordb project, and a proposed API design.

## Constraints
- Research only, no code
- Be specific with numbers (dimensions, latency, recall rates)
- Include a "decisions" section recommending what approach to take for a lightweight single-node vector DB
