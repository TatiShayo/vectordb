# CONTEXT.md — Ubiquitous Domain Language (VectorDB)

## Core Entities
- **EmbeddingVector**: Fixed-dimension floating-point array representing high-dimensional semantic space.
- **MemoryMappedIndex**: Disk-backed vector storage file accessed via OS virtual memory without userspace buffer copying.
- **SimilarityMetric**: Distance function (`COSINE`, `EUCLIDEAN`, `DOT_PRODUCT`) evaluating vector proximity.
- **ConstantTimeAuth**: Security guard utilizing `hmac.compare_digest` to prevent timing analysis.

## Domain Invariants
- All vector insertions within a collection must match the collection's declared dimensionality (e.g. 1536).
- Index writes must complete atomically to prevent partial-write corruption.

## Forbidden Terminology
- Do not call vectors "arrays" or "lists" in storage layers; use "EmbeddingVector".
- Do not use non-constant-time equality operators (`==`) on security credentials.
