# ADR 0001: Constant-Time Token Auth and Atomic Index Persistence

## Context
High-performance vector databases exposed to public APIs face token enumeration timing attacks and disk corruption during crash events.

## Decision
1. **Constant-Time Verification**: Enforce `hmac.compare_digest` for all authentication tokens.
2. **Atomic Index Snapping**: Write memory-mapped indexes to `.tmp` files before atomic filesystem rename.
3. **L2-Normalized SIMD Dot Products**: Accelerate cosine similarity searches via hardware SIMD dot products.

## Consequences
- **Positive**: Complete immunity against auth timing attacks and rock-solid crash durability.
- **Negative**: Requires disk space overhead for dual-file temporary index writing.
