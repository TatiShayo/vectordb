# SPEC 001: High-Speed Vector Index & Constant-Time Security Engine

## Problem Statement
AI applications need low-latency semantic search that resists auth side-channel attacks and survives abrupt node restarts.

## Solution
A lightweight vector database featuring SIMD dot product indexing, constant-time API token evaluation, and atomic persistence.

## User Stories
1. As an AI engineer, I want to query nearest neighbors in under 5ms, so that my search latency is imperceptible.
2. As a security architect, I want API authentication protected against timing attacks, so that credentials cannot be enumerated.

## Implementation Decisions
- Constant-time auth in `vectordb/api/auth.py`.
- Atomic persistence in `vectordb/storage/index.py`.

## Testing Decisions
- Seam: `vectordb/tests/test_versioned_persistence_and_auth_timing.py`.
- Verify constant-time comparison digest logic and index recovery across abrupt process kills.
