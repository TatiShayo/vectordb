# TICKETS — VectorDB Search Engine

## [TICKET-001] Constant-Time Auth Digest Guard
- **Blocked by**: None
- **Delivers**: Constant-time token verification eliminating side-channel leakage.
- **Verification**: `vectordb/tests/test_versioned_persistence_and_auth_timing.py`

## [TICKET-002] Atomic Memory-Mapped Index Snapshotter
- **Blocked by**: TICKET-001
- **Delivers**: Atomic file rename pipeline ensuring index integrity across crashes.
- **Verification**: Index file recovery unit tests.
