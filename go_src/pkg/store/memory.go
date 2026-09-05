// Package store provides key-value metadata storage for vector IDs.
package store

import "sync"

// MemoryStore is a thread-safe in-memory metadata store keyed by vector ID.
type MemoryStore struct {
	mu   sync.RWMutex
	data map[int]map[string]string
}

// NewMemoryStore creates an empty store.
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{data: make(map[int]map[string]string)}
}

// Set associates metadata with a vector ID.
func (s *MemoryStore) Set(id int, meta map[string]string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	copy := make(map[string]string, len(meta))
	for k, v := range meta { copy[k] = v }
	s.data[id] = copy
}

// Get retrieves metadata for a vector ID.
func (s *MemoryStore) Get(id int) (map[string]string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	m, ok := s.data[id]
	return m, ok
}
