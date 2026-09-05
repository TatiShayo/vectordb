// Package index provides goroutine-parallel flat vector similarity search.
package index

import (
	"errors"
	"math"
	"runtime"
	"sort"
	"sync"
)

var (
	// ErrDimensionMismatch is returned when a vector dimension does not match the index.
	ErrDimensionMismatch = errors.New("vector dimension mismatch")
	// ErrEmptyVector is returned when an empty vector is provided.
	ErrEmptyVector = errors.New("vector cannot be empty")
)

// Vector is a slice of float32 values representing an embedding.
type Vector []float32

// SearchResult holds a matched vector ID and its similarity score.
type SearchResult struct {
	ID    int
	Score float32
}

// FlatIndex stores all vectors in memory and supports concurrent search.
type FlatIndex struct {
	mu      sync.RWMutex
	vectors []Vector
	dim     int
}

// NewFlatIndex creates an empty index.
func NewFlatIndex() *FlatIndex {
	return &FlatIndex{}
}

// Dim returns the embedding dimension expected by the index (0 if empty).
func (idx *FlatIndex) Dim() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.dim
}

// Add appends a vector to the index and returns its assigned ID.
// If the vector dimension does not match existing vectors, an error is returned.
func (idx *FlatIndex) Add(v Vector) (int, error) {
	if len(v) == 0 {
		return -1, ErrEmptyVector
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	if len(idx.vectors) == 0 {
		idx.dim = len(v)
	} else if len(v) != idx.dim {
		return -1, ErrDimensionMismatch
	}

	cp := make(Vector, len(v))
	copy(cp, v)
	idx.vectors = append(idx.vectors, cp)
	return len(idx.vectors) - 1, nil
}

// Len returns the number of indexed vectors.
func (idx *FlatIndex) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

// Search returns the topK most similar vectors to query using goroutine fan-out.
// If query dimension does not match the index, nil is returned.
func (idx *FlatIndex) Search(query Vector, topK int) []SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	n := len(idx.vectors)
	if n == 0 || len(query) == 0 {
		return nil
	}
	if idx.dim > 0 && len(query) != idx.dim {
		return nil
	}
	if topK <= 0 {
		topK = 10
	}
	if topK > n {
		topK = n
	}

	nWorkers := runtime.NumCPU()
	if nWorkers > n {
		nWorkers = n
	}
	chunkSize := (n + nWorkers - 1) / nWorkers

	type workerResult struct {
		results []SearchResult
	}
	workerResults := make([]workerResult, nWorkers)
	var wg sync.WaitGroup

	for w := 0; w < nWorkers; w++ {
		wStart := w * chunkSize
		wEnd := wStart + chunkSize
		if wEnd > n {
			wEnd = n
		}
		wg.Add(1)
		go func(workerIdx, s, e int) {
			defer wg.Done()
			if e <= s {
				return
			}
			local := make([]SearchResult, 0, e-s)
			for i := s; i < e; i++ {
				score := cosineSimilarity(query, idx.vectors[i])
				local = append(local, SearchResult{ID: i, Score: score})
			}
			workerResults[workerIdx] = workerResult{results: local}
		}(w, wStart, wEnd)
	}
	wg.Wait()

	// Merge all worker results
	all := make([]SearchResult, 0, n)
	for _, wr := range workerResults {
		all = append(all, wr.results...)
	}
	sort.Slice(all, func(i, j int) bool { return all[i].Score > all[j].Score })
	if len(all) > topK {
		all = all[:topK]
	}
	return all
}

// cosineSimilarity computes cosine similarity between two vectors.
// It guards against slice boundary overflows and division by zero.
func cosineSimilarity(a, b Vector) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		av, bv := float64(a[i]), float64(b[i])
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	denom := math.Sqrt(na) * math.Sqrt(nb)
	if denom < 1e-9 {
		return 0
	}
	return float32(dot / denom)
}
