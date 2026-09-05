// Package index provides goroutine-parallel flat vector similarity search.
package index

import (
	"math"
	"runtime"
	"sort"
	"sync"
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
}

// NewFlatIndex creates an empty index.
func NewFlatIndex() *FlatIndex {
	return &FlatIndex{}
}

// Add appends a vector to the index and returns its assigned ID.
func (idx *FlatIndex) Add(v Vector) int {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	copy := make(Vector, len(v))
	copy_slice(v, copy)
	idx.vectors = append(idx.vectors, copy)
	return len(idx.vectors) - 1
}

func copy_slice(src, dst Vector) {
	for i, v := range src {
		dst[i] = v
	}
}

// Len returns the number of indexed vectors.
func (idx *FlatIndex) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.vectors)
}

// Search returns the topK most similar vectors to query using goroutine fan-out.
func (idx *FlatIndex) Search(query Vector, topK int) []SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	n := len(idx.vectors)
	if n == 0 {
		return nil
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
func cosineSimilarity(a, b Vector) float32 {
	var dot, na, nb float64
	for i := range a {
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
