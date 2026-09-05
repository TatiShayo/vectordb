package index

import (
	"math"
	"testing"
)

func makeVec(vals ...float32) Vector { return Vector(vals) }

func TestEmptyIndexReturnsNil(t *testing.T) {
	idx := NewFlatIndex()
	res := idx.Search(makeVec(1, 0, 0), 5)
	if res != nil {
		t.Errorf("expected nil result, got %v", res)
	}
}

func TestSingleVectorIsTop1(t *testing.T) {
	idx := NewFlatIndex()
	idx.Add(makeVec(1, 0, 0))
	res := idx.Search(makeVec(1, 0, 0), 1)
	if len(res) != 1 || res[0].ID != 0 {
		t.Errorf("unexpected result: %v", res)
	}
}

func TestIdenticalVectorHasScore1(t *testing.T) {
	idx := NewFlatIndex()
	v := makeVec(0.6, 0.8)
	idx.Add(v)
	res := idx.Search(v, 1)
	if math.Abs(float64(res[0].Score)-1.0) > 1e-5 {
		t.Errorf("expected score ~1.0, got %f", res[0].Score)
	}
}

func TestOrthogonalVectorsScoreNearZero(t *testing.T) {
	idx := NewFlatIndex()
	idx.Add(makeVec(1, 0))
	res := idx.Search(makeVec(0, 1), 1)
	if math.Abs(float64(res[0].Score)) > 1e-5 {
		t.Errorf("expected score ~0.0, got %f", res[0].Score)
	}
}

func TestTopK1ReturnsExactlyOne(t *testing.T) {
	idx := NewFlatIndex()
	for i := 0; i < 10; i++ {
		idx.Add(makeVec(float32(i), float32(i+1)))
	}
	res := idx.Search(makeVec(5, 6), 1)
	if len(res) != 1 {
		t.Errorf("expected 1 result, got %d", len(res))
	}
}

func TestTopKLargerThanCorpusReturnsAll(t *testing.T) {
	idx := NewFlatIndex()
	for i := 0; i < 5; i++ {
		idx.Add(makeVec(float32(i), 0))
	}
	res := idx.Search(makeVec(1, 0), 100)
	if len(res) != 5 {
		t.Errorf("expected 5 results, got %d", len(res))
	}
}

func TestResultsAreSortedDescending(t *testing.T) {
	idx := NewFlatIndex()
	for i := 0; i < 20; i++ {
		idx.Add(makeVec(float32(i), float32(20-i)))
	}
	res := idx.Search(makeVec(19, 1), 10)
	for i := 1; i < len(res); i++ {
		if res[i].Score > res[i-1].Score {
			t.Errorf("results not sorted at position %d: %f > %f", i, res[i].Score, res[i-1].Score)
		}
	}
}

func TestZeroVectorHandledGracefully(t *testing.T) {
	idx := NewFlatIndex()
	idx.Add(makeVec(0, 0, 0))
	res := idx.Search(makeVec(1, 0, 0), 1)
	// Should return 0, not NaN or panic
	if math.IsNaN(float64(res[0].Score)) {
		t.Error("got NaN score for zero vector")
	}
}

func TestLargeVectorDimension(t *testing.T) {
	idx := NewFlatIndex()
	v := make(Vector, 1536)
	for i := range v { v[i] = 0.1 }
	idx.Add(v)
	res := idx.Search(v, 1)
	if len(res) != 1 {
		t.Error("expected 1 result for 1536-dim vector")
	}
	if math.Abs(float64(res[0].Score)-1.0) > 1e-4 {
		t.Errorf("expected score ~1.0, got %f", res[0].Score)
	}
}

func TestConcurrentAddAndSearchNoRace(t *testing.T) {
	idx := NewFlatIndex()
	// Pre-populate to avoid empty-index edge case
	for i := 0; i < 50; i++ {
		idx.Add(makeVec(float32(i), float32(i+1)))
	}
	done := make(chan struct{})
	// Concurrent reader
	go func() {
		for i := 0; i < 200; i++ {
			idx.Search(makeVec(1, 2), 3)
		}
		close(done)
	}()
	// Concurrent writer
	for i := 0; i < 50; i++ {
		idx.Add(makeVec(float32(i), 0))
	}
	<-done
}

func TestLenTracksAddedVectors(t *testing.T) {
	idx := NewFlatIndex()
	for i := 0; i < 42; i++ {
		idx.Add(makeVec(float32(i)))
	}
	if idx.Len() != 42 {
		t.Errorf("expected Len()=42, got %d", idx.Len())
	}
}
