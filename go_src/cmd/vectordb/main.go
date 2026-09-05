// vectordb HTTP server — goroutine-parallel vector search engine
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/tatishayo/vectordb/pkg/index"
	"github.com/tatishayo/vectordb/pkg/store"
)

type Server struct {
	idx   *index.FlatIndex
	store *store.MemoryStore
}

type AddRequest struct {
	Vector   []float32         `json:"vector"`
	Metadata map[string]string `json:"metadata"`
}

type AddResponse struct {
	ID int `json:"id"`
}

type SearchRequest struct {
	Query []float32 `json:"query"`
	TopK  int       `json:"top_k"`
}

type SearchResultItem struct {
	ID       int               `json:"id"`
	Score    float32           `json:"score"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type SearchResponse struct {
	Results []SearchResultItem `json:"results"`
}

func (s *Server) handleAdd(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req AddRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if len(req.Vector) == 0 {
		http.Error(w, "vector required", http.StatusBadRequest)
		return
	}
	id, err := s.idx.Add(index.Vector(req.Vector))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.Metadata != nil {
		s.store.Set(id, req.Metadata)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(AddResponse{ID: id})
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if len(req.Query) == 0 {
		http.Error(w, "query required", http.StatusBadRequest)
		return
	}
	if req.TopK <= 0 {
		req.TopK = 10
	}
	rawResults := s.idx.Search(index.Vector(req.Query), req.TopK)
	items := make([]SearchResultItem, len(rawResults))
	for i, r := range rawResults {
		meta, _ := s.store.Get(r.ID)
		items[i] = SearchResultItem{
			ID:       r.ID,
			Score:    r.Score,
			Metadata: meta,
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(SearchResponse{Results: items})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"status":"ok","vectors":%d,"dim":%d}`, s.idx.Len(), s.idx.Dim())
}

func main() {
	srv := &Server{
		idx:   index.NewFlatIndex(),
		store: store.NewMemoryStore(),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/vectors", srv.handleAdd)
	mux.HandleFunc("/search", srv.handleSearch)
	mux.HandleFunc("/health", srv.handleHealth)

	httpSrv := &http.Server{Addr: ":8080", Handler: mux}

	go func() {
		log.Println("vectordb listening on :8080")
		if err := httpSrv.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("server error: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("shutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	httpSrv.Shutdown(ctx)
}
