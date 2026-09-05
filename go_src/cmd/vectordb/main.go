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

type SearchResponse struct {
	Results []index.SearchResult `json:"results"`
}

func (s *Server) handleAdd(w http.ResponseWriter, r *http.Request) {
	var req AddRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if len(req.Vector) == 0 {
		http.Error(w, "vector required", http.StatusBadRequest)
		return
	}
	id := s.idx.Add(index.Vector(req.Vector))
	if req.Metadata != nil {
		s.store.Set(id, req.Metadata)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(AddResponse{ID: id})
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	if len(req.Query) == 0 {
		http.Error(w, "query required", http.StatusBadRequest)
		return
	}
	if req.TopK <= 0 { req.TopK = 10 }
	results := s.idx.Search(index.Vector(req.Query), req.TopK)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(SearchResponse{Results: results})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"status":"ok","vectors":%d}`, s.idx.Len())
}

func main() {
	srv := &Server{
		idx:   index.NewFlatIndex(),
		store: store.NewMemoryStore(),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /vectors", srv.handleAdd)
	mux.HandleFunc("POST /search", srv.handleSearch)
	mux.HandleFunc("GET /health", srv.handleHealth)

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
