"""
Pure Python/NumPy Hierarchical Navigable Small World (HNSW) Graph Index.

Features:
- Multi-layer skip-list graph hierarchy with exponentially distributed layers.
- Full distance metric support: Cosine, Euclidean/L2, Dot Product, Manhattan.
- Concurrency safety with threading.RLock.
- Cycle prevention via visited sets during greedy and beam search.
- Heuristic diverse neighbor selection (pruning) to maintain graph connectivity.
- Soft-deletion and rebuild support.
"""
from __future__ import annotations
import heapq
import logging
import math
import random
import threading
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from utils.normalize import compute_distance, normalize, prepare_vector

logger = logging.getLogger(__name__)


class HNSWIndex:
    """
    Thread-safe Hierarchical Navigable Small World (HNSW) vector index.
    """

    def __init__(
        self,
        dimension: int = 384,
        distance: str = "cosine",
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 64,
        m0: Optional[int] = None,
        seed: Optional[int] = 42,
    ):
        self.dimension = dimension
        self.distance = distance.lower()
        self.m = m
        self.m0 = m0 or (2 * m)
        self.ef_construction = max(ef_construction, m)
        self.ef_search = ef_search
        self.mL = 1.0 / math.log(m)

        self._lock = threading.RLock()
        self._rng = random.Random(seed)

        # Storage
        self.nodes: Dict[int, np.ndarray] = {}
        # layer -> {node_id: list of neighbor_ids}
        self.layers: List[Dict[int, List[int]]] = []
        self.node_levels: Dict[int, int] = {}
        self.entry_point: Optional[int] = None
        self.max_level: int = -1
        self._deleted: Set[int] = set()

    def _get_random_level(self) -> int:
        """Draw level from exponential distribution: floor(-ln(uniform) * mL)."""
        r = self._rng.random()
        while r == 0:
            r = self._rng.random()
        return int(-math.log(r) * self.mL)

    def _dist(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if self.distance == "cosine":
            # Both vectors are already L2 normalized
            dot = float(np.dot(v1, v2))
            return max(0.0, 1.0 - dot)
        elif self.distance in ("euclidean", "l2"):
            diff = v1 - v2
            return float(np.sqrt(np.dot(diff, diff)))
        elif self.distance == "dot":
            return -float(np.dot(v1, v2))
        elif self.distance == "manhattan":
            return float(np.sum(np.abs(v1 - v2)))
        return compute_distance(v1, v2, self.distance)

    def add(self, node_id: int, vector: Union[Sequence[float], np.ndarray]) -> None:
        """Insert a single vector into the HNSW graph."""
        vec = prepare_vector(vector, normalize_vec=(self.distance == "cosine"))
        if len(vec) != self.dimension:
            raise ValueError(f"Vector dimension {len(vec)} != {self.dimension}")

        with self._lock:
            if node_id in self.nodes:
                old_lvl = self.node_levels.get(node_id, 0)
                for lvl in range(min(len(self.layers), old_lvl + 1)):
                    if node_id in self.layers[lvl]:
                        for neighbor in list(self.layers[lvl][node_id]):
                            if neighbor in self.layers[lvl] and node_id in self.layers[lvl][neighbor]:
                                self.layers[lvl][neighbor].remove(node_id)
                        del self.layers[lvl][node_id]
                self._deleted.discard(node_id)

            self.nodes[node_id] = vec
            node_level = self._get_random_level()
            self.node_levels[node_id] = node_level

            # Ensure layers list has enough levels
            while len(self.layers) <= node_level:
                self.layers.append({})

            # Initialize node in all its layers 0..node_level
            for lvl in range(node_level + 1):
                self.layers[lvl][node_id] = []

            if self.entry_point is None:
                self.entry_point = node_id
                self.max_level = node_level
                return

            curr_ep = self.entry_point
            curr_dist = self._dist(vec, self.nodes[curr_ep])

            # 1. Greedy search down from max_level to node_level + 1 (with cycle prevention)
            for lvl in range(self.max_level, node_level, -1):
                changed = True
                visited_greedy: Set[int] = {curr_ep}
                while changed:
                    changed = False
                    neighbors = self.layers[lvl].get(curr_ep, [])
                    for neighbor in neighbors:
                        if neighbor in visited_greedy or neighbor in self._deleted or neighbor not in self.nodes:
                            continue
                        visited_greedy.add(neighbor)
                        d = self._dist(vec, self.nodes[neighbor])
                        if d < curr_dist:
                            curr_dist = d
                            curr_ep = neighbor
                            changed = True

            # 2. Connect from min(max_level, node_level) down to 0
            ep_set = {curr_ep}
            for lvl in range(min(self.max_level, node_level), -1, -1):
                valid_eps = {ep for ep in ep_set if self.node_levels.get(ep, -1) >= lvl and ep not in self._deleted}
                if not valid_eps:
                    valid_eps = {curr_ep} if self.node_levels.get(curr_ep, -1) >= lvl else set(self.layers[lvl].keys()) - {node_id}
                if not valid_eps:
                    valid_eps = {node_id}

                # Search level for ef candidates
                candidates = self._search_layer(vec, valid_eps, self.ef_construction, lvl)
                # Remove self from candidates
                candidates = [(d, c_id) for d, c_id in candidates if c_id != node_id]

                # Select M (or M0) neighbors
                max_conn = self.m0 if lvl == 0 else self.m
                selected_neighbors = self._select_heuristic(vec, candidates, max_conn)

                # Add bidirectional links
                for neighbor in selected_neighbors:
                    if neighbor not in self.layers[lvl]:
                        continue
                    self.layers[lvl][node_id].append(neighbor)
                    self.layers[lvl][neighbor].append(node_id)
                    # Prune neighbor if over max_conn
                    if len(self.layers[lvl][neighbor]) > max_conn:
                        n_cands = [
                            (self._dist(self.nodes[neighbor], self.nodes[n_id]), n_id)
                            for n_id in self.layers[lvl][neighbor]
                            if n_id in self.nodes and n_id in self.layers[lvl]
                        ]
                        n_selected = self._select_heuristic(self.nodes[neighbor], n_cands, max_conn)
                        self.layers[lvl][neighbor] = n_selected

                ep_set = set(selected_neighbors) if selected_neighbors else valid_eps

            if node_level > self.max_level:
                self.max_level = node_level
                self.entry_point = node_id

    def _search_layer(
        self, query: np.ndarray, ep_set: Set[int], ef: int, level: int
    ) -> List[Tuple[float, int]]:
        """
        Beam search at a single layer.
        Returns list of (distance, node_id) candidates.
        """
        visited: Set[int] = set(ep_set)
        # Min-heap of candidates to explore: (dist, node_id)
        candidates: List[Tuple[float, int]] = []
        # Max-heap of nearest found so far: (-dist, node_id)
        w: List[Tuple[float, int]] = []

        for ep in ep_set:
            if ep not in self.nodes or ep in self._deleted:
                continue
            d = self._dist(query, self.nodes[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(w, (-d, ep))

        if not w:
            return []

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_dist = -w[0][0]

            if c_dist > furthest_dist:
                break

            neighbors = self.layers[level].get(c_id, [])
            for neighbor in neighbors:
                if neighbor in visited or neighbor not in self.nodes or neighbor in self._deleted:
                    continue
                visited.add(neighbor)

                n_dist = self._dist(query, self.nodes[neighbor])
                furthest_dist = -w[0][0]

                if n_dist < furthest_dist or len(w) < ef:
                    heapq.heappush(candidates, (n_dist, neighbor))
                    heapq.heappush(w, (-n_dist, neighbor))
                    if len(w) > ef:
                        heapq.heappop(w)

        return sorted([(-d, n_id) for d, n_id in w])

    def _select_heuristic(
        self, base_vec: np.ndarray, candidates: List[Tuple[float, int]], max_conn: int
    ) -> List[int]:
        """
        Selects up to max_conn diverse neighbors to prevent graph clustering/disconnection.
        """
        if len(candidates) <= max_conn:
            return [n_id for _, n_id in candidates]

        sorted_cands = sorted(candidates, key=lambda x: x[0])
        result: List[int] = []

        for d_bc, c_id in sorted_cands:
            if len(result) >= max_conn:
                break
            if c_id not in self.nodes or c_id in self._deleted:
                continue
            c_vec = self.nodes[c_id]
            is_good = True
            for r_id in result:
                d_cr = self._dist(c_vec, self.nodes[r_id])
                if d_cr < d_bc:
                    is_good = False
                    break
            if is_good or not result:
                result.append(c_id)

        if len(result) < max_conn:
            for _, c_id in sorted_cands:
                if c_id not in result and c_id in self.nodes and c_id not in self._deleted:
                    result.append(c_id)
                if len(result) >= max_conn:
                    break

        return result

    def search(
        self, query: Union[Sequence[float], np.ndarray], top_k: int = 10, ef_search: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Search for top_k nearest neighbors.
        Returns list of (node_id, score/distance) sorted by proximity.
        """
        q = prepare_vector(query, normalize_vec=(self.distance == "cosine"))
        ef = max(ef_search or self.ef_search, top_k)

        with self._lock:
            if self.entry_point is None or not self.nodes:
                return []

            curr_ep = self.entry_point
            if curr_ep in self._deleted or curr_ep not in self.nodes:
                active_eps = [nid for nid in self.nodes if nid not in self._deleted]
                if not active_eps:
                    return []
                curr_ep = active_eps[0]

            curr_dist = self._dist(q, self.nodes[curr_ep])

            # 1. Greedy search from max_level down to 1 (with cycle prevention)
            for lvl in range(self.max_level, 0, -1):
                changed = True
                visited_greedy: Set[int] = {curr_ep}
                while changed:
                    changed = False
                    neighbors = self.layers[lvl].get(curr_ep, [])
                    for neighbor in neighbors:
                        if neighbor in visited_greedy or neighbor in self._deleted or neighbor not in self.nodes:
                            continue
                        visited_greedy.add(neighbor)
                        d = self._dist(q, self.nodes[neighbor])
                        if d < curr_dist:
                            curr_dist = d
                            curr_ep = neighbor
                            changed = True

            # 2. Beam search at level 0
            candidates = self._search_layer(q, {curr_ep}, ef, 0)
            valid = [(n_id, d) for d, n_id in candidates if n_id not in self._deleted and n_id in self.nodes]
            valid.sort(key=lambda x: x[1])

            if self.distance == "cosine":
                return [(n_id, 1.0 - d) for n_id, d in valid[:top_k]]
            return [(n_id, d) for n_id, d in valid[:top_k]]

    def remove(self, node_id: int) -> bool:
        """Soft-delete a node from the index."""
        with self._lock:
            if node_id in self.nodes and node_id not in self._deleted:
                self._deleted.add(node_id)
                return True
            return False

    def size(self) -> int:
        with self._lock:
            return len(self.nodes) - len(self._deleted)

    def rebuild(self) -> None:
        """Rebuild graph purging soft-deleted nodes."""
        with self._lock:
            live_nodes = [(n_id, vec) for n_id, vec in self.nodes.items() if n_id not in self._deleted]
            self.nodes.clear()
            self.layers.clear()
            self.node_levels.clear()
            self._deleted.clear()
            self.entry_point = None
            self.max_level = -1
            for n_id, vec in live_nodes:
                self.add(n_id, vec)
