"""
Quantization support — A02/A03/A04.

Three strategies:
  float32  — original, maximum recall
  int8     — 4x smaller, 3.66x faster (scalar quantization)
  binary   — 32x smaller, 24.76x faster (1-bit), needs float32 rescore

Usage:
  q = Quantizer("int8")
  stored = q.encode(vectors_f32)   # compress for storage
  query_enc = q.encode(query)
  candidates = search(query_enc, stored, top_k * RESCORE_FACTOR)
  final = q.rescore(query, candidates_f32, top_k)
"""
from __future__ import annotations
import numpy as np
from typing import Literal, Tuple

QuantMode = Literal["float32", "int8", "binary"]


class Quantizer:
    def __init__(self, mode: QuantMode = "float32", dimension: int = 384):
        self.mode = mode
        self.dimension = dimension
        # Learned during fit()
        self._scale: np.ndarray | None = None
        self._zero:  np.ndarray | None = None
        self._fitted = (mode == "float32")  # float32 never needs fitting

    # ── Fit on a sample of vectors ────────────────────────────────────────────

    def fit(self, vectors: np.ndarray) -> "Quantizer":
        """Learn quantization parameters from a representative sample."""
        if self.mode == "float32":
            return self
        if self.mode == "int8":
            # Per-dimension min/max → scale to [-127, 127]
            vmin = vectors.min(axis=0)
            vmax = vectors.max(axis=0)
            self._scale = np.where(vmax - vmin > 1e-9, (vmax - vmin) / 254.0, 1.0)
            self._zero  = vmin
        elif self.mode == "binary":
            pass  # sign(v) — no learned params needed
        self._fitted = True
        return self

    # ── Encode: float32 → compressed ─────────────────────────────────────────

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        if self.mode == "float32":
            return vectors.astype(np.float32)
        if self.mode == "int8":
            if self._scale is None:
                self.fit(vectors)
            scaled = (vectors - self._zero) / self._scale
            return np.clip(np.round(scaled), -127, 127).astype(np.int8)
        if self.mode == "binary":
            # Pack 8 floats into 1 byte via sign bit
            flat = (vectors > 0).astype(np.uint8)
            n, d = flat.shape
            pad = (8 - d % 8) % 8
            if pad:
                flat = np.pad(flat, ((0,0),(0,pad)))
            return np.packbits(flat.reshape(n, -1, 8), axis=2).reshape(n, -1)
        raise ValueError(f"Unknown mode: {self.mode}")

    # ── Decode: compressed → float32 ─────────────────────────────────────────

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        if self.mode == "float32":
            return encoded
        if self.mode == "int8":
            return encoded.astype(np.float32) * self._scale + self._zero
        if self.mode == "binary":
            unpacked = np.unpackbits(encoded, axis=1)[:, :self.dimension]
            return unpacked.astype(np.float32) * 2 - 1  # {0,1} → {-1,+1}
        raise ValueError(f"Unknown mode: {self.mode}")

    # ── Approximate distance in compressed space ──────────────────────────────

    def inner_product(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Approximate cosine scores in compressed space."""
        if self.mode == "float32":
            return vectors @ query
        if self.mode == "int8":
            return vectors.astype(np.float32) @ query.astype(np.float32)
        if self.mode == "binary":
            # Hamming-based approximation: fewer differing bits → higher similarity
            q_bits = self.encode(query.reshape(1, -1))
            xor = np.bitwise_xor(vectors, q_bits)
            hamming = np.unpackbits(xor, axis=1).sum(axis=1)
            return 1.0 - hamming / self.dimension  # ~cosine
        raise ValueError(f"Unknown mode: {self.mode}")

    # ── Two-stage: binary fast search then float32 rescore ───────────────────

    @staticmethod
    def rescore(
        query_f32: np.ndarray,
        candidates_f32: np.ndarray,
        candidate_ids: list,
        top_k: int,
    ) -> Tuple[list, np.ndarray]:
        """Re-rank float32 candidates by exact cosine. Returns (ids, scores)."""
        scores = candidates_f32 @ query_f32
        order = np.argsort(-scores)[:top_k]
        return [candidate_ids[i] for i in order], scores[order]

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "dimension": self.dimension,
            "scale": self._scale.tolist() if self._scale is not None else None,
            "zero": self._zero.tolist()   if self._zero  is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Quantizer":
        q = cls(d["mode"], d["dimension"])
        if d.get("scale"):
            q._scale = np.array(d["scale"], dtype=np.float32)
            q._zero  = np.array(d["zero"],  dtype=np.float32)
            q._fitted = True
        return q
