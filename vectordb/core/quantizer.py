"""
Quantization engine — Scalar Quantization & Product Quantization.

Strategies:
  - float32: Original uncompressed vectors (lossless).
  - int8: Symmetric scalar quantization to [-127, 127] (4x compression, <0.01 MSE error).
  - uint8: Affine min-max scalar quantization to [0, 255] (4x compression).
  - binary: 1-bit sign quantization with bitpacking (32x compression).
  - Product Quantizer (PQ): Subspace vector quantization with learned codebooks.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

QuantMode = Literal["float32", "int8", "uint8", "binary"]


class Quantizer:
    """Scalar Quantizer supporting float32, int8, uint8, and binary modes."""

    def __init__(self, mode: QuantMode = "float32", dimension: int = 384):
        self.mode = mode
        self.dimension = dimension
        self._scale: Optional[np.ndarray] = None
        self._zero: Optional[np.ndarray] = None
        self._fitted = (mode == "float32")

    def fit(self, vectors: np.ndarray) -> "Quantizer":
        """Learn quantization parameters from a representative vector batch."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self.dimension = vectors.shape[1]

        if self.mode == "float32":
            pass
        elif self.mode == "int8":
            # Symmetric INT8: scale to [-127, 127]
            max_abs = np.maximum(np.abs(vectors).max(axis=0), 1e-8)
            self._scale = (max_abs / 127.0).astype(np.float32)
            self._zero = np.zeros_like(self._scale)
        elif self.mode == "uint8":
            # Affine UINT8: scale to [0, 255]
            vmin = vectors.min(axis=0)
            vmax = vectors.max(axis=0)
            diff = np.maximum(vmax - vmin, 1e-8)
            self._scale = (diff / 255.0).astype(np.float32)
            self._zero = vmin.astype(np.float32)
        elif self.mode == "binary":
            pass
        else:
            raise ValueError(f"Unknown quantization mode: {self.mode}")

        self._fitted = True
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Compress float32 vectors into quantized representations."""
        vectors = np.asarray(vectors, dtype=np.float32)
        is_1d = (vectors.ndim == 1)
        if is_1d:
            vectors = vectors.reshape(1, -1)

        if not self._fitted:
            self.fit(vectors)

        if self.mode == "float32":
            res = vectors
        elif self.mode == "int8":
            scaled = vectors / self._scale
            res = np.clip(np.round(scaled), -127, 127).astype(np.int8)
        elif self.mode == "uint8":
            scaled = (vectors - self._zero) / self._scale
            res = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
        elif self.mode == "binary":
            flat = (vectors > 0).astype(np.uint8)
            n, d = flat.shape
            pad = (8 - d % 8) % 8
            if pad:
                flat = np.pad(flat, ((0, 0), (0, pad)))
            res = np.packbits(flat.reshape(n, -1, 8), axis=2).reshape(n, -1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return res[0] if is_1d and self.mode != "binary" else res

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decompress quantized representations back to float32 vectors."""
        encoded = np.asarray(encoded)
        is_1d = (encoded.ndim == 1 and self.mode != "binary")
        if is_1d:
            encoded = encoded.reshape(1, -1)

        if self.mode == "float32":
            res = encoded.astype(np.float32)
        elif self.mode == "int8":
            if self._scale is None:
                scale = np.ones(self.dimension, dtype=np.float32) / 127.0
            else:
                scale = self._scale
            res = encoded.astype(np.float32) * scale
        elif self.mode == "uint8":
            if self._scale is None:
                scale = np.ones(self.dimension, dtype=np.float32) / 255.0
                zero = np.zeros(self.dimension, dtype=np.float32)
            else:
                scale, zero = self._scale, self._zero
            res = encoded.astype(np.float32) * scale + zero
        elif self.mode == "binary":
            if encoded.ndim == 1:
                encoded = encoded.reshape(1, -1)
            unpacked = np.unpackbits(encoded, axis=1)[:, :self.dimension]
            res = unpacked.astype(np.float32) * 2.0 - 1.0
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return res[0] if is_1d else res

    def inner_product(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Approximate dot product / cosine similarity in compressed space."""
        query = np.asarray(query, dtype=np.float32).ravel()
        if self.mode == "float32":
            return np.dot(vectors, query)
        elif self.mode == "int8":
            q_enc = self.encode(query.reshape(1, -1))[0]
            # Dequantized approximate dot product
            return np.dot(vectors.astype(np.float32), (q_enc.astype(np.float32) * (self._scale ** 2)))
        elif self.mode == "uint8":
            dec_v = self.decode(vectors)
            return np.dot(dec_v, query)
        elif self.mode == "binary":
            q_bits = self.encode(query.reshape(1, -1))
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            xor = np.bitwise_xor(vectors, q_bits)
            hamming = np.unpackbits(xor, axis=1).sum(axis=1)
            return 1.0 - (2.0 * hamming / self.dimension)
        raise ValueError(f"Unknown mode: {self.mode}")

    def reconstruction_error(self, vectors: np.ndarray) -> Dict[str, float]:
        """Measure reconstruction fidelity: MSE and mean cosine similarity."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        encoded = self.encode(vectors)
        decoded = self.decode(encoded)
        mse = float(np.mean((vectors - decoded) ** 2))
        max_err = float(np.max(np.abs(vectors - decoded)))
        # Cosine similarity between original and reconstructed
        norms_orig = np.linalg.norm(vectors, axis=1)
        norms_dec = np.linalg.norm(decoded, axis=1)
        valid = (norms_orig > 1e-9) & (norms_dec > 1e-9)
        if np.any(valid):
            dots = np.sum(vectors[valid] * decoded[valid], axis=1)
            cos_sim = float(np.mean(dots / (norms_orig[valid] * norms_dec[valid])))
        else:
            cos_sim = 1.0
        return {"mse": mse, "max_error": max_err, "mean_cosine_similarity": cos_sim}

    @staticmethod
    def rescore(
        query_f32: np.ndarray,
        candidates_f32: np.ndarray,
        candidate_ids: List[Any],
        top_k: int,
    ) -> Tuple[List[Any], np.ndarray]:
        """Re-rank candidates using exact float32 inner products."""
        scores = np.dot(candidates_f32, query_f32)
        order = np.argsort(-scores)[:top_k]
        return [candidate_ids[i] for i in order], scores[order]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "dimension": self.dimension,
            "scale": self._scale.tolist() if self._scale is not None else None,
            "zero": self._zero.tolist() if self._zero is not None else None,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Quantizer":
        q = cls(d["mode"], d["dimension"])
        if d.get("scale") is not None:
            q._scale = np.array(d["scale"], dtype=np.float32)
            q._zero = np.array(d["zero"], dtype=np.float32)
            q._fitted = True
        return q


class ProductQuantizer:
    """
    Product Quantizer (PQ).
    Decomposes a D-dimensional vector into M subvectors of dimension D/M.
    Learns K = 2^nbits centroids per subspace using K-means.
    """

    def __init__(self, dimension: int = 128, m: int = 8, nbits: int = 8):
        if dimension % m != 0:
            raise ValueError(f"Dimension {dimension} must be divisible by m={m}")
        if nbits not in (4, 8):
            raise ValueError(f"nbits must be 4 or 8 (got {nbits})")
        self.dimension = dimension
        self.m = m
        self.nbits = nbits
        self.k = 2 ** nbits  # 256 for 8-bit, 16 for 4-bit
        self.d_sub = dimension // m
        # Shape: (m, k, d_sub)
        self.codebook: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, vectors: np.ndarray, n_iter: int = 15) -> "ProductQuantizer":
        """Learn codebooks for each subspace using k-means."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n_samples, dim = vectors.shape
        if dim != self.dimension:
            raise ValueError(f"Input dimension {dim} != expected {self.dimension}")

        codebooks = np.zeros((self.m, self.k, self.d_sub), dtype=np.float32)

        for sub_idx in range(self.m):
            start = sub_idx * self.d_sub
            end = start + self.d_sub
            sub_vecs = vectors[:, start:end]

            # K-means initialization
            if n_samples >= self.k:
                indices = np.random.choice(n_samples, size=self.k, replace=False)
                centroids = sub_vecs[indices].copy()
            else:
                # Fewer samples than k: repeat with small noise
                tile_count = int(np.ceil(self.k / n_samples))
                tiled = np.tile(sub_vecs, (tile_count, 1))[:self.k]
                noise = np.random.randn(*tiled.shape).astype(np.float32) * 1e-4
                centroids = tiled + noise

            # Lloyd's algorithm iterations
            for _ in range(n_iter):
                # (N, K) distance matrix
                dists = np.sum((sub_vecs[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
                labels = np.argmin(dists, axis=1)

                new_centroids = np.zeros_like(centroids)
                counts = np.bincount(labels, minlength=self.k)

                for cid in range(self.k):
                    if counts[cid] > 0:
                        new_centroids[cid] = np.mean(sub_vecs[labels == cid], axis=0)
                    else:
                        # Re-seed empty cluster
                        new_centroids[cid] = sub_vecs[np.random.randint(n_samples)]

                centroids = new_centroids

            codebooks[sub_idx] = centroids

        self.codebook = codebooks
        self._fitted = True
        return self

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors into PQ codes.
        Returns uint8 matrix of shape (N, m).
        """
        if not self._fitted or self.codebook is None:
            raise RuntimeError("ProductQuantizer must be fitted before encoding")

        vectors = np.asarray(vectors, dtype=np.float32)
        is_1d = (vectors.ndim == 1)
        if is_1d:
            vectors = vectors.reshape(1, -1)

        n = vectors.shape[0]
        codes = np.zeros((n, self.m), dtype=np.uint8)

        for sub_idx in range(self.m):
            start = sub_idx * self.d_sub
            end = start + self.d_sub
            sub_vecs = vectors[:, start:end]
            centroids = self.codebook[sub_idx]  # (k, d_sub)

            # Distances: (N, K)
            dists = np.sum((sub_vecs[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            codes[:, sub_idx] = np.argmin(dists, axis=1).astype(np.uint8)

        return codes[0] if is_1d else codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode PQ codes back to reconstructed float32 vectors (N, dimension).
        """
        if not self._fitted or self.codebook is None:
            raise RuntimeError("ProductQuantizer must be fitted before decoding")

        codes = np.asarray(codes, dtype=np.int64)
        is_1d = (codes.ndim == 1)
        if is_1d:
            codes = codes.reshape(1, -1)

        n = codes.shape[0]
        reconstructed = np.zeros((n, self.dimension), dtype=np.float32)

        for sub_idx in range(self.m):
            start = sub_idx * self.d_sub
            end = start + self.d_sub
            sub_codes = codes[:, sub_idx]
            reconstructed[:, start:end] = self.codebook[sub_idx, sub_codes]

        return reconstructed[0] if is_1d else reconstructed

    def compute_asymmetric_distances(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """
        Asymmetric Distance Computation (ADC):
        Precomputes query-to-centroid lookup table (LUT) and computes distances in O(N * m).
        """
        if not self._fitted or self.codebook is None:
            raise RuntimeError("ProductQuantizer must be fitted before distance computation")

        query = np.asarray(query, dtype=np.float32).ravel()
        codes = np.asarray(codes, dtype=np.int64)
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)

        # Build LUT: (m, k)
        lut = np.zeros((self.m, self.k), dtype=np.float32)
        for sub_idx in range(self.m):
            start = sub_idx * self.d_sub
            end = start + self.d_sub
            q_sub = query[start:end]
            centroids = self.codebook[sub_idx]  # (k, d_sub)
            lut[sub_idx] = np.sum((centroids - q_sub[np.newaxis, :]) ** 2, axis=1)

        # Sum distances across subspaces for each vector code
        n_vectors = codes.shape[0]
        distances = np.zeros(n_vectors, dtype=np.float32)
        for sub_idx in range(self.m):
            distances += lut[sub_idx, codes[:, sub_idx]]

        return distances

    def reconstruction_error(self, vectors: np.ndarray) -> Dict[str, float]:
        """Compute MSE and cosine reconstruction metrics."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        codes = self.encode(vectors)
        decoded = self.decode(codes)
        mse = float(np.mean((vectors - decoded) ** 2))
        max_err = float(np.max(np.abs(vectors - decoded)))
        norms_orig = np.linalg.norm(vectors, axis=1)
        norms_dec = np.linalg.norm(decoded, axis=1)
        valid = (norms_orig > 1e-9) & (norms_dec > 1e-9)
        if np.any(valid):
            dots = np.sum(vectors[valid] * decoded[valid], axis=1)
            cos_sim = float(np.mean(dots / (norms_orig[valid] * norms_dec[valid])))
        else:
            cos_sim = 1.0
        return {"mse": mse, "max_error": max_err, "mean_cosine_similarity": cos_sim}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "m": self.m,
            "nbits": self.nbits,
            "k": self.k,
            "d_sub": self.d_sub,
            "codebook": self.codebook.tolist() if self.codebook is not None else None,
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProductQuantizer":
        pq = cls(dimension=d["dimension"], m=d["m"], nbits=d["nbits"])
        if d.get("codebook") is not None:
            pq.codebook = np.array(d["codebook"], dtype=np.float32)
            pq._fitted = True
        return pq
