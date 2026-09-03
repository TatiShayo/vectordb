"""
Vector normalization and distance metric utilities.
Supports: Cosine, Euclidean/L2, Dot Product, and Manhattan distances.
"""
from __future__ import annotations
import numpy as np
from typing import Union, Sequence


def normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a single vector or a batch (N, D) in-place-safe."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        norm = float(np.linalg.norm(v))
        return (v / norm).astype(np.float32) if norm > 1e-10 else v
    # Batch
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return (v / norms).astype(np.float32)


def to_float32(v: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Convert list or ndarray to float32 ndarray."""
    return np.asarray(v, dtype=np.float32)


def prepare_vector(v: Union[Sequence[float], np.ndarray], normalize_vec: bool = True) -> np.ndarray:
    """Convert and optionally normalize a vector."""
    arr = to_float32(v)
    return normalize(arr) if normalize_vec else arr


def prepare_batch(vectors: Union[Sequence[Sequence[float]], np.ndarray], normalize_vec: bool = True) -> np.ndarray:
    """Convert a list/batch of vectors to a float32 matrix, optionally normalized."""
    mat = np.asarray(vectors, dtype=np.float32)
    return normalize(mat) if normalize_vec else mat


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors in [-1, 1]."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(sim, -1.0, 1.0))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance in [0, 2]. Distance = 1 - similarity."""
    return 1.0 - cosine_similarity(a, b)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean (L2) distance between two vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Alias for euclidean_distance."""
    return euclidean_distance(a, b)


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Manhattan (L1) distance between two vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.sum(np.abs(a - b)))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Dot Product (Inner Product) between two vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b))


def compute_distance(a: np.ndarray, b: np.ndarray, metric: str = "cosine") -> float:
    """Compute distance/score between two vectors based on metric."""
    metric = metric.lower()
    if metric == "cosine":
        return cosine_distance(a, b)
    elif metric in ("euclidean", "l2"):
        return euclidean_distance(a, b)
    elif metric == "manhattan":
        return manhattan_distance(a, b)
    elif metric == "dot":
        return -dot_product(a, b)  # Lower is closer convention for distance
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def pairwise_distances(X: np.ndarray, Y: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compute pairwise distance matrix between X (N, D) and Y (M, D).
    Returns (N, M) matrix.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    metric = metric.lower()
    
    if metric == "cosine":
        norm_X = np.linalg.norm(X, axis=1, keepdims=True)
        norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)
        norm_X = np.where(norm_X < 1e-10, 1.0, norm_X)
        norm_Y = np.where(norm_Y < 1e-10, 1.0, norm_Y)
        X_norm = X / norm_X
        Y_norm = Y / norm_Y
        sims = np.dot(X_norm, Y_norm.T)
        return 1.0 - np.clip(sims, -1.0, 1.0)
    elif metric in ("euclidean", "l2"):
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
        x2 = np.sum(X ** 2, axis=1, keepdims=True)
        y2 = np.sum(Y ** 2, axis=1, keepdims=True)
        dists_sq = np.maximum(0.0, x2 + y2.T - 2.0 * np.dot(X, Y.T))
        return np.sqrt(dists_sq)
    elif metric == "manhattan":
        # |X_i - Y_j|
        return np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]).sum(axis=2)
    elif metric == "dot":
        return -np.dot(X, Y.T)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
