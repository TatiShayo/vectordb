"""
Vector normalization utilities.
"""
import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a single vector or a batch (N, D) in-place-safe."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else v
    # Batch
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return v / norms


def to_float32(v: list | np.ndarray) -> np.ndarray:
    """Convert list or ndarray to float32."""
    return np.asarray(v, dtype=np.float32)


def prepare_vector(v: list | np.ndarray, normalize_vec: bool = True) -> np.ndarray:
    """Convert and optionally normalize a vector."""
    arr = to_float32(v)
    return normalize(arr) if normalize_vec else arr


def prepare_batch(vectors: list[list], normalize_vec: bool = True) -> np.ndarray:
    """Convert a list of vectors to a float32 matrix, optionally normalized."""
    mat = np.array(vectors, dtype=np.float32)
    return normalize(mat) if normalize_vec else mat
