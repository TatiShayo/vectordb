"""
Lazy-loaded sentence-transformer embedder.

sentence-transformers is optional — if it isn't installed the text-search
endpoint returns 503 with a clear message instead of crashing at startup.
"""
from __future__ import annotations

import logging
import threading
from typing import List

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_model = None
_model_name: str | None = None


def _load(model_name: str):
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model
    with _lock:
        if _model is not None and _model_name == model_name:
            return _model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            _model = SentenceTransformer(model_name)
            _model_name = model_name
            logger.info("Embedding model loaded.")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
    return _model


def embed(text: str | List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed text(s) into float32 vectors. Returns (N, D) or (D,) ndarray."""
    model = _load(model_name)
    if isinstance(text, str):
        return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def is_available(model_name: str = "all-MiniLM-L6-v2") -> bool:
    try:
        _load(model_name)
        return True
    except (RuntimeError, Exception):
        return False
