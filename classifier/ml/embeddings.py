"""Shared MiniLM encoder — used by both training and runtime classifiers.

Lazy-loaded singleton. First call takes ~5–10s (model download on first run).
Subsequent calls return the cached SentenceTransformer instance.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL_NAME = os.environ.get("DMR_EMBEDDING_MODEL", _DEFAULT_MODEL_NAME)
_lock = threading.Lock()
_model = None
_load_failed = False


def set_embedding_model(name: str) -> None:
    """Override the embedding model used by Layer 3.

    Switching models invalidates the loaded singleton. The new model loads on
    next encode() call.

    NOTE: If the new model has a different output dimensionality than the
    trained head_v1.joblib, you MUST retrain (`dmr train`) — otherwise the
    head will reject the embeddings.

    Example:
        set_embedding_model("BAAI/bge-large-en-v1.5")
    """
    global _MODEL_NAME, _model, _load_failed
    with _lock:
        _MODEL_NAME = name
        _model = None
        _load_failed = False
    logger.info("ml.embeddings: switched to %s (retrain L3 head if dim differs)", name)


def current_embedding_model() -> str:
    return _MODEL_NAME


def get_encoder():
    """Return the shared SentenceTransformer or None if unavailable."""
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None
    with _lock:
        if _model is not None:
            return _model
        if _load_failed:
            return None
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_MODEL_NAME)
            logger.info("ml.embeddings: loaded %s", _MODEL_NAME)
            return _model
        except ImportError:
            logger.warning(
                "ml.embeddings: sentence-transformers not installed — "
                "run `pip install sentence-transformers scikit-learn`"
            )
            _load_failed = True
            return None
        except Exception as exc:
            logger.warning("ml.embeddings: load failed — %s", exc)
            _load_failed = True
            return None


def encode(texts: list[str]) -> Optional[np.ndarray]:
    """Encode a list of texts → (N, 384) L2-normalized array. Returns None on failure."""
    enc = get_encoder()
    if enc is None:
        return None
    return enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def encode_one(text: str) -> Optional[np.ndarray]:
    """Encode a single text → (384,) vector. Returns None on failure."""
    arr = encode([text])
    if arr is None:
        return None
    return arr[0]
