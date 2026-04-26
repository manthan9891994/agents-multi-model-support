"""Optional embedding-based semantic cache. Falls back gracefully if
sentence-transformers or numpy are not installed."""
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from classifier.core.types import ClassificationDecision

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    import numpy  # noqa: F401
                    from sentence_transformers import SentenceTransformer
                    _model = SentenceTransformer("all-MiniLM-L6-v2")
                except ImportError:
                    _model = False  # mark as unavailable
    return _model if _model is not False else None


class SemanticCache:
    """Cosine-similarity cache: hits when similarity >= threshold (default 0.92)."""

    def __init__(self, threshold: float = 0.92, max_size: int = 5000):
        self._threshold = threshold
        self._max_size = max_size
        self._embeddings: list = []
        self._decisions: list = []
        self._lock = threading.Lock()

    def get(self, task: str) -> "Optional[ClassificationDecision]":
        model = _get_model()
        if model is None:
            return None
        try:
            import numpy as np
            vec = model.encode(task, normalize_embeddings=True)
            with self._lock:
                if not self._embeddings:
                    return None
                sims = np.array(self._embeddings) @ vec
                best_idx = int(np.argmax(sims))
                if float(sims[best_idx]) >= self._threshold:
                    return self._decisions[best_idx]
        except Exception:
            pass
        return None

    def set(self, task: str, decision: "ClassificationDecision") -> None:
        model = _get_model()
        if model is None:
            return
        try:
            vec = model.encode(task, normalize_embeddings=True)
            with self._lock:
                if len(self._embeddings) >= self._max_size:
                    self._embeddings.pop(0)
                    self._decisions.pop(0)
                self._embeddings.append(vec)
                self._decisions.append(decision)
        except Exception:
            pass

    def clear(self) -> None:
        with self._lock:
            self._embeddings.clear()
            self._decisions.clear()


semantic_cache = SemanticCache()
