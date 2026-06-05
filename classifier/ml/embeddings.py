"""Shared MiniLM encoder — used by both training and runtime classifiers.

The default encoder is **bundled inside the package** (classifier/ml/models/
all-MiniLM-L6-v2/) so Layer 3 runs fully offline out of the box — no download,
no network, works on air-gapped/restricted machines. The L3 head is trained
against this exact embedder, so changing it without retraining will silently
degrade routing.

Resolution order (default model):
  1. Bundled in-package copy   → offline, instant (this is the default path).
  2. Local cache from a prior download (~/.cache/dynamic-model-router/encoders/).
  3. One-time HF download (~90MB) only if the bundled copy is absent.

A custom encoder (DMR_EMBEDDING_MODEL) is honored as-is: a local dir loads
directly; an HF id downloads on first use.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Frozen-by-default encoder. Pinned revision = reproducibility guarantee.
# If you change either field, retrain the L3 head (`dmr train --auto`).
_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_MODEL_REVISION: str | None = None  # None resolves to `main`; set a SHA to pin

_MODEL_NAME = os.environ.get("DMR_EMBEDDING_MODEL", _DEFAULT_MODEL_NAME)
_MODEL_REVISION = os.environ.get("DMR_EMBEDDING_REVISION", _DEFAULT_MODEL_REVISION)

_lock = threading.Lock()
_model = None
_load_failed = False
_local_path: str | None = None

# Encoder shipped inside the package — enables fully-offline Layer 3.
_BUNDLED_DIR = Path(__file__).parent / "models" / "all-MiniLM-L6-v2"


def _bundled_encoder_path() -> str | None:
    """Path to the in-package encoder snapshot, or None if not shipped.

    Validated by the presence of the two files SentenceTransformer needs to
    recognize a local model dir, so a partial copy doesn't masquerade as ready.
    """
    if (_BUNDLED_DIR / "config.json").exists() and (_BUNDLED_DIR / "modules.json").exists():
        return str(_BUNDLED_DIR)
    return None


def _cache_dir() -> Path:
    """Package-controlled encoder cache. Honors DMR_CACHE_DIR override."""
    base = Path(os.environ.get("DMR_CACHE_DIR") or (Path.home() / ".cache" / "dynamic-model-router"))
    target = base / "encoders"
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_encoder_available(
    *, model_name: str | None = None, revision: str | None = None, quiet: bool = False
) -> str | None:
    """Resolve the encoder snapshot path, downloading only if necessary.

    For the default `all-MiniLM-L6-v2`, the bundled in-package copy is used
    instantly with no network call. A custom `DMR_EMBEDDING_MODEL` is resolved
    as a local directory or downloaded from Hugging Face (cached under
    `~/.cache/dynamic-model-router/encoders/`). Returns None if the model is
    unavailable (caller falls back to ST's own resolution).
    """
    name = model_name or _MODEL_NAME
    rev = revision if revision is not None else (
        _MODEL_REVISION if name == _DEFAULT_MODEL_NAME else None
    )

    # Offline-first. For the default MiniLM, prefer the encoder bundled in the
    # package — no network, no huggingface_hub needed, works air-gapped.
    if name in (_DEFAULT_MODEL_NAME, "all-MiniLM-L6-v2"):
        bundled = _bundled_encoder_path()
        if bundled:
            return bundled

    # A custom encoder given as a local directory loads directly.
    if os.path.isdir(name):
        return name

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        if not quiet:
            logger.warning(
                "ml.embeddings: huggingface_hub not installed — "
                "run `pip install dynamic-model-router[ml]`"
            )
        return None

    cache = _cache_dir()

    # Try offline first — if already cached, skip the network call entirely.
    try:
        return snapshot_download(
            repo_id=name,
            revision=rev,
            cache_dir=str(cache),
            local_files_only=True,
        )
    except Exception:
        pass  # not cached yet, proceed to download

    if not quiet:
        logger.info(
            "ml.embeddings: downloading %s (rev=%s) → %s (one-time, ~90MB for MiniLM)",
            name, rev or "main", cache,
        )
    try:
        local = snapshot_download(repo_id=name, revision=rev, cache_dir=str(cache))
        if not quiet:
            logger.info("ml.embeddings: encoder ready at %s", local)
        return local
    except Exception as exc:
        logger.warning("ml.embeddings: encoder download failed — %s", exc)
        return None


def set_embedding_model(name: str, revision: str | None = None) -> None:
    """Override the embedding model used by Layer 3.

    Switching models invalidates the loaded singleton. The new model loads on
    next encode() call.

    NOTE: If the new model has a different output dimensionality than the
    trained head_v1.joblib, you MUST retrain (`dmr train --auto`) — otherwise
    the head will reject the embeddings.

    Example:
        set_embedding_model("BAAI/bge-large-en-v1.5")
    """
    global _MODEL_NAME, _MODEL_REVISION, _model, _load_failed, _local_path
    with _lock:
        _MODEL_NAME = name
        _MODEL_REVISION = revision
        _model = None
        _local_path = None
        _load_failed = False
    logger.info("ml.embeddings: switched to %s (retrain L3 head if dim differs)", name)


def current_embedding_model() -> str:
    return _MODEL_NAME


def get_encoder():
    """Return the shared SentenceTransformer or None if unavailable."""
    global _model, _load_failed, _local_path
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
        except ImportError:
            logger.warning(
                "ml.embeddings: sentence-transformers not installed — "
                "run `pip install dynamic-model-router[ml]`"
            )
            _load_failed = True
            return None

        local = ensure_encoder_available()
        target = local or _MODEL_NAME  # fall back to ST's own HF resolution
        _local_path = local
        try:
            _model = SentenceTransformer(target)
            logger.info("ml.embeddings: loaded %s", _MODEL_NAME)
            return _model
        except Exception as exc:
            logger.warning("ml.embeddings: load failed — %s", exc)
            _load_failed = True
            return None


def encode(texts: list[str]) -> np.ndarray | None:
    """Encode a list of texts → (N, 384) L2-normalized array. Returns None on failure."""
    enc = get_encoder()
    if enc is None:
        return None
    return enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def encode_one(text: str) -> np.ndarray | None:
    """Encode a single text → (384,) vector. Returns None on failure."""
    arr = encode([text])
    if arr is None:
        return None
    return arr[0]
