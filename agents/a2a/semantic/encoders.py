"""
Text Encoders for the Semantic Model
====================================
``trust_eval`` is a frozen encoder + trainable head. This module provides the
encoder side as a small pluggable interface with two implementations:

- ``SentenceTransformerEncoder`` — a real pretrained sentence-transformer
  (default ``all-MiniLM-L6-v2``, 384-d). Best semantics; needs the model
  available locally or downloadable. Used for training and the demo.
- ``HashingEncoder`` — a deterministic, dependency-light feature-hashing
  encoder (numpy only, no network). Reproducible and offline; used for fast
  tests and reproducible FL simulation.

A trained model records which encoder it used (``spec``) so inference rebuilds
the same one. Encoders return L2-normalized ``float32`` vectors of shape
``(n, dim)``.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class Encoder(ABC):
    """Encodes a batch of texts into L2-normalized vectors of fixed dimension."""

    #: A string that identifies this encoder, persisted with a trained model so
    #: that inference can reconstruct an equivalent encoder (see ``get_encoder``).
    spec: str

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a ``(len(texts), dim)`` float32 array of L2-normalized rows."""


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (mat / norms).astype(np.float32)


class HashingEncoder(Encoder):
    """Deterministic feature-hashing encoder — numpy only, no network.

    Each token is hashed to an index and a sign (signed feature hashing), counts
    are accumulated, and the vector is L2-normalized. Captures lexical overlap,
    which is enough signal for the synthetic trust dataset, and is fully
    reproducible — ideal for tests and FL simulation.
    """

    def __init__(self, dim: int = 256):
        self._dim = int(dim)
        self.spec = f"hashing:{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for tok in _tokenize(text):
            h = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[idx] += sign
        return vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.vstack([self._encode_one(t) for t in texts]) if texts else np.zeros((0, self._dim), np.float32)
        return _l2_normalize(mat)


class SentenceTransformerEncoder(Encoder):
    """Wraps a pretrained sentence-transformer (lazy import of the heavy dep)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy / heavy

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        self.spec = f"sentence-transformers:{model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)


def get_encoder(spec: str = "sentence-transformers:all-MiniLM-L6-v2") -> Encoder:
    """Build an encoder from a spec string.

    Specs:
      - ``"hashing:<dim>"`` or ``"hashing"`` → ``HashingEncoder``
      - ``"sentence-transformers:<model_name>"`` → ``SentenceTransformerEncoder``
    """
    if spec == "hashing" or spec.startswith("hashing:"):
        dim = int(spec.split(":", 1)[1]) if ":" in spec else 256
        return HashingEncoder(dim=dim)
    if spec.startswith("sentence-transformers:"):
        return SentenceTransformerEncoder(model_name=spec.split(":", 1)[1])
    raise ValueError(f"unknown encoder spec: {spec!r}")
