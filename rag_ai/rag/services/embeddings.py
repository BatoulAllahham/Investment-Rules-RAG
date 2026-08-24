from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class EmbeddingProvider(ABC):
    name: str
    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Dependency-free embedding backend for local development.

    This is useful for building and testing the RAG plumbing. For production
    semantic retrieval, replace it with OpenAI, sentence-transformers, or
    another real embedding model.
    """

    name = "local-hash-v1"

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_RE.findall(text.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai-text-embedding-3-small"
    dimensions = 1536

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.name = f"openai-{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use OpenAI embeddings.") from exc

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before using OpenAI embeddings.")

        client = OpenAI()
        response = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


def get_embedding_provider(name: str = "local") -> EmbeddingProvider:
    normalized = name.strip().lower()
    if normalized in {"local", "hash", "local-hash"}:
        return LocalHashEmbeddingProvider()
    if normalized in {"openai", "text-embedding-3-small"}:
        return OpenAIEmbeddingProvider()
    raise ValueError(f"Unknown embedding provider: {name}")

