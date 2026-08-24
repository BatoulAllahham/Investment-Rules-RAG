from __future__ import annotations

from pathlib import Path

from .embeddings import get_embedding_provider
from .vector_store import ChromaVectorStore, SearchResult


def search_chunks(
    question: str,
    persist_path: str | Path,
    collection_name: str = "investment_rules",
    embedding_provider: str = "local",
    top_k: int = 5,
) -> list[SearchResult]:
    provider = get_embedding_provider(embedding_provider)
    query_vector = provider.embed([question])[0]
    store = ChromaVectorStore(
        persist_path=persist_path,
        collection_name=collection_name,
    )
    return store.similarity_search(
        query_vector=query_vector,
        embedding_provider=provider.name,
        top_k=top_k,
    )
