from __future__ import annotations

from pathlib import Path

from .llm import RAGAnswer, answer_with_openrouter
from .search import search_chunks


def ask_question(
    question: str,
    persist_path: str | Path,
    collection_name: str,
    embedding_provider: str,
    chat_model: str,
    top_k: int = 5,
    temperature: float = 0.1,
) -> RAGAnswer:
    sources = search_chunks(
        question=question,
        persist_path=persist_path,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        top_k=top_k,
    )
    return answer_with_openrouter(
        question=question,
        sources=sources,
        model=chat_model,
        temperature=temperature,
    )
