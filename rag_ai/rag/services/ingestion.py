from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .chunker import TextChunk, chunk_pdf_pages
from .embeddings import EmbeddingProvider, get_embedding_provider
from .pdf_loader import extract_pdf_text
from .vector_store import ChromaVectorStore


@dataclass(frozen=True)
class IngestionResult:
    source_path: Path
    persist_path: Path
    collection_name: str
    page_count: int
    chunk_count: int
    embedding_provider: str
    chunks: list[TextChunk]


def ingest_pdf(
    pdf_path: str | Path,
    persist_path: str | Path,
    collection_name: str = "investment_rules",
    embedding_provider: str | EmbeddingProvider = "local",
    max_tokens: int = 800,
    overlap_tokens: int = 120,
    batch_size: int = 64,
) -> IngestionResult:
    path = Path(pdf_path).expanduser()
    provider = (
        get_embedding_provider(embedding_provider)
        if isinstance(embedding_provider, str)
        else embedding_provider
    )

    pages = extract_pdf_text(path)
    chunks = chunk_pdf_pages(
        pages=pages,
        source_path=path,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    vectors = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(provider.embed([chunk.text for chunk in batch]))

    store = ChromaVectorStore(
        persist_path=persist_path,
        collection_name=collection_name,
    )
    store.upsert_document(
        source_path=path,
        source_sha256=_sha256_file(path),
        page_count=len(pages),
        chunks=chunks,
        vectors=vectors,
        embedding_provider=provider.name,
    )

    return IngestionResult(
        source_path=path,
        persist_path=Path(persist_path),
        collection_name=collection_name,
        page_count=len(pages),
        chunk_count=len(chunks),
        embedding_provider=provider.name,
        chunks=chunks,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
