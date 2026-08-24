from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunker import TextChunk


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    score: float
    text: str
    page_start: int
    page_end: int
    section_title: str
    metadata: dict[str, Any]


class ChromaVectorStore:
    def __init__(self, persist_path: str | Path, collection_name: str):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Chroma is not installed. Install it with: pip install chromadb"
            ) from exc

        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_document(
        self,
        source_path: str | Path,
        source_sha256: str,
        page_count: int,
        chunks: list[TextChunk],
        vectors: list[list[float]],
        embedding_provider: str,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length.")

        source_path = str(Path(source_path))
        source_name = Path(source_path).name

        self.collection.delete(where={"source_path": source_path})
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "source": source_name,
                    "source_path": source_path,
                    "source_sha256": source_sha256,
                    "page_count": page_count,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_title": chunk.section_title,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "embedding_provider": embedding_provider,
                }
            )

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )

    def similarity_search(
        self,
        query_vector: list[float],
        embedding_provider: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"embedding_provider": embedding_provider},
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        search_results: list[SearchResult] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = metadata or {}
            page_start = int(metadata.get("page_start", 0))
            page_end = int(metadata.get("page_end", page_start))
            section_title = str(metadata.get("section_title", ""))
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    score=1.0 - float(distance),
                    text=document or "",
                    page_start=page_start,
                    page_end=page_end,
                    section_title=section_title,
                    metadata=dict(metadata),
                )
            )

        return search_results

    def count_chunks(self) -> int:
        return self.collection.count()
