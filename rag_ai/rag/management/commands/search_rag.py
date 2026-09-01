from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from rag.services.collections import collection_name_for_provider
from rag.services.embeddings import get_embedding_provider
from rag.services.search import search_chunks


class Command(BaseCommand):
    help = "Search indexed RAG chunks for a question."

    def add_arguments(self, parser):
        parser.add_argument("question", help="Customer question to search for.")
        parser.add_argument(
            "--chroma-path",
            default=None,
            help="Chroma persistence directory. Defaults to RAG_CHROMA_PATH.",
        )
        parser.add_argument(
            "--collection",
            default=None,
            help="Chroma collection name. Defaults to RAG_CHROMA_COLLECTION.",
        )
        parser.add_argument(
            "--embedding-provider",
            default=None,
            choices=["local", "openai", "openrouter", "bge-m3", "baai/bge-m3", "text-embedding-3-small"],
            help="Embedding backend used during ingestion. Defaults to RAG_EMBEDDING_PROVIDER.",
        )
        parser.add_argument("--top-k", type=int, default=5)

    def handle(self, *args, **options):
        embedding_provider = options["embedding_provider"] or settings.RAG_EMBEDDING_PROVIDER
        provider = get_embedding_provider(embedding_provider)
        chroma_path = Path(options["chroma_path"] or settings.RAG_CHROMA_PATH)
        collection_name = options["collection"] or collection_name_for_provider(
            settings.RAG_CHROMA_COLLECTION,
            provider.name,
        )
        results = search_chunks(
            question=options["question"],
            persist_path=chroma_path,
            collection_name=collection_name,
            embedding_provider=embedding_provider,
            top_k=options["top_k"],
        )

        if not results:
            self.stdout.write(self.style.WARNING("No chunks found. Run ingest_pdf first."))
            return

        for index, result in enumerate(results, start=1):
            source_name = result.metadata.get("source", "unknown source")
            pages = (
                f"page {result.page_start}"
                if result.page_start == result.page_end
                else f"pages {result.page_start}-{result.page_end}"
            )
            self.stdout.write("")
            self.stdout.write(
                self.style.SUCCESS(f"{index}. {source_name}, {pages}, score={result.score:.4f}")
            )
            if result.section_title:
                self.stdout.write(f"Section: {result.section_title}")
            legal_label = _legal_label(result.metadata)
            if legal_label:
                self.stdout.write(f"Legal reference: {legal_label}")
            self.stdout.write(result.text[:1200])


def _legal_label(metadata) -> str:
    parts = []
    source_type = metadata.get("source_type")
    document_number = metadata.get("document_number")
    document_year = metadata.get("document_year")
    article_number = metadata.get("article_number")
    if source_type and document_number:
        label = f"{source_type} رقم {document_number}"
        if document_year:
            label += f" لعام {document_year}"
        parts.append(label)
    if article_number:
        parts.append(f"المادة ({article_number})")
    return " - ".join(parts)
