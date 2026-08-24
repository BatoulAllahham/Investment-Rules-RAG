from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from rag.services.collections import collection_name_for_provider
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
            default="local",
            choices=["local", "openai", "openrouter", "text-embedding-3-small"],
            help="Embedding backend used during ingestion.",
        )
        parser.add_argument("--top-k", type=int, default=5)

    def handle(self, *args, **options):
        chroma_path = Path(options["chroma_path"] or settings.RAG_CHROMA_PATH)
        collection_name = options["collection"] or collection_name_for_provider(
            settings.RAG_CHROMA_COLLECTION,
            options["embedding_provider"],
        )
        results = search_chunks(
            question=options["question"],
            persist_path=chroma_path,
            collection_name=collection_name,
            embedding_provider=options["embedding_provider"],
            top_k=options["top_k"],
        )

        if not results:
            self.stdout.write(self.style.WARNING("No chunks found. Run ingest_pdf first."))
            return

        for index, result in enumerate(results, start=1):
            pages = (
                f"page {result.page_start}"
                if result.page_start == result.page_end
                else f"pages {result.page_start}-{result.page_end}"
            )
            self.stdout.write("")
            self.stdout.write(self.style.SUCCESS(f"{index}. score={result.score:.4f}, {pages}"))
            if result.section_title:
                self.stdout.write(f"Section: {result.section_title}")
            self.stdout.write(result.text[:1200])
