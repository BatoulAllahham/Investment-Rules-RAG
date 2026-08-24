from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from rag.services.collections import collection_name_for_provider
from rag.services.ingestion import ingest_pdf


class Command(BaseCommand):
    help = "Extract, chunk, embed, and store a selectable PDF for RAG search."

    def add_arguments(self, parser):
        parser.add_argument("pdf_path", help="Path to the source PDF.")
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
            help="Embedding backend to use.",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=800,
            help="Approximate maximum tokens per chunk.",
        )
        parser.add_argument(
            "--overlap-tokens",
            type=int,
            default=120,
            help="Approximate token overlap between neighboring chunks.",
        )

    def handle(self, *args, **options):
        chroma_path = Path(options["chroma_path"] or settings.RAG_CHROMA_PATH)
        collection_name = options["collection"] or collection_name_for_provider(
            settings.RAG_CHROMA_COLLECTION,
            options["embedding_provider"],
        )
        result = ingest_pdf(
            pdf_path=options["pdf_path"],
            persist_path=chroma_path,
            collection_name=collection_name,
            embedding_provider=options["embedding_provider"],
            max_tokens=options["max_tokens"],
            overlap_tokens=options["overlap_tokens"],
        )

        self.stdout.write(self.style.SUCCESS("PDF indexed successfully."))
        self.stdout.write(f"Source: {result.source_path}")
        self.stdout.write(f"Pages: {result.page_count}")
        self.stdout.write(f"Chunks: {result.chunk_count}")
        self.stdout.write(f"Embedding provider: {result.embedding_provider}")
        self.stdout.write(f"Chroma path: {result.persist_path}")
        self.stdout.write(f"Collection: {result.collection_name}")
