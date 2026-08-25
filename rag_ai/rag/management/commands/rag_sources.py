from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from rag.services.collections import collection_name_for_provider
from rag.services.embeddings import get_embedding_provider
from rag.services.vector_store import ChromaVectorStore


class Command(BaseCommand):
    help = "Show which source PDFs are indexed in the active Chroma collection."

    def add_arguments(self, parser):
        parser.add_argument(
            "--chroma-path",
            default=None,
            help="Chroma persistence directory. Defaults to RAG_CHROMA_PATH.",
        )
        parser.add_argument(
            "--collection",
            default=None,
            help="Chroma collection name. Defaults to the provider-specific collection.",
        )
        parser.add_argument(
            "--embedding-provider",
            default=None,
            choices=["local", "openai", "openrouter", "text-embedding-3-small"],
            help="Embedding backend used during ingestion. Defaults to RAG_EMBEDDING_PROVIDER.",
        )

    def handle(self, *args, **options):
        embedding_provider = options["embedding_provider"] or settings.RAG_EMBEDDING_PROVIDER
        provider = get_embedding_provider(embedding_provider)
        chroma_path = Path(options["chroma_path"] or settings.RAG_CHROMA_PATH)
        collection_name = options["collection"] or collection_name_for_provider(
            settings.RAG_CHROMA_COLLECTION,
            provider.name,
        )
        store = ChromaVectorStore(chroma_path, collection_name)
        stats = store.source_stats()

        self.stdout.write(f"Collection: {collection_name}")
        self.stdout.write(f"Total chunks: {store.count_chunks()}")
        if not stats:
            self.stdout.write(self.style.WARNING("No indexed sources found."))
            return

        for item in stats:
            self.stdout.write("")
            self.stdout.write(self.style.SUCCESS(item["source"]))
            self.stdout.write(f"Chunks: {item['chunks']}")
            self.stdout.write(f"Pages: {item['page_count']}")
            self.stdout.write(f"Embedding provider: {item['embedding_provider']}")
            self.stdout.write(f"Path: {item['source_path']}")
