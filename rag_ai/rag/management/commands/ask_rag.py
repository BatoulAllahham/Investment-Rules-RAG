from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from rag.services.collections import collection_name_for_provider
from rag.services.embeddings import get_embedding_provider
from rag.services.qa import ask_question


class Command(BaseCommand):
    help = "Ask a question, retrieve relevant Chroma chunks, and generate an answer."

    def add_arguments(self, parser):
        parser.add_argument("question", help="Customer question to answer.")
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
            choices=["local", "openai", "openrouter", "bge-m3", "baai/bge-m3", "text-embedding-3-small"],
            help="Embedding backend used during ingestion.",
        )
        parser.add_argument(
            "--model",
            default=None,
            help="OpenRouter chat model. Defaults to RAG_CHAT_MODEL.",
        )
        parser.add_argument("--top-k", type=int, default=None)
        parser.add_argument("--temperature", type=float, default=0.1)

    def handle(self, *args, **options):
        embedding_provider = options["embedding_provider"] or settings.RAG_EMBEDDING_PROVIDER
        provider = get_embedding_provider(embedding_provider)
        chroma_path = Path(options["chroma_path"] or settings.RAG_CHROMA_PATH)
        collection_name = options["collection"] or collection_name_for_provider(
            settings.RAG_CHROMA_COLLECTION,
            provider.name,
        )
        chat_model = options["model"] or settings.RAG_CHAT_MODEL

        result = ask_question(
            question=options["question"],
            persist_path=chroma_path,
            collection_name=collection_name,
            embedding_provider=embedding_provider,
            chat_model=chat_model,
            top_k=options["top_k"] or settings.RAG_DEFAULT_TOP_K,
            temperature=options["temperature"],
        )

        self.stdout.write(self.style.SUCCESS(f"Model: {result.model}"))
        self.stdout.write("")
        self.stdout.write(result.answer)
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Retrieved sources:"))
        for index, source in enumerate(result.sources, start=1):
            source_name = source.metadata.get("source", "unknown source")
            pages = (
                f"page {source.page_start}"
                if source.page_start == source.page_end
                else f"pages {source.page_start}-{source.page_end}"
            )
            legal_label = _legal_label(source.metadata)
            if legal_label:
                self.stdout.write(
                    f"{index}. {source_name}, {pages}, {legal_label} | score={source.score:.4f}"
                )
            else:
                self.stdout.write(f"{index}. {source_name}, {pages} | score={source.score:.4f}")


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
