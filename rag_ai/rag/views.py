from __future__ import annotations

import json
import re
from json import JSONDecodeError
from pathlib import Path

from django.conf import settings
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from rag.services.collections import collection_name_for_provider
from rag.services.embeddings import get_embedding_provider
from rag.services.qa import ask_question


@csrf_exempt
@require_POST
def ask_rag_api(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except JSONDecodeError:
        return JsonResponse({"error": "Request body must be valid JSON."}, status=400)

    question = str(payload.get("question", "")).strip()
    if not question:
        return JsonResponse({"error": "Field 'question' is required."}, status=400)

    embedding_provider_name = payload.get(
        "embedding_provider",
        settings.RAG_EMBEDDING_PROVIDER,
    )
    provider = get_embedding_provider(str(embedding_provider_name))
    collection_name = payload.get("collection") or collection_name_for_provider(
        settings.RAG_CHROMA_COLLECTION,
        provider.name,
    )

    try:
        top_k = int(payload.get("top_k", 5))
        temperature = float(payload.get("temperature", 0.1))
    except (TypeError, ValueError):
        return JsonResponse(
            {"error": "Fields 'top_k' and 'temperature' must be numbers."},
            status=400,
        )

    if top_k < 1:
        return JsonResponse({"error": "Field 'top_k' must be at least 1."}, status=400)
    chat_model = str(payload.get("model") or settings.RAG_CHAT_MODEL)
    chroma_path = Path(payload.get("chroma_path") or settings.RAG_CHROMA_PATH)

    try:
        result = ask_question(
            question=question,
            persist_path=chroma_path,
            collection_name=collection_name,
            embedding_provider=str(embedding_provider_name),
            chat_model=chat_model,
            top_k=top_k,
            temperature=temperature,
        )
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)

    return JsonResponse(
        {
            "question": question,
            "answer": result.answer,
            "model": result.model,
            "collection": collection_name,
            "top_k": top_k,
            "sources": [
                {
                    "source_number": index,
                    "source": source.metadata.get("source", ""),
                    "source_path": source.metadata.get("source_path", ""),
                    "score": source.score,
                    "page_start": source.page_start,
                    "page_end": source.page_end,
                    "section_title": source.section_title,
                    "snippet": _single_line(source.text[:700]),
                    "metadata": source.metadata,
                }
                for index, source in enumerate(result.sources, start=1)
            ],
        },
        json_dumps_params={"ensure_ascii": False},
    )


def _single_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
