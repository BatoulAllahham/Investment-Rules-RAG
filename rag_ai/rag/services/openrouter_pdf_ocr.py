from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from .pdf_loader import PageText


class OpenRouterOCRError(RuntimeError):
    pass


def extract_pdf_with_openrouter_ocr(pdf_path: str | Path) -> list[PageText]:
    path = Path(pdf_path).expanduser()
    reader = PdfReader(str(path))
    pages: list[PageText] = []

    for page_index, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        output = BytesIO()
        writer.write(output)
        text = _extract_page_text(
            pdf_bytes=output.getvalue(),
            filename=f"{path.stem}-page-{page_index}.pdf",
        )
        pages.append(PageText(page_number=page_index, text=text.strip()))

    return pages


def _extract_page_text(pdf_bytes: bytes, filename: str) -> str:
    try:
        import requests
    except ImportError as exc:
        raise OpenRouterOCRError("Install requests to use OpenRouter OCR.") from exc

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterOCRError("Set OPENROUTER_API_KEY to use OpenRouter OCR.")

    data_url = "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("utf-8")
    response = requests.post(
        os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        + "/chat/completions",
        headers=_headers(api_key),
        json={
            "model": os.getenv("RAG_OCR_MODEL", "openrouter/free"),
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all readable Arabic and English text from this PDF page. "
                                "Return only the extracted text. Do not summarize."
                            ),
                        },
                        {
                            "type": "file",
                            "file": {
                                "filename": filename,
                                "file_data": data_url,
                            },
                        },
                    ],
                }
            ],
            "plugins": [
                {
                    "id": "file-parser",
                    "pdf": {
                        "engine": os.getenv("RAG_OCR_PDF_ENGINE", "cloudflare-ai"),
                    },
                }
            ],
        },
        timeout=120,
    )

    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenRouterOCRError(f"OpenRouter OCR returned non-JSON response: {response.text}") from exc

    if response.status_code >= 400:
        message = payload.get("error", {}).get("message") or response.text
        raise OpenRouterOCRError(f"OpenRouter OCR failed: {message}")

    message = payload.get("choices", [{}])[0].get("message", {})
    annotation_text = _extract_annotation_text(message.get("annotations", []))
    if annotation_text:
        return annotation_text
    return str(message.get("content") or "")


def _extract_annotation_text(annotations: list[dict]) -> str:
    text_parts = []
    for annotation in annotations:
        if annotation.get("type") != "file":
            continue
        for part in annotation.get("file", {}).get("content", []):
            if part.get("type") == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
    return "\n\n".join(text_parts).strip()


def _headers(api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers
