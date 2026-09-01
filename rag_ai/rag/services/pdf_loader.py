from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


class PDFExtractionError(RuntimeError):
    pass


def extract_pdf_text(pdf_path: str | Path) -> list[PageText]:
    """Extract selectable PDF text page by page.

    pypdf is the default because it handles this Arabic source PDF better than
    pdfplumber's plain text extraction. pdfplumber remains a fallback.
    """
    path = Path(pdf_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = _extract_with_pypdf(path)
    if _has_usable_text(pages):
        return pages

    pages = _extract_with_pdfplumber(path)
    if _has_usable_text(pages):
        return pages

    raise PDFExtractionError(
        "No selectable text was extracted from this PDF."
    )


def _extract_with_pypdf(path: Path) -> list[PageText]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PDFExtractionError("Install pypdf to extract selectable PDF text.") from exc

    reader = PdfReader(str(path))
    return [
        PageText(page_number=index + 1, text=page.extract_text() or "")
        for index, page in enumerate(reader.pages)
    ]


def _extract_with_pdfplumber(path: Path) -> list[PageText]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise PDFExtractionError("Install pdfplumber to use the fallback extractor.") from exc

    with pdfplumber.open(str(path)) as pdf:
        return [
            PageText(page_number=index + 1, text=page.extract_text() or "")
            for index, page in enumerate(pdf.pages)
        ]


def _has_usable_text(pages: Iterable[PageText]) -> bool:
    return sum(len(page.text.strip()) for page in pages) > 100
