from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

from .pdf_loader import PageText
from .text_cleaning import clean_page_texts


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟؛])\s+")
ARTICLE_RE = re.compile(r"^(المادة|الفصل|الباب|Article|Chapter|Section)\b", re.IGNORECASE)


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    section_title: str
    token_count: int


@dataclass(frozen=True)
class TextBlock:
    text: str
    page_number: int
    section_title: str
    token_count: int


def chunk_pdf_pages(
    pages: list[PageText],
    source_path: str | Path,
    max_tokens: int = 800,
    overlap_tokens: int = 120,
) -> list[TextChunk]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")

    cleaned_pages = clean_page_texts(pages)
    blocks = list(_build_blocks(cleaned_pages))
    chunks: list[TextChunk] = []
    current: list[TextBlock] = []
    current_tokens = 0

    for block in blocks:
        split_blocks = _split_large_block(block, max_tokens)
        for piece in split_blocks:
            if current and current_tokens + piece.token_count > max_tokens:
                chunks.append(_make_chunk(chunks, current, source_path))
                current = _overlap_blocks(current, overlap_tokens)
                current_tokens = sum(item.token_count for item in current)

            current.append(piece)
            current_tokens += piece.token_count

    if current:
        chunks.append(_make_chunk(chunks, current, source_path))

    return chunks


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _build_blocks(pages: list[PageText]):
    section_title = ""

    for page in pages:
        paragraphs = _paragraphs(page.text)
        for paragraph in paragraphs:
            first_line = paragraph.splitlines()[0].strip()
            if _looks_like_heading(first_line):
                section_title = first_line
            yield TextBlock(
                text=paragraph,
                page_number=page.page_number,
                section_title=section_title,
                token_count=count_tokens(paragraph),
            )


def _paragraphs(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        if _looks_like_heading(line) and current:
            paragraphs.append("\n".join(current).strip())
            current = [line]
            continue
        current.append(line)

        if _ends_paragraph(line):
            paragraphs.append("\n".join(current).strip())
            current = []

    if current:
        paragraphs.append("\n".join(current).strip())

    return paragraphs


def _looks_like_heading(line: str) -> bool:
    if not line:
        return False
    if ARTICLE_RE.match(line):
        return True
    if len(line) <= 90 and not _ends_paragraph(line):
        return True
    return False


def _ends_paragraph(line: str) -> bool:
    return line.endswith((".", ":", "؛", "؟", "!", "?"))


def _split_large_block(block: TextBlock, max_tokens: int) -> list[TextBlock]:
    if block.token_count <= max_tokens:
        return [block]

    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(block.text) if part.strip()]
    if len(sentences) <= 1:
        return _split_by_tokens(block, max_tokens)

    pieces: list[TextBlock] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            pieces.append(_copy_block(block, "\n".join(current)))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        pieces.append(_copy_block(block, "\n".join(current)))

    return pieces


def _split_by_tokens(block: TextBlock, max_tokens: int) -> list[TextBlock]:
    tokens = TOKEN_RE.findall(block.text)
    pieces = []
    for index in range(0, len(tokens), max_tokens):
        text = " ".join(tokens[index : index + max_tokens])
        pieces.append(_copy_block(block, text))
    return pieces


def _copy_block(block: TextBlock, text: str) -> TextBlock:
    return TextBlock(
        text=text,
        page_number=block.page_number,
        section_title=block.section_title,
        token_count=count_tokens(text),
    )


def _overlap_blocks(blocks: list[TextBlock], overlap_tokens: int) -> list[TextBlock]:
    if overlap_tokens == 0:
        return []

    selected: list[TextBlock] = []
    total = 0
    for block in reversed(blocks):
        selected.append(block)
        total += block.token_count
        if total >= overlap_tokens:
            break
    return list(reversed(selected))


def _make_chunk(
    chunks: list[TextChunk],
    blocks: list[TextBlock],
    source_path: str | Path,
) -> TextChunk:
    text = "\n\n".join(block.text for block in blocks).strip()
    chunk_index = len(chunks)
    chunk_hash = sha1(f"{Path(source_path).name}:{chunk_index}:{text}".encode("utf-8")).hexdigest()
    section_title = next((block.section_title for block in blocks if block.section_title), "")
    return TextChunk(
        chunk_id=chunk_hash,
        chunk_index=chunk_index,
        text=text,
        page_start=min(block.page_number for block in blocks),
        page_end=max(block.page_number for block in blocks),
        section_title=section_title,
        token_count=count_tokens(text),
    )

