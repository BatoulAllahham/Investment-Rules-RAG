from __future__ import annotations

import re
from dataclasses import dataclass, replace
from hashlib import sha1
from pathlib import Path

from .pdf_loader import PageText
from .text_cleaning import clean_page_texts


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟؛])\s+")
ARTICLE_RE = re.compile(r"^(المادة|الفصل|الباب|Article|Chapter|Section)\b", re.IGNORECASE)
NORMAL_ARTICLE_HEADER_RE = re.compile(
    r"^\s*المادة\s*(?:[\(/]?\s*([0-9٠-٩]+)\s*[\)/:\s]*)",
    re.UNICODE,
)
REVERSED_ARTICLE_HEADER_RE = re.compile(
    r"^\s*[:/()\s]*([0-9٠-٩]+)\s+المادة\s*$",
    re.UNICODE,
)
CHAPTER_RE = re.compile(r"^\s*(الفصل|الباب)\b.*", re.UNICODE)
DOCUMENT_RE = re.compile(
    r"\b(القانون|المرسوم|القرار)\s+رقم\s*[\(/ ]*\s*([0-9٠-٩]+)\s*[\)/ ]*"
    r"(?:لعام\s*([0-9٠-٩]{4}|[12][0-9]{3}))?",
    re.UNICODE,
)
REVERSED_DOCUMENT_RE = re.compile(
    r"[\(/ ]*([0-9٠-٩]+)\s*(القانون|المرسوم|القرار)\s+رقم",
    re.UNICODE,
)
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    section_title: str
    token_count: int
    chunk_type: str = "text"
    source_type: str = ""
    document_number: str = ""
    document_year: str = ""
    chapter: str = ""
    article_number: str = ""


@dataclass(frozen=True)
class TextBlock:
    text: str
    page_number: int
    page_end: int
    section_title: str
    token_count: int
    chunk_type: str = "text"
    source_type: str = ""
    document_number: str = ""
    document_year: str = ""
    chapter: str = ""
    article_number: str = ""


@dataclass(frozen=True)
class ArticleHeader:
    article_number: str
    header_text: str


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
    legal_blocks = list(_build_legal_blocks(cleaned_pages))
    article_blocks = [block for block in legal_blocks if block.chunk_type == "article"]
    if len(article_blocks) >= 3:
        return _make_hybrid_chunks(
            blocks=legal_blocks,
            source_path=source_path,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

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


def _build_legal_blocks(pages: list[PageText]) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    current_lines: list[str] = []
    current_page = 0
    current_page_end = 0
    current_section_title = ""
    current_chunk_type = "section"
    current_article_number = ""
    current_chapter = ""
    current_source_type = ""
    current_document_number = ""
    current_document_year = ""
    previous_article: int | None = None

    def flush() -> None:
        nonlocal current_lines
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            current_lines = []
            return
        blocks.append(
            TextBlock(
                text=text,
                page_number=current_page,
                page_end=current_page_end,
                section_title=current_section_title,
                token_count=count_tokens(text),
                chunk_type=current_chunk_type,
                source_type=current_source_type,
                document_number=current_document_number,
                document_year=current_document_year,
                chapter=current_chapter,
                article_number=current_article_number,
            )
        )
        current_lines = []

    for page in pages:
        for line in _non_empty_lines(page.text):
            clean_line = _clean_line(line)
            if not clean_line:
                continue

            document_info = _document_info_from_line(clean_line)
            if document_info and _should_update_document_info(
                clean_line,
                current_source_type,
                bool(current_lines),
            ):
                current_source_type, current_document_number, current_document_year = document_info

            chapter = _chapter_from_line(clean_line)
            if chapter:
                if current_lines:
                    flush()
                current_chapter = chapter
                current_page = page.page_number
                current_page_end = page.page_number
                current_section_title = current_chapter
                current_chunk_type = "section"
                current_article_number = ""

            header = _article_header_from_line(clean_line, previous_article)
            if header:
                flush()
                previous_article = _safe_int(header.article_number)
                current_page = page.page_number
                current_page_end = page.page_number
                current_article_number = header.article_number
                current_section_title = _legal_section_title(
                    chapter=current_chapter,
                    article_number=current_article_number,
                )
                current_chunk_type = "article"
                current_lines = [header.header_text]
                continue

            if not current_lines:
                current_page = page.page_number
                current_page_end = page.page_number
                current_section_title = current_chapter
                current_chunk_type = "section"
                current_article_number = ""

            current_page_end = page.page_number
            current_lines.append(clean_line)

    flush()
    return blocks


def _make_hybrid_chunks(
    blocks: list[TextBlock],
    source_path: str | Path,
    max_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    section_blocks: list[TextBlock] = []
    section_tokens = 0

    def flush_sections() -> None:
        nonlocal section_blocks, section_tokens
        if section_blocks:
            chunks.append(_make_chunk(chunks, section_blocks, source_path))
            section_blocks = []
            section_tokens = 0

    for block in blocks:
        if block.chunk_type == "article":
            flush_sections()
            if _is_marker_only_article(block):
                continue
            pieces = _split_large_block(block, max_tokens)
            is_split = len(pieces) > 1
            for piece in pieces:
                if is_split:
                    piece = replace(piece, chunk_type="article_part")
                chunks.append(_make_chunk(chunks, [piece], source_path))
            continue

        for piece in _split_large_block(block, max_tokens):
            if section_blocks and section_tokens + piece.token_count > max_tokens:
                chunks.append(_make_chunk(chunks, section_blocks, source_path))
                section_blocks = _overlap_blocks(section_blocks, overlap_tokens)
                section_tokens = sum(item.token_count for item in section_blocks)

            section_blocks.append(piece)
            section_tokens += piece.token_count

    flush_sections()
    return chunks


def _is_marker_only_article(block: TextBlock) -> bool:
    return bool(block.article_number) and block.token_count <= 5


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
                page_end=page.page_number,
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


def _non_empty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _article_header_from_line(line: str, previous_article: int | None) -> ArticleHeader | None:
    reversed_match = REVERSED_ARTICLE_HEADER_RE.match(line)
    if reversed_match:
        article_number = _normalize_article_number(
            reversed_match.group(1),
            previous_article,
        )
        return ArticleHeader(
            article_number=article_number,
            header_text=f"المادة ({article_number}):",
        )

    normal_match = NORMAL_ARTICLE_HEADER_RE.match(line)
    if not normal_match:
        return None

    raw_number = normal_match.group(1)
    if not raw_number:
        return None

    article_number = _normalize_article_number(raw_number, previous_article)
    header_text = _replace_article_number(line, normal_match, article_number)
    return ArticleHeader(article_number=article_number, header_text=header_text)


def _replace_article_number(line: str, match: re.Match, article_number: str) -> str:
    suffix = line[match.end() :].strip()
    if suffix:
        return f"المادة ({article_number}): {suffix}"
    return f"المادة ({article_number}):"


def _chapter_from_line(line: str) -> str:
    match = CHAPTER_RE.match(line)
    if not match:
        return ""
    return line[:120]


def _document_info_from_line(line: str) -> tuple[str, str, str] | None:
    match = DOCUMENT_RE.search(line)
    if match:
        source_type = match.group(1)
        raw_number = match.group(2)
        raw_year = match.group(3) or ""
        year = _normalize_document_year(raw_year)
        reverse_number = bool(raw_year and year != _digits_only(raw_year))
        return (
            source_type,
            _normalize_document_number(raw_number, reverse_number=reverse_number),
            year,
        )

    reversed_match = REVERSED_DOCUMENT_RE.search(line)
    if reversed_match:
        return (
            reversed_match.group(2),
            _normalize_document_number(reversed_match.group(1), reverse_number=True),
            "",
        )

    return None


def _should_update_document_info(line: str, current_source_type: str, has_current_text: bool) -> bool:
    if not current_source_type:
        return True
    if has_current_text:
        return False
    if "تعدل المادة" in line:
        return False
    return line.startswith(("القانون رقم", "المرسوم رقم", "القرار رقم"))


def _legal_section_title(chapter: str, article_number: str) -> str:
    article_title = f"المادة ({article_number})"
    if chapter:
        return f"{chapter} - {article_title}"
    return article_title


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
        page_end=block.page_end,
        section_title=block.section_title,
        token_count=count_tokens(text),
        chunk_type=block.chunk_type,
        source_type=block.source_type,
        document_number=block.document_number,
        document_year=block.document_year,
        chapter=block.chapter,
        article_number=block.article_number,
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
        page_end=max(block.page_end for block in blocks),
        section_title=section_title,
        token_count=count_tokens(text),
        chunk_type=_first_value(blocks, "chunk_type") or "text",
        source_type=_first_value(blocks, "source_type"),
        document_number=_first_value(blocks, "document_number"),
        document_year=_first_value(blocks, "document_year"),
        chapter=_first_value(blocks, "chapter"),
        article_number=_first_value(blocks, "article_number"),
    )


def _first_value(blocks: list[TextBlock], attribute: str) -> str:
    return next((str(getattr(block, attribute)) for block in blocks if getattr(block, attribute)), "")


def _normalize_article_number(raw: str, previous_article: int | None) -> str:
    candidates = _number_candidates(raw)
    if not candidates:
        return ""
    if previous_article is None:
        selected = candidates[0]
    else:
        expected = previous_article + 1
        selected = min(candidates, key=lambda item: (abs(item - expected), item < previous_article))
    return str(selected)


def _normalize_document_year(raw: str) -> str:
    if not raw:
        return ""
    candidates = _number_candidates(raw)
    plausible = [item for item in candidates if 1900 <= item <= 2100]
    if plausible:
        return str(plausible[0])
    return str(candidates[0]) if candidates else ""


def _normalize_document_number(raw: str, reverse_number: bool = False) -> str:
    digits = _digits_only(raw)
    if not digits:
        return ""
    if reverse_number and _has_arabic_digits(raw) and len(digits) > 1:
        return digits[::-1]
    return digits


def _number_candidates(raw: str) -> list[int]:
    digits = _digits_only(raw)
    if not digits:
        return []
    candidates = [int(digits)]
    if _has_arabic_digits(raw) and len(digits) > 1:
        reversed_digits = digits[::-1]
        if reversed_digits != digits:
            candidates.append(int(reversed_digits))
    return candidates


def _digits_only(raw: str) -> str:
    return re.sub(r"\D+", "", raw.translate(ARABIC_DIGITS))


def _has_arabic_digits(raw: str) -> bool:
    return any("٠" <= char <= "٩" for char in raw)


def _safe_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
