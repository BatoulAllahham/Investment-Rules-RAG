from __future__ import annotations

import re
from collections import Counter

from .pdf_loader import PageText


_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_page_texts(pages: list[PageText]) -> list[PageText]:
    repeated_lines = _find_repeated_lines(pages)
    cleaned: list[PageText] = []

    for page in pages:
        lines = []
        for raw_line in page.text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = _WHITESPACE_RE.sub(" ", raw_line.replace("\u00a0", " ")).strip()
            if not line or line in repeated_lines:
                continue
            lines.append(line)

        text = "\n".join(lines)
        text = _BLANK_LINES_RE.sub("\n\n", text).strip()
        cleaned.append(PageText(page_number=page.page_number, text=text))

    return cleaned


def _find_repeated_lines(pages: list[PageText]) -> set[str]:
    counts: Counter[str] = Counter()
    for page in pages:
        page_lines = {
            _WHITESPACE_RE.sub(" ", line.replace("\u00a0", " ")).strip()
            for line in page.text.splitlines()
        }
        counts.update(line for line in page_lines if 4 <= len(line) <= 120)

    threshold = max(4, int(len(pages) * 0.25))
    return {line for line, count in counts.items() if count >= threshold}

