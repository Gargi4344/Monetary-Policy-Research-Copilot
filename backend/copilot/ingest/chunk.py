"""Split documents into ~500-token chunks with 15% overlap, by heading and paragraph.

Paragraphs are kept whole where possible; a heading (a short line without final
punctuation) starts a new section and is repeated at the top of each chunk of that
section, so a chunk read on its own still says what it is about. Token counts use
~0.75 words per token, which is close enough for the bge tokenizer on English prose.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from copilot.config import get_settings
from copilot.ingest.scrape import Doc

WORDS_PER_TOKEN = 0.75


@dataclass
class Chunk:
    """A passage plus the metadata of its document."""

    chunk_id: str
    doc_id: str
    url: str
    doc_type: str
    title: str
    date: str
    section: str
    text: str
    position: int


def is_heading(line: str) -> bool:
    """A short line with no sentence-ending punctuation looks like a heading."""
    s = line.strip()
    return 0 < len(s.split()) <= 10 and not re.search(r"[.!?:;,]$", s) and not s[0].islower()


def sections(text: str) -> list[tuple[str, list[str]]]:
    """Group paragraphs under their nearest heading."""
    out: list[tuple[str, list[str]]] = [("", [])]
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if is_heading(line) and out[-1][1]:
            out.append((line, []))
        elif is_heading(line) and not out[-1][1]:
            out[-1] = (line, [])
        else:
            out[-1][1].append(line)
    return [(h, ps) for h, ps in out if ps]


def _words(n_tokens: int) -> int:
    return max(20, int(n_tokens * WORDS_PER_TOKEN))


DOC_TYPE_LABEL = {
    "announcement": "interest rate announcement",
    "opening_statement": "opening statement",
    "deliberations": "summary of Governing Council deliberations",
    "mpr": "Monetary Policy Report",
}


def context_header(doc: Doc) -> str:
    """'Title — 5 June 2024 (interest rate announcement)'.

    The date is written in words so questions such as "in June 2024" match lexically and
    semantically; without it a chunk only carries the date as metadata.
    """
    from datetime import date

    d = date.fromisoformat(doc.date)
    return f"{doc.title} — {d.day} {d:%B %Y} ({DOC_TYPE_LABEL.get(doc.doc_type, doc.doc_type)})"


def chunk_doc(doc: Doc, tokens: int = 500, overlap: float = 0.15) -> list[Chunk]:
    """Chunk one document; consecutive chunks of a section share ~`overlap` of their words."""
    size = _words(tokens)
    keep = int(size * overlap)
    chunks: list[Chunk] = []

    header = context_header(doc)

    def emit(section: str, words: list[str]) -> None:
        body = " ".join(words)
        prefix = f"{header}\n" + (f"{section}\n" if section else "")
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}-{len(chunks):03d}", doc_id=doc.doc_id, url=doc.url,
                            doc_type=doc.doc_type, title=doc.title, date=doc.date, section=section,
                            text=prefix + body, position=len(chunks)))  # fmt: skip

    for section, paragraphs in sections(doc.text):
        buf: list[str] = []
        for para in paragraphs:
            words = para.split()
            if buf and len(buf) + len(words) > size:
                emit(section, buf)
                buf = buf[-keep:] if keep else []
            # A single paragraph longer than the window is split on its own.
            while len(words) > size:
                take = size - len(buf)
                emit(section, buf + words[:take])
                buf = (buf + words[:take])[-keep:] if keep else []
                words = words[take:]
            buf += words
        if buf and (not chunks or chunks[-1].section != section or len(buf) > keep):
            emit(section, buf)
    return chunks


def chunk_all(docs: list[Doc]) -> list[Chunk]:
    """Chunk every document and write data/chunks.jsonl."""
    s = get_settings()
    out = [c for d in docs for c in chunk_doc(d, s.chunk_tokens, s.chunk_overlap)]
    with s.chunks_path.open("w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return out


def load_chunks(path: Path | None = None) -> list[Chunk]:
    """Read data/chunks.jsonl."""
    path = path or get_settings().chunks_path
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
