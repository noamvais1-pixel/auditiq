"""Structure-aware chunking for regulatory Hebrew documents.

A fixed 800-character window is the wrong tool here. These documents are written
as numbered hierarchies -- ``17. הגדרות`` / ``17(א)`` / ``17(א)(2)`` -- and the
number *is* how an accountant cites the answer. Splitting mid-clause produces
chunks that are individually unciteable, which then makes the citation guardrail
reject perfectly good answers.

So we split on heading and clause boundaries first, and only fall back to size
based splitting inside an over-long clause. Each chunk keeps a breadcrumb of the
headings above it, which is prepended to the embedded text -- a bare clause like
"שיעור זה יחול גם על נכס שנרכש בשנת המס" is meaningless without knowing which
section it sits in, both to the embedding model and to the reader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

# Clause labels as they actually appear in Hebrew regulatory text:
#   "4."   "4)"   "4(א)."   "4(א)(2)"   "(א)"   "(2)"
# The trailing separator is optional and may be "." or ")", because
# "4(א). לישות" and "4(א) לישות" are both in use -- an earlier version of this
# pattern required whitespace immediately after the paren, which silently
# swallowed every sub-clause into the preceding chunk and left it uncitable.
_CLAUSE = re.compile(
    r"^\s*("
    r"\d+(?:\([א-ת0-9]+\))+"   # 4(א), 17(א)(2)
    r"|\d+"                    # 4
    r"|\([א-ת0-9]+\)"          # (א), (2)
    r")[.)]?\s+"
)

MAX_CHARS = 1100
OVERLAP = 150


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    breadcrumb: list[str] = field(default_factory=list)
    clause: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Human-readable source label shown next to the answer."""
        parts = [self.meta.get("title", self.doc_id)]
        if self.breadcrumb:
            parts.append(" › ".join(self.breadcrumb))
        if self.clause:
            parts.append(f"סעיף {self.clause}")
        return " · ".join(parts)

    @property
    def embed_text(self) -> str:
        prefix = " › ".join([self.meta.get("title", "")] + self.breadcrumb).strip(" ›")
        return f"{prefix}\n{self.text}" if prefix else self.text


def _split_long(text: str) -> list[str]:
    """Last-resort splitter: prefer sentence ends, keep a small overlap."""
    if len(text) <= MAX_CHARS:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            window = text.rfind(". ", start + MAX_CHARS // 2, end)
            if window == -1:
                window = text.rfind(" ", start + MAX_CHARS // 2, end)
            if window != -1:
                end = window + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)
    return [p for p in parts if p]


def chunk_document(doc_id: str, text: str, meta: dict | None = None) -> list[Chunk]:
    meta = meta or {}
    breadcrumb: list[str] = []
    chunks: list[Chunk] = []
    buffer: list[str] = []
    current_clause: str | None = None
    seq = 0

    def flush() -> None:
        nonlocal buffer, seq
        body = "\n".join(buffer).strip()
        buffer = []
        if not body:
            return
        for piece in _split_long(body):
            seq += 1
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{seq}",
                    text=piece,
                    breadcrumb=list(breadcrumb),
                    clause=current_clause,
                    meta=meta,
                )
            )

    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            breadcrumb = breadcrumb[: level - 1] + [title]
            current_clause = None
            continue

        clause = _CLAUSE.match(line)
        if clause and buffer:
            flush()
        if clause:
            # group(1) excludes the trailing separator, so no stripping is needed --
            # rstrip(".)") here would eat the closing paren of "4(א)".
            current_clause = clause.group(1)
        buffer.append(line)

    flush()
    return chunks
