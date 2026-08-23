"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion.

RRF rather than a weighted score sum, because BM25 scores and cosine
similarities live on incomparable scales; any fixed weight you pick is tuned to
one corpus and silently wrong on the next. RRF only uses *rank*, so it needs no
calibration -- and in practice it is within a point or two of a tuned blend.

After fusion we apply one domain rule that pure ranking cannot express: if the
query names a specific identifier (a standard number, a section, a form), chunks
carrying that identifier are pinned to the top. An accountant asking about
"תקן 27" wants תקן 27, and a semantically-adjacent passage about תקן 29 is not a
near miss -- it is a wrong answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25
from .chunker import Chunk
from .embeddings import EmbeddingProvider, cosine, get_provider

RRF_K = 60
# Redaction placeholders such as "[ת.ז-1]" are not search terms. Left in, they
# pull every chunk that happens to mention a phone number or an ID to the top --
# found by driving the real UI with a question that contained both.
_PLACEHOLDER = re.compile(r"\[[^\[\]]{1,24}-\d+\]")
_IDENTIFIER = re.compile(r"(?:תקן|תקנה|סעיף|טופס|חוזר|iAS|IFRS|IAS)\s*[-']?\s*(\d+[א-ת]?)", re.IGNORECASE)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    pinned: bool = False


class Index:
    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self.provider = provider or get_provider()
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self.bm25 = BM25()

    # ---------------------------------------------------------------- build
    def build(self, chunks: list[Chunk]) -> "Index":
        self.chunks = chunks
        texts = [c.embed_text for c in chunks]
        self.bm25.fit(texts)
        self.vectors = self.provider.embed(texts, task="document") if texts else []
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": self.provider.name,
            "dim": self.provider.dim,
            "chunks": [
                {
                    "doc_id": c.doc_id,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "breadcrumb": c.breadcrumb,
                    "clause": c.clause,
                    "meta": c.meta,
                }
                for c in self.chunks
            ],
            "vectors": self.vectors,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, provider: EmbeddingProvider | None = None) -> "Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = cls(provider)
        if payload.get("provider") != idx.provider.name:
            raise ValueError(
                f"index was built with embedding provider {payload.get('provider')!r} "
                f"but {idx.provider.name!r} is configured; re-run `make ingest`"
            )
        idx.chunks = [Chunk(**c) for c in payload["chunks"]]
        idx.vectors = payload["vectors"]
        idx.bm25.fit([c.embed_text for c in idx.chunks])
        return idx

    # ------------------------------------------------------------- retrieve
    def search(self, query: str, k: int = 6, candidates: int = 30) -> list[Hit]:
        if not self.chunks:
            return []
        query = _PLACEHOLDER.sub(" ", query)

        lexical = self.bm25.score(query)
        lex_order = sorted(range(len(lexical)), key=lambda i: -lexical[i])[:candidates]

        q_vec = self.provider.embed([query], task="query")[0]
        dense = [cosine(q_vec, v) for v in self.vectors]
        dense_order = sorted(range(len(dense)), key=lambda i: -dense[i])[:candidates]

        lex_rank = {doc: r for r, doc in enumerate(lex_order)}
        dense_rank = {doc: r for r, doc in enumerate(dense_order)}

        w_dense = getattr(self.provider, "fusion_weight", 1.0)
        fused: dict[int, float] = {}
        for doc in set(lex_order) | set(dense_order):
            score = 0.0
            if doc in lex_rank:
                score += 1 / (RRF_K + lex_rank[doc] + 1)
            if doc in dense_rank:
                score += w_dense / (RRF_K + dense_rank[doc] + 1)
            fused[doc] = score

        wanted = {m.group(1) for m in _IDENTIFIER.finditer(query)}
        pinned: set[int] = set()
        if wanted:
            for doc, chunk in enumerate(self.chunks):
                haystack = f"{chunk.meta.get('title','')} {' '.join(chunk.breadcrumb)} {chunk.text}"
                if any(_mentions(haystack, ident) for ident in wanted):
                    pinned.add(doc)
                    fused[doc] = fused.get(doc, 0.0) + 1.0

        ordered = sorted(fused, key=lambda d: -fused[d])[:k]
        return [
            Hit(
                chunk=self.chunks[d],
                score=fused[d],
                bm25_rank=lex_rank.get(d),
                dense_rank=dense_rank.get(d),
                pinned=d in pinned,
            )
            for d in ordered
        ]


def _mentions(haystack: str, identifier: str) -> bool:
    """Word-boundary match so "תקן 2" does not match "תקן 27"."""
    return re.search(rf"(?<!\d){re.escape(identifier)}(?!\d)", haystack) is not None


def load_corpus(corpus_dir: str | Path) -> list[Chunk]:
    """Read the markdown corpus, honouring a simple YAML-ish front matter."""
    from .chunker import chunk_document

    chunks: list[Chunk] = []
    for md in sorted(Path(corpus_dir).glob("*.md")):
        raw = md.read_text(encoding="utf-8")
        meta: dict = {}
        body = raw
        if raw.startswith("---"):
            _, fm, body = raw.split("---", 2)
            for line in fm.strip().splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    meta[key.strip()] = val.strip()
        chunks.extend(chunk_document(md.stem, body.strip(), meta))
    return chunks
