"""Pluggable embedding providers, with an offline fallback.

Reviewers should be able to `git clone && make demo` and get a working system
without an API key, so the default provider is a local, deterministic one. It is
a character n-gram hashing vectoriser -- essentially lexical, not semantic. That
is an honest limitation, not a hidden one: `GET /api/health` reports which
provider is live, and the eval suite records scores per provider so the gap is
measurable rather than assumed.

With `EMBEDDING_PROVIDER=voyage` (Anthropic's recommended embedding partner) or
`openai`, the same interface returns real semantic vectors and recall on
paraphrased questions improves markedly -- see docs/EVALUATION.md.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol, Sequence

from .hebrew import normalise, strip_diacritics


class EmbeddingProvider(Protocol):
    name: str
    dim: int
    # Weight this provider's ranking carries in hybrid fusion. A provider that is
    # not actually semantic should not get an equal vote against BM25 -- see
    # retriever.Index.search and docs/EVALUATION.md for the measurement.
    fusion_weight: float

    # ``task`` is "document" when indexing and "query" when searching. Providers
    # that support asymmetric embeddings use it; the rest ignore it. Getting this
    # wrong costs real recall on retrieval models trained for the distinction,
    # and it is invisible when it is wrong -- nothing errors, results just get
    # slightly worse.
    def embed(self, texts: Sequence[str], task: str = "document") -> list[list[float]]: ...


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashingEmbeddings:
    """Offline fallback: hashed character 3-grams + whole words.

    Character n-grams give partial credit for Hebrew inflection (הפחת/פחת share
    most of their trigrams) which a pure bag-of-words vector would miss.
    """

    name = "local-hashing"
    # Zero, and that is a measurement, not an oversight. Swept over the eval set
    # (docs/EVALUATION.md): recall@6 / MRR go 0.959/0.747 at weight 0, 0.924/0.728
    # at 0.3, and 0.890/0.691 at 1.0 -- monotonically worse the more this provider
    # is trusted. It is a lexical vectoriser, so it agrees with BM25 on the easy
    # questions and adds noise on the hard ones, which is the worst possible
    # profile for rank fusion. The dense side is still computed and its ranks are
    # still reported, so the offline path exercises the same code as the live one;
    # it just does not get a vote it has not earned.
    fusion_weight = 0.0

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        clean = strip_diacritics(text)
        words = [normalise(w) for w in clean.split() if w]
        feats = list(words)
        for w in words:
            padded = f"^{w}$"
            feats += [padded[i : i + 3] for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: Sequence[str], task: str = "document") -> list[list[float]]:
        del task  # symmetric: this vectoriser has no notion of query vs document
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for feat in self._features(text):
                h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if h[4] % 2 == 0 else -1.0
                vec[idx] += sign
            out.append(_l2(vec))
        return out


class VoyageEmbeddings:
    name = "voyage"
    fusion_weight = 1.0

    def __init__(self, model: str = "voyage-3", dim: int = 1024) -> None:
        import voyageai  # imported lazily so the offline path needs no dep

        self.client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self.model = model
        self.dim = dim

    def embed(self, texts: Sequence[str], task: str = "document") -> list[list[float]]:
        resp = self.client.embed(
            list(texts), model=self.model, input_type="query" if task == "query" else "document"
        )
        return [_l2(e) for e in resp.embeddings]


class OpenAIEmbeddings:
    name = "openai"
    fusion_weight = 1.0

    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.dim = dim

    def embed(self, texts: Sequence[str], task: str = "document") -> list[list[float]]:
        del task  # OpenAI embeddings are symmetric
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        return [_l2(d.embedding) for d in resp.data]


class GeminiEmbeddings:
    """Gemini embeddings, used asymmetrically.

    The model is trained with distinct task types for the two sides of a search:
    a stored clause is embedded as RETRIEVAL_DOCUMENT, the user's question as
    RETRIEVAL_QUERY. Using one type for both still "works" -- it just quietly
    retrieves worse, which is the kind of bug that never surfaces as an error.

    Requests are batched because the endpoint caps how many inputs it accepts per
    call, and a 400 halfway through an ingest is an annoying way to find that out.
    """

    name = "gemini"
    fusion_weight = 1.0
    BATCH = 64

    _TASK = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}

    def __init__(self, model: str | None = None, dim: int = 3072) -> None:
        from google import genai

        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = model or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.dim = dim

    def embed(self, texts: Sequence[str], task: str = "document") -> list[list[float]]:
        from google.genai import types

        config = types.EmbedContentConfig(task_type=self._TASK.get(task, "RETRIEVAL_DOCUMENT"))
        out: list[list[float]] = []
        batch: list[str] = list(texts)
        for i in range(0, len(batch), self.BATCH):
            resp = self.client.models.embed_content(
                model=self.model, contents=batch[i : i + self.BATCH], config=config
            )
            out.extend(_l2(list(e.values)) for e in resp.embeddings)
        if out:
            self.dim = len(out[0])
        return out


def get_provider(name: str | None = None) -> EmbeddingProvider:
    name = (name or os.getenv("EMBEDDING_PROVIDER") or "local").lower()
    try:
        if name == "gemini":
            return GeminiEmbeddings()
        if name == "voyage":
            return VoyageEmbeddings()
        if name == "openai":
            return OpenAIEmbeddings()
    except Exception as exc:  # missing dep or key -- degrade loudly, not silently
        import logging

        logging.getLogger(__name__).warning(
            "embedding provider %r unavailable (%s); falling back to local", name, exc
        )
    return HashingEmbeddings()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are L2-normalised
