"""BM25 over the Hebrew-normalised token stream.

Why lexical retrieval at all, when embeddings exist: in this corpus the user's
question very often contains the exact string that identifies the answer -- a
standard number ("תקן 27"), a section ("סעיף 17(א)"), a form number ("טופס 106").
Dense retrieval is notoriously weak on those, because a number carries almost no
semantic signal. BM25 nails them. The two are complementary, which is why the
retriever fuses both rather than picking one.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .hebrew import tokenize


@dataclass
class BM25:
    k1: float = 1.5
    b: float = 0.75
    doc_tokens: list[list[str]] = field(default_factory=list)
    doc_freq: Counter[str] = field(default_factory=Counter)
    doc_len: list[int] = field(default_factory=list)
    avg_len: float = 0.0

    def fit(self, documents: list[str]) -> "BM25":
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.doc_freq = Counter()
        for toks in self.doc_tokens:
            self.doc_freq.update(set(toks))
        return self

    def _idf(self, term: str) -> float:
        n = len(self.doc_tokens)
        df = self.doc_freq.get(term, 0)
        # BM25+ style flooring keeps very common terms from going negative.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        scores = [0.0] * len(self.doc_tokens)
        for i, toks in enumerate(self.doc_tokens):
            if not toks:
                continue
            tf = Counter(toks)
            dl = self.doc_len[i]
            total = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
                total += self._idf(term) * (f * (self.k1 + 1)) / denom
            scores[i] = total
        return scores
