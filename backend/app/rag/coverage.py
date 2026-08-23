"""Does the corpus actually cover this question?

Answering a question the corpus does not cover is the worst failure this system
has, so it is worth being precise about how it is prevented.

The obvious approach -- threshold the retrieval score -- does not work, and the
eval suite is what showed that. RRF scores are rank-derived and therefore
near-constant. Raw BM25, cosine and query-term-overlap were each measured across
the eval set (see docs/EVALUATION.md): on every one of them, the out-of-scope
question "מה שיעור דמי הביטוח הלאומי למעסיק" scores *inside* the range of
legitimate in-scope questions. There is no threshold that rejects it without
also rejecting real questions, because on a small corpus a generic term like
"שיעור" carries enough lexical mass to look like a match.

So coverage is enforced in two layers, and they are kept distinct because they
have different costs and different error profiles:

**Layer 1, here: a lexical pre-filter.** Deliberately conservative. It fires only
on questions that share essentially nothing with the corpus, and refuses them
without a model call at all -- faster and cheaper.

The thresholds are set below where the data says they could be, on purpose.
Measured over the eval set after Hebrew stemming landed, in-scope questions
bottom out at BM25 mass 10.1 and 0.69 unseen-term rate, while the worst
out-of-scope case sits at 6.7 and 0.75. A threshold at 8.0 would therefore score
a perfect refusal rate on this set. It is not used, because a boundary fitted to
three negative examples is overfitting, and the cost of the two error types is
wildly asymmetric: a missed refusal is caught downstream by the model gate and
the citation guardrail, whereas a false refusal makes the tool look broken to an
accountant who asked a perfectly good question, and they stop using it.

So the thresholds sit ~2.5x clear of the worst legitimate question. Layer 1
catches only the unambiguous cases; anything arguable is the model's call.

**Layer 2, the model.** The semantic judgement -- "these clauses are about
leases, the question is about payroll tax" -- needs to read the text, so it is
the model's job, enforced after the fact by the citation guardrail. An answer
that cites nothing valid is downgraded to a refusal regardless of what it says.

The eval suite reports the two layers separately rather than blending them, so a
regression in either is visible.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hebrew import tokenize
from .retriever import _PLACEHOLDER

# See the module docstring: these are deliberately *not* the separating values.
MAX_OOV_RATE = 0.90      # worst legitimate question measured at 0.69
MIN_LEXICAL_MASS = 4.0   # worst legitimate question measured at 10.1


@dataclass
class Coverage:
    sufficient: bool
    reason: str
    oov_rate: float
    lexical_mass: float

    def as_dict(self) -> dict:
        return {
            "sufficient": self.sufficient,
            "reason": self.reason,
            "oov_rate": round(self.oov_rate, 3),
            "lexical_mass": round(self.lexical_mass, 3),
        }


def assess(query: str, index, hits) -> Coverage:
    if not hits:
        return Coverage(False, "no_hits", 1.0, 0.0)

    query = _PLACEHOLDER.sub(" ", query)
    terms = tokenize(query)
    if not terms:
        return Coverage(False, "empty_query", 1.0, 0.0)

    unseen = sum(1 for t in terms if index.bm25.doc_freq.get(t, 0) == 0)
    oov_rate = unseen / len(terms)
    lexical_mass = max(index.bm25.score(query), default=0.0)

    if lexical_mass <= MIN_LEXICAL_MASS:
        return Coverage(False, "no_lexical_overlap", oov_rate, lexical_mass)
    if oov_rate >= MAX_OOV_RATE:
        return Coverage(False, "vocabulary_not_in_corpus", oov_rate, lexical_mass)

    return Coverage(True, "ok", oov_rate, lexical_mass)
