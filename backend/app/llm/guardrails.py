"""Post-hoc verification that the answer is actually grounded.

Instructing a model to cite its sources gets you citations. It does not get you
*correct* citations -- the common failure is a well-formed id that was never in
the context, which is indistinguishable from a real one to anyone reading
quickly. That is the specific hallucination this domain cannot tolerate, so it
is checked mechanically after generation rather than merely requested before it.

Three checks, in increasing strictness:

* **Well-formedness** -- every ``[[id]]`` in the answer must be an id that was
  actually placed in the context. Unknown ids are stripped and reported.
* **Presence** -- a substantive answer with zero valid citations is downgraded to
  a refusal, because it is by construction ungrounded.
* **Coverage** -- paragraphs containing a factual claim but no citation are
  flagged. This one warns rather than blocks: it is a heuristic, and blocking on
  a heuristic teaches users to route around the tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION = re.compile(r"\[\[([^\[\]]+?)\]\]")

# Phrases that mark the model correctly declining -- these must not be
# downgraded for lacking citations, or the refusal path becomes unusable.
_REFUSAL_MARKERS = (
    "לא נמצא", "אין במקורות", "לא מכסים", "לא מכוסה", "אין לי מקור",
    "המקורות שסופקו אינם", "לא ניתן לענות", "חסר מידע",
)

_CLAIM_MARKERS = re.compile(
    r"(יש ל|חובה|נדרש|אין ל|ניתן ל|שיעור|אחוז|%|מחויב|תוכר|יוכר|תיזקף|נמדד|מתחיל)"
)

REFUSAL_TEXT = (
    "לא מצאתי במאגר מקור שמכסה את השאלה הזו, ולכן אני לא עונה עליה.\n\n"
    "אפשר לנסות לנסח מחדש עם מונח מקצועי מדויק יותר, או לפנות למקור הרשמי."
)


@dataclass
class GroundingReport:
    valid_citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)
    is_refusal: bool = False
    blocked: bool = False

    @property
    def ok(self) -> bool:
        return not self.blocked and not self.invalid_citations


def verify(answer: str, allowed_ids: set[str]) -> tuple[str, GroundingReport]:
    """Returns the cleaned answer and a report. May replace the answer entirely."""
    report = GroundingReport()

    found = CITATION.findall(answer)
    for cid in found:
        cid = cid.strip()
        (report.valid_citations if cid in allowed_ids else report.invalid_citations).append(cid)

    # Drop fabricated ids rather than showing the user a citation that goes
    # nowhere. The report keeps them so the failure stays visible in the logs.
    cleaned = answer
    for bad in set(report.invalid_citations):
        cleaned = cleaned.replace(f"[[{bad}]]", "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    report.is_refusal = any(m in cleaned for m in _REFUSAL_MARKERS)

    if not report.valid_citations and not report.is_refusal and len(cleaned) > 120:
        report.blocked = True
        return REFUSAL_TEXT, report

    if not report.is_refusal:
        for para in (p.strip() for p in cleaned.split("\n\n")):
            if len(para) > 80 and _CLAIM_MARKERS.search(para) and not CITATION.search(para):
                report.uncited_claims.append(para[:120])

    return cleaned, report


def citation_ids(hits) -> set[str]:
    return {h.chunk.chunk_id for h in hits}
