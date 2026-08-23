"""Detection and redaction of Israeli PII before text is sent to an external LLM.

Design notes
------------
The firm's users paste raw client material into the chat (trial balances, emails
from clients, ledger excerpts). That material routinely contains תעודת זהות,
ח.פ, bank details and phone numbers. None of it is needed to answer a question
about accounting treatment, so it never leaves the perimeter.

Two properties matter more than raw recall here:

1. *Reversibility*. We replace each finding with a stable placeholder such as
   ``[ת.ז-1]`` and keep the mapping in memory for the duration of the request.
   The model reasons over the placeholder; we re-hydrate the answer afterwards so
   the user still sees their own data. Hashing or dropping the value would make
   the assistant unusable for anything document-shaped.

2. *Precision*. A redactor that fires on every 9-digit number destroys invoice
   numbers and journal references, users stop trusting it, and they work around
   it by disabling it. So identifiers that carry a check digit are validated
   against it: Israeli ID and ח.פ use the Luhn-style ניכוי ביקורת, credit cards
   use Luhn proper. This trades a little recall for a lot of trust.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

# Hebrew label used in the placeholder, so the model still knows *what kind* of
# entity it is looking at and can reason about it ("the ID belongs to...").
_LABELS = {
    "israeli_id": "ת.ז",
    "company_id": "ח.פ",
    "phone": "טלפון",
    "email": "אימייל",
    "bank_account": "חשבון-בנק",
    "credit_card": "כרטיס-אשראי",
}


def _luhn_israeli(digits: str) -> bool:
    """Israeli ID / ח.פ check digit (ספרת ביקורת).

    Each digit is multiplied alternately by 1 and 2; products above 9 have their
    own digits summed. The total must be divisible by 10.
    """
    if len(digits) != 9 or not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(digits):
        val = int(ch) * (1 if i % 2 == 0 else 2)
        total += val if val < 10 else val - 9
    return total % 10 == 0


def _luhn_card(digits: str) -> bool:
    if not (13 <= len(digits) <= 19) or not digits.isdigit():
        return False
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        val = int(ch)
        if i % 2 == parity:
            val *= 2
            if val > 9:
                val -= 9
        total += val
    return total % 10 == 0


def _valid_israeli_phone(digits: str) -> bool:
    """05x / 0[2-4,8-9] landline, optionally written with +972."""
    d = digits
    if d.startswith("972"):
        d = "0" + d[3:]
    if not d.startswith("0"):
        return False
    return len(d) in (9, 10)


@dataclass(frozen=True)
class Rule:
    kind: str
    pattern: re.Pattern[str]
    # Returns True if the raw match really is an entity of this kind.
    validate: Callable[[str], bool] = lambda _: True
    # Extracts the digits/text to validate, defaults to the whole match.
    normalize: Callable[[str], str] = lambda s: re.sub(r"\D", "", s)


# Order matters: the first rule to claim a span wins, so put the
# check-digit-validated, high-precision rules first.
RULES: tuple[Rule, ...] = (
    Rule(
        kind="email",
        pattern=re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
        normalize=lambda s: s,
    ),
    Rule(
        kind="credit_card",
        pattern=re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
        validate=_luhn_card,
    ),
    Rule(
        # ח.פ is 9 digits starting with 5 (חברה בע"מ) or 51/52.
        kind="company_id",
        pattern=re.compile(r"(?<!\d)5\d{8}(?!\d)"),
        validate=_luhn_israeli,
    ),
    Rule(
        kind="israeli_id",
        pattern=re.compile(r"(?<!\d)\d{9}(?!\d)"),
        validate=_luhn_israeli,
    ),
    Rule(
        kind="bank_account",
        # "בנק 12 סניף 345 חשבון 123456" and IBAN IL##...
        pattern=re.compile(r"\bIL\d{2}[ ]?(?:\d[ ]?){17,21}\b", re.IGNORECASE),
        normalize=lambda s: s,
    ),
    Rule(
        kind="phone",
        pattern=re.compile(r"(?:\+?972[\- ]?|\b0)(?:\d[\- ]?){8,9}\b"),
        validate=_valid_israeli_phone,
    ),
)


@dataclass
class Finding:
    kind: str
    value: str
    start: int
    end: int
    placeholder: str


@dataclass
class RedactionResult:
    text: str
    findings: list[Finding] = field(default_factory=list)
    _map: dict[str, str] = field(default_factory=dict)

    @property
    def redacted(self) -> bool:
        return bool(self.findings)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.kind] = counts.get(f.kind, 0) + 1
        return counts

    def rehydrate(self, text: str) -> str:
        """Put the user's real values back into a model-produced answer."""
        for placeholder, value in self._map.items():
            text = text.replace(placeholder, value)
        return text


def _overlaps(start: int, end: int, taken: Iterable[tuple[int, int]]) -> bool:
    return any(start < t_end and t_start < end for t_start, t_end in taken)


def redact(text: str, rules: tuple[Rule, ...] = RULES) -> RedactionResult:
    """Replace PII with stable, reversible placeholders.

    The same value seen twice gets the same placeholder, so the model can still
    tell that two mentions refer to one person.
    """
    findings: list[Finding] = []
    taken: list[tuple[int, int]] = []
    counters: dict[str, int] = {}
    by_value: dict[tuple[str, str], str] = {}

    for rule in rules:
        for m in rule.pattern.finditer(text):
            raw = m.group(0)
            if _overlaps(m.start(), m.end(), taken):
                continue
            if not rule.validate(rule.normalize(raw)):
                continue
            key = (rule.kind, rule.normalize(raw))
            if key in by_value:
                placeholder = by_value[key]
            else:
                counters[rule.kind] = counters.get(rule.kind, 0) + 1
                placeholder = f"[{_LABELS[rule.kind]}-{counters[rule.kind]}]"
                by_value[key] = placeholder
            taken.append((m.start(), m.end()))
            findings.append(
                Finding(rule.kind, raw, m.start(), m.end(), placeholder)
            )

    findings.sort(key=lambda f: f.start)

    out, cursor = [], 0
    for f in findings:
        out.append(text[cursor : f.start])
        out.append(f.placeholder)
        cursor = f.end
    out.append(text[cursor:])

    return RedactionResult(
        text="".join(out),
        findings=findings,
        _map={f.placeholder: f.value for f in findings},
    )
