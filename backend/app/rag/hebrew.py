"""Hebrew text normalisation for lexical retrieval.

Off-the-shelf tokenisers treat Hebrew as a bag of opaque strings, which quietly
destroys recall in a corpus like this one. Three specifics matter:

* **Final forms.** ``מס`` and ``מ ס`` are fine, but ``הכנסות`` vs ``הכנסה``, and
  more importantly ``נכסים`` written mid-sentence vs ``נכסיםם`` -- any word whose
  last letter is one of ךםןףץ must be folded to its base form or it will not
  match the same word appearing inside a compound.

* **Clitic prefixes.** Hebrew glues ו/ה/ב/ל/כ/מ/ש onto the front of nouns, so a
  query for ``פחת`` misses every occurrence of ``הפחת`` and ``לפחת``. We index
  both the surface form and the stripped form, and only strip when the remainder
  is still a plausible word (>= 3 letters). Stripping blindly turns ``מס`` into
  ``ס``.

* **Number and construct state.** ``מבנה`` and ``מבנים`` are the same noun, and a
  question asking about "מבנה" must reach a table listing rates for "מבנים".
  Likewise ``שיעור`` vs the construct ``שיעורי``. A short suffix list folds both
  to a shared stem. The remainder must stay >= 3 letters, which is what stops
  ``פחת`` from being stemmed to ``פח``.

* **Niqqud and gershayim.** Legal and accounting Hebrew is full of ר"ח, בע"מ,
  מע"מ. The quote marks are part of the token, not punctuation, so they survive
  tokenisation -- but we also index the de-quoted form so ``מעמ`` finds ``מע"מ``.

This is a heuristic, not a morphological analyser. It is the right size for the
job: no model to download, no license to clear, and it recovers most of the
recall a real analyser would.
"""

from __future__ import annotations

import re
import unicodedata

FINAL_FORMS = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})
CLITICS = ("ו", "ה", "ב", "ל", "כ", "מ", "ש", "כש", "מש", "לכש")

# Longest first: "מהותיות" must lose "יות", not "ות" then strand a "י".
# Note these apply to already-folded text, so plural "ים" appears here as "ימ".
SUFFIXES = ("יות", "ימ", "ות", "ית", "ינ", "ה", "ת", "י")
MIN_STEM = 3
_NIQQUD = re.compile(r"[֑-ׇ]")
_GERSH = re.compile(r"[׳״\"']")
_TOKEN = re.compile(r"[A-Za-z0-9֐-׿][A-Za-z0-9֐-׿\"'׳״.\-]*")

# Words that are common enough to add noise instead of signal.
STOPWORDS = {
    "של", "על", "את", "אם", "או", "גם", "כל", "לא", "כי", "זה", "היא", "הוא",
    "אשר", "עם", "אך", "אין", "יש", "כפי", "לפי", "בין", "אלא", "רק", "בו",
    "בה", "הם", "הן", "אנו", "היה", "הייתה", "יהיה", "ניתן", "אפשר", "וכן",
}


def strip_diacritics(text: str) -> str:
    return _NIQQUD.sub("", unicodedata.normalize("NFC", text))


def normalise(word: str) -> str:
    """Fold a single surface word to its index key."""
    w = strip_diacritics(word).lower().strip(".,;:()[]{}־-")
    return w.translate(FINAL_FORMS)


def _stem(word: str) -> str | None:
    """Fold number and construct-state suffixes. Heuristic, deliberately shallow."""
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= MIN_STEM:
            return word[: -len(suf)]
    return None


def _strip_clitic(word: str) -> str | None:
    for c in sorted(CLITICS, key=len, reverse=True):
        if word.startswith(c) and len(word) - len(c) >= 3:
            return word[len(c) :]
    return None


def tokenize(text: str, expand: bool = True) -> list[str]:
    """Tokenise into index keys.

    ``expand`` adds morphological variants (clitic-stripped, de-quoted). We use
    it when indexing *and* when querying, so a query written one way still hits a
    document written the other way.
    """
    tokens: list[str] = []
    for raw in _TOKEN.findall(text):
        base = normalise(raw)
        if not base or base in STOPWORDS:
            continue
        tokens.append(base)
        if not expand:
            continue
        dequoted = _GERSH.sub("", base)
        if dequoted != base and len(dequoted) >= 2:
            tokens.append(dequoted)
        stripped = _strip_clitic(base)
        if stripped and stripped not in STOPWORDS:
            tokens.append(stripped)
        for form in (base, stripped):
            stem = _stem(form) if form else None
            if stem and stem not in STOPWORDS:
                tokens.append(stem)
    return tokens
