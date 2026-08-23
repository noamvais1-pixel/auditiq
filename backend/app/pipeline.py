"""The request path, in one place.

Ordering is the whole design here:

    redact -> retrieve -> generate -> verify -> audit -> rehydrate

*Redact before retrieve*, not just before generate: the retrieval query is also
text, and with a hosted embedding provider it leaves the perimeter too. Redacting
only at the model boundary would leak client identifiers to the embedding
endpoint -- an easy detail to miss and an awkward one to explain afterwards.

*Rehydrate last*, after the audit record is written, so the log never holds the
raw values it exists to protect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import settings
from .llm import LLMClient, build_user_turn, citation_ids, verify
from .rag import Index, assess_coverage, load_corpus
from .security import AuditLog, redact, sha256_text

log = logging.getLogger(__name__)


@dataclass
class Source:
    chunk_id: str
    citation: str
    text: str
    score: float
    pinned: bool


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    mode: str = "offline"
    model: str = ""
    pii_summary: dict[str, int] = field(default_factory=dict)
    grounding: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    audit_hash: str = ""


class Assistant:
    def __init__(self) -> None:
        self.index = self._load_index()
        self.llm = LLMClient()
        self.audit = AuditLog(settings.audit_log_path)

    @staticmethod
    def _load_index() -> Index:
        if settings.index_path.exists():
            try:
                return Index.load(settings.index_path)
            except ValueError as exc:
                log.warning("%s -- rebuilding in memory", exc)
        log.info("no usable index on disk; building from corpus")
        return Index().build(load_corpus(settings.corpus_dir))

    def ask(self, question: str, session_id: str = "anon", user: str = "anon") -> Answer:
        redaction = (
            redact(question) if settings.redact_pii else type("R", (), {
                "text": question, "findings": [], "summary": lambda: {},
                "rehydrate": staticmethod(lambda t: t)})()
        )
        safe_question = redaction.text

        hits = self.index.search(safe_question, k=settings.top_k)
        coverage = assess_coverage(safe_question, self.index, hits)

        if not coverage.sufficient:
            # Refuse before spending a model call. Layer 1 of the coverage gate;
            # see app/rag/coverage.py for why this is deliberately conservative.
            return self._refuse(question, safe_question, redaction, coverage,
                                session_id, user)

        user_turn = build_user_turn(safe_question, hits)
        gen = self.llm.generate(user_turn, hits)
        cleaned, report = verify(gen.text, citation_ids(hits))

        # Only sources the answer actually leaned on are shown. A citations panel
        # listing every retrieved chunk trains users to ignore it.
        cited = set(report.valid_citations)
        sources = [
            Source(h.chunk.chunk_id, h.chunk.citation, h.chunk.text, round(h.score, 4), h.pinned)
            for h in hits
            if h.chunk.chunk_id in cited
        ] or [
            Source(h.chunk.chunk_id, h.chunk.citation, h.chunk.text, round(h.score, 4), h.pinned)
            for h in hits[:3]
        ]

        record = self.audit.append(
            event="ask",
            session_id=session_id,
            user=user,
            input_sha256=sha256_text(question),
            redacted_prompt=safe_question,
            pii_summary=redaction.summary(),
            source_ids=[h.chunk.chunk_id for h in hits],
            model=gen.model,
            answer_sha256=sha256_text(cleaned),
            meta={
                "mode": gen.mode,
                "coverage": coverage.as_dict(),
                "blocked": report.blocked,
                "invalid_citations": report.invalid_citations,
                "uncited_claims": len(report.uncited_claims),
                "tools_used": [{"name": t["name"], "input": t["input"]} for t in gen.tool_calls],
            },
        )

        return Answer(
            text=redaction.rehydrate(cleaned),
            sources=sources,
            mode=gen.mode,
            model=gen.model,
            pii_summary=redaction.summary(),
            grounding={
                "valid_citations": report.valid_citations,
                "invalid_citations": report.invalid_citations,
                "uncited_claims": report.uncited_claims,
                "blocked": report.blocked,
                "is_refusal": report.is_refusal,
            },
            tool_calls=gen.tool_calls,
            usage=gen.usage,
            audit_hash=record.hash,
            coverage=coverage.as_dict(),
        )

    def _refuse(self, raw_question, safe_question, redaction, coverage, session_id, user) -> Answer:
        text = (
            "לא מצאתי במאגר מקור שמכסה את השאלה הזו, ולכן אני לא עונה עליה.\n\n"
            "המאגר כולל כרגע: מע\"מ, פחת, הכרה בהכנסה, חכירות, צדדים קשורים, "
            "מסים נדחים ומהותיות בביקורת."
        )
        record = self.audit.append(
            event="refuse",
            session_id=session_id,
            user=user,
            input_sha256=sha256_text(raw_question),
            redacted_prompt=safe_question,
            pii_summary=redaction.summary(),
            source_ids=[],
            model="",
            answer_sha256=sha256_text(text),
            meta={"mode": "gate", "coverage": coverage.as_dict()},
        )
        return Answer(
            text=text,
            sources=[],
            mode="gate",
            pii_summary=redaction.summary(),
            grounding={"valid_citations": [], "invalid_citations": [],
                       "uncited_claims": [], "blocked": True, "is_refusal": True},
            coverage=coverage.as_dict(),
            audit_hash=record.hash,
        )
