"""FastAPI surface."""

from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import settings
from .pipeline import Assistant
from .security import redact

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="AuditIQ",
    description="עוזר AI מבוסס RAG לרואי חשבון — תשובות עם ציטוט מקור בלבד.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

assistant = Assistant()


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=8000)
    session_id: str = "anon"
    user: str = "anon"


class RedactRequest(BaseModel):
    text: str = Field(max_length=20000)


@app.get("/api/health")
def health() -> dict:
    ok, broken_at = assistant.audit.verify()
    return {
        "status": "ok",
        "mode": assistant.llm.mode,
        "model": assistant.llm.model if assistant.llm.mode == "live" else None,
        "embedding_provider": assistant.index.provider.name,
        "chunks_indexed": len(assistant.index.chunks),
        "documents": len({c.doc_id for c in assistant.index.chunks}),
        "pii_redaction": settings.redact_pii,
        "audit_chain_valid": ok,
        "audit_chain_broken_at": broken_at,
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    try:
        return asdict(assistant.ask(req.question, req.session_id, req.user))
    except Exception as exc:
        logging.exception("ask failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}") from exc


@app.post("/api/redact/preview")
def redact_preview(req: RedactRequest) -> dict:
    """Lets the UI show what would be stripped *before* the user hits send.

    Users trust a redactor they can watch working far more than one that
    promises to have worked.
    """
    result = redact(req.text)
    return {
        "redacted_text": result.text,
        "summary": result.summary(),
        "findings": [
            {"kind": f.kind, "placeholder": f.placeholder, "start": f.start, "end": f.end}
            for f in result.findings
        ],
    }


@app.get("/api/sources")
def sources() -> dict:
    docs: dict[str, dict] = {}
    for chunk in assistant.index.chunks:
        entry = docs.setdefault(
            chunk.doc_id,
            {
                "doc_id": chunk.doc_id,
                "title": chunk.meta.get("title", chunk.doc_id),
                "doc_type": chunk.meta.get("doc_type"),
                "effective_from": chunk.meta.get("effective_from"),
                "source_note": chunk.meta.get("source_note"),
                "chunks": 0,
            },
        )
        entry["chunks"] += 1
    return {"documents": sorted(docs.values(), key=lambda d: d["title"])}


@app.get("/api/audit")
def audit(limit: int = 50) -> dict:
    records = list(assistant.audit)[-limit:]
    ok, broken_at = assistant.audit.verify()
    return {"chain_valid": ok, "broken_at": broken_at, "records": records}
