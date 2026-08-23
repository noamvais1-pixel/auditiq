"""Tamper-evident audit trail.

An audit firm has to be able to answer "what did the assistant tell this client,
and on what basis, on 14 March?" long after the fact. A plain log file answers
that only if you trust that nobody edited it.

Each record therefore carries the SHA-256 of the previous record. Changing or
deleting any past line breaks the chain from that point on, and ``verify()``
reports exactly where. This is deliberately not a blockchain -- it is the
cheapest construction that makes silent edits detectable, which is the actual
requirement.

What is *not* stored: the user's raw text. Only the redacted prompt, the
retrieved source ids, and a hash of the raw input. That way the log can prove
which sources an answer rested on without becoming a second copy of the client
data it was meant to protect.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

GENESIS = "0" * 64


def _sha(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class AuditRecord:
    ts: str
    event: str
    session_id: str
    user: str
    # Hash of the raw user input -- lets us prove "this exact question was asked"
    # without retaining the question's PII.
    input_sha256: str
    redacted_prompt: str
    pii_summary: dict[str, int] = field(default_factory=dict)
    source_ids: list[str] = field(default_factory=list)
    model: str = ""
    answer_sha256: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    prev_hash: str = GENESIS
    hash: str = ""

    def compute_hash(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k != "hash"}
        return _sha(json.dumps(body, sort_keys=True, ensure_ascii=False))


class AuditLog:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = GENESIS
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    last = json.loads(line)["hash"]
        return last

    def append(self, **kwargs: Any) -> AuditRecord:
        with self._lock:
            rec = AuditRecord(
                ts=datetime.now(timezone.utc).isoformat(),
                prev_hash=self._last_hash(),
                **kwargs,
            )
            rec.hash = rec.compute_hash()
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            return rec

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return iter(())
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def verify(self) -> tuple[bool, str | None]:
        """Returns (ok, first_broken_record_ts)."""
        prev = GENESIS
        for raw in self:
            stored = raw.pop("hash")
            rec = AuditRecord(**raw)
            if rec.prev_hash != prev or rec.compute_hash() != stored:
                return False, rec.ts
            prev = stored
        return True, None


def sha256_text(text: str) -> str:
    return _sha(text)
