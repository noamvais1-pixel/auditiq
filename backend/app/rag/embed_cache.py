"""Disk cache and rate-limit handling for hosted embedding providers.

Both of these exist because the eval suite hit them, not because they were
planned. Sweeping four fusion weights over 29 questions re-embedded the same 29
strings four times and blew through the free tier's 100-requests-per-minute
limit mid-run.

The cache is keyed on (provider, model, task, text). All four matter: the same
sentence embedded as a query and as a document is deliberately *not* the same
vector, and silently sharing one entry between them would reintroduce exactly the
asymmetry bug the task types exist to avoid.

Wrapping rather than subclassing, so it composes with any provider -- including
the offline one, where it is harmless.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

RETRY_HINT = re.compile(r"retryDelay['\"]?[:\s]+['\"]?(\d+)")
MAX_ATTEMPTS = 4
MAX_SLEEP = 65.0


def _key(provider: str, model: str, task: str, text: str) -> str:
    payload = f"{provider}|{model}|{task}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedEmbeddings:
    """Wraps an embedding provider with a JSON disk cache and 429 backoff."""

    def __init__(self, inner, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self._store: dict[str, list[float]] = {}
        self._dirty = False
        if self.path.exists():
            try:
                self._store = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("embedding cache unreadable (%s); starting empty", exc)

    # provider interface passthrough
    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def fusion_weight(self) -> float:
        return self.inner.fusion_weight

    @fusion_weight.setter
    def fusion_weight(self, value: float) -> None:
        self.inner.fusion_weight = value

    def _model(self) -> str:
        return getattr(self.inner, "model", self.inner.name)

    def _call(self, texts: list[str], task: str) -> list[list[float]]:
        """Call the provider, backing off on rate limits using the server's hint."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self.inner.embed(texts, task=task)
            except Exception as exc:
                message = str(exc)
                is_rate_limit = "429" in message or "RESOURCE_EXHAUSTED" in message
                if not is_rate_limit or attempt == MAX_ATTEMPTS:
                    raise
                hint = RETRY_HINT.search(message)
                delay = min(float(hint.group(1)) + 1 if hint else 2 ** attempt, MAX_SLEEP)
                log.warning(
                    "rate limited by %s (attempt %d/%d); sleeping %.0fs",
                    self.name, attempt, MAX_ATTEMPTS, delay,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")

    def embed(self, texts: Sequence[str], task: str = "document") -> list[list[float]]:
        texts = list(texts)
        model = self._model()
        keys = [_key(self.name, model, task, t) for t in texts]

        missing = [i for i, k in enumerate(keys) if k not in self._store]
        if missing:
            fresh = self._call([texts[i] for i in missing], task)
            if len(fresh) != len(missing):
                raise RuntimeError(
                    f"{self.name} returned {len(fresh)} vectors for {len(missing)} inputs"
                )
            for i, vec in zip(missing, fresh):
                self._store[keys[i]] = vec
            self._dirty = True
            self.flush()

        return [self._store[k] for k in keys]

    def flush(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._store), encoding="utf-8")
        tmp.replace(self.path)  # atomic, so a crash mid-write cannot corrupt it
        self._dirty = False
