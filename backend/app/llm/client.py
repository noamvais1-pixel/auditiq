"""Thin façade over the configured model provider.

Everything provider-specific lives in ``providers.py``. What stays here is the
part the rest of the app depends on: a stable ``generate``/``stream`` pair that
never raises, because a model outage should degrade the assistant to its
extractive fallback rather than return a 500 to an accountant mid-question.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Iterator

from .prompts import SYSTEM_PROMPT
from .providers import Generation, OfflineProvider, build_provider

log = logging.getLogger(__name__)

RETRY_HINT = re.compile(r"retryDelay['\"]?[:\s]+['\"]?(\d+)")
MAX_ATTEMPTS = 4
MAX_SLEEP = 65.0


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "rate_limit" in text.lower()


def _backoff_seconds(exc: Exception, attempt: int) -> float:
    """Prefer the server's own retry hint over a guessed exponential delay."""
    hint = RETRY_HINT.search(str(exc))
    return min(float(hint.group(1)) + 1 if hint else 2 ** attempt, MAX_SLEEP)


class LLMClient:
    def __init__(self, provider=None) -> None:
        self.provider = provider or build_provider()
        self._fallback = OfflineProvider()

    @property
    def mode(self) -> str:
        return "offline" if isinstance(self.provider, OfflineProvider) else "live"

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def provider_name(self) -> str:
        return self.provider.name

    def _offline(self, hits) -> Generation:
        self._fallback.set_hits(hits)
        return self._fallback.generate(SYSTEM_PROMPT, "")

    def generate(self, user_turn: str, hits) -> Generation:
        """Generate, retrying rate limits before falling back.

        Without the retry a burst of questions -- or an eval run -- silently
        degrades to the extractive fallback, and the degradation is easy to miss
        because the fallback produces a plausible answer with real citations. A
        run that looks like it measured the model can end up having measured the
        fallback instead. Retry first; fall back only when the model is genuinely
        unavailable.
        """
        if isinstance(self.provider, OfflineProvider):
            return self._offline(hits)
        exc: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self.provider.generate(SYSTEM_PROMPT, user_turn)
            except Exception as err:
                exc = err
                if not _is_rate_limit(err) or attempt == MAX_ATTEMPTS:
                    break
                delay = _backoff_seconds(err, attempt)
                log.warning(
                    "rate limited by %s (attempt %d/%d); sleeping %.0fs",
                    self.provider_name, attempt, MAX_ATTEMPTS, delay,
                )
                time.sleep(delay)

        if exc is not None:
            log.error("generation failed: %s: %s", type(exc).__name__, str(exc)[:200])
        gen = self._offline(hits)
        gen.mode = "degraded"  # never let a fallback be counted as a live answer
        gen.text = (
            f"שגיאה בפנייה למודל ({type(exc).__name__ if exc else 'Unknown'}). "
            f"מוצגת תשובת גיבוי מהמקורות:\n\n{gen.text}"
        )
        return gen

    def stream(self, user_turn: str, hits) -> Iterator[str]:
        if isinstance(self.provider, OfflineProvider):
            yield from self._offline_stream(hits)
            return
        try:
            yield from self.provider.stream(SYSTEM_PROMPT, user_turn)
        except Exception as exc:
            log.exception("streaming failed")
            yield f"\n\n[שגיאה בפנייה למודל: {type(exc).__name__}]"

    def _offline_stream(self, hits) -> Iterator[str]:
        self._fallback.set_hits(hits)
        yield from self._fallback.stream(SYSTEM_PROMPT, "")
