"""Thin façade over the configured model provider.

Everything provider-specific lives in ``providers.py``. What stays here is the
part the rest of the app depends on: a stable ``generate``/``stream`` pair that
never raises, because a model outage should degrade the assistant to its
extractive fallback rather than return a 500 to an accountant mid-question.
"""

from __future__ import annotations

import logging
from typing import Iterator

from .prompts import SYSTEM_PROMPT
from .providers import Generation, OfflineProvider, build_provider

log = logging.getLogger(__name__)


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
        if isinstance(self.provider, OfflineProvider):
            return self._offline(hits)
        try:
            return self.provider.generate(SYSTEM_PROMPT, user_turn)
        except Exception as exc:
            log.exception("generation failed")
            gen = self._offline(hits)
            gen.text = (
                f"שגיאה בפנייה למודל ({type(exc).__name__}). מוצגת תשובת גיבוי מהמקורות:\n\n"
                f"{gen.text}"
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
