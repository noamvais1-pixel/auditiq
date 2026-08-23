"""Anthropic client with tool-use loop, streaming, and an offline demo mode.

The offline mode is not a toy. `ANTHROPIC_API_KEY` is absent on a reviewer's
machine and inside CI, and a project that cannot be run is a project that does
not get read. In offline mode the retrieval, redaction, guardrail and audit
paths all execute for real -- only generation is replaced by an extractive
answer built from the retrieved chunks. The API reports `mode: "offline"` and
the UI shows a banner, so nobody mistakes it for the real thing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

from .prompts import SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, run_tool

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
MAX_TOOL_ROUNDS = 4


@dataclass
class Generation:
    text: str
    model: str
    mode: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._client = None
        self.mode = "offline"
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                from anthropic import Anthropic

                self._client = Anthropic()
                self.mode = "live"
            except Exception as exc:
                log.warning("anthropic client unavailable (%s); running offline", exc)

    # ------------------------------------------------------------------ live
    def _run_tool_loop(self, user_turn: str) -> Generation:
        """Standard Anthropic tool-use loop: call, execute, feed results back."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_turn}]
        tool_calls: list[dict[str, Any]] = []
        usage = {"input_tokens": 0, "output_tokens": 0}

        for _ in range(MAX_TOOL_ROUNDS):
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            usage["input_tokens"] += resp.usage.input_tokens
            usage["output_tokens"] += resp.usage.output_tokens

            if resp.stop_reason != "tool_use":
                text = "".join(b.text for b in resp.content if b.type == "text")
                return Generation(text, self.model, "live", tool_calls, usage, resp.stop_reason)

            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                output = run_tool(block.name, dict(block.input))
                tool_calls.append({"name": block.name, "input": dict(block.input), "output": output})
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output, ensure_ascii=False),
                        "is_error": "error" in output,
                    }
                )
            messages.append({"role": "user", "content": results})

        return Generation(
            "החישוב דרש יותר שלבים מהמותר. נסה/י לפצל את השאלה.",
            self.model, "live", tool_calls, usage, "max_tool_rounds",
        )

    # --------------------------------------------------------------- offline
    @staticmethod
    def _offline_answer(hits) -> Generation:
        if not hits:
            return Generation(
                "לא נמצאו מקורות רלוונטיים במאגר לשאלה זו.", "offline", "offline"
            )
        lines = [
            "**מצב הדגמה — ללא מפתח API.** התשובה הבאה מורכבת מקטעי המקור שאותרו, "
            "ללא ניסוח מחדש על ידי מודל. האחזור, הסרת המידע המזהה, אכיפת הציטוט "
            "ולוג הביקורת פועלים במלואם.\n",
        ]
        for hit in hits[:3]:
            snippet = " ".join(hit.chunk.text.split())[:400]
            lines.append(f"{snippet} [[{hit.chunk.chunk_id}]]\n")
        return Generation("\n".join(lines), "offline", "offline")

    # ------------------------------------------------------------------ api
    def generate(self, user_turn: str, hits) -> Generation:
        if self._client is None:
            return self._offline_answer(hits)
        try:
            return self._run_tool_loop(user_turn)
        except Exception as exc:
            log.exception("generation failed")
            gen = self._offline_answer(hits)
            gen.text = f"שגיאה בפנייה למודל ({type(exc).__name__}). מוצגת תשובת גיבוי:\n\n{gen.text}"
            return gen

    def stream(self, user_turn: str, hits) -> Iterator[str]:
        """Token stream for the UI.

        Tool use and streaming do not compose cleanly in one pass, so when the
        model wants a tool we finish the loop non-streamed and emit the result.
        Users care that text appears promptly, not that every path streams.
        """
        if self._client is None:
            for chunk in self._offline_answer(hits).text.split(" "):
                yield chunk + " "
            return
        try:
            with self._client.messages.stream(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=[{"role": "user", "content": user_turn}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
            if final.stop_reason == "tool_use":
                yield "\n\n"
                yield self._run_tool_loop(user_turn).text
        except Exception as exc:
            log.exception("streaming failed")
            yield f"\n\n[שגיאה בפנייה למודל: {type(exc).__name__}]"
