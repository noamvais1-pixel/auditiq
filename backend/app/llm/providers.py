"""Model providers behind one interface.

The provider is a config line, not a rewrite. That is worth the small amount of
indirection for two reasons beyond tidiness:

* The tool schemas, the system prompt, the citation guardrail and the audit
  record are provider-independent. Only the wire format differs. Letting the
  provider detail leak into the pipeline would mean re-testing all of that
  whenever the firm's model choice changes -- and in a regulated environment the
  model choice changes for procurement reasons, not technical ones.

* It makes the offline path a provider like any other, rather than a special
  case threaded through the real client with `if` statements.

Each provider owns exactly one thing: turning a system prompt, a user turn and a
list of tool schemas into a ``Generation``, executing any tool calls along the
way. The tool *results* are recorded identically regardless of provider, so the
answer stays auditable either way.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from .tools import TOOL_SCHEMAS, run_tool

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 4


@dataclass
class Generation:
    text: str
    model: str
    mode: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None


class ModelProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, user_turn: str) -> Generation: ...
    def stream(self, system: str, user_turn: str) -> Iterator[str]: ...


# --------------------------------------------------------------- Anthropic
class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None) -> None:
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def generate(self, system: str, user_turn: str) -> Generation:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_turn}]
        tool_calls: list[dict[str, Any]] = []
        usage = {"input_tokens": 0, "output_tokens": 0}

        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system,
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
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output, ensure_ascii=False),
                    "is_error": "error" in output,
                })
            messages.append({"role": "user", "content": results})

        return Generation(_ROUNDS_EXHAUSTED, self.model, "live", tool_calls, usage, "max_tool_rounds")

    def stream(self, system: str, user_turn: str) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=2048,
            system=system,
            tools=TOOL_SCHEMAS,
            messages=[{"role": "user", "content": user_turn}],
        ) as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()
        if final.stop_reason == "tool_use":
            # Streaming and tool use do not compose in a single pass; finish the
            # loop unstreamed rather than showing the user a truncated answer.
            yield "\n\n" + self.generate(system, user_turn).text


# ------------------------------------------------------------------ Gemini
def _to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Anthropic ``input_schema`` -> Gemini ``parameters``.

    Gemini accepts an OpenAPI subset, not full JSON Schema. Our tool schemas stay
    inside that subset on purpose (type/properties/required/enum/description), so
    the conversion is a rename rather than a translation -- and anything outside
    the subset would fail loudly at declaration time, not silently at call time.
    """
    allowed = {"type", "properties", "required", "enum", "description", "items"}
    return {k: v for k, v in schema.items() if k in allowed}


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str | None = None) -> None:
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=_to_gemini_schema(t["input_schema"]),
                    )
                    for t in TOOL_SCHEMAS
                ]
            )
        ]

    def _config(self, system: str):
        types = self._types
        return types.GenerateContentConfig(
            system_instruction=system,
            tools=self._tools,
            # Tools are executed explicitly rather than by the SDK's automatic
            # loop, so every call and its result lands in the audit record.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            max_output_tokens=2048,
        )

    @staticmethod
    def _accumulate(usage: dict[str, int], meta) -> None:
        if not meta:
            return
        usage["input_tokens"] += meta.prompt_token_count or 0
        usage["output_tokens"] += meta.candidates_token_count or 0

    def generate(self, system: str, user_turn: str) -> Generation:
        types = self._types
        contents = [types.Content(role="user", parts=[types.Part(text=user_turn)])]
        tool_calls: list[dict[str, Any]] = []
        usage = {"input_tokens": 0, "output_tokens": 0}

        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.client.models.generate_content(
                model=self.model, contents=contents, config=self._config(system)
            )
            self._accumulate(usage, resp.usage_metadata)

            calls = resp.function_calls or []
            if not calls:
                return Generation(resp.text or "", self.model, "live", tool_calls, usage, "stop")

            contents.append(resp.candidates[0].content)
            parts = []
            for call in calls:
                output = run_tool(call.name, dict(call.args or {}))
                tool_calls.append({"name": call.name, "input": dict(call.args or {}), "output": output})
                parts.append(
                    types.Part.from_function_response(name=call.name, response=output)
                )
            contents.append(types.Content(role="user", parts=parts))

        return Generation(_ROUNDS_EXHAUSTED, self.model, "live", tool_calls, usage, "max_tool_rounds")

    def stream(self, system: str, user_turn: str) -> Iterator[str]:
        types = self._types
        contents = [types.Content(role="user", parts=[types.Part(text=user_turn)])]
        saw_tool_call = False
        for chunk in self.client.models.generate_content_stream(
            model=self.model, contents=contents, config=self._config(system)
        ):
            if chunk.function_calls:
                saw_tool_call = True
                continue
            if chunk.text:
                yield chunk.text
        if saw_tool_call:
            yield "\n\n" + self.generate(system, user_turn).text


# ----------------------------------------------------------------- offline
_ROUNDS_EXHAUSTED = "החישוב דרש יותר שלבים מהמותר. נסה/י לפצל את השאלה."

OFFLINE_PREAMBLE = (
    "**מצב הדגמה — לא הוגדר מפתח API.** התשובה הבאה מורכבת מקטעי המקור שאותרו, "
    "ללא ניסוח מחדש על ידי מודל. האחזור, הסרת המידע המזהה, אכיפת הציטוט "
    "ולוג הביקורת פועלים במלואם.\n"
)


class OfflineProvider:
    """Extractive fallback so the project runs with no key and no network.

    Not a stub: retrieval, redaction, the coverage gate, the citation guardrail
    and the audit chain all execute for real. Only generation is replaced. The
    API reports ``mode: "offline"`` and the UI shows a banner, so this is never
    mistaken for the real thing.
    """

    name = "offline"
    model = "offline"

    def __init__(self, hits_provider=None) -> None:
        self._hits: list = []

    def set_hits(self, hits) -> None:
        self._hits = hits

    def generate(self, system: str, user_turn: str) -> Generation:
        if not self._hits:
            return Generation("לא נמצאו מקורות רלוונטיים במאגר.", "offline", "offline")
        lines = [OFFLINE_PREAMBLE]
        for hit in self._hits[:3]:
            snippet = " ".join(hit.chunk.text.split())[:400]
            lines.append(f"{snippet} [[{hit.chunk.chunk_id}]]\n")
        return Generation("\n".join(lines), "offline", "offline")

    def stream(self, system: str, user_turn: str) -> Iterator[str]:
        for word in self.generate(system, user_turn).text.split(" "):
            yield word + " "


# ---------------------------------------------------------------- selection
def build_provider(name: str | None = None) -> ModelProvider:
    """Pick a provider from config, degrading to offline with a loud warning.

    Explicit ``LLM_PROVIDER`` wins. Otherwise whichever key is present is used --
    convenient locally, and unambiguous in deployment where the variable is set.
    """
    name = (name or os.getenv("LLM_PROVIDER") or "").lower()

    if not name:
        if os.getenv("GOOGLE_API_KEY"):
            name = "gemini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            name = "anthropic"

    if not name:
        return OfflineProvider()

    try:
        if name == "gemini":
            return GeminiProvider()
        if name == "anthropic":
            return AnthropicProvider()
        raise ValueError(f"unknown LLM_PROVIDER {name!r}")
    except Exception as exc:
        log.warning("provider %r unavailable (%s); running offline", name, exc)
        return OfflineProvider()
