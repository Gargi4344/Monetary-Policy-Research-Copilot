"""One LLM interface for every provider, through LiteLLM.

Default `ollama/qwen2.5:7b` runs locally and free. Any LiteLLM model string works through
.env (for example `openai/gpt-4o-mini`, `azure/<deployment>`, `anthropic/claude-...`) with
that provider's key. Tests install `FakeLLM` with `set_llm`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from copilot.config import get_settings

log = logging.getLogger("llm")
Messages = list[dict[str, str]]


class LLM(Protocol):
    """What the agent, tone index and evaluation need from a model."""

    model: str

    def complete(
        self, messages: Messages, *, json_mode: bool = False, max_tokens: int = 1024
    ) -> str:
        """Return the full reply."""

    def stream(self, messages: Messages, *, max_tokens: int = 1024) -> Iterator[str]:
        """Yield reply tokens."""


class LiteLLMModel:
    """LiteLLM-backed model (Ollama by default)."""

    def __init__(self, model: str | None = None) -> None:
        """Use the configured model unless one is given."""
        s = get_settings()
        self.model = model or s.llm_model
        self.kwargs: dict[str, Any] = {
            "temperature": s.llm_temperature,
            "timeout": s.llm_timeout_seconds,
        }
        if self.model.startswith("ollama"):
            self.kwargs["api_base"] = s.ollama_base_url

    def complete(
        self, messages: Messages, *, json_mode: bool = False, max_tokens: int = 1024
    ) -> str:
        """One completion; `json_mode` asks the provider for a JSON object."""
        import litellm

        t0 = time.perf_counter()
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = litellm.completion(
            model=self.model, messages=messages, max_tokens=max_tokens, **self.kwargs, **extra
        )
        text = r.choices[0].message.content or ""
        log.info("llm", extra={"model": self.model, "ms": round((time.perf_counter() - t0) * 1000),
                               "tokens_in": getattr(r.usage, "prompt_tokens", None),
                               "tokens_out": getattr(r.usage, "completion_tokens", None)})  # fmt: skip
        return text

    def stream(self, messages: Messages, *, max_tokens: int = 1024) -> Iterator[str]:
        """Stream tokens."""
        import litellm

        for part in litellm.completion(model=self.model, messages=messages, max_tokens=max_tokens,
                                       stream=True, **self.kwargs):  # fmt: skip
            if tok := part.choices[0].delta.content:
                yield tok


@dataclass
class FakeLLM:
    """Deterministic stand-in: the first rule whose key appears in the last message wins."""

    model: str = "fake"
    rules: dict[str, str | Callable[[Messages], str]] = field(default_factory=dict)
    default: str = "{}"
    calls: list[Messages] = field(default_factory=list)

    def complete(
        self, messages: Messages, *, json_mode: bool = False, max_tokens: int = 1024
    ) -> str:
        """Return the matching rule's reply."""
        self.calls.append(messages)
        last = messages[-1]["content"]
        for key, reply in self.rules.items():
            if key in last:
                return reply(messages) if callable(reply) else reply
        return self.default

    def stream(self, messages: Messages, *, max_tokens: int = 1024) -> Iterator[str]:
        """Stream the matching reply word by word."""
        for word in re.split(r"(\s+)", self.complete(messages)):
            if word:
                yield word


_override: LLM | None = None


def set_llm(llm: LLM | None) -> None:
    """Install a model for tests (None restores the configured one)."""
    global _override
    _override = llm


def get_llm(model: str | None = None) -> LLM:
    """The configured model, or the test override."""
    return _override if _override is not None else LiteLLMModel(model)


def parse_json(text: str) -> Any:
    """Parse a JSON reply, tolerating code fences or text around the object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}|\[.*\]", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise
