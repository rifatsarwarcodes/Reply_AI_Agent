"""Base agent with role-based LLM construction and Langfuse observability.

Each agent specifies its `name` (matching a key in config.MODELS) to
automatically get the right model.  Direct audio / multimodal calls
go through `_call_multimodal` which uses requests against OpenRouter.
"""

from __future__ import annotations

import base64
import json as _json
import logging
import os
from typing import Any

import requests
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.langchain import CallbackHandler

import config

log = logging.getLogger(__name__)


def build_llm(role: str | None = None) -> ChatOpenAI:
    """Build a ChatOpenAI pointed at the model assigned to *role*."""
    cfg = config.MODELS.get(role or "prefilter", {})
    model_id = cfg.get("id", config.DEFAULT_MODEL)
    temperature = cfg.get("temperature", 0.2)
    max_tokens = cfg.get("max_tokens", 4096)

    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=config.OPENROUTER_BASE_URL,
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class BaseAgent:
    """Provides LLM helpers that every specialist agent inherits."""

    name: str = "base"

    def __init__(self, llm: ChatOpenAI | None = None):
        self.llm = llm or build_llm(role=self.name)

    # ------------------------------------------------------------------
    # Text-based LLM helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        handler = CallbackHandler()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages, config={"callbacks": [handler]})
        return response.content

    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> Any:
        """Call LLM and parse the response as JSON."""
        raw = self._call_llm(system_prompt, user_prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        # DeepSeek R1 sometimes wraps output in <think>...</think> tags
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            log.warning("LLM returned non-JSON (%s): %.200s", self.name, raw)
            return raw

    # ------------------------------------------------------------------
    # Direct OpenRouter call (for multimodal: audio, images)
    # ------------------------------------------------------------------

    @staticmethod
    def _call_openrouter_raw(
        model: str,
        messages: list[dict],
        temperature: float = 0.15,
        max_tokens: int = 4096,
    ) -> str:
        """Low-level OpenRouter call supporting multimodal content parts."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        resp = requests.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    @staticmethod
    def _encode_audio_b64(filepath: str) -> str:
        """Read an MP3 file and return its base64 encoding."""
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
