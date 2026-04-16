"""Base agent with LLM access and Langfuse observability."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse.langchain import CallbackHandler

import config

log = logging.getLogger(__name__)


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )


class BaseAgent:
    """Provides LLM helpers that every specialist agent inherits."""

    name: str = "base"

    def __init__(self, llm: ChatOpenAI | None = None):
        self.llm = llm or build_llm()

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        handler = CallbackHandler()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages, config={"callbacks": [handler]})
        return response.content

    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> Any:
        """Call LLM and attempt to parse the response as JSON."""
        import json as _json

        raw = self._call_llm(system_prompt, user_prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            log.warning("LLM returned non-JSON (%s): %.200s", self.name, raw)
            return raw
