"""Optional language model for COURTSIDE Coach through any OpenAI-compatible chat-completions endpoint.

Works with OpenAI, Azure OpenAI's v1 endpoint, and local servers such as Ollama or LM Studio. Configuration comes
from environment variables only; the API key is sent to the configured endpoint and never logged or returned.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, key_header: str = "Authorization",
                 timeout: float = 45.0, transport: httpx.BaseTransport | None = None):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("COURTSIDE_LLM_BASE_URL must be an http(s) URL such as https://api.openai.com/v1.")
        if not model.strip():
            raise ValueError("COURTSIDE_LLM_MODEL must name a model.")
        self.base_url = base_url.rstrip("/")
        self.model = model.strip()
        self.api_key = api_key or None
        self.key_header = key_header
        self.timeout = timeout
        self.client = httpx.Client(transport=transport, timeout=timeout)

    @classmethod
    def from_env(cls, env: dict | None = None) -> "LLMClient | None":
        """None unless COURTSIDE_LLM_MODEL is set. Defaults to OpenAI with a key, or a local Ollama server without one."""
        env = os.environ if env is None else env
        model = (env.get("COURTSIDE_LLM_MODEL") or "").strip()
        if not model:
            return None
        key = (env.get("COURTSIDE_LLM_API_KEY") or env.get("OPENAI_API_KEY") or "").strip() or None
        base = (env.get("COURTSIDE_LLM_BASE_URL") or "").strip() or ("https://api.openai.com/v1" if key else "http://127.0.0.1:11434/v1")
        header = (env.get("COURTSIDE_LLM_KEY_HEADER") or "Authorization").strip()
        return cls(base, model, key, header)

    @property
    def endpoint(self) -> str:
        return urlparse(self.base_url).netloc

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        """One chat-completions call; returns the assistant message ({content, tool_calls})."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.key_header] = f"Bearer {self.api_key}" if self.key_header.lower() == "authorization" else self.api_key
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": 0.2}
        try:
            response = self.client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach {self.endpoint} ({type(exc).__name__})") from exc
        if response.status_code >= 400:
            raise LLMError(f"{self.endpoint} returned HTTP {response.status_code}")
        try:
            message = response.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.endpoint} returned an unexpected response") from exc
        if not isinstance(message, dict):
            raise LLMError(f"{self.endpoint} returned an unexpected response")
        return message
