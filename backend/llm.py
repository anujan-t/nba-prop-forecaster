"""Optional language model for COURTSIDE through any OpenAI-compatible chat-completions endpoint.

Works with Google Gemini's, OpenAI's, Anthropic's and OpenRouter's OpenAI-compatible endpoints, Azure OpenAI's v1
endpoint, and local servers such as Ollama or LM Studio. Configuration comes from environment variables only; the API
key is sent to the configured endpoint and never logged or returned.

Rate limits (HTTP 429/503) are retried after the provider's suggested delay within a bounded wait, calls can be paced
to a requests-per-minute budget, identical requests can be answered from a cache, and token usage is counted.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Protocol
from urllib.parse import urlparse

import httpx

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_HOST = urlparse(GEMINI_BASE_URL).netloc
GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"
RETRY_STATUSES = (429, 503)


class LLMError(RuntimeError):
    pass


def gemini_tool(tool: dict) -> dict:
    """A tool definition in the JSON-schema subset Gemini's function declarations accept: no additionalProperties, no
    empty required lists, one type per field (the server still validates every argument)."""
    def clean(schema):
        if isinstance(schema, list):
            return [clean(item) for item in schema]
        if not isinstance(schema, dict):
            return schema
        out = {}
        for key, value in schema.items():
            if key == "additionalProperties" or key == "required" and not value:
                continue
            if key == "type" and isinstance(value, list):
                value = next((kind for kind in value if kind != "null"), "string")
            out[key] = clean(value)
        return out
    return clean(tool)


class ResponseCache(Protocol):
    def get(self, key: str, ttl: float) -> dict | None: ...
    def put(self, key: str, message: dict) -> None: ...


class StoreCache:
    """LLM responses in the app's SQLite provider cache (keys are hashes of the request; nothing secret is stored).
    Entries older than `keep_days` are pruned now and then."""

    def __init__(self, store, prefix: str = "llm:", keep_days: float = 2.0):
        self.store, self.prefix, self.keep_days = store, prefix, keep_days
        self._puts = 0

    def get(self, key: str, ttl: float) -> dict | None:
        saved = self.store.resource(self.prefix + key)
        if not saved or "message" not in saved:
            return None
        try:
            fetched = datetime.fromisoformat(saved["fetched_at"])
        except (KeyError, TypeError, ValueError):
            return None
        return saved["message"] if datetime.now(timezone.utc) - fetched <= timedelta(seconds=ttl) else None

    def put(self, key: str, message: dict) -> None:
        self.store.save_resource(self.prefix + key, {"message": message})
        self._puts += 1
        if self._puts % 50 == 1:
            self.prune()

    def prune(self) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.keep_days)).isoformat(timespec="microseconds")
        with self.store.connect() as db:
            return db.execute("DELETE FROM provider_cache WHERE key LIKE ? AND json_extract(payload, '$.fetched_at') < ?",
                              (self.prefix + "%", cutoff)).rowcount


class LLMClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, key_header: str = "Authorization",
                 timeout: float = 45.0, transport: httpx.BaseTransport | None = None, *, min_interval: float = 0.0,
                 max_retries: int = 3, max_wait: float = 40.0, cache: ResponseCache | None = None, cache_ttl: float = 0.0,
                 reasoning_effort: str | None = None, price_in: float | None = None, price_out: float | None = None):
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
        self.temperature: float | None = 0.2
        self.client = httpx.Client(transport=transport, timeout=timeout)
        self.min_interval, self.max_retries, self.max_wait = max(0.0, min_interval), max(0, max_retries), max_wait
        self.cache, self.cache_ttl = cache, cache_ttl
        self.reasoning_effort = reasoning_effort or None
        self.price_in, self.price_out = price_in, price_out  # USD per million prompt / completion tokens, if known
        self.usage = {"calls": 0, "cached": 0, "retries": 0, "rate_limited": 0,
                      "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.last_usage: dict = {}
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self._sleep = time.sleep

    @property
    def provider(self) -> str:
        return "gemini" if urlparse(self.base_url).netloc == GEMINI_HOST else "openai-compatible"

    @classmethod
    def from_env(cls, env: dict | None = None) -> "LLMClient | None":
        """None unless a model is configured.

        COURTSIDE_LLM_MODEL (+ COURTSIDE_LLM_API_KEY / COURTSIDE_LLM_BASE_URL) picks any OpenAI-compatible endpoint:
        OpenAI with a key, a local Ollama server without one. Otherwise GEMINI_API_KEY alone selects Google Gemini
        (model COURTSIDE_GEMINI_MODEL, default gemini-3.8-flash). Optional: COURTSIDE_LLM_RPM (pace calls),
        COURTSIDE_LLM_REASONING_EFFORT, COURTSIDE_LLM_PRICE_IN / _OUT (USD per million tokens, for cost estimates).
        """
        env = os.environ if env is None else env
        model = (env.get("COURTSIDE_LLM_MODEL") or "").strip()
        gemini_key = (env.get("GEMINI_API_KEY") or "").strip() or None
        if model:
            key = (env.get("COURTSIDE_LLM_API_KEY") or env.get("OPENAI_API_KEY") or "").strip() or None
            base = (env.get("COURTSIDE_LLM_BASE_URL") or "").strip() or ("https://api.openai.com/v1" if key else "http://127.0.0.1:11434/v1")
            if key is None and urlparse(base).netloc == GEMINI_HOST:
                key = gemini_key
        elif gemini_key:
            model = (env.get("COURTSIDE_GEMINI_MODEL") or "").strip() or GEMINI_DEFAULT_MODEL
            key, base = gemini_key, GEMINI_BASE_URL
        else:
            return None
        header = (env.get("COURTSIDE_LLM_KEY_HEADER") or "Authorization").strip()

        def number(name: str) -> float | None:
            try:
                value = float((env.get(name) or "").strip())
            except ValueError:
                return None
            return value if value > 0 else None

        rpm = number("COURTSIDE_LLM_RPM")
        return cls(base, model, key, header, min_interval=60.0 / rpm if rpm else 0.0,
                   reasoning_effort=(env.get("COURTSIDE_LLM_REASONING_EFFORT") or "").strip() or None,
                   price_in=number("COURTSIDE_LLM_PRICE_IN"), price_out=number("COURTSIDE_LLM_PRICE_OUT"))

    @property
    def endpoint(self) -> str:
        return urlparse(self.base_url).netloc

    def complete(self, messages: list[dict], tools: list[dict] | None = None, *, cache_ttl: float | None = None,
                 response_format: dict | None = None, meter: dict | None = None) -> dict:
        """One chat-completions call; returns the assistant message ({content, tool_calls}).

        With a cache and a positive TTL (per call or the client default) an identical request is answered from the
        cache. Rate-limited calls are retried after the provider's suggested delay, within max_wait seconds. A `meter`
        dict, if given, accumulates this call's calls, cache hits, tokens and seconds (per-answer accounting).
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers[self.key_header] = f"Bearer {self.api_key}" if self.key_header.lower() == "authorization" else self.api_key
        body = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = [gemini_tool(tool) for tool in tools] if self.provider == "gemini" else tools
            body["tool_choice"] = "auto"
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if response_format:
            body["response_format"] = response_format
        ttl = self.cache_ttl if cache_ttl is None else cache_ttl
        key = self._cache_key(body) if self.cache is not None and ttl and ttl > 0 else None
        if key is not None:
            hit = self.cache.get(key, ttl)
            if hit is not None:
                self._record({}, 0.0, cached=True, meter=meter)
                return hit
        if self.temperature is not None:
            body["temperature"] = self.temperature
        started = time.monotonic()
        response = self._send(body, headers)
        if response.status_code == 400 and "temperature" in body and "temperature" in response.text:
            # Some models (e.g. reasoning models) accept only their default temperature.
            self.temperature = None
            del body["temperature"]
            response = self._send(body, headers)
        if response.status_code >= 400:
            raise LLMError(f"{self.endpoint} returned HTTP {response.status_code}{self._reason(response)}")
        try:
            payload = response.json()
            message = payload["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.endpoint} returned an unexpected response") from exc
        if not isinstance(message, dict):
            raise LLMError(f"{self.endpoint} returned an unexpected response")
        self._record(payload.get("usage") if isinstance(payload.get("usage"), dict) else {}, time.monotonic() - started, meter=meter)
        if key is not None:
            self.cache.put(key, message)
        return message

    @property
    def local(self) -> bool:
        """Whether the endpoint runs on this computer (Ollama, LM Studio), so nothing sent to it leaves the machine."""
        return (urlparse(self.base_url).hostname or "").lower() in ("127.0.0.1", "localhost", "::1")

    def estimate_cost(self, usage: dict) -> float | None:
        """US dollars for a usage record at the configured per-million-token prices, or None if prices are unknown."""
        if self.price_in is None or self.price_out is None:
            return None
        return (usage.get("prompt_tokens", 0) * self.price_in + usage.get("completion_tokens", 0) * self.price_out) / 1e6

    def _cache_key(self, body: dict) -> str:
        return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()

    def _record(self, usage: dict, seconds: float, cached: bool = False, meter: dict | None = None) -> None:
        tokens = {name: int(usage.get(name) or 0) for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
        with self._lock:
            self.usage["cached" if cached else "calls"] += 1
            for name, value in tokens.items():
                self.usage[name] += value
            self.last_usage = {**tokens, "cached": cached, "seconds": round(seconds, 3)}
        if meter is not None:
            meter["cached" if cached else "calls"] = meter.get("cached" if cached else "calls", 0) + 1
            for name, value in tokens.items():
                meter[name] = meter.get(name, 0) + value
            meter["model_seconds"] = round(meter.get("model_seconds", 0.0) + seconds, 3)

    def _send(self, body: dict, headers: dict) -> httpx.Response:
        deadline = time.monotonic() + self.max_wait
        attempt = 0
        while True:
            self._pace()
            response = self._post(body, headers)
            if response.status_code not in RETRY_STATUSES:
                return response
            delay = self._retry_delay(response, attempt)
            if attempt >= self.max_retries or time.monotonic() + delay > deadline:
                with self._lock:
                    self.usage["rate_limited"] += 1
                return response
            attempt += 1
            with self._lock:
                self.usage["retries"] += 1
            self._sleep(delay)

    def _pace(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self.min_interval
        if wait > 0:
            self._sleep(wait)

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        """The provider's Retry-After header or Gemini's RetryInfo.retryDelay, else exponential backoff."""
        header = response.headers.get("retry-after")
        if header:
            try:
                return min(max(float(header), 0.0), 60.0)
            except ValueError:
                pass
        match = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', response.text or "")
        if match:
            return min(float(match.group(1)), 60.0)
        return float(min(2 ** attempt, 8))

    def _post(self, body: dict, headers: dict) -> httpx.Response:
        try:
            return self.client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"could not reach {self.endpoint} ({type(exc).__name__})") from exc

    @staticmethod
    def _reason(response: httpx.Response) -> str:
        """The provider's error code (e.g. invalid_api_key, RESOURCE_EXHAUSTED); never its message, which can quote the key."""
        try:
            data = response.json()
            if isinstance(data, list) and data:
                data = data[0]
            error = data.get("error") or {}
        except (ValueError, AttributeError):
            return ""
        if not isinstance(error, dict):
            return ""
        code = error.get("code") if isinstance(error.get("code"), str) else error.get("status") or error.get("type")
        if not isinstance(code, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{3,40}", code) or code.startswith(("sk-", "AIza")):
            return ""
        return f" ({code})"
