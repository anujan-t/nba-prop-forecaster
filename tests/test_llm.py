"""LLMClient: Gemini configuration, optional tools, token accounting, pacing, rate-limit retries and caching.

Everything runs against httpx.MockTransport; no request leaves the machine.
"""
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from backend.data import Store
from backend.llm import GEMINI_DEFAULT_MODEL, GEMINI_HOST, LLMClient, LLMError, StoreCache, gemini_tool

OK = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}}


def client(handler, **options) -> LLMClient:
    model = LLMClient("https://llm.example/v1", "m", "sk-test", transport=httpx.MockTransport(handler), **options)
    model.waits = []
    model._sleep = model.waits.append  # record instead of sleeping
    return model


class ConfigurationTests(unittest.TestCase):
    def test_gemini_key_alone_selects_gemini(self):
        gemini = LLMClient.from_env({"GEMINI_API_KEY": "g-key"})
        self.assertEqual((gemini.endpoint, gemini.model, gemini.provider, gemini.api_key), (GEMINI_HOST, GEMINI_DEFAULT_MODEL, "gemini", "g-key"))
        self.assertEqual(LLMClient.from_env({"GEMINI_API_KEY": "g", "COURTSIDE_GEMINI_MODEL": "gemini-x-flash"}).model, "gemini-x-flash")

    def test_an_explicit_model_wins_and_can_point_at_gemini(self):
        explicit = LLMClient.from_env({"COURTSIDE_LLM_MODEL": "gpt-x", "OPENAI_API_KEY": "sk-1", "GEMINI_API_KEY": "g"})
        self.assertEqual((explicit.endpoint, explicit.api_key, explicit.provider), ("api.openai.com", "sk-1", "openai-compatible"))
        routed = LLMClient.from_env({"COURTSIDE_LLM_MODEL": "gemini-y", "GEMINI_API_KEY": "g",
                                     "COURTSIDE_LLM_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai"})
        self.assertEqual((routed.provider, routed.api_key), ("gemini", "g"))

    def test_pacing_reasoning_and_prices_from_the_environment(self):
        tuned = LLMClient.from_env({"GEMINI_API_KEY": "g", "COURTSIDE_LLM_RPM": "12", "COURTSIDE_LLM_REASONING_EFFORT": "low",
                                    "COURTSIDE_LLM_PRICE_IN": "0.5", "COURTSIDE_LLM_PRICE_OUT": "3"})
        self.assertEqual((tuned.min_interval, tuned.reasoning_effort, tuned.price_in, tuned.price_out), (5.0, "low", 0.5, 3.0))
        loose = LLMClient.from_env({"GEMINI_API_KEY": "g", "COURTSIDE_LLM_RPM": "lots", "COURTSIDE_LLM_PRICE_IN": "-1"})
        self.assertEqual((loose.min_interval, loose.price_in), (0.0, None))


class RequestTests(unittest.TestCase):
    def test_tools_are_optional_and_reasoning_effort_is_sent(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=OK)
        model = client(handler, reasoning_effort="low")
        model.complete([{"role": "user", "content": "hi"}])
        model.complete([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "f", "parameters": {}}}])
        self.assertNotIn("tools", bodies[0])
        self.assertEqual((bodies[1]["tool_choice"], bodies[1]["reasoning_effort"]), ("auto", "low"))

    def test_usage_is_counted_and_priced(self):
        model = client(lambda request: httpx.Response(200, json=OK), price_in=0.5, price_out=3.0)
        model.complete([])
        model.complete([])
        self.assertEqual((model.usage["calls"], model.usage["prompt_tokens"], model.usage["total_tokens"]), (2, 240, 300))
        self.assertEqual(model.last_usage["completion_tokens"], 30)
        self.assertAlmostEqual(model.estimate_cost(model.last_usage), (120 * 0.5 + 30 * 3.0) / 1e6)
        self.assertIsNone(client(lambda request: httpx.Response(200, json=OK)).estimate_cost(model.last_usage))

    def test_a_meter_accounts_for_one_answer_only(self):
        model = client(lambda request: httpx.Response(200, json=OK))
        model.complete([])  # someone else's call
        meter = {}
        model.complete([], meter=meter)
        model.complete([], meter=meter)
        self.assertEqual((meter["calls"], meter["prompt_tokens"], meter["completion_tokens"]), (2, 240, 60))
        self.assertIn("model_seconds", meter)
        self.assertEqual(model.usage["calls"], 3)

    def test_gemini_gets_tool_schemas_it_accepts(self):
        tool = {"type": "function", "function": {"name": "f", "description": "d", "parameters": {
            "type": "object", "required": [], "additionalProperties": False,
            "properties": {"odds": {"type": ["integer", "null"]}, "side": {"type": "string", "enum": ["over", "under"]}}}}}
        cleaned = gemini_tool(tool)["function"]["parameters"]
        self.assertEqual(cleaned, {"type": "object", "properties": {"odds": {"type": "integer"},
                                                                    "side": {"type": "string", "enum": ["over", "under"]}}})
        self.assertIn("additionalProperties", tool["function"]["parameters"])  # the original is untouched
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json=OK)
        gemini = LLMClient("https://generativelanguage.googleapis.com/v1beta/openai", "g", "key", transport=httpx.MockTransport(handler))
        gemini.complete([], [tool])
        client(handler).complete([], [tool])
        self.assertNotIn("additionalProperties", json.dumps(bodies[0]["tools"]))
        self.assertIn("additionalProperties", json.dumps(bodies[1]["tools"]))  # other providers get the schema as written

    def test_local_endpoints_are_recognised(self):
        self.assertTrue(LLMClient("http://127.0.0.1:11434/v1", "llama").local)
        self.assertTrue(LLMClient("http://localhost:1234/v1", "m").local)
        self.assertFalse(LLMClient("https://generativelanguage.googleapis.com/v1beta/openai", "g").local)


class RateLimitTests(unittest.TestCase):
    def test_retry_after_header_then_success(self):
        replies = [httpx.Response(429, headers={"retry-after": "2"}, json={}), httpx.Response(200, json=OK)]
        model = client(lambda request: replies.pop(0))
        self.assertEqual(model.complete([])["content"], "ok")
        self.assertEqual((model.waits, model.usage["retries"], model.usage["rate_limited"]), ([2.0], 1, 0))

    def test_gemini_retry_delay_and_exhaustion_report_a_safe_code(self):
        body = [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota for key AIzaSECRET exceeded",
                           "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3s"}]}}]
        model = client(lambda request: httpx.Response(429, json=body), max_retries=2)
        with self.assertRaises(LLMError) as raised:
            model.complete([])
        self.assertEqual(model.waits, [3.0, 3.0])
        self.assertIn("HTTP 429 (RESOURCE_EXHAUSTED)", str(raised.exception))
        self.assertNotIn("AIza", str(raised.exception))
        self.assertEqual(model.usage["rate_limited"], 1)

    def test_a_delay_beyond_the_wait_budget_fails_fast(self):
        model = client(lambda request: httpx.Response(429, headers={"retry-after": "50"}, json={}), max_wait=10)
        with self.assertRaisesRegex(LLMError, "HTTP 429"):
            model.complete([])
        self.assertEqual(model.waits, [])

    def test_backoff_without_a_hint_and_pacing_between_calls(self):
        replies = [httpx.Response(503, json={}), httpx.Response(200, json=OK), httpx.Response(200, json=OK)]
        model = client(lambda request: replies.pop(0), min_interval=10.0)
        model.complete([])
        model.complete([])
        self.assertEqual(model.waits[0], 1.0)  # exponential backoff starts at one second
        self.assertTrue(any(wait > 8 for wait in model.waits[1:]))  # paced to one call per ten seconds


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.directory.name))

    def tearDown(self):
        self.directory.cleanup()

    def test_identical_requests_are_served_from_the_cache_until_the_ttl_passes(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(200, json=OK)
        model = client(handler, cache=StoreCache(self.store), cache_ttl=3600)
        messages = [{"role": "user", "content": "who leads the league in threes?"}]
        model.complete(messages)
        self.assertEqual(model.complete(messages)["content"], "ok")
        self.assertEqual((len(calls), model.usage["cached"], model.last_usage["cached"]), (1, 1, True))
        model.complete([{"role": "user", "content": "something else"}])
        model.complete(messages, cache_ttl=0)  # a per-call TTL of zero bypasses the cache
        self.assertEqual(len(calls), 3)
        with self.store.connect() as db:
            db.execute("UPDATE provider_cache SET payload = json_set(payload, '$.fetched_at', '2000-01-01T00:00:00+00:00')")
            keys = [row[0] for row in db.execute("SELECT key FROM provider_cache")]
        model.complete(messages)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(key.startswith("llm:") and "sk-" not in key for key in keys))

    def test_old_entries_are_pruned_and_other_cache_rows_kept(self):
        cache = StoreCache(self.store)
        cache.put("old", {"content": "x"})
        self.store.save_resource("roster:2026-27", {"players": []})
        with self.store.connect() as db:
            db.execute("UPDATE provider_cache SET payload = json_set(payload, '$.fetched_at', '2000-01-01T00:00:00+00:00')")
        cache.put("new", {"content": "y"})
        self.assertEqual(cache.prune(), 1)
        self.assertIsNotNone(self.store.resource("roster:2026-27"))
        self.assertIsNotNone(cache.get("new", 3600))


if __name__ == "__main__":
    unittest.main()
