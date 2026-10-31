"""Optional live check against Google Gemini. Skipped unless you opt in:

    $env:COURTSIDE_LIVE_TESTS = "1"   # and GEMINI_API_KEY set (e.g. as a Windows user environment variable)
    .\\.venv\\Scripts\\python.exe -m unittest tests.test_gemini_live -v

It makes a handful of real Gemini calls (free tier is fine) on the evaluation's fixed fictional league: no NBA or ESPN
requests, and nothing about your own slip or bets. The key is read from the environment and never printed.
"""
import os
import sys
import unittest

LIVE = os.environ.get("COURTSIDE_LIVE_TESTS") == "1"


def gemini_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key and sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as registry:
                key = winreg.QueryValueEx(registry, "GEMINI_API_KEY")[0]
        except OSError:
            key = None
    return key or None


@unittest.skipUnless(LIVE, "set COURTSIDE_LIVE_TESTS=1 to run live Gemini tests")
class GeminiLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        key = gemini_key()
        if not key:
            raise unittest.SkipTest("GEMINI_API_KEY is not set")
        from backend.llm import LLMClient
        cls.llm = LLMClient.from_env({"GEMINI_API_KEY": key, "COURTSIDE_GEMINI_MODEL": os.environ.get("COURTSIDE_GEMINI_MODEL", "")})
        cls.llm.min_interval = 7.0  # stay under free-tier per-minute limits

    def test_gemini_calls_a_coach_tool(self):
        from backend.assistant import PUBLIC_TOOLS
        reply = self.llm.complete([{"role": "system", "content": "Use the tools to answer."},
                                   {"role": "user", "content": "Use odds_math to convert -135 American odds."}], PUBLIC_TOOLS)
        calls = reply.get("tool_calls") or []
        self.assertTrue(calls, "Gemini answered without calling a tool")
        self.assertEqual(calls[0]["function"]["name"], "odds_math")
        self.assertGreater(self.llm.last_usage["prompt_tokens"], 0)

    def test_coach_answers_from_the_tools_and_admits_missing_data(self):
        from evals.coach_eval import run
        result = run("gemini", rpm=8, only=["B1", "G4", "M1"], llm=self.llm, verbose=False)
        answers = {answer["id"]: answer for answer in result["answers"]}
        self.assertTrue(any(answer["engine"] == "llm" for answer in answers.values()), "Gemini answered nothing")
        for answer in answers.values():
            self.assertNotIn(self.llm.api_key, answer["reply"])
        self.assertTrue(answers["M1"]["correct"], answers["M1"]["reply"])


if __name__ == "__main__":
    unittest.main()
