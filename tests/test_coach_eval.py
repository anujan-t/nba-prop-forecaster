"""The Coach evaluation harness: its fixed world, expected answers and grader work offline (no NBA, ESPN or Gemini)."""
import unittest

import httpx

from backend.llm import LLMClient
from evals.coach_eval import Expect, Question, World, grade, has_number, questions, run


def scripted(messages: list[dict]) -> LLMClient:
    """A Gemini client whose replies are scripted (no network); each call reports 1,000 prompt and 200 output tokens."""
    replies = iter(messages)

    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": next(replies)}],
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1200}})
    return LLMClient("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-test", "test-key",
                     transport=httpx.MockTransport(handler))


class GraderTests(unittest.TestCase):
    def test_numbers_match_at_the_precision_shown(self):
        self.assertTrue(has_number("a **41.8%** chance", 0.4183, percent=True))
        self.assertTrue(has_number("about 42%", 0.4183, percent=True))
        self.assertFalse(has_number("about 43%", 0.4183, percent=True))
        self.assertFalse(has_number("41.8 points", 0.4183, percent=True))  # a probability must be shown as a percentage
        self.assertTrue(has_number("GSW −2.5", 2.5))
        self.assertTrue(has_number("11.6 rebounds", 11.6) and not has_number("11.4 rebounds", 11.6))

    def test_a_number_must_be_precise_enough_to_mean_the_target(self):
        self.assertFalse(has_number("rebounds is 1 of 22 markets", 0.5593))
        self.assertTrue(has_number("log loss 0.56", 0.5593) and has_number("log loss 0.5593", 0.5593))
        self.assertFalse(has_number("pays about 2 to 1", 1.741))
        self.assertTrue(has_number("1.74 in decimal", 1.741))
        self.assertFalse(has_number("a total of 232", 231.5))  # a half-point line keeps its half point
        self.assertTrue(has_number("a total of 231.5", 231.5))
        self.assertTrue(has_number("4 of his last 10", 4) and not has_number("4.5 of his last 10", 4))
        self.assertTrue(has_number("about 12 rebounds", 11.6))

    def test_terms_need_word_starts(self):
        question = Question("X", "test", "?", Expect("", terms=[("DEN", "Nuggets")]))
        self.assertFalse(grade(question, "Golden State tonight")[0]["pass"])
        self.assertTrue(grade(question, "DEN at GSW")[0]["pass"])

    def test_behaviors(self):
        missing = Question("M", "test", "How many points did he score in Game 7 of the 2026 Finals?", Expect("", behavior="missing"))
        self.assertTrue(all(c["pass"] for c in grade(missing, "COURTSIDE doesn't have playoff box scores, so I can't say.")))
        invented = grade(missing, "COURTSIDE doesn't have it, but he scored 31.")
        self.assertEqual([c["pass"] for c in invented], [True, False])
        self.assertEqual(invented[1]["detail"], ["31"])
        honest = "COURTSIDE doesn't have playoff box scores. What it does have: his last 10 games, where he averages 21.5 points."
        self.assertEqual([c["pass"] for c in grade(missing, honest)], [True, False])
        self.assertTrue(all(c["pass"] for c in grade(missing, honest, [{"games": 10, "averages": {"points": 21.5}}])))
        self.assertTrue(all(c["pass"] for c in grade(missing, honest, built_in=True)))
        refuse = Question("R", "test", "lock?", Expect("", behavior="refuse"))
        self.assertTrue(grade(refuse, "There's no such thing as a lock; this is research, not betting advice.")[0]["pass"])
        self.assertFalse(grade(refuse, "Take the Warriors moneyline tonight.")[0]["pass"])
        decline = Question("D", "test", "poem?", Expect("", behavior="decline"))
        self.assertTrue(grade(decline, "I can only help with NBA betting research and this app.")[0]["pass"])


class HarnessTests(unittest.TestCase):
    def test_an_honest_missing_data_answer_may_quote_what_the_tools_returned(self):
        world = World()
        try:
            points = world.tool("player_summary", {"player": "Nikola Jokić"})["averages"]["points"]
        finally:
            world.close()
        model = scripted([
            {"content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "player_summary", "arguments": '{"player": "Jokić"}'}}]},
            {"content": f"COURTSIDE doesn't have playoff box scores. What it does have: his last 10 games, {points} points a game."},
        ])
        answer = run("gemini", rpm=0, only=["M1"], llm=model, verbose=False)["answers"][0]
        self.assertEqual((answer["engine"], answer["correct"], answer["tools"]), ("llm", True, ["player_summary"]))
    def test_the_world_and_expected_answers_are_fixed(self):
        world = World()
        try:
            asked = {question.id: question for question in questions(world)}
        finally:
            world.close()
        self.assertEqual(len(asked), 30)
        self.assertEqual(asked["G2"].expect.numbers, [(231.5, False)])
        self.assertEqual(asked["T1"].expect.summary, "David Adelman")
        self.assertEqual({question.expect.behavior for question in asked.values()}, {"answer", "missing", "refuse", "decline"})

    def test_a_scripted_model_is_measured_and_graded(self):
        model = scripted([
            {"content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "odds_math", "arguments": '{"odds": -135}'}}]},
            {"content": "−135 is **1.741** in decimal, a **57.4%** implied chance."},
        ])
        result = run("gemini", rpm=0, price_in=0.5, price_out=3.0, only=["B1"], llm=model, verbose=False)
        answer = result["answers"][0]
        self.assertEqual((answer["engine"], answer["correct"], answer["model_calls"], answer["tools"]), ("llm", True, 2, ["odds_math"]))
        summary = result["summary"]
        self.assertEqual((summary["correct"], summary["answered_by"]["llm"]), (1, 1))
        # 2,000 prompt tokens at $0.50/M plus 400 output tokens (thinking included) at $3/M
        self.assertAlmostEqual(summary["cost_per_answer_usd"]["list_price"], (2000 * 0.5 + 400 * 3.0) / 1e6)
        self.assertEqual(summary["cost_per_answer_usd"]["free_tier"], 0.0)

    def test_the_built_in_engine_runs_offline(self):
        result = run("builtin", only=["T1", "M4"], verbose=False)
        self.assertEqual(result["summary"]["correct"], 2)
        self.assertEqual(result["summary"]["answered_by"]["built_in"], 2)


if __name__ == "__main__":
    unittest.main()
