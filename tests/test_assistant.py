import json
import unittest
from datetime import date, timedelta

import httpx
import numpy as np
import pandas as pd

from backend.assistant import Coach, LocalMemory, leg_house, parlay_house
from backend.data import metadata, validate_frame
from backend.llm import LLMClient, LLMError
from backend import nlu

PLAYERS = [
    {"id": "203999", "name": "Nikola Jokić", "team": "DEN"}, {"id": "1627750", "name": "Jamal Murray", "team": "DEN"},
    {"id": "1631099", "name": "Keegan Murray", "team": "SAC"}, {"id": "1627749", "name": "Dejounte Murray", "team": "NOP"},
    {"id": "1628983", "name": "Shai Gilgeous-Alexander", "team": "OKC"}, {"id": "1641705", "name": "Victor Wembanyama", "team": "SAS"},
    {"id": "1630224", "name": "Jalen Green", "team": "PHX"}, {"id": "203110", "name": "Draymond Green", "team": "GSW"},
    {"id": "201939", "name": "Stephen Curry", "team": "GSW"}, {"id": "1630178", "name": "Tyrese Maxey", "team": "PHI"},
    {"id": "1629029", "name": "Luka Dončić", "team": "LAL"}, {"id": "1630169", "name": "Tyrese Haliburton", "team": "IND"},
]
MEANS = {"203999": (27, 12, 10, 1.4, 0.8), "1627750": (21, 4, 6, 1.0, 0.3), "1628983": (31, 5, 6, 1.7, 0.9),
         "1641705": (24, 11, 3, 1.1, 3.6), "201939": (26, 4, 6, 1.0, 0.4)}


def league(games: int = 40) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(11)
    start = date(2025, 11, 1)
    frames = {}
    for player in PLAYERS:
        means = MEANS.get(player["id"], (14, 5, 3, 0.8, 0.5))
        rows = []
        for game in range(games):
            day = start + timedelta(days=2 * game + (0 if player["team"] == "DEN" else hash(player["id"]) % 2))
            pace = rng.normal()
            values = [max(0, int(round(m + (0.25 * m + 1) * (0.6 * pace + 0.8 * rng.normal())))) for m in means]
            rows.append(dict(date=day.isoformat(), player_id=player["id"], player_name=player["name"], team=player["team"],
                             opponent="BOS", minutes=34.0, home=bool(game % 2),
                             **dict(zip(("points", "rebounds", "assists", "steals", "blocks"), values))))
        frames[player["id"]] = validate_frame(pd.DataFrame(rows))
    return frames


class FakeLearning(LocalMemory):
    def __init__(self):
        super().__init__()
        self.tracked, self.settled = [], 0

    def calibrate_over(self, p, stat=None):
        return None

    def calibration_status(self):
        return {"samples": 0}

    def track(self, legs, book_odds, stake, event_date, priced, note):
        bet = {"id": "b1", "event_date": event_date.isoformat(), "book_odds": book_odds, "stake": stake, "legs": legs,
               "probability": priced["probability"], "status": "pending"}
        self.tracked.append(bet)
        return bet

    def settle(self):
        self.settled += 1
        return {"settled_legs": 2, "updated_bets": 1, "warnings": [], "record": self.track_record()}

    def track_record(self):
        return {"bets": 1, "pending": 0, "won": 1, "lost": 0, "void": 0, "reduced": 0, "staked": 10.0, "profit": 45.0, "roi": 4.5,
                "legs_graded": 2, "legs_won": 2, "expected_legs_won": 1.2, "brier": 0.2}

    def status(self):
        return {"calibration": {"message": "Needs at least 800 graded predictions (has 0)."},
                "memory": {"aliases": list(self.alias_map.values()), "turns": len(self.turns), "liked": 0, "disliked": 0},
                "record": self.track_record(), "training": False}


class CoachTestCase(unittest.TestCase):
    def setUp(self):
        self.frames = league()
        self.learning = FakeLearning()
        self.history_calls = []
        self.coach = Coach(lambda: PLAYERS, self.history, self.load, learning=self.learning, today=lambda: date(2026, 10, 28))
        self.slip = {"legs": [], "book_odds": None}

    def history(self, player_id, refresh):
        self.history_calls.append(player_id)
        frame = self.frames[player_id]
        return f"ds-{player_id}", frame, metadata(frame, f"ds-{player_id}", "History", "nba_api")

    def load(self, dataset_id):
        if not dataset_id.startswith("ds-") or dataset_id[3:] not in self.frames:
            raise KeyError(dataset_id)
        frame = self.frames[dataset_id[3:]]
        return frame, metadata(frame, dataset_id, "History", "nba_api")

    def say(self, message, focus=None, conversation="conv-test-1"):
        response = self.coach.chat(message, conversation, self.slip, focus)
        json.dumps(response)
        if response["slip"] is not None:
            self.slip = response["slip"]
        return response

    def legs(self):
        return [(leg["label"]["player"], leg["request"]["stat"], leg["request"]["side"], leg["request"]["line"]) for leg in self.slip["legs"]]


class SlipCommandTests(CoachTestCase):
    def test_add_two_legs_in_one_message(self):
        response = self.say("add jokic over 26.5 points and jamal murray 21+ points")
        self.assertEqual(self.legs(), [("Nikola Jokić", "points", "over", 26.5), ("Jamal Murray", "points", "over", 20.5)])
        self.assertEqual([a["type"] for a in response["actions"]], ["add_leg", "add_leg"])
        card = response["cards"][-1]
        self.assertEqual((card["type"], card["house"]["basis"], card["house"]["odds"]), ("parlay", "standard", 264))
        self.assertEqual(card["result"]["pairs"][0]["basis"], "teammates")
        self.assertIn("Added leg 1", response["reply"])
        self.assertEqual(self.slip["legs"][0]["request"]["dataset_id"], "ds-203999")

    def test_same_player_carries_into_the_next_segment(self):
        self.say("add the joker over 26.5 points and over 9.5 dimes")
        self.assertEqual(self.legs(), [("Nikola Jokić", "points", "over", 26.5), ("Nikola Jokić", "assists", "over", 9.5)])

    def test_ambiguous_name_asks_then_learns(self):
        response = self.say("add murray over 3.5 assists")
        self.assertEqual(self.slip["legs"], [])
        options = response["clarification"]["options"]
        self.assertEqual({o["label"].split(" · ")[0] for o in options}, {"Jamal Murray", "Keegan Murray", "Dejounte Murray"})
        choice = next(o for o in options if o["label"].startswith("Jamal"))
        learned = self.say(choice["message"])
        self.assertEqual(self.legs(), [("Jamal Murray", "assists", "over", 3.5)])
        self.assertTrue(learned["learned"])
        third = self.say("add murray under 22.5 points")
        self.assertIsNone(third["clarification"])
        self.assertEqual(self.legs()[-1], ("Jamal Murray", "points", "under", 22.5))

    def test_slip_player_is_preferred_for_ambiguous_names(self):
        self.say("add jamal murray over 20.5 points")
        response = self.say("murray over 5.5 assists?")
        self.assertIsNone(response["clarification"])
        self.assertEqual(response["cards"][0]["player"]["name"], "Jamal Murray")

    def test_remove_flip_change_undo_clear(self):
        self.say("add jokic o26.5 pts, sga 30+ points and wemby over 3.5 blocks")
        self.assertEqual(len(self.slip["legs"]), 3)
        self.say("flip leg 2")
        self.assertEqual(self.legs()[1][2], "under")
        self.say("change leg 1 line to 27.5")
        self.assertEqual(self.legs()[0][3], 27.5)
        self.say("make it the under")
        self.assertEqual(self.legs()[0][2], "under")
        self.say("remove wemby")
        self.assertEqual([leg[0] for leg in self.legs()], ["Nikola Jokić", "Shai Gilgeous-Alexander"])
        undone = self.say("undo")
        self.assertEqual(undone["actions"][0]["type"], "undo")
        self.assertEqual(len(self.slip["legs"]), 3)
        self.say("remove leg 3")
        self.assertEqual(len(self.slip["legs"]), 2)
        cleared = self.say("clear my slip")
        self.assertEqual((self.slip, cleared["actions"][0]["type"]), ({"legs": [], "book_odds": None}, "clear_slip"))

    def test_remove_then_add_in_one_message_only_touches_each_part(self):
        self.say("add jokic over 26.5 points and jamal murray over 20.5 points")
        self.say("remove jamal murray and add sga over 30.5 points")
        self.assertEqual([leg[0] for leg in self.legs()], ["Nikola Jokić", "Shai Gilgeous-Alexander"])

    def test_adding_the_same_stat_replaces_the_leg(self):
        self.say("add jokic over 26.5 points")
        response = self.say("add jokic under 28.5 points")
        self.assertEqual(self.legs(), [("Nikola Jokić", "points", "under", 28.5)])
        self.assertEqual(response["actions"][0]["type"], "replace_leg")
        again = self.say("add jokic under 28.5 points")
        self.assertEqual((again["actions"], again["slip"]), ([], None))
        self.assertIn("already", again["reply"])

    def test_undo_refuses_when_the_slip_changed_outside_the_chat(self):
        self.say("add jokic over 26.5 points")
        self.slip = {"legs": [], "book_odds": None}  # the user cleared it in the UI
        response = self.say("undo")
        self.assertIsNone(response["slip"])
        self.assertIn("changed since my last edit", response["reply"])

    def test_slip_limit_and_unsupported_stats(self):
        for player in ("jokic", "jamal murray"):
            for stat, line in zip(("points", "rebounds", "assists", "steals", "blocks"), (20.5, 5.5, 4.5, 0.5, 0.5)):
                self.say(f"add {player} over {line} {stat}")
        self.assertEqual(len(self.slip["legs"]), 10)
        full = self.say("add sga over 30.5 points")
        self.assertEqual(len(self.slip["legs"]), 10)
        self.assertIn("maximum", full["reply"])
        unsupported = self.say("add steph over 30.5 minutes")
        self.assertIn("can't price minutes", unsupported["reply"])
        self.assertIsNone(unsupported["slip"])
        self.say("clear my slip")
        no_splits = self.say("add steph 4+ threes")
        self.assertIn("box-score details", no_splits["reply"])
        self.assertIsNone(no_splits["slip"])
        combo = self.say("add jokic over 45.5 PRA")
        self.assertEqual(self.legs(), [("Nikola Jokić", "pts_reb_ast", "over", 45.5)])
        self.assertIn("PRA", combo["reply"])

    def test_focus_is_used_for_this_prop(self):
        focus = {"dataset_id": "ds-203999", "player_id": "203999", "player": "Nikola Jokić", "team": "DEN", "stat": "rebounds",
                 "line": 11.5, "model": "poisson", "lookback": 20, "expected_minutes": None, "rest_days": None, "opponent": None,
                 "use_context": False}
        self.say("add the under", focus=focus)
        self.assertEqual(self.legs(), [("Nikola Jokić", "rebounds", "under", 11.5)])
        request = self.slip["legs"][0]["request"]
        self.assertEqual((request["model"], request["lookback"], request["use_context"]), ("poisson", 20, False))


class QuestionTests(CoachTestCase):
    def test_parlay_odds_against_the_house(self):
        self.say("add jokic over 26.5 points and sga over 30.5 points")
        response = self.say("book has my parlay at +450, is that good?")
        self.assertEqual(self.slip["book_odds"], 450)
        card = next(c for c in response["cards"] if c["type"] == "parlay")
        house = card["house"]
        self.assertEqual((house["basis"], house["odds"]), ("entered", 450))
        self.assertAlmostEqual(house["implied"], 1 / 5.5)
        self.assertAlmostEqual(house["edge"], card["result"]["probability"] - 1 / 5.5)
        self.assertAlmostEqual(house["house_edge"], -(card["result"]["probability"] * 4.5 - (1 - card["result"]["probability"])))
        self.assertIn("break even", response["reply"])
        odds_only = self.say("what are my odds")
        self.assertIsNone(odds_only["slip"])
        self.assertEqual(odds_only["cards"][0]["house"]["odds"], 450)

    def test_single_prop_question_with_a_price_leaves_the_slip_alone(self):
        response = self.say("jokic over 26.5 points at -115?")
        self.assertIsNone(response["slip"])
        card = response["cards"][0]
        self.assertEqual((card["type"], card["side"], card["line"]), ("forecast", "over", 26.5))
        self.assertAlmostEqual(card["house"]["implied"], 115 / 215)
        self.assertAlmostEqual(card["over"] + card["under"] + card["push"], 1.0)

    def test_hit_rate_uses_the_requested_window(self):
        response = self.say("how often has jokic gone over 26.5 points in his last 10 games")
        card = response["cards"][0]
        expected = int((self.frames["203999"].tail(10)["points"] > 26.5).sum())
        self.assertEqual((card["hit_rate"]["games"], card["hit_rate"]["over"]), (10, expected))

    def test_player_summary(self):
        response = self.say("what's SGA averaging")
        card = response["cards"][0]
        recent = self.frames["1628983"].tail(10)
        self.assertEqual(card["type"], "player")
        self.assertAlmostEqual(card["averages"]["points"], round(recent["points"].mean(), 1))
        self.assertEqual(len(card["last_games"]), 5)
        self.assertEqual(card["last_games"][0]["date"], recent["date"].iloc[-1])

    def test_explain_links(self):
        self.say("add jokic over 26.5 points and jamal murray over 20.5 points")
        response = self.say("why are my legs linked?")
        self.assertIn("Teammates on DEN", response["reply"])

    def test_teach_and_forget_nicknames(self):
        response = self.say("remember chef means stephen curry")
        self.assertIn("Stephen Curry", response["learned"][0])
        self.say("add chef over 5.5 assists")
        self.assertEqual(self.legs(), [("Stephen Curry", "assists", "over", 5.5)])
        self.say("forget chef")
        self.assertNotIn("chef", self.learning.aliases())

    def test_track_and_grade(self):
        self.say("add jokic over 26.5 points and sga over 30.5 points")
        response = self.say("track this for $10 at +450 tonight")
        self.assertEqual(self.learning.tracked[0]["book_odds"], 450)
        self.assertEqual((self.learning.tracked[0]["stake"], self.learning.tracked[0]["event_date"]), (10.0, "2026-10-28"))
        self.assertEqual(response["cards"][-1]["type"], "bet")
        graded = self.say("grade my bets")
        self.assertEqual(self.learning.settled, 1)
        self.assertEqual(graded["cards"][0]["type"], "track_record")
        learning = self.say("are you getting smarter?")
        self.assertEqual(learning["cards"][0]["type"], "learning")

    def test_help_unknown_and_wellbeing(self):
        self.assertIn("Build & price the slip", self.say("help")["reply"])
        unknown = self.say("purple monkey dishwasher")
        self.assertIn("didn't catch", unknown["reply"])
        self.assertTrue(unknown["suggestions"])
        self.assertIn("1-800-GAMBLER", self.say("I keep chasing my losses, what are my odds")["reply"])


class FollowUpTests(CoachTestCase):
    def test_a_bare_stat_answers_which_stat(self):
        asked = self.say("should i bet on jamal murray over 20.5 tonight")
        self.assertIn("Which stat for Jamal Murray?", asked["reply"])
        self.assertIn("add Jamal Murray over 20.5 points", asked["suggestions"])
        self.assertEqual(self.slip["legs"], [])
        answered = self.say("points")
        self.assertEqual(self.legs(), [("Jamal Murray", "points", "over", 20.5)])
        self.assertIn("Added leg 1", answered["reply"])

    def test_short_replies_finish_a_missing_line_or_side(self):
        self.assertIn("What line", self.say("add jokic rebounds")["reply"])
        self.assertIn("Over or under 12 rebounds", self.say("12")["reply"])
        self.say("the under")
        self.assertEqual(self.legs(), [("Nikola Jokić", "rebounds", "under", 12.0)])

    def test_a_bare_line_follows_the_player_being_discussed(self):
        self.assertEqual(self.say("how about curry")["cards"][0]["type"], "player")
        both = self.say("over or under 25.5?")
        card = both["cards"][0]
        self.assertEqual((card["type"], card["player"]["name"], card["stat"], card["line"], card["side"]),
                         ("forecast", "Stephen Curry", "points", 25.5, None))
        self.assertIn("Assuming you mean **points**", both["reply"])
        self.assertRegex(both["reply"], r"the model leans \*\*(?:over|under)\*\*|close to a coin flip")
        self.assertIsNone(both["slip"])
        under = self.say("what about the under?")["cards"][0]
        self.assertEqual((under["player"]["name"], under["stat"], under["line"], under["side"]), ("Stephen Curry", "points", 25.5, "under"))
        higher = self.say("and 27.5?")
        self.assertEqual((higher["cards"][0]["stat"], higher["cards"][0]["line"]), ("points", 27.5))
        self.assertNotIn("Assuming", higher["reply"])
        self.say("over 27.5")
        self.assertEqual(self.legs(), [("Stephen Curry", "points", "over", 27.5)])

    def test_an_ambiguous_line_asks_which_stat_then_takes_a_bare_reply(self):
        self.frames["1630178"] = self.frames["1630178"].assign(rebounds=5, assists=5)
        self.say("what's tyrese maxey averaging?")
        asked = self.say("over 4.5?")
        self.assertIn("Which stat for Tyrese Maxey at 4.5?", asked["reply"])
        self.assertEqual(asked["cards"], [])
        self.assertIn("Tyrese Maxey over 4.5 rebounds?", asked["suggestions"])
        card = self.say("assists")["cards"][0]
        self.assertEqual((card["player"]["name"], card["stat"], card["line"], card["side"]), ("Tyrese Maxey", "assists", 4.5, "over"))
        self.assertEqual(self.slip["legs"], [])

    def test_new_requests_are_not_read_as_follow_ups(self):
        self.say("how about curry")
        self.assertNotIn("Stephen Curry", self.say("who has the most rebounds?")["reply"])
        self.say("how about curry")
        self.assertIn("10 or more in two of", self.say("what's a double-double?")["reply"])
        self.say("how about curry")
        self.say("add jokic over 26.5 points")
        self.assertEqual(self.legs(), [("Nikola Jokić", "points", "over", 26.5)])
        self.assertEqual(self.say("what are my odds?")["cards"][0]["type"], "parlay")
        self.say("how about curry")
        self.say("remove leg 1")
        self.assertEqual(self.slip["legs"], [])
        self.say("how about curry")
        self.assertIn("didn't", self.say("purple monkey dishwasher")["reply"])
        self.assertNotIn("Stephen Curry", self.say("over or under 25.5?", conversation="conv-test-2")["reply"])

    def test_a_question_after_a_prompt_is_answered_not_added(self):
        self.say("should i bet on jamal murray over 20.5 tonight")
        definition = self.say("what's PRA?")
        self.assertIn("points + rebounds + assists", definition["reply"])
        self.assertEqual(self.slip["legs"], [])
        self.say("add jokic over or under 12.5 rebounds")
        hit_rate = self.say("what's the over hit rate?")
        self.assertEqual((hit_rate["cards"][0]["type"], hit_rate["cards"][0]["stat"]), ("forecast", "rebounds"))
        self.assertEqual(self.slip["legs"], [])
        self.frames["1630178"] = self.frames["1630178"].assign(rebounds=5, assists=5)
        self.say("what's tyrese maxey averaging?")
        self.say("over 4.5?")
        self.say("add assists")
        self.assertEqual(self.legs(), [("Tyrese Maxey", "assists", "over", 4.5)])

    def test_the_market_being_discussed_carries_to_a_new_line(self):
        self.frames["201939"] = self.frames["201939"].assign(turnovers=lambda frame: frame["assists"] // 2)
        self.say("stephen curry over 2.5 turnovers?")
        card = self.say("what about 3.5?")["cards"][0]
        self.assertEqual((card["stat"], card["line"]), ("turnovers", 3.5))


class HouseMathTests(unittest.TestCase):
    def test_standard_and_entered_parlay_prices(self):
        standard = parlay_house(0.2, 3, None)
        self.assertEqual((standard["basis"], standard["odds"]), ("standard", 596))
        self.assertAlmostEqual(standard["decimal"], (1 + 100 / 110) ** 3)
        entered = parlay_house(0.2, 3, 450)
        self.assertAlmostEqual(entered["expected_roi"], 0.2 * 4.5 - 0.8)
        self.assertEqual(entered["verdict"], "value")
        self.assertEqual(parlay_house(0.1, 3, 450)["verdict"], "house")
        self.assertIn("$1.75 per $1", parlay_house(0.25, 4, 1000)["note"])

    def test_single_prop_push_refund(self):
        house = leg_house(0.45, 0.45, -110)
        self.assertAlmostEqual(house["model"], 0.5)
        self.assertAlmostEqual(house["expected_roi"], 0.45 * (100 / 110) - 0.45)
        self.assertIn("excluding pushes", house["note"])


def fake_llm(responses: list[dict], seen: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": responses.pop(0)}]})
    return LLMClient("https://llm.example/v1", "test-model", "sk-test", transport=httpx.MockTransport(handler))


class LanguageModelTests(CoachTestCase):
    def test_tool_calls_build_the_slip_and_numbers_come_from_tools(self):
        seen = []
        self.coach.llm = fake_llm([
            {"content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "add_leg", "arguments": json.dumps(
                {"player": "Nikola Jokić", "stat": "rebounds", "line": 11.5, "side": "over"})}}]},
            {"content": None, "tool_calls": [{"id": "2", "type": "function", "function": {"name": "price_slip", "arguments": "{}"}}]},
            {"content": "Added **Jokić over 11.5 rebounds**; the slip is priced above."},
        ], seen)
        response = self.say("build me something with the joker on the glass")
        self.assertEqual(response["engine"], {"mode": "llm", "model": "test-model"})
        self.assertEqual(self.legs(), [("Nikola Jokić", "rebounds", "over", 11.5)])
        self.assertIn("priced above", response["reply"])
        self.assertTrue(any(card["type"] == "parlay" for card in response["cards"]))
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[0]["tools"][0]["function"]["name"], "get_slip")
        self.assertEqual(seen[1]["messages"][-1]["role"], "tool")
        self.assertIn("never answer current NBA facts from memory", seen[0]["messages"][0]["content"])

    def test_failures_fall_back_to_the_built_in_coach(self):
        def broken(request):
            raise httpx.ConnectError("refused")
        self.coach.llm = LLMClient("http://127.0.0.1:9/v1", "local", transport=httpx.MockTransport(broken))
        response = self.say("build me something spicy")
        self.assertEqual(response["engine"]["mode"], "built_in")
        self.assertIn("Parlay tab", response["reply"])
        status = self.coach.status()
        self.assertFalse(status["ready"])
        self.assertNotIn("sk-", json.dumps(status))

    def test_bad_tool_arguments_are_reported_to_the_model(self):
        seen = []
        self.coach.llm = fake_llm([
            {"content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "add_leg", "arguments": json.dumps(
                {"player": "Nobody Atall", "stat": "points", "line": 10.5, "side": "over"})}}]},
            {"content": "I couldn't find that player."},
        ], seen)
        response = self.say("put that one guy in")
        self.assertEqual(self.slip["legs"], [])
        self.assertIn("error", seen[1]["messages"][-1]["content"])
        self.assertEqual(response["reply"], "I couldn't find that player.")

    def test_configuration_from_environment(self):
        self.assertIsNone(LLMClient.from_env({}))
        openai = LLMClient.from_env({"COURTSIDE_LLM_MODEL": "gpt-x", "OPENAI_API_KEY": "sk-1"})
        self.assertEqual((openai.endpoint, openai.model), ("api.openai.com", "gpt-x"))
        local = LLMClient.from_env({"COURTSIDE_LLM_MODEL": "llama3.2"})
        self.assertEqual(local.endpoint, "127.0.0.1:11434")
        with self.assertRaises(ValueError):
            LLMClient("ftp://bad", "m")
        error = LLMClient("https://llm.example/v1", "m", transport=httpx.MockTransport(lambda r: httpx.Response(401, json={})))
        with self.assertRaisesRegex(LLMError, "HTTP 401"):
            error.complete([], [])

    def test_models_without_temperature_and_safe_error_codes(self):
        calls = []

        def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            if "temperature" in body:
                return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature' does not support 0.2.",
                                                           "code": "unsupported_value"}})
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        client = LLMClient("https://llm.example/v1", "m", "sk-test", transport=httpx.MockTransport(handler))
        self.assertEqual(client.complete([], [])["content"], "ok")
        client.complete([], [])
        self.assertEqual(["temperature" in body for body in calls], [True, False, False])
        denied = LLMClient("https://llm.example/v1", "m", "sk-secret", transport=httpx.MockTransport(lambda r: httpx.Response(
            401, json={"error": {"message": "Incorrect API key provided: sk-secr***cret.", "code": "invalid_api_key"}})))
        with self.assertRaises(LLMError) as raised:
            denied.complete([], [])
        self.assertIn("HTTP 401 (invalid_api_key)", str(raised.exception))
        self.assertNotIn("sk-", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
