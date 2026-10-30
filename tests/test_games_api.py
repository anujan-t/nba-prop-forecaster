"""API tests for moneylines and spreads: the team model endpoints, game legs in /api/parlay and the bet ledger."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import news
from backend.app import create_app
from test_coach_api import ROSTER
from test_coach_league import TODAY, FeedPatch
from test_teams import FakeFetch, TRUE


def leg(**changes) -> dict:
    base = {"kind": "game", "game_id": "1", "date": TODAY.isoformat(), "team": "DEN", "opponent": "GSW", "home": False,
            "market": "moneyline", "line": None}
    return {**base, **changes}


def final_board(day):
    games = []
    if day == TODAY:
        games = [{"id": "1", "start": "2026-10-28T23:30Z", "state": "post", "detail": "Final",
                  "home": {"team": "GSW", "name": "Golden State Warriors", "record": "3-2", "score": 108, "winner": False},
                  "away": {"team": "DEN", "name": "Denver Nuggets", "record": "3-2", "score": 112, "winner": True},
                  "odds": None, "broadcast": None, "venue": None, "note": None}]
    return {"date": day.isoformat(), "games": games, "provider": "fixture"}


class GamesApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.feeds = FeedPatch().__enter__()
        self.patches = [patch("backend.newsroom.eastern_today", return_value=TODAY), patch("backend.teams.fetch_team_games", FakeFetch()),
                        patch("backend.app.fetch_roster", return_value=ROSTER)]
        for item in self.patches:
            item.start()
        self.client = TestClient(create_app(Path(self.directory.name)), base_url="http://127.0.0.1")

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.feeds.__exit__()
        self.directory.cleanup()

    def test_model_endpoint_and_the_games_feed(self):
        model = self.client.get("/api/games/model")
        self.assertEqual(model.status_code, 200, model.text)
        status = model.json()
        self.assertEqual((len(status["ratings"]), status["season"], status["sigma"]), (len(TRUE), "2026-27", 13.0))
        self.assertEqual(status["backtest"]["season"], "2025-26")
        feed = self.client.get(f"/api/news/games?day={TODAY.isoformat()}").json()
        self.assertIsNone(feed["model_error"])
        view = feed["games"][0]["model"]
        self.assertEqual(view["markets"]["moneyline"]["away"]["book"]["odds"], 115)
        self.assertEqual(view["markets"]["spread"]["home"]["line"], -2.5)
        self.assertAlmostEqual(view["home_win"] + view["away_win"], 1.0)
        past = self.client.get(f"/api/news/games?day={(TODAY.replace(day=27)).isoformat()}").json()
        self.assertIsNone(past["games"][0]["model"])

    def test_parlay_prices_game_legs_next_to_props(self):
        response = self.client.post("/api/parlay", json={"legs": [leg(), leg(market="spread", line=2.5)], "book_odds": 200})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual([item["kind"] for item in result["legs"]], ["game", "game"])
        self.assertEqual((result["legs"][0]["book"]["odds"], result["legs"][1]["book"]["line"]), (115, 2.5))
        self.assertEqual(result["pairs"][0]["basis"], "same_game")
        self.assertNotIn("pmf", result["legs"][0])
        self.assertAlmostEqual(result["probability"], result["legs"][0]["probability"], places=6)
        legacy = {"dataset_id": "demo", "player_id": "203999", "stat": "points", "line": 26.5, "side": "over"}
        mixed = self.client.post("/api/parlay", json={"legs": [legacy, leg(game_id="2", team="BOS", opponent="NYK", home=True)]})
        self.assertEqual(mixed.status_code, 200, mixed.text)
        self.assertEqual([item["kind"] for item in mixed.json()["legs"]], ["player", "game"])

    def test_game_leg_validation(self):
        cases = [leg(line=2.5), leg(market="spread"), leg(market="spread", line=2.3), leg(opponent="DEN"), leg(kind="total"),
                 leg(game_id="abc"), leg(market="spread", line=75.0), {**leg(), "extra": 1}]
        for item in cases:
            with self.subTest(leg=item):
                self.assertEqual(self.client.post("/api/parlay", json={"legs": [item]}).status_code, 422)
        contradictory = self.client.post("/api/parlay", json={"legs": [leg(), leg(team="GSW", opponent="DEN", home=True, market="spread", line=-2.5)]})
        self.assertEqual(contradictory.status_code, 422)
        self.assertIn("can't both win", contradictory.text)
        twice = {"legs": [{"request": leg(), "label": {"player": "Denver Nuggets", "team": "DEN"}},
                          {"request": leg(team="GSW", opponent="DEN", home=True), "label": {"player": "Golden State Warriors", "team": "GSW"}}],
                 "book_odds": None}
        response = self.client.post("/api/ledger", json={"slip": twice})
        self.assertEqual(response.status_code, 422)
        self.assertIn("each game's moneyline and spread", response.text)

    def test_a_tracked_moneyline_is_graded_from_the_final_score(self):
        slip = {"legs": [{"request": leg(), "label": {"player": "Denver Nuggets", "team": "DEN"}},
                         {"request": leg(market="spread", line=2.5), "label": {"player": "Denver Nuggets", "team": "DEN"}}], "book_odds": 150}
        tracked = self.client.post("/api/ledger", json={"slip": slip, "stake": 10, "event_date": TODAY.isoformat()})
        self.assertEqual(tracked.status_code, 201, tracked.text)
        self.assertEqual([item["kind"] for item in tracked.json()["legs"]], ["game", "game"])
        with patch.object(news, "fetch_scoreboard", side_effect=final_board), patch("backend.newsroom.fresh", return_value=False):
            settled = self.client.post("/api/ledger/settle").json()
        self.assertEqual(settled["settled_legs"], 2)
        bet = self.client.get("/api/ledger").json()["bets"][0]
        self.assertEqual((bet["status"], bet["profit"]), ("won", 15.0))
        self.assertEqual([item["actual"] for item in bet["legs"]], [4, 4])

    def test_coach_game_legs_round_trip_through_the_api(self):
        def chat(message, slip):
            response = self.client.post("/api/assistant/chat", json={"message": message, "conversation_id": "games-conversation",
                                                                     "slip": slip, "focus": None})
            self.assertEqual(response.status_code, 200, response.text)
            return response.json()
        slip = chat("add nuggets moneyline", {"legs": [], "book_odds": None})["slip"]
        self.assertEqual([(item["request"]["team"], item["request"]["market"]) for item in slip["legs"]], [("DEN", "moneyline")])
        again = chat("add nuggets moneyline", slip)
        self.assertEqual((again["actions"], again["slip"]), ([], None))
        self.assertIn("already", again["reply"])
        priced = self.client.post("/api/parlay", json={"legs": [item["request"] for item in slip["legs"]], "book_odds": None})
        self.assertEqual(priced.status_code, 200, priced.text)


if __name__ == "__main__":
    unittest.main()
