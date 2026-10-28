"""API tests for the coach, bet tracker and learning endpoints, with NBA Stats replaced by local fixtures."""
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.data import validate_frame

ROSTER = {"season": "2026-27", "players": [
    {"id": "203999", "name": "Nikola Jokić", "team": "DEN", "team_id": "1610612743", "team_name": "Denver Nuggets", "from_year": 2015},
    {"id": "1627750", "name": "Jamal Murray", "team": "DEN", "team_id": "1610612743", "team_name": "Denver Nuggets", "from_year": 2016},
], "teams": [], "total_players": 2, "total_teams": 1, "provider": "test"}


def game_logs(player_ids, season, season_type, player_names=None):
    rng = np.random.default_rng(int(player_ids[0]) % 1000 + int(season[:4]))
    start = date(int(season[:4]), 10, 22)
    rows = []
    for game in range(35):
        for player_id in player_ids:
            name = (player_names or {}).get(player_id, player_id)
            base = 27 if player_id == "203999" else 21
            rows.append(dict(date=(start + timedelta(days=2 * game)).isoformat(), player_id=player_id, player_name=name, team="DEN",
                             opponent="LAL", minutes=34.0, home=bool(game % 2), points=int(rng.poisson(base)), rebounds=int(rng.poisson(8)),
                             assists=int(rng.poisson(6)), steals=int(rng.poisson(1)), blocks=int(rng.poisson(0.7)), season=season))
    return validate_frame(pd.DataFrame(rows))


class CoachApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.patches = [patch("backend.app.fetch_roster", return_value=ROSTER), patch("backend.app.import_game_logs", side_effect=game_logs)]
        for item in self.patches:
            item.start()
        self.client = TestClient(create_app(Path(self.directory.name)), base_url="http://127.0.0.1")

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.directory.cleanup()

    def chat(self, message, slip=None, conversation="api-conversation"):
        response = self.client.post("/api/assistant/chat", json={"message": message, "conversation_id": conversation,
                                                                 "slip": slip or {"legs": [], "book_odds": None}, "focus": None})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_chat_builds_a_slip_that_the_parlay_and_ledger_endpoints_accept(self):
        first = self.chat("add jokic over 26.5 points and jamal murray over 20.5 points")
        slip = first["slip"]
        self.assertEqual([leg["label"]["player"] for leg in slip["legs"]], ["Nikola Jokić", "Jamal Murray"])
        priced = self.client.post("/api/parlay", json={"legs": [leg["request"] for leg in slip["legs"]], "book_odds": None})
        self.assertEqual(priced.status_code, 200)
        self.assertIsNone(priced.json()["calibration"])
        chat_card = next(card for card in first["cards"] if card["type"] == "parlay")
        self.assertAlmostEqual(chat_card["result"]["probability"], priced.json()["probability"])
        second = self.chat("book has it at +450, good price?", slip)
        self.assertEqual(second["slip"]["book_odds"], 450)
        tracked = self.client.post("/api/ledger", json={"slip": second["slip"], "stake": 10, "event_date": "2026-10-28", "note": None})
        self.assertEqual(tracked.status_code, 201, tracked.text)
        bets = self.client.get("/api/ledger").json()
        self.assertEqual((len(bets["bets"]), bets["record"]["pending"]), (1, 1))
        self.assertEqual(self.client.delete(f"/api/ledger/{tracked.json()['id']}").status_code, 204)
        self.assertEqual(self.client.delete(f"/api/ledger/{tracked.json()['id']}").status_code, 404)

    def test_feedback_status_and_learning(self):
        reply = self.chat("help")
        self.assertEqual(self.client.post("/api/assistant/feedback", json={"turn_id": reply["turn_id"], "rating": 1, "note": None}).status_code, 200)
        self.assertEqual(self.client.post("/api/assistant/feedback", json={"turn_id": "missing123", "rating": 1, "note": None}).status_code, 404)
        status = self.client.get("/api/assistant/status").json()
        self.assertEqual((status["mode"], status["ready"]), ("built_in", True))
        learning = self.client.get("/api/learning").json()
        self.assertEqual(learning["memory"]["liked"], 1)
        self.assertFalse(learning["calibration"]["active"])
        self.chat("remember the joker means nikola jokic")
        self.assertEqual(self.client.delete("/api/learning/aliases/the%20joker").status_code, 204)
        self.assertEqual(self.client.delete("/api/learning/aliases/the%20joker").status_code, 404)
        retrained = self.client.post("/api/learning/retrain")
        self.assertEqual(retrained.status_code, 200)
        self.assertIn("calibration", retrained.json())

    def test_chat_survives_a_roster_outage_with_the_saved_roster(self):
        self.chat("what's jokic averaging")  # caches the roster
        store = self.client.app.state.store
        with store.connect() as db:
            db.execute("UPDATE provider_cache SET payload = json_set(payload, '$.fetched_at', '2000-01-01T00:00:00+00:00') WHERE key LIKE 'roster:%'")
        from backend.nba import NBAUnavailableError
        with patch("backend.app.fetch_roster", side_effect=NBAUnavailableError("NBA Stats timed out.")):
            reply = self.chat("what's jokic averaging")
        self.assertEqual(reply["cards"][0]["type"], "player")

    def test_validation(self):
        leg = {"request": {"dataset_id": "x", "player_id": "203999", "stat": "points", "line": 26.5, "side": "over"},
               "label": {"player": "Nikola Jokić", "team": "DEN"}}
        cases = [
            {"message": "", "conversation_id": "api-conversation"},
            {"message": "hi", "conversation_id": "short"},
            {"message": "x" * 1001, "conversation_id": "api-conversation"},
            {"message": "hi", "conversation_id": "api-conversation", "slip": {"legs": [leg, leg], "book_odds": None}},
            {"message": "hi", "conversation_id": "api-conversation", "slip": {"legs": [], "book_odds": 50}},
        ]
        for body in cases:
            with self.subTest(body=str(body)[:60]):
                self.assertEqual(self.client.post("/api/assistant/chat", json=body).status_code, 422)
        empty = self.client.post("/api/ledger", json={"slip": {"legs": [], "book_odds": None}, "stake": None, "event_date": None, "note": None})
        self.assertEqual(empty.status_code, 422)


if __name__ == "__main__":
    unittest.main()
