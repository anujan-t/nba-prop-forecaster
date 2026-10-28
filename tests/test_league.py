from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from nba_api.stats.static import teams
import pandas as pd

from backend.app import create_app
from backend.data import demo_frame, validate_frame
from backend.league import career_seasons, current_season, fetch_profile, fetch_roster, shooting
from backend.nba import NBAUnavailableError, NoGamesError, import_game_logs


def roster_fixture():
    franchises = teams.get_teams()
    players = []
    for index in range(521):
        team = franchises[index % len(franchises)]
        players.append({
            "id": str(2000000000 + index), "name": f"Player {index}",
            "team": team["abbreviation"], "team_id": str(team["id"]), "team_name": team["full_name"],
            "from_year": 2010, "to_year": 2026, "has_played": index != 1,
        })
    players[0].update(id="203999", name="Nikola Jokic")
    players[1].update(id="1643411", name="Darius Acuff Jr.", from_year=2026)
    return {
        "season": "2026-27", "players": players, "total_players": 521, "total_teams": 30,
        "teams": [{"id": str(team["id"]), "abbreviation": team["abbreviation"], "name": team["full_name"],
                   "count": sum(p["team_id"] == str(team["id"]) for p in players)} for team in franchises],
        "provider": "Unit-test roster fixture",
    }


def game_fixture(player_id, season, count):
    frame = demo_frame().query("player_id == '203999'").head(count).copy()
    frame["player_id"] = player_id
    frame["player_name"] = "Darius Acuff Jr." if player_id == "1643411" else "Nikola Jokic"
    frame["date"] = pd.date_range(f"{season[:4]}-10-01", periods=count, freq="2D").strftime("%Y-%m-%d")
    frame["season"] = season
    frame["game_id"] = [f"{season[:4]}-{i}" for i in range(count)]
    frame["win"] = "W"
    for field, value in {
        "field_goals_made": 5, "field_goals_attempted": 12, "three_pointers_made": 1,
        "three_pointers_attempted": 4, "free_throws_made": 2, "free_throws_attempted": 3,
        "turnovers": 2, "fouls": 2, "plus_minus": -4, "offensive_rebounds": 1, "defensive_rebounds": 4,
    }.items():
        frame[field] = value
    return validate_frame(frame)


class LeagueProviderTests(unittest.TestCase):
    @patch("backend.league.commonallplayers.CommonAllPlayers")
    def test_every_current_roster_row_is_preserved_including_nonparticipants(self, endpoint):
        fixture = roster_fixture()
        rows = [{
            "PERSON_ID": int(p["id"]), "DISPLAY_FIRST_LAST": p["name"], "ROSTERSTATUS": 1,
            "TEAM_ID": int(p["team_id"]), "TEAM_ABBREVIATION": p["team"], "TEAM_CITY": "",
            "TEAM_NAME": p["team_name"], "FROM_YEAR": str(p["from_year"]), "TO_YEAR": "2026",
            "GAMES_PLAYED_FLAG": "Y" if p["has_played"] else "N",
        } for p in fixture["players"]]
        rows.append({**rows[0], "PERSON_ID": 999, "ROSTERSTATUS": 0})
        endpoint.return_value.get_data_frames.return_value = [pd.DataFrame(rows)]
        roster = fetch_roster()
        self.assertEqual(roster["total_players"], 521)
        self.assertEqual(roster["total_teams"], 30)
        self.assertEqual(sum(t["count"] for t in roster["teams"]), 521)
        self.assertFalse(next(p for p in roster["players"] if p["id"] == "1643411")["has_played"])
        self.assertFalse(any(p["id"] == "999" for p in roster["players"]))
        endpoint.return_value.get_data_frames.return_value = [pd.DataFrame(rows[:1])]
        with self.assertRaisesRegex(NBAUnavailableError, "partial directory"):
            fetch_roster()

    @patch("backend.nba.playergamelog.PlayerGameLog")
    def test_new_rookie_missing_from_static_directory_can_load_logs(self, endpoint):
        endpoint.return_value.get_data_frames.return_value = [pd.DataFrame([{
            "GAME_DATE": "Oct 25, 2026", "MATCHUP": "SAC vs. MEM", "MIN": 36,
            "PTS": 22, "REB": 3, "AST": 7, "STL": 0, "BLK": 1,
            "FGM": 9, "FGA": 23, "FG3M": 2, "FG3A": 9, "FTM": 2, "FTA": 2,
            "TOV": 1, "PF": 4, "PLUS_MINUS": -8, "Game_ID": "0022600118", "WL": "L",
        }])]
        with patch("backend.nba.players.get_players", return_value=[]):
            frame = import_game_logs(["1643411"], "2026-27", "Regular Season", player_names={"1643411": "Darius Acuff Jr."})
        self.assertEqual(frame.iloc[0]["player_name"], "Darius Acuff Jr.")
        self.assertEqual(frame.iloc[0]["three_pointers_made"], 2)
        self.assertEqual(frame.iloc[0]["plus_minus"], -8)
        self.assertEqual(frame.iloc[0]["game_id"], "0022600118")

    def test_shooting_uses_total_attempts_and_missing_values_stay_unknown(self):
        frame = pd.DataFrame({"field_goals_made": [5, 0], "field_goals_attempted": [5, 10],
                              "three_pointers_made": [0, 0], "three_pointers_attempted": [0, 0]})
        result = shooting(frame)
        self.assertAlmostEqual(result["field_goal"], 1 / 3)
        self.assertIsNone(result["three_point"])
        self.assertIsNone(result["free_throw"])

    def test_traded_career_season_uses_total_once(self):
        frame = pd.DataFrame([
            {"SEASON_ID": "2025-26", "TEAM_ID": 1, "TEAM_ABBREVIATION": "AAA", "GP": 10, "PTS": 100},
            {"SEASON_ID": "2025-26", "TEAM_ID": 2, "TEAM_ABBREVIATION": "BBB", "GP": 20, "PTS": 400},
            {"SEASON_ID": "2025-26", "TEAM_ID": 0, "TEAM_ABBREVIATION": "TOT", "GP": 30, "PTS": 500},
        ])
        result = career_seasons(frame)
        self.assertEqual(result[0]["games"], 30)
        self.assertAlmostEqual(result[0]["averages"]["points"], 500 / 30)
        self.assertIsNone(result[0]["averages"]["blocks"])
        self.assertIsNone(result[0]["shooting"]["field_goal"])

    @patch("backend.league.commonplayerinfo.CommonPlayerInfo")
    def test_profile_keeps_rookie_zero_experience_and_unknown_fields(self, endpoint):
        endpoint.return_value.get_data_frames.return_value = [
            pd.DataFrame([{"PERSON_ID": 1643411, "DISPLAY_FIRST_LAST": "Darius Acuff Jr.",
                           "SEASON_EXP": 0, "JERSEY": "0", "POSITION": None, "BIRTHDATE": None}]),
            pd.DataFrame(), pd.DataFrame({"SEASON_ID": ["12026", "22026"]}),
        ]
        profile = fetch_profile("1643411")
        self.assertEqual(profile["experience"], 0)
        self.assertEqual(profile["jersey"], "0")
        self.assertIsNone(profile["age"])
        self.assertIsNone(profile["position"])
        self.assertEqual(profile["available_seasons"], ["2026-27"])


class LeagueAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.provider = patch("backend.app.fetch_roster", return_value=roster_fixture()).start()
        self.season = patch("backend.app.current_season", return_value="2026-27").start()
        self.client = TestClient(create_app(Path(self.temp.name)), base_url="http://127.0.0.1")

    def tearDown(self):
        self.client.close()
        patch.stopall()
        self.temp.cleanup()

    def test_roster_search_is_not_limited_to_cached_players_or_25_matches(self):
        result = self.client.get("/api/nba/roster")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["total_players"], 521)
        self.assertEqual(len(self.client.get("/api/nba/players").json()["players"]), 521)
        self.assertEqual(len(self.client.get("/api/nba/players?search=Player").json()["players"]), 519)
        self.assertEqual(self.client.get("/api/nba/players?search=Acuff").json()["players"][0]["id"], "1643411")
        self.provider.assert_called_once()

    @patch("backend.app.import_game_logs")
    def test_recent_veteran_history_pools_seasons_and_caches_details(self, games):
        games.side_effect = lambda ids, season, kind, player_names: game_fixture(ids[0], season, 60 if season == "2025-26" else 3)
        body = {"player_id": "203999", "window": "recent"}
        response = self.client.post("/api/nba/player-history", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["summary"]["games"], 63)
        self.assertEqual(data["queried_seasons"], ["2025-26", "2026-27"])
        self.assertEqual(len(data["games"]), 63)
        self.assertTrue(any(split["label"] == "Season 2026-27" and split["games"] == 3 for split in data["splits"]))
        self.assertTrue(any("pools" in warning for warning in data["warnings"]))
        repeated = self.client.post("/api/nba/player-history", json=body).json()
        self.assertTrue(repeated["cache_hit"])
        self.assertEqual(games.call_count, 2)
        refreshed = self.client.post("/api/nba/player-history", json={**body, "refresh": True}).json()
        self.assertEqual(refreshed["history"]["id"], data["history"]["id"])
        self.assertEqual(games.call_count, 4)

    @patch("backend.app.import_game_logs")
    def test_current_rookie_is_available_with_limited_history(self, games):
        games.side_effect = lambda ids, season, kind, player_names: game_fixture(ids[0], season, 3)
        response = self.client.post("/api/nba/player-history", json={"player_id": "1643411"})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["status"], "limited")
        self.assertEqual(data["summary"]["games"], 3)
        self.assertEqual(data["queried_seasons"], ["2026-27"])
        self.assertEqual(games.call_count, 1)
        self.assertIn("1643411", games.call_args.kwargs["player_names"])
        forecast = self.client.post("/api/forecast", json={"player_id": "1643411", "dataset_id": data["history"]["id"]})
        self.assertEqual(forecast.status_code, 422)

    @patch("backend.app.fetch_profile", return_value={"id": "1643411", "name": "Darius Acuff Jr."})
    @patch("backend.app.import_game_logs", side_effect=NoGamesError("No completed games"))
    def test_no_games_is_not_an_invalid_player_and_does_not_make_fake_dataset(self, games, profile):
        first = self.client.post("/api/nba/player-history", json={"player_id": "1643411"}).json()
        self.assertEqual(first["status"], "empty")
        self.assertIsNone(first["history"])
        self.assertEqual(first["games"], [])
        self.assertEqual(len(self.client.get("/api/datasets").json()["datasets"]), 1)
        self.assertEqual(self.client.get("/api/nba/players/1643411/profile").status_code, 200)
        second = self.client.post("/api/nba/player-history", json={"player_id": "1643411"}).json()
        self.assertTrue(second["cache_hit"])
        games.assert_called_once()

    @patch("backend.app.import_game_logs")
    def test_upstream_failure_does_not_silently_drop_a_requested_season(self, games):
        games.side_effect = [game_fixture("203999", "2025-26", 20), NBAUnavailableError("Blocked")]
        response = self.client.post("/api/nba/player-history", json={"player_id": "203999"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(self.client.get("/api/datasets").json()["datasets"]), 1)

    def test_invalid_window_and_unknown_roster_player(self):
        self.assertEqual(self.client.post("/api/nba/player-history", json={"player_id": "203999", "window": "2025-27"}).status_code, 422)
        self.assertEqual(self.client.post("/api/nba/player-history", json={"player_id": "not-an-id"}).status_code, 422)
        self.assertEqual(self.client.post("/api/nba/player-history", json={"player_id": "999"}).status_code, 404)
