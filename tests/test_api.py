from contextlib import closing
from datetime import datetime, timedelta, timezone
from io import StringIO
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import pandas as pd
from requests.exceptions import Timeout

import without_llm  # noqa: F401  (first: unit tests never reach a real language model)
from backend.app import create_app
from backend.data import Store, demo_frame, metadata, read_csv, validate_frame
from backend.nba import NBAUnavailableError, import_game_logs
from backend.stats import PROP_STATS


class DataTests(unittest.TestCase):
    def test_demo_is_reproducible_and_complete(self):
        first = demo_frame().copy()
        self.assertEqual(len(first), 864)
        self.assertEqual(first.player_id.nunique(), 12)
        self.assertTrue((first.groupby("player_id").size() == 72).all())
        demo_frame.cache_clear()
        pd.testing.assert_frame_equal(first, demo_frame())
        roundtrip = read_csv(first.to_csv(index=False))
        pd.testing.assert_frame_equal(first, roundtrip)

    def test_bad_data_rejected(self):
        sample = demo_frame().head(20)
        cases = []
        for column, value in [("minutes", 0), ("points", -1), ("assists", 1.5),
                              ("rebounds", float("inf")), ("date", "tomorrow"),
                              ("date", "2025-02-30"), ("player_name", "=CMD()"),
                              ("player_id", "../escape"), ("home", "maybe")]:
            frame = sample.copy()
            frame[column] = frame[column].astype(object)
            frame.loc[0, column] = value
            cases.append(frame)
        cases.append(sample.drop(columns=["blocks"]))
        cases.append(pd.concat([sample, sample.head(1)]))
        for index, frame in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_frame(frame)

    def test_csv_empty_and_malformed(self):
        for text in ("", "foo,bar\n1,2"):
            with self.assertRaises(ValueError):
                read_csv(text)

    def test_missing_home_is_unknown_not_away(self):
        sample = demo_frame().head(20).drop(columns=["home"])
        normalized = validate_frame(sample)
        self.assertTrue(normalized.home.isna().all())
        self.assertTrue(read_csv(normalized.to_csv(index=False)).home.isna().all())

    def test_native_record_storage_does_not_use_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            frame = demo_frame().query("player_id == '203999'").copy()
            with patch("backend.data.read_csv", side_effect=AssertionError("CSV parser must not run")), \
                    patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("CSV serializer must not run")):
                info = store.save(frame, "API records", "nba_api")
                loaded, _ = store.load(info["id"])
                pd.testing.assert_frame_equal(frame.reset_index(drop=True), loaded)

    def test_legacy_data_migrates_once_without_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            frame = demo_frame().head(20)
            info = metadata(frame, "legacy-id", "Existing history", "nba_api")
            with closing(sqlite3.connect(path / "courtside.sqlite3")) as db, db:
                db.execute("CREATE TABLE datasets (id TEXT PRIMARY KEY, metadata TEXT NOT NULL, csv TEXT NOT NULL)")
                db.execute("INSERT INTO datasets VALUES (?, ?, ?)", ("legacy-id", json.dumps(info), frame.to_csv(index=False)))
            store = Store(path)
            with patch("backend.data.read_csv", side_effect=AssertionError("Migration must not repeat")):
                store = Store(path)
                loaded, original = store.load("legacy-id")
            pd.testing.assert_frame_equal(frame.reset_index(drop=True), loaded)
            self.assertEqual(original, info)
            with store.connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM datasets").fetchone()[0], 1)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM game_data").fetchone()[0], 1)


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.client = TestClient(create_app(self.path), base_url="http://127.0.0.1")
        self.request = {"dataset_id": "demo", "player_id": "203999", "stat": "points", "line": 26.5}

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def test_discovery_and_forecast(self):
        self.assertEqual(self.client.get("/api/health").json(), {"status": "ok"})
        datasets = self.client.get("/api/datasets").json()
        self.assertEqual(datasets["datasets"][0]["rows"], 864)
        self.assertEqual([stat["id"] for stat in datasets["stats"]], list(PROP_STATS))
        self.assertEqual({stat["group"] for stat in datasets["stats"]}, {"core", "combo", "shooting", "other", "milestone"})
        self.assertEqual(datasets["simulation"]["default"], 250_000)
        self.assertEqual(len(self.client.get("/api/players").json()["players"]), 12)
        response = self.client.post("/api/forecast", json={**self.request, "simulations": 20_000})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["sample_size"], 60)
        self.assertEqual(result["resolved_model"], "simulation")
        self.assertEqual(result["simulation"]["runs"], 20_000)
        self.assertIsNone(result["calibrated"])
        self.assertIsNone(result["injury"])
        combo = self.client.post("/api/forecast", json={**self.request, "stat": "pts_reb_ast", "line": 47.5, "simulations": 20_000})
        self.assertEqual(combo.status_code, 200, combo.text)
        self.assertEqual(combo.json()["stat"], "pts_reb_ast")

    def test_bad_requests(self):
        for extra in ({"odds_over": 0}, {"line": 2.25}, {"lookback": 1},
                      {"expected_minutes": -2}, {"stat": "turnovers"}, {"stat": "minutes"},
                      {"stat": "double_double", "line": 1.5}, {"simulations": 5_000_000},
                      {"model": "magic"}, {"rest_days": 31}, {"unknown": True}):
            with self.subTest(extra=extra):
                result = self.client.post("/api/forecast", json={**self.request, **extra})
                self.assertEqual(result.status_code, 422, result.text)
        self.assertEqual(self.client.post("/api/forecast", json={**self.request, "player_id": "missing"}).status_code, 404)
        self.assertEqual(self.client.get("/api/players?dataset_id=missing").status_code, 404)

    def test_csv_import_export_and_persistence(self):
        csv = demo_frame().query("player_id == '203999'").to_csv(index=False)
        response = self.client.post("/api/datasets/import", files={"file": ("games.csv", csv, "text/csv")}, data={"name": "My fixture"})
        self.assertEqual(response.status_code, 201, response.text)
        dataset = response.json()
        self.assertEqual(dataset["source"], "csv")
        self.assertEqual(dataset["rows"], 72)
        persisted, info = Store(self.path).load(dataset["id"])
        self.assertEqual(len(persisted), 72)
        self.assertEqual(info["name"], "My fixture")
        export = self.client.get(f"/api/datasets/{dataset['id']}/export")
        self.assertEqual(len(pd.read_csv(StringIO(export.text))), 72)
        bad = self.client.post("/api/datasets/import", files={"file": ("bad.csv", "bad,data\nx,y")})
        self.assertEqual(bad.status_code, 422)

    def test_manual_lines_persist_and_filter_by_event(self):
        line = {**self.request, "event_date": "2026-11-01", "book": "Manual entry"}
        first = self.client.post("/api/lines", json=line)
        self.assertEqual(first.status_code, 201, first.text)
        self.client.post("/api/lines", json={**line, "line": 27.5})
        self.client.post("/api/lines", json={**line, "event_date": "2026-11-02", "line": 28.5})
        records = Store(self.path).lines("demo", "203999", "points", "2026-11-01")
        self.assertEqual([row["line"] for row in records], [26.5, 27.5])
        self.assertLess(records[0]["id"], records[1]["id"])
        response = self.client.get("/api/lines", params={**{key: line[key] for key in ("dataset_id", "player_id", "stat", "event_date")}})
        self.assertEqual(len(response.json()["lines"]), 2)

    def test_external_origin_cannot_write(self):
        result = self.client.post("/api/forecast", json=self.request, headers={"Origin": "https://evil.example"})
        self.assertEqual(result.status_code, 403)
        result = self.client.post("/api/forecast", json=self.request, headers={"Origin": "http://127.0.0.1"})
        self.assertEqual(result.status_code, 200)

    def test_nba_validation_and_failure_are_visible(self):
        for request in ({"player_ids": ["not-a-number"]}, {"player_ids": ["203999"], "season": "2025-27"},
                        {"player_ids": ["203999", "203999"]}):
            self.assertEqual(self.client.post("/api/datasets/nba", json=request).status_code, 422)
        with patch("backend.app.import_game_logs", side_effect=NBAUnavailableError("NBA unavailable")):
            result = self.client.post("/api/datasets/nba", json={"player_ids": ["203999"]})
            self.assertEqual(result.status_code, 502)
            self.assertIn("NBA unavailable", result.json()["detail"])
        self.assertEqual(len(self.client.get("/api/datasets").json()["datasets"]), 1)

    @patch("backend.app.fetch_roster", return_value={"players": [{"id": "203999", "name": "Nikola Jokić"}]})
    def test_nba_search(self, provider):
        response = self.client.get("/api/nba/players?search=Jokic")
        self.assertTrue(any(player["id"] == "203999" for player in response.json()["players"]))

    @patch("backend.app.import_game_logs")
    def test_direct_nba_cache_and_native_forecast(self, provider):
        provider.return_value = demo_frame().query("player_id == '203999'").copy()
        request = {"player_ids": ["203999"], "season": "2025-26"}
        first = self.client.post("/api/nba/history", json=request)
        self.assertEqual(first.status_code, 200, first.text)
        info = first.json()
        self.assertFalse(info["cache_hit"])
        self.assertTrue(info["fetched_at"])
        self.assertEqual(info["nba_query"]["season"], "2025-26")
        second = self.client.post("/api/nba/history", json=request)
        self.assertTrue(second.json()["cache_hit"])
        self.assertEqual(second.json()["id"], info["id"])
        provider.assert_called_once()
        with patch("backend.data.read_csv", side_effect=AssertionError("No CSV parsing in NBA forecasting")):
            forecast = self.client.post("/api/forecast", json={**self.request, "dataset_id": info["id"]})
        self.assertEqual(forecast.status_code, 200, forecast.text)
        self.assertEqual(forecast.json()["dataset"]["fetched_at"], info["fetched_at"])

    @patch("backend.app.import_game_logs")
    def test_refresh_preserves_id_and_notebook_and_failed_refresh_keeps_cache(self, provider):
        frame = demo_frame().query("player_id == '203999'").copy()
        provider.return_value = frame
        request = {"player_ids": ["203999"], "season": "2025-26"}
        first = self.client.post("/api/nba/history", json=request).json()
        line = {**self.request, "dataset_id": first["id"], "event_date": "2026-11-01"}
        self.assertEqual(self.client.post("/api/lines", json=line).status_code, 201)
        revised = frame.copy()
        revised["points"] += 1
        provider.return_value = revised
        refreshed = self.client.post("/api/nba/history", json={**request, "refresh": True}).json()
        self.assertEqual(refreshed["id"], first["id"])
        self.assertFalse(refreshed["cache_hit"])
        self.assertNotEqual(refreshed["fetched_at"], first["fetched_at"])
        self.assertEqual(len(self.client.get("/api/datasets").json()["datasets"]), 2)
        self.assertEqual(len(Store(self.path).lines(first["id"], "203999", "points", None)), 1)
        provider.side_effect = NBAUnavailableError("Upstream blocked")
        failure = self.client.post("/api/nba/history", json={**request, "refresh": True})
        self.assertEqual(failure.status_code, 502)
        saved, saved_info = Store(self.path).load(first["id"])
        self.assertEqual(saved_info["fetched_at"], refreshed["fetched_at"])
        self.assertEqual(saved["points"].tolist(), revised["points"].tolist())
        self.assertIn("remains unchanged", failure.json()["detail"])

    @patch("backend.app.import_game_logs")
    def test_expired_cache_refetches_and_seasons_are_separate(self, provider):
        provider.return_value = demo_frame().query("player_id == '203999'").copy()
        request = {"player_ids": ["203999"], "season": "2025-26"}
        first = self.client.post("/api/nba/history", json=request).json()
        first["fetched_at"] = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
        with Store(self.path).connect() as db:
            db.execute("UPDATE game_data SET metadata = ? WHERE id = ?", (json.dumps(first), first["id"]))
        refreshed = self.client.post("/api/nba/history", json=request).json()
        self.assertFalse(refreshed["cache_hit"])
        self.assertEqual(provider.call_count, 2)
        other = self.client.post("/api/nba/history", json={**request, "season": "2024-25"}).json()
        self.assertNotEqual(first["id"], other["id"])

    @patch("backend.app.import_game_logs")
    def test_player_order_does_not_create_duplicate_cache(self, provider):
        provider.return_value = demo_frame().query("player_id in ['203999', '201939']").copy()
        request = {"player_ids": ["203999", "201939"], "season": "2025-26"}
        first = self.client.post("/api/nba/history", json=request).json()
        second = self.client.post("/api/nba/history", json={**request, "player_ids": ["201939", "203999"]}).json()
        self.assertEqual(first["id"], second["id"])
        self.assertTrue(second["cache_hit"])
        provider.assert_called_once()


class NBAConnectorTests(unittest.TestCase):
    @patch("backend.nba.playergamelog.PlayerGameLog")
    def test_normalizes_real_api_schema(self, mock_endpoint):
        mock_endpoint.return_value.get_data_frames.return_value = [pd.DataFrame([
            {"GAME_DATE": "OCT 25, 2025", "MATCHUP": "DEN vs. MIN", "MIN": 36,
             "PTS": 28, "REB": 12, "AST": 10, "STL": 2, "BLK": 1},
            {"GAME_DATE": "OCT 23, 2025", "MATCHUP": "DEN @ LAL", "MIN": 35,
             "PTS": 25, "REB": 11, "AST": 8, "STL": 1, "BLK": 0},
        ])]
        frame = import_game_logs(["203999"], "2025-26", "Regular Season")
        self.assertEqual(frame["date"].tolist(), ["2025-10-23", "2025-10-25"])
        self.assertEqual(frame["rest_days"].tolist(), [2, 1])
        self.assertEqual(frame["home"].tolist(), [False, True])
        self.assertEqual(frame["points"].tolist(), [25, 28])

    @patch("backend.nba.playergamelog.PlayerGameLog", side_effect=Timeout("timeout"))
    def test_timeout_does_not_substitute_demo(self, mock_endpoint):
        with self.assertRaisesRegex(NBAUnavailableError, "No synthetic"):
            import_game_logs(["203999"], "2025-26", "Regular Season")
