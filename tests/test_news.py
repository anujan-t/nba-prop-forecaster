from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import without_llm  # noqa: F401  (first: unit tests never reach a real language model)
from backend import news
from backend.app import create_app

ROSTER = [{"id": "1630169", "name": "Tyrese Haliburton", "team": "IND"}, {"id": "1630162", "name": "Anthony Edwards", "team": "MIN"},
          {"id": "1629029", "name": "Luka Dončić", "team": "LAL"}, {"id": "203999", "name": "Nikola Jokić", "team": "DEN"}]

HEADLINES = {"articles": [
    {"id": 1, "type": "HeadlineNews", "headline": "Pacers' Tyrese Haliburton (hamstring) out at least 2 weeks",
     "description": "Indiana will be without its star guard.", "published": "2026-10-28T22:13:47Z",
     "links": {"web": {"href": "https://www.espn.com/nba/story/_/id/1"}}, "images": [{"url": "https://a.espncdn.com/x.jpg"}],
     "categories": [{"type": "team", "teamId": 11, "description": "Indiana Pacers"},
                    {"type": "athlete", "athleteId": 4396993, "description": "Tyrese Haliburton"}]},
    {"id": 2, "type": "HeadlineNews", "headline": "Mavericks exercise 2027-28 option on Zaccharie Risacher deal",
     "published": "2026-10-28T23:09:25Z", "categories": [{"type": "team", "teamId": 6}]},
    {"id": 3, "type": "Recap", "headline": "Spurs beat Kings behind Wembanyama", "published": "2026-10-28T02:42:47Z", "categories": []},
    {"id": 4, "type": "HeadlineNews", "headline": "Sources: Knicks assistant coach Chris Jent steps away", "published": "2026-10-28T01:03:38Z",
     "categories": [{"type": "team", "teamId": 18}]},
    {"id": 5, "type": "Story", "headline": "Wednesday's fantasy hoops adds and bets", "published": "2026-10-28T17:01:59Z", "categories": []},
    {"id": 6, "type": "Media", "headline": "Ernie Johnson announces his last year", "description": "Ernie Johnson announces his last year",
     "published": "2026-10-28T18:33:05Z", "links": {"web": {"href": "javascript:alert(1)"}}},
]}
INJURIES = {"injuries": [
    {"id": "11", "displayName": "Indiana Pacers", "injuries": [
        {"status": "Day-To-Day", "date": "2026-10-27T10:00Z", "shortComment": "Probable for Friday.",
         "athlete": {"displayName": "Bench Guy", "team": {"abbreviation": "IND"}, "position": {"abbreviation": "G"}}, "details": {}},
        {"status": "Out", "date": "2026-10-28T22:00Z", "shortComment": "Hamstring strain.", "longComment": "Out at least two weeks.",
         "athlete": {"displayName": "Tyrese Haliburton", "team": {"abbreviation": "IND"}, "position": {"abbreviation": "PG"},
                     "headshot": {"href": "https://a.espncdn.com/h.png"}},
         "details": {"type": "Hamstring", "detail": "Strain", "side": "Left", "returnDate": "2026-11-12"}},
    ]},
    {"id": "26", "displayName": "Utah Jazz", "injuries": [
        {"status": "Questionable", "athlete": {"displayName": "Someone Else", "team": {"abbreviation": "UTAH"}}, "details": {}}]},
]}
TRANSACTIONS = {"transactions": [
    {"date": "2026-10-27T07:00Z", "description": "Signed G Jane Doe.", "team": {"id": "9", "abbreviation": "GS"}},
    {"date": "2026-10-28T07:00Z", "description": "Waived F AK Okereke.", "team": {"id": "13", "abbreviation": "LAL"}},
]}


def fake_get(path: str, params=None):
    return {"news": HEADLINES, "injuries": INJURIES, "transactions": TRANSACTIONS}[path]


class ParsingTests(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(news.classify("Lakers rule out Luka Doncic vs. Clippers", None, "HeadlineNews"), "injury")
        self.assertEqual(news.classify("Wizards upgrade No. 1 pick AJ Dybantsa to available vs. Cavaliers", None, "HeadlineNews"), "injury")
        self.assertEqual(news.classify("Mavericks exercise 2027-28 option on Risacher deal", None, "HeadlineNews"), "transaction")
        self.assertEqual(news.classify("Knicks assistant coach steps away", None, "HeadlineNews"), "coaching")
        self.assertEqual(news.classify("Updated NBA championship odds", None, "Story"), "betting")
        self.assertEqual(news.classify("Anything at all", None, "Recap"), "game")
        self.assertEqual(news.classify("Hornets to retire Muggsy Bogues' jersey", None, "HeadlineNews"), "other")

    @patch.object(news, "_get", side_effect=fake_get)
    def test_headlines(self, _):
        result = news.fetch_headlines(ROSTER)
        first = result["articles"][0]
        self.assertEqual(first["headline"], "Mavericks exercise 2027-28 option on Zaccharie Risacher deal")  # newest first
        injury = next(item for item in result["articles"] if item["id"] == "1")
        self.assertEqual((injury["kind"], injury["teams"]), ("injury", ["IND"]))
        self.assertEqual(injury["athletes"], [{"name": "Tyrese Haliburton", "player_id": "1630169"}])
        self.assertEqual(injury["url"], "https://www.espn.com/nba/story/_/id/1")
        video = next(item for item in result["articles"] if item["id"] == "6")
        self.assertTrue(video["video"])
        self.assertIsNone(video["url"])  # only https links are passed through
        self.assertIsNone(video["description"])  # a description that repeats the headline is dropped
        self.assertEqual(result["counts"]["injury"], 1)
        self.assertEqual(result["counts"]["coaching"], 1)

    @patch.object(news, "_get", side_effect=fake_get)
    def test_injuries(self, _):
        result = news.fetch_injuries(ROSTER)
        self.assertEqual([team["team"] for team in result["teams"]], ["IND", "UTA"])
        pacers = result["teams"][0]["injuries"]
        self.assertEqual([entry["player"] for entry in pacers], ["Tyrese Haliburton", "Bench Guy"])  # out before day-to-day
        star = pacers[0]
        self.assertEqual((star["player_id"], star["status_key"], star["injury"], star["return_date"]), ("1630169", "out", "Hamstring", "2026-11-12"))
        self.assertIsNone(pacers[1]["player_id"])
        self.assertEqual(result["counts"], {"day_to_day": 1, "out": 1, "questionable": 1})
        self.assertEqual(result["teams"][1]["injuries"][0]["team"], "UTA")

    @patch.object(news, "_get", side_effect=fake_get)
    def test_transactions(self, _):
        result = news.fetch_transactions()
        self.assertEqual([item["team"] for item in result["transactions"]], ["LAL", "GSW"])

    def test_bad_payloads_are_reported(self):
        with patch.object(news, "_get", return_value={"unexpected": True}):
            for fetch in (news.fetch_headlines, news.fetch_injuries, news.fetch_transactions):
                with self.subTest(fetch=fetch.__name__), self.assertRaises(news.NewsUnavailableError):
                    fetch()

    def test_style_tags_and_sentence(self):
        metrics = {"pace": {"rank": 2}, "three_rate": {"rank": 29}, "defense": {"rank": 1}, "offense": {"rank": 15}}
        plays = [{"play_type": "Isolation", "label": "Isolation", "frequency": 0.12, "frequency_rank": 1}]
        self.assertEqual(news._style_tags(metrics, plays, 30), ["Plays fast", "Few threes", "Elite defense", "Isolation-heavy"])
        sentence = news.style_sentence({"metrics": metrics, "play_types": plays})
        self.assertIn("2nd in pace", sentence)
        self.assertIn("15th on offense and 1st on defense", sentence)


class NewsApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.client = TestClient(create_app(Path(self.directory.name)), base_url="http://127.0.0.1")
        self.store = self.client.app.state.store
        self.store.save_resource("roster:" + "2026-27", {"players": ROSTER, "teams": [], "season": "2026-27"})

    def tearDown(self):
        self.client.close()
        self.directory.cleanup()

    def test_feeds_are_cached_and_fall_back_to_the_last_copy(self):
        with patch.object(news, "_get", side_effect=fake_get) as fetch:
            first = self.client.get("/api/news/injuries").json()
            second = self.client.get("/api/news/injuries").json()
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(first["total"], 3)
        self.assertFalse(second["stale"])
        with patch.object(news, "_get", side_effect=news.NewsUnavailableError("ESPN is down.")):
            stale = self.client.get("/api/news/injuries?refresh=true").json()
            missing = self.client.get("/api/news/transactions")
        self.assertTrue(stale["stale"])
        self.assertIn("last saved copy", stale["error"])
        self.assertEqual(stale["total"], 3)
        self.assertEqual(missing.status_code, 502)

    def test_headlines_endpoint(self):
        with patch.object(news, "_get", side_effect=fake_get):
            result = self.client.get("/api/news/headlines").json()
        self.assertEqual(len(result["articles"]), 6)
        self.assertIn("fetched_at", result)

    def test_unknown_team(self):
        self.assertEqual(self.client.get("/api/news/teams/XYZ").status_code, 404)

    def test_stale_cache_expires(self):
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        self.store.save_resource("news:headlines", {"articles": [], "counts": {}})
        with self.store.connect() as db:
            row = db.execute("SELECT payload FROM provider_cache WHERE key = 'news:headlines'").fetchone()
            payload = row["payload"].replace('"fetched_at": "', f'"fetched_at": "{old}", "old": "')
            db.execute("UPDATE provider_cache SET payload = ? WHERE key = 'news:headlines'", (payload,))
        with patch.object(news, "_get", side_effect=fake_get) as fetch:
            result = self.client.get("/api/news/headlines").json()
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(len(result["articles"]), 6)


if __name__ == "__main__":
    unittest.main()
