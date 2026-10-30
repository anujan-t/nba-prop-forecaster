"""The team model behind moneylines and spreads: ratings, margin probabilities, same-game joints and the service."""
import json
import unittest
from datetime import date, datetime, timedelta, timezone

import numpy as np
from scipy.stats import norm

from backend import teams
from backend.nba import NBAUnavailableError


class MemoryStore:
    def __init__(self):
        self.saved: dict[str, dict] = {}

    def resource(self, key):
        return self.saved.get(key)

    def save_resource(self, key, payload):
        self.saved[key] = {**payload, "fetched_at": datetime.now(timezone.utc).isoformat()}
        return self.saved[key]


TODAY = date(2026, 10, 28)
TRUE = {"BOS": 6.0, "DEN": 4.0, "OKC": 8.0, "NYK": 2.0, "LAL": 0.0, "MIA": -3.0, "GSW": 1.0, "PHX": -6.0, "CHA": -8.0, "SAS": 3.0}
HCA, B2B = 2.5, 3.0


def synthetic_season(season: str, start: date, days: int, seed: int) -> list[dict]:
    """A season where margins follow the model's own structure: ratings, home court, a back-to-back penalty and noise."""
    rng = np.random.default_rng(seed)
    codes = sorted(TRUE)
    last: dict[str, date] = {}
    games = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        order = rng.permutation(codes)
        for pair in range(int(rng.integers(2, 6))):
            home, away = str(order[2 * pair]), str(order[2 * pair + 1])
            tired = {team: last.get(team) == day - timedelta(days=1) for team in (home, away)}
            margin = HCA + TRUE[home] - TRUE[away] + B2B * (tired[away] - tired[home]) + rng.normal(0, 8)
            margin = int(round(margin)) or int(rng.choice([-1, 1]))
            games.append({"game_id": f"{season}-{offset}-{pair}", "date": day.isoformat(), "season": season, "home": home, "away": away,
                          "home_points": 110 + max(margin, 0), "away_points": 110 + max(-margin, 0)})
            last[home] = last[away] = day
    return games


SEASONS = {
    "2026-27": synthetic_season("2026-27", date(2026, 10, 20), 8, 3),
    "2025-26": synthetic_season("2025-26", date(2025, 10, 21), 170, 2),
    "2024-25": synthetic_season("2024-25", date(2024, 10, 22), 170, 1),
}


class FakeFetch:
    def __init__(self, fail: bool = False):
        self.calls: list[str] = []
        self.fail = fail

    def __call__(self, season: str) -> dict:
        self.calls.append(season)
        if self.fail:
            raise NBAUnavailableError("NBA Stats is down.")
        return {"season": season, "games": json.loads(json.dumps(SEASONS.get(season, []))), "provider": "fixture"}


ODDS = {"provider": "DraftKings", "home_spread": -2.5, "away_spread": 2.5, "total": 231.5, "favorite": "GSW",
        "home_moneyline": -135, "away_moneyline": 115, "home_spread_odds": -110, "away_spread_odds": -110}


def board(day: date, games: list[dict] | None = None) -> dict:
    return {"date": day.isoformat(), "games": games if games is not None else [], "provider": "fixture"}


def scheduled(game_id: str = "401", home: str = "GSW", away: str = "DEN", odds: dict | None = ODDS) -> dict:
    return {"id": game_id, "start": "2026-10-28T23:30Z", "state": "pre", "detail": None,
            "home": {"team": home, "name": home, "record": None, "score": None, "winner": None},
            "away": {"team": away, "name": away, "record": None, "score": None, "winner": None}, "odds": odds}


def model(fetch: FakeFetch | None = None, schedule=None, store=None) -> teams.GameModel:
    return teams.GameModel(store or MemoryStore(), today=lambda: TODAY, schedule=schedule or (lambda day: board(day)),
                           fetch=fetch or FakeFetch())


class DataTests(unittest.TestCase):
    def test_league_log_rows_pair_into_games(self):
        rows = [
            {"GAME_ID": "1", "GAME_DATE": "2026-10-20", "TEAM_ABBREVIATION": "BOS", "MATCHUP": "BOS @ DET", "PTS": 113},
            {"GAME_ID": "1", "GAME_DATE": "2026-10-20", "TEAM_ABBREVIATION": "DET", "MATCHUP": "DET vs. BOS", "PTS": 121},
            {"GAME_ID": "2", "GAME_DATE": "2026-10-20", "TEAM_ABBREVIATION": "LAL", "MATCHUP": "LAL vs. GSW", "PTS": 109},
        ]
        self.assertEqual(teams.pair_games(rows, "2026-27"), [{"game_id": "1", "date": "2026-10-20", "season": "2026-27", "home": "DET",
                                                               "away": "BOS", "home_points": 121, "away_points": 113}])

    def test_rest_counts_days_since_the_last_game_in_the_season(self):
        games = [{"game_id": "a", "date": "2026-10-20", "season": "2026-27", "home": "BOS", "away": "NYK"},
                 {"game_id": "b", "date": "2026-10-21", "season": "2026-27", "home": "BOS", "away": "MIA"},
                 {"game_id": "c", "date": "2026-10-24", "season": "2026-27", "home": "NYK", "away": "BOS"}]
        rest = teams.rest_days(games)
        self.assertEqual((rest[("a", "BOS")], rest[("b", "BOS")], rest[("c", "BOS")], rest[("c", "NYK")]), (4, 1, 3, 4))


class RatingTests(unittest.TestCase):
    def test_ratings_recover_team_strength_home_court_and_back_to_backs(self):
        games = sorted(SEASONS["2025-26"] + SEASONS["2026-27"], key=lambda game: game["date"])
        ratings = teams.fit_ratings(games, TODAY)
        codes = sorted(TRUE)
        fitted = [ratings.rating(code) for code in codes]
        self.assertGreater(np.corrcoef(fitted, [TRUE[code] for code in codes])[0, 1], 0.9)
        self.assertAlmostEqual(ratings.hca, HCA, delta=1.2)
        self.assertGreater(ratings.b2b, 0.5)
        self.assertEqual(ratings.season, "2026-27")
        self.assertTrue(ratings.early("BOS", "DEN"))
        self.assertGreater(teams.expected_margin(ratings, "OKC", "CHA"), 10)

    def test_too_few_games_is_unavailable(self):
        with self.assertRaises(teams.GameModelUnavailable):
            teams.fit_ratings(SEASONS["2026-27"], TODAY)


class ProbabilityTests(unittest.TestCase):
    def test_margin_distribution_has_no_ties_and_matches_the_win_chance(self):
        for mean in (-7.0, 0.0, 3.2, 12.0):
            pmf = teams.margin_pmf(mean)
            self.assertAlmostEqual(pmf.sum(), 1.0)
            self.assertEqual(pmf[teams.MARGINS == 0][0], 0.0)
            self.assertAlmostEqual(pmf[teams.MARGINS > 0].sum(), norm.cdf(mean / teams.SIGMA), places=9)

    def test_spreads_push_only_on_whole_numbers(self):
        pmf = teams.margin_pmf(5.0)
        half = {"home": True, "market": "spread", "line": -5.5}
        wins, pushes = teams.leg_outcomes(half)
        self.assertEqual(pmf[pushes].sum(), 0.0)
        self.assertAlmostEqual(pmf[wins].sum(), pmf[teams.MARGINS >= 6].sum())
        whole = {"home": True, "market": "spread", "line": -5.0}
        wins, pushes = teams.leg_outcomes(whole)
        self.assertAlmostEqual(pmf[pushes].sum(), pmf[teams.MARGINS == 5][0])
        dog = {"home": False, "market": "spread", "line": 5.5}
        self.assertAlmostEqual(pmf[teams.leg_outcomes(dog)[0]].sum() + pmf[teams.leg_outcomes(half)[0]].sum(), 1.0)
        self.assertAlmostEqual(pmf[teams.leg_outcomes({"home": True, "market": "spread", "line": 0.0})[0]].sum(),
                               pmf[teams.MARGINS > 0].sum())

    def test_same_game_legs_join_exactly(self):
        pmf = teams.margin_pmf(4.0)
        home_ml = {"home": True, "market": "moneyline", "line": None}
        home_minus = {"home": True, "market": "spread", "line": -6.5}
        away_ml = {"home": False, "market": "moneyline", "line": None}
        away_plus = {"home": False, "market": "spread", "line": 6.5}
        self.assertAlmostEqual(teams.joint(pmf, [home_ml, home_minus]), pmf[teams.MARGINS >= 7].sum())
        self.assertEqual(teams.joint(pmf, [home_ml, away_ml]), 0.0)
        self.assertAlmostEqual(teams.joint(pmf, [home_ml, away_plus]), pmf[(teams.MARGINS > 0) & (teams.MARGINS < 6.5)].sum())
        whole = {"home": True, "market": "spread", "line": -4.0}
        self.assertAlmostEqual(teams.joint(pmf, [whole], pushes_count=True), pmf[teams.MARGINS >= 4].sum())

    def test_book_prices(self):
        side = teams.two_way(-135, 115)
        self.assertAlmostEqual(side["implied"], 135 / 235)
        self.assertAlmostEqual(side["no_vig"], (135 / 235) / (135 / 235 + 100 / 215))
        priced = teams.price_side(0.6, 0.0, side)
        self.assertAlmostEqual(priced["edge"], 0.6 - 135 / 235)
        self.assertAlmostEqual(priced["expected_roi"], 0.6 * (100 / 135) - 0.4)
        self.assertEqual(teams.fair_line(4.2), -4.0)
        self.assertEqual(teams.fair_line(-2.8), 3.0)
        self.assertIsNone(teams.two_way(None, 115))


class ServiceTests(unittest.TestCase):
    def test_results_are_cached_and_a_failed_refresh_serves_the_saved_copy(self):
        store = MemoryStore()
        fetch = FakeFetch()
        service = model(fetch, store=store)
        service.ratings()
        service.ratings()
        self.assertEqual(sorted(fetch.calls), ["2025-26", "2026-27"])
        down = model(FakeFetch(fail=True), store=store)
        games, meta = down.games(refresh=True)
        self.assertTrue(meta["stale"])
        self.assertIn("NBA Stats is down", meta["errors"][0])
        self.assertEqual(len(games), len(SEASONS["2025-26"]) + len(SEASONS["2026-27"]))
        with self.assertRaises(teams.GameModelUnavailable):
            model(FakeFetch(fail=True)).ratings()

    def test_annotation_compares_both_sides_with_the_book(self):
        view = model().annotate(scheduled())
        moneyline, spread = view["markets"]["moneyline"], view["markets"]["spread"]
        self.assertAlmostEqual(moneyline["home"]["probability"] + moneyline["away"]["probability"], 1.0)
        self.assertAlmostEqual(view["home_win"], moneyline["home"]["probability"])
        self.assertEqual((spread["home"]["line"], spread["away"]["line"]), (-2.5, 2.5))
        self.assertAlmostEqual(spread["home"]["probability"] + spread["away"]["probability"], 1.0)
        self.assertAlmostEqual(moneyline["home"]["book"]["implied"], 135 / 235)
        self.assertAlmostEqual(moneyline["home"]["book"]["edge"], moneyline["home"]["probability"] - 135 / 235)
        self.assertEqual(view["fair_spread"]["home"], teams.fair_line(view["home_margin"]))
        self.assertNotIn("pmf", view)
        self.assertIsNone(model().annotate(scheduled(odds=None))["markets"]["moneyline"]["home"]["book"])

    def test_a_leg_carries_the_book_price_only_on_the_listed_line(self):
        schedule = lambda day: board(day, [scheduled()] if day == TODAY else [])
        service = model(schedule=schedule)
        base = {"kind": "game", "game_id": "401", "date": TODAY.isoformat(), "team": "DEN", "opponent": "GSW", "home": False}
        listed = service.leg({**base, "market": "spread", "line": 2.5})
        self.assertEqual((listed["book"]["odds"], listed["book"]["line"]), (-110, 2.5))
        self.assertIsNone(service.leg({**base, "market": "spread", "line": 4.5})["book"])
        moneyline = service.leg({**base, "market": "moneyline", "line": None})
        self.assertEqual(moneyline["book"]["odds"], 115)
        self.assertAlmostEqual(moneyline["probability"], moneyline["win_probability"])
        self.assertAlmostEqual(moneyline["margin"], -service.matchup("GSW", "DEN", TODAY)["home_margin"])
        self.assertTrue(moneyline["early_season"])

    def test_back_to_backs_come_from_the_schedule(self):
        yesterday = scheduled("400", home="DEN", away="LAL")
        schedule = lambda day: board(day, [yesterday] if day == TODAY - timedelta(days=1) else [])
        rested = model().matchup("GSW", "DEN", TODAY)
        tired = model(schedule=schedule).matchup("GSW", "DEN", TODAY)
        self.assertTrue(tired["away_b2b"])
        self.assertFalse(rested["away_b2b"])
        self.assertAlmostEqual(tired["home_margin"] - rested["home_margin"], tired["b2b_penalty"])

    def test_team_margins_and_status(self):
        service = model()
        margins = service.team_margins("BOS")
        game = next(game for game in SEASONS["2026-27"] if "BOS" in (game["home"], game["away"]))
        sign = 1 if game["home"] == "BOS" else -1
        self.assertEqual(margins[game["date"]], sign * (game["home_points"] - game["away_points"]))
        status = service.status()
        self.assertEqual([row["rank"] for row in status["ratings"]], list(range(1, len(TRUE) + 1)))
        self.assertIn("OKC", [row["team"] for row in status["ratings"][:2]])
        self.assertIn("CHA", [row["team"] for row in status["ratings"][-2:]])
        backtest = status["backtest"]
        self.assertEqual(backtest["season"], "2025-26")
        self.assertGreater(backtest["games"], 200)
        self.assertLess(backtest["log_loss"], backtest["baseline_log_loss"])


if __name__ == "__main__":
    unittest.main()
