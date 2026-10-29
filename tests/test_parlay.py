import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.data import metadata, validate_frame
from backend.engine import forecast, market_simulation, profit_per_unit
from backend.parlay import LegNotFound, fair_odds, orthant, price_parlay, tetrachoric, valid_correlation


def season(seed: int = 7, games: int = 60, guard_games: list[int] | None = None) -> pd.DataFrame:
    """Two DEN teammates whose minutes and production share a game-level factor, plus an unrelated BOS player."""
    rng = np.random.default_rng(seed)
    start = date(2025, 10, 21)
    rows = []
    for game in range(games):
        day = (start + timedelta(days=2 * game)).isoformat()
        pace, other = rng.normal(size=2)
        minutes = float(np.clip(34 + 6 * pace, 18, 46))
        center = dict(points=0.74 * minutes + 2.5 * rng.normal(), rebounds=12 + 3 * rng.normal(),
                      assists=0.27 * minutes + 1.2 * (0.5 * other + rng.normal()), steals=rng.poisson(1.3), blocks=rng.poisson(0.8))
        guard = dict(points=20 + 5 * (0.7 * pace + 0.7 * rng.normal()), rebounds=4 + 1.5 * rng.normal(),
                     assists=6 + 2 * rng.normal(), steals=rng.poisson(1.0), blocks=rng.poisson(0.3))
        rows.append(dict(date=day, player_id="center", player_name="Center", team="DEN", opponent="LAL", minutes=round(minutes, 1), home=True,
                         **{k: max(0, int(round(v))) for k, v in center.items()}))
        if guard_games is None or game in guard_games:
            rows.append(dict(date=day, player_id="guard", player_name="Guard", team="DEN", opponent="LAL", minutes=33.0, home=True,
                             **{k: max(0, int(round(v))) for k, v in guard.items()}))
        rows.append(dict(date=(start + timedelta(days=2 * game + 1)).isoformat(), player_id="wing", player_name="Wing", team="BOS",
                         opponent="NYK", minutes=35.0, home=False, points=int(rng.poisson(27)), rebounds=int(rng.poisson(8)),
                         assists=int(rng.poisson(5)), steals=int(rng.poisson(1)), blocks=int(rng.poisson(0.7))))
    return validate_frame(pd.DataFrame(rows))


def loader(frame: pd.DataFrame):
    datasets = {"logs": (frame, metadata(frame, "logs", "Test logs", "nba_api"))}

    def load(dataset_id: str):
        return datasets[dataset_id]
    return load


def leg(player_id: str, stat: str, line: float, side: str = "over", **extra) -> dict:
    return {"dataset_id": "logs", "player_id": player_id, "stat": stat, "line": line, "side": side, "model": "auto",
            "lookback": 30, "expected_minutes": None, "rest_days": None, "opponent": None, "use_context": False, **extra}


class OddsAndProbabilityTests(unittest.TestCase):
    def test_fair_odds(self):
        self.assertEqual(fair_odds(0.25), {"decimal": 4.0, "american": 300})
        self.assertEqual(fair_odds(0.75)["american"], -300)
        self.assertEqual(fair_odds(0.5)["american"], 100)
        self.assertIsNone(fair_odds(0.0))
        self.assertIsNone(fair_odds(1.0))

    def test_orthant_matches_the_bivariate_closed_form(self):
        for rho in (-0.6, 0.0, 0.3, 0.8):
            expected = 0.25 + math.asin(rho) / (2 * math.pi)
            self.assertAlmostEqual(orthant([0.5, 0.5], np.array([[1, rho], [rho, 1]])), expected, places=6)

    def test_tetrachoric_inverts_the_orthant(self):
        for rho in (-0.7, -0.2, 0.0, 0.35, 0.9):
            joint = 0.25 + math.asin(rho) / (2 * math.pi)
            self.assertAlmostEqual(tetrachoric(0.5, 0.5, joint), rho, places=3)
        self.assertEqual(tetrachoric(0.0, 0.5, 0.0), 0.0)
        self.assertAlmostEqual(tetrachoric(0.3, 0.6, 0.3), 0.98, places=3)  # the Fréchet upper bound

    def test_orthant_edge_cases(self):
        correlation = np.array([[1, 0.5, 0.2], [0.5, 1, 0.1], [0.2, 0.1, 1]])
        self.assertEqual(orthant([0.4, 0.0, 0.7], correlation), 0.0)
        self.assertAlmostEqual(orthant([0.4, 1.0, 1.0], correlation), 0.4)
        joint = orthant([0.4, 0.6, 0.7], correlation)
        self.assertLessEqual(joint, 0.4)
        self.assertGreater(joint, 0.4 * 0.6 * 0.7)
        self.assertEqual(joint, orthant([0.4, 0.6, 0.7], correlation))

    def test_invalid_correlations_are_repaired(self):
        broken = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])
        fixed, changed = valid_correlation(broken)
        self.assertTrue(changed)
        self.assertGreaterEqual(np.linalg.eigvalsh(fixed).min(), 0)
        np.testing.assert_allclose(np.diag(fixed), 1)
        same, unchanged = valid_correlation(np.eye(3))
        self.assertFalse(unchanged)
        np.testing.assert_array_equal(same, np.eye(3))


class ParlayTests(unittest.TestCase):
    def setUp(self):
        self.frame = season()
        self.load = loader(self.frame)

    def price(self, legs, book_odds=None):
        return price_parlay(legs, book_odds, self.load)

    def test_single_leg_matches_the_forecast(self):
        result = self.price([leg("center", "points", 24.5)])
        expected = forecast(self.frame, self.load("logs")[1], {**leg("center", "points", 24.5), "odds_over": -110, "odds_under": -110})
        self.assertAlmostEqual(result["legs"][0]["probability"], expected["probabilities"]["over"])
        self.assertAlmostEqual(result["probability"], result["independent_probability"])
        self.assertAlmostEqual(result["fair_odds"]["decimal"], 1 / result["probability"])
        self.assertEqual(result["pairs"], [])
        self.assertEqual(result["legs"][0]["hit_rate"]["games"], 30)

    def test_players_on_different_teams_are_independent(self):
        result = self.price([leg("center", "points", 24.5), leg("wing", "points", 26.5)])
        pair = result["pairs"][0]
        self.assertEqual((pair["basis"], pair["games"], pair["correlation"]), ("independent", 0, 0.0))
        self.assertAlmostEqual(result["probability"], result["legs"][0]["probability"] * result["legs"][1]["probability"])
        self.assertEqual(result["history"], [])
        self.assertTrue(any("different teams" in warning for warning in result["warnings"]))

    def test_same_player_correlation_raises_overs_and_lowers_mixed_sides(self):
        together = self.price([leg("center", "points", 24.5), leg("center", "assists", 8.5)])
        pair = together["pairs"][0]
        self.assertEqual((pair["basis"], pair["games"], pair["source"]), ("same_player", 30, "simulation"))
        self.assertEqual(pair["simulations"], 250_000)
        self.assertGreater(pair["rank_correlation"], 0.4)
        self.assertGreater(together["probability"], together["independent_probability"])
        self.assertLessEqual(together["probability"], min(leg["probability"] for leg in together["legs"]))
        self.assertGreater(together["correlation_lift"], 1)
        self.assertIn("priced jointly from 250,000 simulated games", pair["note"])
        self.assertIn("win together", pair["note"])
        mixed = self.price([leg("center", "points", 24.5), leg("center", "assists", 8.5, "under")])
        self.assertLess(mixed["probability"], mixed["independent_probability"])
        self.assertIn("work against each other", mixed["pairs"][0]["note"])

    def test_same_player_block_is_the_exact_simulated_joint(self):
        legs = [leg("center", "points", 24.5), leg("center", "assists", 8.5), leg("center", "rebounds", 11.5, "under")]
        result = self.price(legs)
        frame, info = self.load("logs")
        draws = market_simulation(frame, {**legs[0], "simulations": 250_000})
        wins = (draws.market("points") > 24.5) & (draws.market("assists") > 8.5) & (draws.market("rebounds") < 11.5)
        self.assertAlmostEqual(result["probability"], float(wins.mean()), places=12)
        self.assertEqual(result["history"][0]["model_probability"], float(wins.mean()))

    def test_copula_links_same_player_legs_on_analytic_models(self):
        together = self.price([leg("center", "points", 24.5, model="normal"), leg("center", "assists", 8.5, model="normal")])
        pair = together["pairs"][0]
        self.assertEqual((pair["source"], pair["simulations"]), ("history", None))
        self.assertIn("rank correlation", pair["note"])
        self.assertGreater(together["probability"], together["independent_probability"])

    def test_teammates_are_linked_through_shared_games(self):
        result = self.price([leg("center", "points", 24.5), leg("guard", "points", 19.5)])
        pair = result["pairs"][0]
        self.assertEqual((pair["basis"], pair["games"]), ("teammates", 30))
        self.assertGreater(pair["correlation"], 0)
        self.assertGreater(result["probability"], result["independent_probability"])

    def test_history_counts_games_where_every_linked_leg_won(self):
        legs = [leg("center", "points", 24.5), leg("guard", "points", 19.5), leg("wing", "points", 26.5)]
        result = self.price(legs)
        self.assertEqual(len(result["history"]), 1)
        group = result["history"][0]
        recent = self.frame[self.frame.player_id.isin(["center", "guard"])].pivot(index="date", columns="player_id", values="points").tail(30)
        expected = int(((recent["center"] > 24.5) & (recent["guard"] > 19.5)).sum())
        self.assertEqual((group["legs"], group["games"], group["hits"]), ([0, 1], 30, expected))
        self.assertGreater(group["model_probability"], 0)

    def test_few_shared_games_assume_no_linkage(self):
        # The guard has 12 games, but only the last 6 fall inside the center's recent 30-game window.
        self.load = loader(season(guard_games=[*range(6), *range(54, 60)]))
        result = self.price([leg("center", "points", 24.5), leg("guard", "points", 19.5, lookback=10)])
        pair = result["pairs"][0]
        self.assertEqual((pair["basis"], pair["games"], pair["rank_correlation"], pair["correlation"]), ("teammates", 6, None, 0.0))
        self.assertIn("only 6 shared games", pair["note"])

    def test_whole_number_lines_report_pushes(self):
        result = self.price([leg("center", "points", 25), leg("center", "assists", 8.5)])
        self.assertGreater(result["legs"][0]["push"], 0)
        self.assertGreater(result["no_loss_probability"], result["probability"])
        self.assertTrue(any("push" in warning for warning in result["warnings"]))
        self.assertIsNone(self.price([leg("center", "points", 24.5)])["no_loss_probability"])

    def test_book_odds_comparison(self):
        result = self.price([leg("center", "points", 24.5), leg("wing", "points", 26.5)], book_odds=250)
        p = result["probability"]
        self.assertAlmostEqual(result["book"]["implied"], 100 / 350)
        self.assertAlmostEqual(result["book"]["edge"], p - 100 / 350)
        self.assertAlmostEqual(result["book"]["expected_roi"], p * profit_per_unit(250) - (1 - p))

    def test_duplicate_player_stat_and_missing_data_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Leg 2 \\(Center points\\): leg 1 already uses"):
            self.price([leg("center", "points", 24.5), leg("center", "points", 30.5, "under")])
        with self.assertRaisesRegex(LegNotFound, "Leg 1"):
            self.price([leg("center", "points", 24.5, dataset_id="missing")])
        with self.assertRaisesRegex(LegNotFound, "Leg 2"):
            self.price([leg("center", "points", 24.5), leg("nobody", "points", 24.5)])
        self.load = loader(season(guard_games=list(range(54, 60))))
        with self.assertRaisesRegex(ValueError, "Leg 2 \\(Guard points\\): At least 10 completed games"):
            self.price([leg("center", "points", 24.5), leg("guard", "points", 19.5)])

    def test_results_are_deterministic(self):
        legs = [leg("center", "points", 24.5), leg("center", "assists", 8.5), leg("guard", "points", 19.5), leg("center", "rebounds", 11.5)]
        first, second = self.price(legs), self.price(legs)
        self.assertEqual(first["probability"], second["probability"])
        self.assertLessEqual(first["probability"], min(leg["probability"] for leg in first["legs"]))


class ParlayApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.client = TestClient(create_app(Path(self.directory.name)), base_url="http://127.0.0.1")

    def tearDown(self):
        self.directory.cleanup()

    def body(self, **changes):
        base = {"dataset_id": "demo", "player_id": "203999", "stat": "points", "line": 26.5, "side": "over"}
        return {**base, **changes}

    def test_parlay_endpoint(self):
        response = self.client.post("/api/parlay", json={"legs": [self.body(), self.body(player_id="1628369", stat="rebounds", line=7.5, side="under")],
                                                         "book_odds": 260})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(len(result["legs"]), 2)
        self.assertEqual(result["pairs"][0]["basis"], "independent")
        self.assertAlmostEqual(result["probability"], result["legs"][0]["probability"] * result["legs"][1]["probability"])
        self.assertTrue(result["computed_at"].endswith("Z"))
        self.assertTrue(any("Synthetic" in warning for warning in result["warnings"]))

    def test_parlay_validation(self):
        cases = [
            ({"legs": []}, 422), ({"legs": [self.body(player_id=str(i)) for i in range(11)]}, 422),
            ({"legs": [self.body(side="both")]}, 422), ({"legs": [self.body(line=26.3)]}, 422),
            ({"legs": [self.body()], "book_odds": 50}, 422), ({"legs": [self.body(), self.body()]}, 422),
            ({"legs": [self.body(dataset_id="missing")]}, 404), ({"legs": [self.body(player_id="999")]}, 404),
            ({"legs": [self.body(extra=1)]}, 422),
        ]
        for payload, status in cases:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/api/parlay", json=payload).status_code, status)


if __name__ == "__main__":
    unittest.main()
