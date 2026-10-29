from datetime import date, timedelta
import unittest

import numpy as np
import pandas as pd

from backend import simulation
from backend.data import validate_frame
from backend.engine import CountDistribution
from backend.stats import CATALOG, PROP_STATS, STAT_DEFS, stat_values, with_stat


def box_scores(games: int = 60, seed: int = 3, minutes: float = 34.0, step_at: int | None = None) -> pd.DataFrame:
    """Full NBA-style box scores built from attempts and makes, so every market is defined."""
    rng = np.random.default_rng(seed)
    rows = []
    for game in range(games):
        scale = 1.6 if step_at is not None and game >= step_at else 1.0
        played = float(np.clip(rng.normal(minutes, 3), 12, 44))
        fg2a = int(rng.poisson(0.32 * played * scale))
        fg3a = int(rng.poisson(0.20 * played * scale))
        fta = 2 * int(rng.poisson(0.09 * played * scale))
        fg2m, fg3m, ftm = int(rng.binomial(fg2a, 0.55)), int(rng.binomial(fg3a, 0.37)), int(rng.binomial(fta, 0.8))
        oreb, dreb = int(rng.poisson(0.04 * played)), int(rng.poisson(0.18 * played))
        rows.append({
            "date": (date(2025, 10, 21) + timedelta(days=2 * game)).isoformat(), "player_id": "1",
            "player_name": "Test Player", "team": "DEN", "opponent": ("BOS", "LAL", "MIA")[game % 3], "minutes": round(played, 1),
            "points": 2 * fg2m + 3 * fg3m + ftm, "rebounds": oreb + dreb, "assists": int(rng.poisson(0.2 * played)),
            "steals": int(rng.poisson(0.03 * played)), "blocks": int(rng.poisson(0.02 * played)),
            "field_goals_made": fg2m + fg3m, "field_goals_attempted": fg2a + fg3a, "three_pointers_made": fg3m,
            "three_pointers_attempted": fg3a, "free_throws_made": ftm, "free_throws_attempted": fta,
            "offensive_rebounds": oreb, "defensive_rebounds": dreb, "turnovers": int(rng.poisson(0.07 * played)),
            "fouls": int(min(6, rng.poisson(0.06 * played))), "plus_minus": 0,
        })
    return validate_frame(pd.DataFrame(rows))


class CatalogTests(unittest.TestCase):
    def test_every_market_has_a_definition_and_values(self):
        frame = box_scores()
        self.assertEqual(len(PROP_STATS), 22)
        self.assertEqual(len(set(PROP_STATS)), len(PROP_STATS))
        for spec in CATALOG:
            with self.subTest(stat=spec.id):
                values = stat_values(frame, spec.id)
                self.assertTrue(values.notna().all())
                self.assertTrue((values >= 0).all())
        np.testing.assert_array_equal(stat_values(frame, "pts_reb_ast"), frame["points"] + frame["rebounds"] + frame["assists"])
        np.testing.assert_array_equal(stat_values(frame, "stocks"), frame["steals"] + frame["blocks"])
        np.testing.assert_array_equal(stat_values(frame, "threes"), frame["three_pointers_made"])

    def test_milestones(self):
        frame = pd.DataFrame([
            {"points": 25, "rebounds": 12, "assists": 4, "steals": 1, "blocks": 0},
            {"points": 18, "rebounds": 11, "assists": 10, "steals": 2, "blocks": 1},
            {"points": 30, "rebounds": 9, "assists": 9, "steals": 0, "blocks": 0},
        ])
        self.assertEqual(stat_values(frame, "double_double").tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(stat_values(frame, "triple_double").tolist(), [0.0, 1.0, 0.0])

    def test_missing_box_score_columns_are_unknown_not_zero(self):
        frame = box_scores(games=12).drop(columns=["three_pointers_made"])
        self.assertTrue(stat_values(frame, "threes").isna().all())
        self.assertTrue(with_stat(frame, "threes").empty)
        self.assertEqual(len(with_stat(frame, "pts_reb_ast")), 12)


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = box_scores()
        cls.model = simulation.fit_game_model(cls.history, use_context=False)
        cls.draws = simulation.simulate(cls.model, 200_000)

    def test_uses_full_box_scores_and_reproduces_the_means(self):
        self.assertTrue(self.model.full)
        weights = simulation.recency_weights(len(self.history), simulation.MINUTES_HALF_LIFE)
        for stat in ("points", "rebounds", "assists", "threes", "free_throws_attempted", "pts_reb_ast"):
            with self.subTest(stat=stat):
                observed = float(stat_values(self.history, stat).mean())
                self.assertAlmostEqual(float(self.draws.market(stat).mean()), observed, delta=0.12 * observed + 0.3)
        self.assertGreater(weights[-1], weights[0])

    def test_box_score_identities_hold_in_every_simulated_game(self):
        made = self.draws.base("field_goals_made")
        attempts = self.draws.base("field_goals_attempted")
        self.assertTrue((made <= attempts).all())
        self.assertTrue((self.draws.base("three_pointers_made") <= self.draws.base("three_pointers_attempted")).all())
        self.assertTrue((self.draws.base("free_throws_made") <= self.draws.base("free_throws_attempted")).all())
        points = 2 * (made - self.draws.base("three_pointers_made")) + 3 * self.draws.base("three_pointers_made") + self.draws.base("free_throws_made")
        np.testing.assert_array_equal(self.draws.market("points"), points)
        np.testing.assert_array_equal(self.draws.market("pts_reb_ast"),
                                      self.draws.market("points") + self.draws.market("rebounds") + self.draws.market("assists"))
        self.assertLessEqual(int(self.draws.market("fouls").max()), simulation.FOUL_LIMIT)
        double = self.draws.market("double_double")
        self.assertTrue(set(np.unique(double)) <= {0, 1})

    def test_seeded_and_cached(self):
        again = simulation.simulate(self.model, 200_000)
        np.testing.assert_array_equal(again.market("points"), self.draws.market("points"))
        cache = simulation.SimulationCache(budget=500_000)
        first = cache.get(self.model, 100_000)
        self.assertIs(cache.get(self.model, 100_000), first)
        self.assertIsNot(cache.get(self.model, 100_000, seed=7), first)
        with self.assertRaises(ValueError):
            simulation.simulate(self.model, simulation.MAX_SIMULATIONS + 1)

    def test_count_sampler_matches_mean_and_dispersion(self):
        rng = np.random.default_rng(1)
        lam = np.full(400_000, 6.0)
        for dispersion in (0.8, 1.0, 1.9):
            with self.subTest(dispersion=dispersion):
                counts = simulation._counts(rng, lam, dispersion)
                self.assertAlmostEqual(counts.mean(), 6.0, delta=0.03)
                self.assertAlmostEqual(counts.var() / counts.mean(), dispersion, delta=0.05)

    def test_recent_role_change_moves_the_forecast(self):
        stepped = box_scores(step_at=50)
        flat = simulation.simulate(simulation.fit_game_model(box_scores(), use_context=False), 50_000).market("points").mean()
        moved = simulation.simulate(simulation.fit_game_model(stepped, use_context=False), 50_000).market("points").mean()
        self.assertGreater(moved, flat * 1.1)

    def test_minutes_scenario_scales_production(self):
        base = simulation.fit_game_model(self.history, use_context=True)
        more = simulation.fit_game_model(self.history, expected_minutes=float(self.history["minutes"].mean()) * 1.2, use_context=True)
        ratio = simulation.simulate(more, 100_000).market("points").mean() / simulation.simulate(base, 100_000).market("points").mean()
        self.assertAlmostEqual(ratio, 1.2, delta=0.06)
        with self.assertRaisesRegex(ValueError, "outside the model"):
            simulation.fit_game_model(self.history, expected_minutes=200.0, use_context=True)

    def test_basic_box_scores_still_simulate_core_markets(self):
        basic = self.history.loc[:, ["date", "player_id", "player_name", "team", "opponent", "minutes", "points", "rebounds",
                                      "assists", "steals", "blocks", "rest_days"]]
        model = simulation.fit_game_model(basic, use_context=False)
        self.assertFalse(model.full)
        draws = simulation.simulate(model, 20_000)
        self.assertTrue(draws.supports("pts_reb_ast"))
        self.assertTrue(draws.supports("double_double"))
        self.assertFalse(draws.supports("threes"))

    def test_distribution_from_draws(self):
        fitted = CountDistribution.from_draws(self.draws.market("points"))
        over, under, push = fitted.probabilities(20)
        self.assertAlmostEqual(over + under + push, 1)
        self.assertAlmostEqual(over, float(np.mean(self.draws.market("points") > 20)))
        self.assertAlmostEqual(push, float(np.mean(self.draws.market("points") == 20)))
        self.assertLessEqual(fitted.quantile(0.1), fitted.quantile(0.9))
        self.assertAlmostEqual(sum(row["probability"] for row in fitted.histogram()), 1, delta=0.001)
        self.assertLess(simulation.standard_error(0.5, 250_000), 0.0011)


if __name__ == "__main__":
    unittest.main()
