import math
import unittest

import numpy as np

from backend.data import demo_frame, metadata
from backend.engine import (
    MODELS, american_probability, backtest, context_adjustment,
    fit_distribution, forecast, market_comparison, resolve_model,
)


class OddsTests(unittest.TestCase):
    def test_american_odds(self):
        self.assertAlmostEqual(american_probability(-110), 110 / 210)
        self.assertAlmostEqual(american_probability(150), 0.4)
        self.assertEqual(american_probability(-100), 0.5)
        for odds in (0, 99, -99, 110.5, True, 10001):
            with self.subTest(odds=odds), self.assertRaises(ValueError):
                american_probability(odds)

    def test_no_vig_and_push_adjustment(self):
        result = market_comparison(0.4, 0.3, 0.3, -110, -110)
        self.assertEqual(result["no_vig_over"], 0.5)
        self.assertAlmostEqual(result["no_vig_over"] + result["no_vig_under"], 1)
        self.assertAlmostEqual(result["edge_over"], 0.4 / 0.7 - 0.5)
        self.assertAlmostEqual(result["expected_roi_over"], 0.4 * 100 / 110 - 0.3)
        all_push = market_comparison(0, 0, 1, 100, -100)
        self.assertIsNone(all_push["model_over_conditional"])
        self.assertIsNone(all_push["edge_over"])
        self.assertEqual(all_push["expected_roi_over"], 0)


class DistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = demo_frame().query("player_id == '203999'").tail(30)

    def test_probability_conservation_and_count_support(self):
        for model in MODELS:
            for stat in ("points", "rebounds", "assists", "steals", "blocks"):
                distribution = fit_distribution(self.history, stat, model)
                for line in (0, 0.5, 1, 1.5, 26, 26.5, 200):
                    with self.subTest(model=model, stat=stat, line=line):
                        over, under, push = distribution.probabilities(line)
                        self.assertAlmostEqual(over + under + push, 1)
                        self.assertTrue(all(0 <= p <= 1 for p in (over, under, push)))
                        if line % 1:
                            self.assertAlmostEqual(push, 0)
                masses = distribution.histogram()
                self.assertGreaterEqual(sum(row["probability"] for row in masses), 0.9995 - 1e-12)
                self.assertLessEqual(sum(row["probability"] for row in masses), 1 + 1e-12)
                self.assertGreaterEqual(distribution.quantile(0.1), 0)

    def test_poisson_known_value_and_push(self):
        history = self.history.copy()
        history["blocks"] = 2
        distribution = fit_distribution(history, "blocks", "poisson")
        over, under, push = distribution.probabilities(1)
        self.assertAlmostEqual(under, math.exp(-2))
        self.assertAlmostEqual(push, 2 * math.exp(-2))
        self.assertAlmostEqual(over, 1 - 3 * math.exp(-2))

    def test_empirical_hand_calculation(self):
        history = self.history.head(4).copy()
        history["points"] = [0, 1, 1, 2]
        distribution = fit_distribution(history, "points", "empirical")
        self.assertEqual(distribution.probabilities(1), (0.25, 0.25, 0.5))
        self.assertEqual(distribution.probabilities(1.5), (0.25, 0.75, 0))

    def test_zero_count_and_underdispersion(self):
        history = self.history.copy()
        history["blocks"] = 0
        distribution = fit_distribution(history, "blocks", "negative_binomial")
        self.assertEqual(distribution.model, "poisson")
        self.assertEqual(distribution.probabilities(0), (0, 0, 1))
        self.assertEqual(distribution.quantile(0.9), 0)

    def test_negative_binomial_overdispersion(self):
        history = self.history.copy()
        history["blocks"] = [0, 0, 0, 0, 10] * 6
        distribution = fit_distribution(history, "blocks", "negative_binomial")
        self.assertEqual(distribution.model, "negative_binomial")
        self.assertAlmostEqual(distribution.mean, 2)
        self.assertGreater(distribution.std ** 2, distribution.mean)
        # Auto runs the game simulator when it can build the market; otherwise the analytic rule picks a count model.
        self.assertEqual(fit_distribution(history, "blocks", "auto").model, "simulation")
        fouls = history.assign(fouls=[0, 0, 0, 0, 10] * 6)
        self.assertEqual(resolve_model(fouls, "fouls", "auto"), "negative_binomial")
        self.assertEqual(resolve_model(fouls.assign(fouls=2), "fouls", "auto"), "poisson")
        self.assertEqual(resolve_model(history, "points", "normal"), "normal")
        with self.assertRaisesRegex(ValueError, "yes/no market"):
            resolve_model(history.assign(double_double=0), "double_double", "normal")

    def test_normal_summary_matches_discrete_distribution(self):
        distribution = fit_distribution(self.history, "blocks", "normal")
        masses = [(k, distribution.cdf(k) - distribution.cdf(k - 1)) for k in range(100)]
        self.assertAlmostEqual(sum(k * p for k, p in masses), distribution.mean, places=7)

    def test_context_scaling_and_disabled_features(self):
        base = float(self.history.minutes.mean())
        factor, details, _ = context_adjustment(self.history, "points", base * 1.1, None, None, True)
        self.assertAlmostEqual(factor, 1.1)
        self.assertEqual(details["rest_factor"], 1)
        disabled, _, _ = context_adjustment(self.history, "points", base * 1.1, 0, "BOS", False)
        self.assertEqual(disabled, 1)
        _, unknown, notes = context_adjustment(self.history, "points", None, None, "XYZ", True)
        self.assertEqual(unknown["opponent_factor"], 1)
        self.assertTrue(any("No prior games" in note for note in notes))
        with self.assertRaisesRegex(ValueError, "outside the model"):
            context_adjustment(self.history, "points", base * 4, None, None, True)

    def test_invalid_prop_lines(self):
        distribution = fit_distribution(self.history, "points", "poisson")
        for line in (-1, 1.2, float("nan"), float("inf"), 201):
            with self.subTest(line=line), self.assertRaises(ValueError):
                distribution.probabilities(line)


class ForecastTests(unittest.TestCase):
    def test_forecast_metadata_and_explanations(self):
        frame = demo_frame()
        info = metadata(frame, "demo", "Synthetic", "synthetic")
        result = forecast(frame, info, {
            "player_id": "203999", "stat": "points", "model": "auto", "lookback": 30,
            "line": 26.5, "use_context": True, "odds_over": -110, "odds_under": -110, "simulations": 20_000,
        })
        self.assertEqual(result["sample_size"], 30)
        self.assertEqual(result["resolved_model"], "simulation")
        self.assertEqual(result["simulation"]["runs"], 20_000)
        self.assertEqual(result["simulation"]["box_score"], "basic")
        self.assertLess(result["simulation"]["standard_error"], 0.004)
        self.assertEqual([row["model"] for row in result["comparisons"]], ["empirical", "normal", "poisson", "negative_binomial", "simulation"])
        self.assertTrue(any("Synthetic" in text for text in result["warnings"]))
        self.assertEqual(result["history"], sorted(result["history"], key=lambda row: row["date"]))
        self.assertLessEqual(result["interval"]["low"], result["interval"]["high"])
        probabilities = result["probabilities"]
        self.assertAlmostEqual(probabilities["over"] + probabilities["under"] + probabilities["push"], 1)
        # The simulator is seeded: the same request gives the same answer.
        again = forecast(frame, info, {
            "player_id": "203999", "stat": "points", "model": "auto", "lookback": 30,
            "line": 26.5, "use_context": True, "odds_over": -110, "odds_under": -110, "simulations": 20_000,
        })
        self.assertEqual(again["probabilities"], probabilities)

    def test_combos_and_milestones(self):
        frame = demo_frame()
        info = metadata(frame, "demo", "Synthetic", "synthetic")
        base = {"player_id": "203999", "model": "auto", "lookback": 30, "use_context": True, "odds_over": -110,
                "odds_under": -110, "simulations": 20_000}
        pra = forecast(frame, info, {**base, "stat": "pts_reb_ast", "line": 47.5})
        history = frame[frame["player_id"] == "203999"].tail(30)
        self.assertEqual(pra["summary"]["observed_mean"], float((history["points"] + history["rebounds"] + history["assists"]).mean()))
        self.assertAlmostEqual(pra["summary"]["mean"], pra["summary"]["observed_mean"], delta=2.5)
        double = forecast(frame, info, {**base, "stat": "double_double", "line": 0.5})
        self.assertEqual(double["resolved_model"], "simulation")
        self.assertTrue(0 <= double["probabilities"]["over"] <= 1)
        self.assertEqual(double["probabilities"]["push"], 0)
        with self.assertRaisesRegex(ValueError, "yes/no market"):
            forecast(frame, info, {**base, "stat": "double_double", "line": 1.5})
        with self.assertRaisesRegex(ValueError, "box-score details"):
            forecast(frame, info, {**base, "stat": "threes", "line": 2.5})

    def test_insufficient_data_is_explicit(self):
        frame = demo_frame().query("player_id == '203999'").head(5)
        with self.assertRaisesRegex(ValueError, "10 completed"):
            forecast(frame, {}, {"player_id": "203999", "stat": "points"})


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = demo_frame().query("player_id == '203999'").head(32).copy()
        cls.request = {
            "player_id": "203999", "stat": "points", "model": "auto",
            "lookback": 20, "line": None, "use_context": True,
        }
        cls.result = backtest(cls.frame, cls.request)

    def test_score_and_bins_agree(self):
        result = self.result
        self.assertEqual(result["evaluations"], len(self.frame) - 15)
        score = np.mean([(row["probability"] - row["outcome"]) ** 2 for row in result["predictions"]])
        self.assertAlmostEqual(result["brier_score"], score)
        self.assertEqual(sum(row["count"] for row in result["calibration"]), result["evaluations"])
        self.assertEqual([row["model"] for row in result["model_scores"]],
                         ["empirical", "normal", "poisson", "negative_binomial", "simulation", "auto"])
        self.assertTrue(all(row["evaluations"] == result["evaluations"] for row in result["model_scores"]))
        self.assertAlmostEqual(result["skill_score"], 1 - result["brier_score"] / 0.25)

    def test_future_and_target_minutes_do_not_leak(self):
        changed = self.frame.copy()
        changed.loc[changed.index[-1], "points"] = 100
        changed.loc[changed.index[-1], "minutes"] = 1
        result = backtest(changed, self.request)
        old = self.result["predictions"]
        new = result["predictions"]
        self.assertEqual(old[:-1], new[:-1])
        self.assertEqual(old[-1]["probability"], new[-1]["probability"])
        self.assertEqual(old[-1]["line"], new[-1]["line"])

    def test_pushes_are_excluded(self):
        frame = self.frame.copy()
        frame["points"] = [20] * 16 + [19, 21] * 8
        result = backtest(frame, {**self.request, "line": 20, "model": "empirical"})
        self.assertEqual(result["pushes_excluded"], 1)
        self.assertEqual(result["evaluations"], 16)
        self.assertTrue(math.isfinite(result["log_loss"]))

    def test_empty_evaluation_rejected(self):
        frame = self.frame.copy()
        frame["points"] = 20
        with self.assertRaisesRegex(ValueError, "No settled evaluation"):
            backtest(frame, {**self.request, "line": 20})
