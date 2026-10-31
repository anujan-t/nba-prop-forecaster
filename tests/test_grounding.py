"""Grounding check: numbers in a model's reply must round from numbers the tools returned."""
import json
import unittest

from backend.grounding import dates_in, evidence_values, numbers_in, unsupported_numbers


class NumberExtractionTests(unittest.TestCase):
    def test_figures_percentages_and_ordinals(self):
        found = [(n.text, n.value, n.decimals, n.percent) for n in numbers_in("Jokić is at 54.3% (26.8 avg), 1,293 games, 3rd in the West")]
        self.assertEqual(found, [("54.3", 54.3, 1, True), ("26.8", 26.8, 1, False), ("1,293", 1293.0, 0, False), ("3", 3.0, 0, False)])
        self.assertEqual([n.value for n in numbers_in("a 61 percent chance at −110")], [61.0, 110.0])

    def test_names_list_markers_and_the_helpline_are_not_figures(self):
        text = "1. The 76ers shoot 3-pointers.\n2. Call 1-800-GAMBLER (1-800-426-2537), 24/7."
        self.assertEqual(numbers_in(text), [])


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.evidence = evidence_values(
            {"player": "Nikola Jokić", "over": 0.5432, "mean": 26.83, "games": 60, "interval_80": [19.0, 34.0],
             "record": "23-15", "season": "2025-26", "tipoff": "7:30 PM ET", "teams": ["BOS", "OKC", "CLE", "NYK", "DEN"]},
            "Jokic over 26.5 points tonight?")

    def test_rounded_tool_numbers_pass(self):
        reply = ("The model gives Jokić a **54%** chance (54.3% exactly) to go over 26.5; he projects for about 27 points "
                 "(26.8 on average, 80% range 19-34) over 60 games. Denver are 23-15 in 2025-26; tipoff 7:30 PM. Top 5 below.")
        self.assertEqual(unsupported_numbers(reply, self.evidence), [])

    def test_invented_numbers_are_reported_once_each(self):
        reply = "He's averaged 31.2 points and hit 8 of his last 12, a 67% rate; 31.2 is elite."
        self.assertEqual(unsupported_numbers(reply, self.evidence), ["31.2", "8", "12", "67%"])

    def test_precision_matters(self):
        self.assertEqual(unsupported_numbers("about 26.4 points", self.evidence), ["26.4"])
        self.assertEqual(unsupported_numbers("a 55% chance", self.evidence), ["55%"])
        self.assertEqual(unsupported_numbers("a 54.32% chance", self.evidence), [])

    def test_probabilities_only_become_percentages_with_a_percent_sign(self):
        evidence = evidence_values({"push": 0.031})
        self.assertEqual(unsupported_numbers("3 blocks", evidence), ["3"])
        self.assertEqual(unsupported_numbers("a 3.1% push chance", evidence), [])

    def test_dates_and_times_are_matched_as_dates_and_times(self):
        evidence = evidence_values("Today (US Eastern): 2026-10-28.", {"start": "2026-10-28T23:30Z", "tipoff": "7:30 PM ET",
                                                                       "detail": "10/28 - 7:30 PM EDT", "return": "Nov 5"})
        self.assertEqual(unsupported_numbers("Tonight, Wednesday, October 28, at 7:30 PM (19:30); back around Nov. 5.", evidence), [])
        self.assertEqual(unsupported_numbers("He's averaging 28 points and 10 rebounds.", evidence), ["28", "10"])
        self.assertEqual(unsupported_numbers("Next up Oct 29 at 8 pm.", evidence), ["Oct 29", "8 pm"])
        self.assertEqual(unsupported_numbers("2027-10-28", evidence_values("Today (US Eastern): 2026-10-28.")), ["2027-10-28"])
        self.assertEqual(unsupported_numbers("a 3/2 price", evidence), ["3", "2"])  # fractional odds, not March 2
        self.assertEqual(unsupported_numbers("Nuggets at Warriors on 10/28.", evidence), [])
        self.assertEqual(unsupported_numbers("Nuggets at Warriors on 10/29.", evidence), ["10", "29"])

    def test_betting_phrases_are_not_dates_or_times(self):
        evidence = evidence_values("Today (US Eastern): 2026-01-18.", {"line": 26.5, "threes": 4.8, "last_game": "2026-01-18",
                                                                       "tipoff": "10:00 PM ET", "headline": "Lakers' top 3 may sit"})
        self.assertEqual(unsupported_numbers("Jokić's 26.5 may be a touch high; the top 3 may sit.", evidence), [])
        self.assertEqual(unsupported_numbers("Curry is hitting 4.8 3PM a game.", evidence), [])
        self.assertEqual(unsupported_numbers("Suns at Warriors tips at 10 ET (10 PM).", evidence), [])
        self.assertEqual(unsupported_numbers("He went 1/18 from the field last game.", evidence), ["1", "18"])
        self.assertEqual(unsupported_numbers("His last game was on 1/18, Sun 1/18 (1/18).", evidence), [])
        self.assertEqual(dates_in("Back by 5 May 2026, or the 5th of May"), [("5 May 2026", 2026, 5, 5), ("5th of May", None, 5, 5)])

    def test_tool_results_as_text_lose_their_field_names(self):
        result = {"mean": 3.6, "interval_80": [1.0, 6.0]}
        self.assertEqual(unsupported_numbers("an 80% range of 1-6", evidence_values(result)), [])
        self.assertEqual(unsupported_numbers("an 80% range of 1-6", evidence_values(json.dumps(result))), ["80%"])


if __name__ == "__main__":
    unittest.main()
