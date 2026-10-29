import unittest

from backend import nlu
from backend.stats import PROP_STATS


class MarketParsingTests(unittest.TestCase):
    def test_every_market_has_words_and_a_phrase_that_parses_back(self):
        self.assertEqual(set(nlu.STATS), set(PROP_STATS))
        for stat in PROP_STATS:
            with self.subTest(stat=stat):
                self.assertTrue(nlu.STAT_WORDS[stat])
                line = "" if stat in nlu.MILESTONES else " 4.5"
                spec = nlu.parse_leg(f"over{line} {nlu.phrase(stat)}")
                self.assertEqual(spec.stat, stat)

    def test_combo_and_shooting_wording(self):
        cases = {
            "Jokic over 45.5 PRA": ("pts_reb_ast", "over", 45.5),
            "Jokic o45.5 pts+reb+ast": ("pts_reb_ast", "over", 45.5),
            "Jokic over 45.5 points rebounds and assists": ("pts_reb_ast", "over", 45.5),
            "Murray under 30.5 points + assists": ("pts_ast", "under", 30.5),
            "Murray under 30.5 PA": ("pts_ast", "under", 30.5),
            "Gobert over 16.5 rebounds and assists": ("reb_ast", "over", 16.5),
            "Gobert over 2.5 steals + blocks": ("stocks", "over", 2.5),
            "Curry over 4.5 threes": ("threes", "over", 4.5),
            "Curry over 4.5 3pm": ("threes", "over", 4.5),
            "Curry 5+ 3-pointers": ("threes", "over", 4.5),
            "Curry over 11.5 three point attempts": ("threes_attempted", "over", 11.5),
            "SGA over 8.5 free throws made": ("free_throws_made", "over", 8.5),
            "SGA over 9.5 free throw attempts": ("free_throws_attempted", "over", 9.5),
            "SGA under 3.5 turnovers": ("turnovers", "under", 3.5),
            "Gobert over 3.5 offensive rebounds": ("offensive_rebounds", "over", 3.5),
            "Brunson over 9.5 field goals made": ("field_goals_made", "over", 9.5),
            "Brunson over 20.5 FGA": ("field_goals_attempted", "over", 20.5),
            "Jokic under 26.5 points": ("points", "under", 26.5),
            "Wemby over 3.5 blocks": ("blocks", "over", 3.5),
        }
        for text, (stat, side, line) in cases.items():
            with self.subTest(text=text):
                spec = nlu.parse_leg(text)
                self.assertEqual((spec.stat, spec.side, spec.line), (stat, side, line))

    def test_milestones_are_yes_no(self):
        self.assertEqual(vars(nlu.parse_leg("Jokic double-double")), vars(nlu.LegSpec("double_double", "over", 0.5)))
        self.assertEqual((nlu.parse_leg("Jokic triple double").stat, nlu.parse_leg("Jokic triple double").line), ("triple_double", 0.5))
        no = nlu.parse_leg("Jokic no double double")
        self.assertEqual((no.stat, no.side, no.line), ("double_double", "under", 0.5))

    def test_numbers_are_not_read_as_markets(self):
        self.assertEqual(nlu.parse_leg("Jokic over 13 points").stat, "points")
        self.assertEqual(nlu.parse_leg("Jokic over 13 points").line, 13)
        self.assertIsNone(nlu.parse_leg("Jokic over 3 minutes").stat)
        self.assertEqual(nlu.parse_leg("Jokic over 3 minutes").unsupported, "minutes")

    def test_over_or_under_asks_about_both_sides(self):
        for text, stat in (("over or under 23.5?", None), ("should i be over or under 23.5 points for curry?", "points"),
                           ("o/u 23.5 pts", "points"), ("Curry over/under 23.5", None), ("under or over 23.5 points", "points")):
            with self.subTest(text=text):
                spec = nlu.parse_leg(text)
                self.assertEqual((spec.stat, spec.side, spec.line), (stat, None, 23.5))
        self.assertEqual(nlu.parse_leg("the under?").side, "under")

    def test_a_bare_number_is_a_line_only_on_request(self):
        self.assertIsNone(nlu.parse_leg("what about 25.5?").line)
        self.assertEqual(nlu.parse_leg("what about 25.5?", bare_number=True).line, 25.5)
        self.assertEqual(nlu.parse_leg("what about 25.5 at -115?", bare_number=True).line, 25.5)

    def test_player_names_are_not_markets(self):
        self.assertEqual(nlu.parse_leg("add Trey Murphy III over 20.5 points").stat, "points")
        self.assertIsNone(nlu.parse_leg("Trey Lyles over 6.5").stat)
        self.assertEqual(nlu.parse_leg("Curry over 4.5 treys").stat, "threes")


if __name__ == "__main__":
    unittest.main()
