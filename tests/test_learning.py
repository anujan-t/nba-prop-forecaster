from collections import Counter
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import tempfile
import threading
import unittest

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from backend.data import STATS, Store, validate_frame
from backend.learning import (
    Learning, bet_outcome, calibration_curve, fit_calibration, fit_recalibration, grade_leg, nba_game_logs,
    normalize_phrase, recalibrate, tokens, walk_forward_samples,
)
from backend.nlu import name_key
from backend.stats import stat_values

PLAYERS = {"201": ("Alpha One", "DEN"), "202": ("Beta Two", "BOS"), "203": ("Gamma Three", "LAL"), "204": ("Delta Four", "MIA")}
TEAMS = ("BOS", "LAL", "MIA", "NYK", "PHX", "DEN")
BASE = {"points": 14.0, "rebounds": 5.0, "assists": 3.5, "steals": 0.9, "blocks": 0.6}


def utc(*parts) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


class Clock:
    def __init__(self, moment: datetime):
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment


def game_row(player_id: str, day: str, **stats) -> dict:
    name, team = PLAYERS[player_id]
    return {"date": day, "player_id": player_id, "player_name": name, "team": team, "opponent": "PHX", "minutes": 32.0,
            **{stat: stats.get(stat, 0) for stat in STATS}}


class FakeHistory:
    """Stands in for the NBA provider: per-player game rows, optional failures, and a call log."""

    def __init__(self, games: dict[str, list[dict]], failures: dict[str, Exception] | None = None):
        self.games = games
        self.failures = dict(failures or {})
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, player_id: str, refresh: bool):
        self.calls.append((player_id, refresh))
        if player_id in self.failures:
            raise self.failures[player_id]
        frame = validate_frame(pd.DataFrame(self.games[player_id]))
        return f"nba-{player_id}", frame, {"id": f"nba-{player_id}", "source": "nba_api"}


def slip_leg(player_id: str, stat: str, line: float, side: str) -> dict:
    name, team = PLAYERS[player_id]
    request = {"dataset_id": f"nba-{player_id}", "player_id": player_id, "stat": stat, "line": line, "side": side,
               "model": "auto", "lookback": 30, "expected_minutes": None, "rest_days": None, "opponent": None,
               "use_context": True}
    return {"request": request, "label": {"player": name, "team": team}}


def priced_for(legs: list[dict], probabilities: list[float], pushes: list[float] | None = None) -> dict:
    """A ParlayResult-shaped dict for exactly these legs, as price_parlay would return it."""
    pushes = pushes or [0.0] * len(legs)
    quotes = []
    for index, (leg, probability, push) in enumerate(zip(legs, probabilities, pushes)):
        request = leg["request"]
        name, team = PLAYERS[request["player_id"]]
        quotes.append({"index": index, "player": {"id": request["player_id"], "name": name, "team": team},
                       "stat": request["stat"], "line": request["line"], "side": request["side"],
                       "probability": probability, "push": push})
    independent = math.prod(probabilities)
    return {"legs": quotes, "probability": independent * 1.05, "independent_probability": independent}


def track(learning: Learning, specs: list[tuple], probabilities: list[float], stake=None, odds=None,
          event_date="2026-01-15", pushes=None, note=None) -> dict:
    legs = [slip_leg(*spec) for spec in specs]
    return learning.track(legs, odds, stake, event_date, priced_for(legs, probabilities, pushes), note)


def game_log(player_id: str, seed: int, games: int = 72, growth: float = 0.015) -> pd.DataFrame:
    """Poisson box scores; with growth > 0 each mean rises by 1.5% of its starting level every game."""
    rng = np.random.default_rng(seed)
    start = date(2025, 10, 22)
    rows = []
    for t in range(games):
        scale = 1 + growth * t
        rows.append({"date": (start + timedelta(days=2 * t)).isoformat(), "player_id": player_id,
                     "player_name": f"Player {player_id}", "team": "OKC", "opponent": TEAMS[t % len(TEAMS)],
                     "minutes": float(rng.uniform(28, 36)),
                     **{stat: int(rng.poisson(mean * scale)) for stat, mean in BASE.items()}})
    return pd.DataFrame(rows)


def save_nba(store: Store, frame: pd.DataFrame, name: str) -> dict:
    query = {"player_ids": sorted(frame["player_id"].unique().tolist()), "season": "2025-26", "season_type": "Regular Season"}
    return store.save(frame, name, "nba_api", nba_query=query)


class TempStoreCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(Path(self.directory.name))
        self.clock = Clock(utc(2026, 1, 16, 18, 0))


class TextTests(unittest.TestCase):
    def test_normalize_phrase(self):
        self.assertEqual(normalize_phrase("  The   JOKER?! "), "the joker")
        self.assertEqual(normalize_phrase("¡Luka Dončić!"), "luka doncic")
        self.assertEqual(normalize_phrase("Nikola\tJokić."), "nikola jokic")
        self.assertEqual(normalize_phrase("\"SGA\""), "sga")
        self.assertEqual(normalize_phrase("A.J. Green"), "a.j. green")  # inner punctuation is kept

    def test_tokens(self):
        self.assertEqual(tokens("Jokić points over 25.5?"), {"jokic", "points", "over", "25", "5"})
        self.assertEqual(tokens("!!!"), set())


class MemoryTests(TempStoreCase):
    def setUp(self):
        super().setUp()
        self.learning = Learning(self.store, clock=self.clock)

    def test_alias_learning_overwrite_use_and_forget(self):
        alias = self.learning.learn_alias("  The Joker! ", "203999", "Nikola Jokić", "taught")
        self.assertEqual(alias, {"phrase": "the joker", "player_id": "203999", "player": "Nikola Jokić", "source": "taught", "uses": 0})
        self.assertEqual(list(self.learning.aliases()), ["the joker"])
        self.assertEqual(self.learning.use_alias("THE JOKER")["uses"], 1)
        self.assertEqual(self.learning.use_alias("the joker.")["uses"], 2)
        self.assertIsNone(self.learning.use_alias("nobody"))
        # Same player: the phrase is overwritten but keeps its usage count.
        self.assertEqual(self.learning.learn_alias("the joker", "203999", "Nikola Jokić", "clarified")["uses"], 2)
        # A different player overwrites the phrase and starts counting again.
        moved = self.learning.learn_alias("The Joker", "1628983", "Shai Gilgeous-Alexander", "clarified")
        self.assertEqual((moved["player_id"], moved["player"], moved["source"], moved["uses"]),
                         ("1628983", "Shai Gilgeous-Alexander", "clarified", 0))
        self.assertEqual(len(self.learning.aliases()), 1)
        self.assertTrue(self.learning.forget_alias("the JOKER"))
        self.assertFalse(self.learning.forget_alias("the joker"))
        self.assertEqual(self.learning.aliases(), {})

    def test_alias_keys_are_the_assistants_name_key(self):
        cases = {"A.J.": "aj", "D'Angelo": "dangelo", "De\u2019Aaron Fox": "deaaron fox", "Triple J Jr.": "triple j",
                 "Gilgeous-Alexander": "gilgeous alexander", "Big Rob III": "big rob", "  Luka   Dončić! ": "luka doncic"}
        for number, (phrase, key) in enumerate(cases.items()):
            self.assertEqual(name_key(phrase), key)
            self.assertEqual(self.learning.learn_alias(phrase, str(1000 + number), f"Player {number}")["phrase"], key)
        self.assertEqual(sorted(self.learning.aliases()), sorted(cases.values()))
        self.assertEqual(self.learning.aliases()["aj"]["player_id"], "1000")

        # Spellings with the same name key are one nickname, matching the assistant's lookups.
        self.assertEqual(self.learning.use_alias("aj")["uses"], 1)
        self.assertEqual(self.learning.use_alias("A.J.")["uses"], 2)
        self.assertEqual(self.learning.alias("a-j"), None)
        self.assertEqual(self.learning.alias("AJ")["player"], "Player 0")
        relearned = self.learning.learn_alias("AJ", "1000", "Player 0", "clarified")
        self.assertEqual((relearned["phrase"], relearned["source"], relearned["uses"]), ("aj", "clarified", 2))
        moved = self.learning.learn_alias("a.j.", "1628960", "Grayson Allen")
        self.assertEqual((moved["player"], moved["uses"]), ("Grayson Allen", 0))
        self.assertEqual(self.learning.use_alias("triple j")["player_id"], "1003")
        self.assertIsNone(self.learning.use_alias(None))
        self.assertTrue(self.learning.forget_alias("A.J."))
        self.assertTrue(self.learning.forget_alias("triple j jr"))
        self.assertFalse(self.learning.forget_alias("aj"))
        self.assertEqual(len(self.learning.aliases()), len(cases) - 2)

    def test_alias_validation(self):
        for phrase in ("a", "!!", "x" * 41):
            with self.assertRaisesRegex(ValueError, "2 to 40 characters"):
                self.learning.learn_alias(phrase, "203999", "Nikola Jokić")
        for phrase in ("23", "25.5", "#30", "1 000"):
            with self.assertRaisesRegex(ValueError, r"just a number.*can't be a nickname"):
                self.learning.learn_alias(phrase, "203999", "Nikola Jokić")
        for phrase in ("Points", "PTS", "reb", "Boards", "assists", "ast", "steals", "blocks", "pra", "threes"):
            with self.assertRaisesRegex(ValueError, "is a stat word"):
                self.learning.learn_alias(phrase, "203999", "Nikola Jokić")
        for player_id in ("", "   ", None):
            with self.assertRaisesRegex(ValueError, "needs a player"):
                self.learning.learn_alias("joker", player_id, "Nikola Jokić")
        with self.assertRaises(ValueError):
            self.learning.learn_alias("joker", "203999", "Nikola Jokić", "guessed")
        with self.assertRaises(ValueError):
            self.learning.learn_alias("joker", "20 3999", "Nikola Jokić")
        self.assertEqual(self.learning.aliases(), {})
        self.assertEqual(self.learning.learn_alias("x" * 40, "1", "Forty")["phrase"], "x" * 40)
        self.assertEqual(self.learning.learn_alias("Point God", "2", "Chris Paul")["phrase"], "point god")
        self.assertEqual(self.learning.learn_alias("KD 35", "3", "Kevin Durant")["phrase"], "kd 35")

    def test_aliases_saved_under_an_older_normalization_are_rekeyed(self):
        rows = [("a.j", "1631260", "AJ Green", "taught", 3, "2026-01-01T00:00:00Z"),
                ("d'angelo", "1626156", "D'Angelo Russell", "taught", 1, "2026-01-02T00:00:00Z"),
                ("dangelo", "1626156", "D'Angelo Russell", "clarified", 2, "2026-01-01T00:00:00Z"),
                ("the joker", "203999", "Nikola Jokić", "taught", 4, "2026-01-03T00:00:00Z")]
        with self.store.connect() as db:
            db.executemany("""INSERT INTO chat_aliases (phrase, player_id, player, source, uses, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""", [(*row, row[-1]) for row in rows])
        aliases = Learning(self.store, clock=self.clock).aliases()
        self.assertEqual(sorted(aliases), ["aj", "dangelo", "the joker"])
        self.assertEqual((aliases["aj"]["player"], aliases["aj"]["uses"]), ("AJ Green", 3))
        self.assertEqual((aliases["dangelo"]["source"], aliases["dangelo"]["uses"]), ("taught", 3))
        self.assertEqual(aliases["the joker"]["uses"], 4)

    def turn(self, message: str, conversation_id: str = "conv-1", **extra) -> str:
        return self.learning.record_turn({
            "conversation_id": conversation_id, "message": message, "reply": f"Reply to {message}", "intents": ["forecast"],
            "entities": {"players": ["203999"]}, "actions": [], "engine": "built_in", "slip_before": None,
            "slip_after": None, "tool_calls": [], **extra,
        })

    def test_record_and_read_turns(self):
        slip = {"legs": [slip_leg("201", "points", 22.5, "over")], "book_odds": None}
        tool_calls = [{"name": "forecast", "arguments": {"player_id": "201"}}]
        turn_id = self.turn("Alpha points over 22.5", actions=[{"type": "add_leg", "leg": 1, "description": "Added leg 1"}],
                            slip_after=slip, tool_calls=tool_calls, engine="llm")
        self.assertRegex(turn_id, r"^[0-9a-f]{16}$")
        saved = self.learning.turn(turn_id)
        self.assertEqual(saved, {
            "conversation_id": "conv-1", "message": "Alpha points over 22.5", "reply": "Reply to Alpha points over 22.5",
            "intents": ["forecast"], "entities": {"players": ["203999"]},
            "actions": [{"type": "add_leg", "leg": 1, "description": "Added leg 1"}], "engine": "llm",
            "slip_before": None, "slip_after": slip, "tool_calls": tool_calls,
            "turn_id": turn_id, "rating": None, "created_at": "2026-01-16T18:00:00Z",
        })
        self.assertEqual(self.learning.last_turns("conv-1"), [saved])
        self.assertIsNone(self.learning.turn("0" * 16))
        with self.assertRaises(ValueError):
            self.learning.record_turn({"conversation_id": "conv-1", "message": "no reply"})

    def test_turn_fields_round_trip_as_json_with_none_kept(self):
        empty = self.learning.record_turn({"conversation_id": "conv-1", "message": "hi", "reply": "Hello.", "intents": None,
                                           "entities": None, "actions": None, "engine": None, "tool_calls": None})
        saved = self.learning.turn(empty)
        for field in ("intents", "entities", "actions", "engine", "slip_before", "slip_after", "tool_calls"):
            self.assertIn(field, saved)
            self.assertIsNone(saved[field], field)
        odd = self.turn("odd values", intents=("forecast", "parlay"), entities={"day": date(2026, 1, 16), "count": np.int64(3),
                                                                                 "line": np.float64(22.5), "pair": (1, 2)},
                        actions=[], tool_calls=[], engine="")
        saved = self.learning.turn(odd)
        self.assertEqual(saved["intents"], ["forecast", "parlay"])
        self.assertEqual(saved["entities"], {"day": "2026-01-16", "count": 3, "line": 22.5, "pair": [1, 2]})
        self.assertEqual((saved["actions"], saved["tool_calls"], saved["engine"]), ([], [], ""))

    def test_last_turns_are_newest_first_per_conversation(self):
        ids = [self.turn(f"message {index}") for index in range(4)]
        self.turn("elsewhere", conversation_id="conv-2")
        recent = self.learning.last_turns("conv-1", limit=3)
        self.assertEqual([turn["turn_id"] for turn in recent], ids[::-1][:3])
        self.assertEqual(len(self.learning.last_turns("conv-1")), 4)
        self.assertEqual([turn["message"] for turn in self.learning.last_turns("conv-2")], ["elsewhere"])
        self.assertEqual(self.learning.last_turns("missing"), [])

    def test_turns_are_capped(self):
        self.learning.max_turns = 3
        ids = [self.turn(f"message {index}") for index in range(5)]
        self.assertIsNone(self.learning.turn(ids[0]))
        self.assertIsNone(self.learning.turn(ids[1]))
        self.assertEqual([turn["turn_id"] for turn in self.learning.last_turns("conv-1")], ids[:1:-1])
        self.assertEqual(self.learning.memory_summary()["turns"], 3)

    def test_feedback_replaces_rating_and_rejects_unknown_turns(self):
        turn_id = self.turn("Alpha points")
        self.assertEqual(self.learning.feedback(turn_id, 1), {"turn_id": turn_id, "rating": 1})
        self.assertEqual(self.learning.feedback(turn_id, -1, "Wrong player"), {"turn_id": turn_id, "rating": -1})
        saved = self.learning.turn(turn_id)
        self.assertEqual(saved["rating"], -1)
        self.assertNotIn("note", saved)
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT note FROM chat_turns WHERE id = ?", (turn_id,)).fetchone()["note"], "Wrong player")
        summary = self.learning.memory_summary()
        self.assertEqual((summary["turns"], summary["liked"], summary["disliked"]), (1, 0, 1))
        with self.assertRaises(KeyError):
            self.learning.feedback("f" * 16, 1)
        for rating in (0, 2, True):
            with self.assertRaises(ValueError):
                self.learning.feedback(turn_id, rating)

    def test_examples_use_liked_turns_ranked_by_similarity(self):
        best = self.turn("Jokic points over 25.5")
        second = self.turn("add Jokic rebounds")
        unrelated = self.turn("what is the weather")
        disliked = self.turn("Jokic points over 25.5 tonight please")
        self.turn("Jokic points")  # unrated
        for turn_id in (best, second, unrelated):
            self.learning.feedback(turn_id, 1)
        self.learning.feedback(disliked, -1)
        examples = self.learning.examples("Jokić points over?")
        self.assertEqual([example["message"] for example in examples], ["Jokic points over 25.5", "add Jokic rebounds"])
        self.assertEqual(set(examples[0]), {"message", "reply", "intents", "tool_calls"})
        self.assertEqual(examples[0]["intents"], ["forecast"])
        self.assertEqual(len(self.learning.examples("Jokić points over?", limit=1)), 1)
        self.assertEqual(self.learning.examples("zzz qqq"), [])
        # A repeated liked message appears once, with its newest reply.
        repeat = self.learning.record_turn({"conversation_id": "conv-1", "message": "jokic points over 25.5!",
                                            "reply": "Newer reply", "intents": [], "entities": {}, "actions": [],
                                            "engine": "built_in", "slip_before": None, "slip_after": None, "tool_calls": []})
        self.learning.feedback(repeat, 1)
        examples = self.learning.examples("Jokić points over?")
        self.assertEqual([example["reply"] for example in examples], ["Newer reply", "Reply to add Jokic rebounds"])

    def test_memory_summary(self):
        self.learning.learn_alias("the joker", "203999", "Nikola Jokić")
        self.learning.learn_alias("Ant", "1630162", "Anthony Edwards", "clarified")
        liked = self.turn("one")
        self.turn("two")
        self.learning.feedback(liked, 1)
        summary = self.learning.memory_summary()
        self.assertEqual([alias["phrase"] for alias in summary["aliases"]], ["ant", "the joker"])
        self.assertEqual(summary["aliases"][0], {"phrase": "ant", "player_id": "1630162", "player": "Anthony Edwards",
                                                 "source": "clarified", "uses": 0})
        self.assertEqual((summary["turns"], summary["liked"], summary["disliked"]), (2, 1, 0))

    def test_concurrent_turns(self):
        ids: list[str] = []

        def worker(offset: int):
            for index in range(10):
                ids.append(self.turn(f"thread {offset} message {index}"))

        threads = [threading.Thread(target=worker, args=(offset,)) for offset in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(set(ids)), 40)
        self.assertEqual(self.learning.memory_summary()["turns"], 40)


class TrackTests(TempStoreCase):
    def setUp(self):
        super().setUp()
        self.learning = Learning(self.store, clock=self.clock)

    def test_track_snapshots_the_priced_slip(self):
        legs = [slip_leg("201", "points", 22.5, "over"), slip_leg("202", "rebounds", 10, "under")]
        priced = priced_for(legs, [0.6, 0.45], pushes=[0.0, 0.08])
        bet = self.learning.track(legs, 264, 10, date(2026, 1, 15), priced, "  Friday parlay ")
        self.assertRegex(bet["id"], r"^[0-9a-f]{12}$")
        self.assertEqual(bet["created_at"], "2026-01-16T18:00:00Z")
        self.assertEqual(bet["event_date"], "2026-01-15")
        self.assertEqual((bet["book_odds"], bet["stake"], bet["status"], bet["profit"], bet["note"]),
                         (264, 10.0, "pending", None, "Friday parlay"))
        self.assertAlmostEqual(bet["independent_probability"], 0.27)
        self.assertAlmostEqual(bet["probability"], 0.27 * 1.05)
        self.assertEqual(bet["legs"][0], {"player_id": "201", "player": "Alpha One", "team": "DEN", "stat": "points",
                                          "line": 22.5, "side": "over", "probability": 0.6, "status": "pending",
                                          "actual": None, "settled_at": None})
        self.assertEqual((bet["legs"][1]["line"], bet["legs"][1]["side"], bet["legs"][1]["probability"]), (10.0, "under", 0.45))
        self.assertEqual(self.learning.bets(), [bet])

    def test_event_date_defaults_to_today_in_eastern_time(self):
        learning = Learning(self.store, clock=Clock(utc(2026, 1, 16, 3, 0)))  # 10 PM on Jan 15 in New York
        legs = [slip_leg("201", "points", 22.5, "over")]
        self.assertEqual(learning.track(legs, None, None, None, priced_for(legs, [0.6]))["event_date"], "2026-01-15")
        injected = Learning(self.store, clock=self.clock, today=lambda: date(2026, 2, 1))
        self.assertEqual(injected.track(legs, None, None, None, priced_for(legs, [0.6]))["event_date"], "2026-02-01")
        self.assertEqual(learning.track(legs, None, None, "2026-01-20", priced_for(legs, [0.6]))["event_date"], "2026-01-20")

    def test_bare_requests_use_priced_player_labels(self):
        request = slip_leg("203", "assists", 4.5, "over")
        bet = self.learning.track([request["request"]], None, None, None, priced_for([request], [0.5]))
        self.assertEqual((bet["legs"][0]["player"], bet["legs"][0]["team"]), ("Gamma Three", "LAL"))

    def test_track_validation(self):
        legs = [slip_leg("201", "points", 22.5, "over")]
        priced = priced_for(legs, [0.6])
        invalid = [
            dict(slip_legs=[], priced={"legs": [], "probability": 1, "independent_probability": 1}),
            dict(slip_legs=[slip_leg("201", "points", 10.5 + k, "over") for k in range(11)]),
            dict(stake=0), dict(stake=-5), dict(stake=1_000_001), dict(stake=True), dict(stake="10"),
            dict(note="x" * 201), dict(book_odds=50), dict(book_odds=10001), dict(book_odds=-99), dict(book_odds=150.5),
            dict(event_date="01/15/2026"), dict(event_date="2026-02-30"),
            dict(priced=priced_for(legs + [slip_leg("202", "points", 20.5, "over")], [0.6, 0.5])),
            dict(priced=priced_for([slip_leg("201", "points", 22.5, "under")], [0.4])),
            dict(priced=priced_for([slip_leg("201", "points", 23.5, "over")], [0.4])),
            dict(slip_legs=[slip_leg("201", "points", 22.5, "over"), slip_leg("201", "points", 25.5, "over")],
                 priced=priced_for([slip_leg("201", "points", 22.5, "over"), slip_leg("201", "points", 25.5, "over")], [0.6, 0.4])),
            dict(slip_legs=[slip_leg("201", "minutes", 2.5, "over")], priced=priced_for([slip_leg("201", "minutes", 2.5, "over")], [0.5])),
            dict(slip_legs=[slip_leg("201", "double_double", 1.5, "over")], priced=priced_for([slip_leg("201", "double_double", 1.5, "over")], [0.5])),
            dict(slip_legs=[slip_leg("201", "points", 22.3, "over")], priced=priced_for([slip_leg("201", "points", 22.3, "over")], [0.5])),
            dict(priced=priced_for(legs, [1.5])),
        ]
        for overrides in invalid:
            arguments = {"slip_legs": legs, "book_odds": None, "stake": None, "event_date": None, "priced": priced,
                         "note": None, **overrides}
            with self.subTest(overrides=str(overrides)[:80]), self.assertRaises(ValueError):
                self.learning.track(**arguments)
        self.assertEqual(self.learning.bets(), [])
        ten = [slip_leg("201", stat, 0.5 + k, "over") for k, stat in enumerate(STATS)]
        ten += [slip_leg("202", stat, 0.5 + k, "under") for k, stat in enumerate(STATS)]
        bet = self.learning.track(ten, -10000, 1_000_000, None, priced_for(ten, [0.9] * 10), "x" * 200)
        self.assertEqual((len(bet["legs"]), bet["book_odds"], bet["stake"]), (10, -10000, 1_000_000.0))

    def test_bets_are_newest_first_and_deletable(self):
        first = track(self.learning, [("201", "points", 22.5, "over")], [0.6])
        second = track(self.learning, [("202", "points", 20.5, "under")], [0.5])
        self.assertEqual([bet["id"] for bet in self.learning.bets()], [second["id"], first["id"]])
        self.assertTrue(self.learning.delete_bet(first["id"]))
        self.assertFalse(self.learning.delete_bet(first["id"]))
        self.assertIsNone(self.learning.bet(first["id"]))
        self.assertEqual([bet["id"] for bet in self.learning.bets()], [second["id"]])
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM ledger_legs WHERE bet_id = ?", (first["id"],)).fetchone()[0], 0)


class GradingTests(unittest.TestCase):
    def test_grade_leg(self):
        self.assertEqual(grade_leg(25, 22.5, "over"), "won")
        self.assertEqual(grade_leg(18, 20.5, "over"), "lost")
        self.assertEqual(grade_leg(10, 10, "over"), "push")
        self.assertEqual(grade_leg(10, 10, "under"), "push")
        self.assertEqual(grade_leg(5, 6.5, "under"), "won")
        self.assertEqual(grade_leg(7, 6.5, "under"), "lost")
        self.assertEqual(grade_leg(0, 0.5, "under"), "won")

    def test_bet_outcome(self):
        self.assertEqual(bet_outcome(["won", "lost"], 10.0, 200), ("lost", -10.0))
        self.assertEqual(bet_outcome(["lost", "pending"], 10.0, 200), ("lost", -10.0))
        self.assertEqual(bet_outcome(["lost"], None, 200), ("lost", None))
        self.assertEqual(bet_outcome(["won", "pending"], 10.0, 200), ("pending", None))
        self.assertEqual(bet_outcome(["void", "void"], 10.0, 200), ("void", 0.0))
        self.assertEqual(bet_outcome(["void"], None, 200), ("void", None))
        self.assertEqual(bet_outcome(["push"], 10.0, -110), ("void", 0.0))  # a lone push is refunded
        self.assertEqual(bet_outcome(["won", "won"], 10.0, 264), ("won", 26.4))
        self.assertEqual(bet_outcome(["won"], 11.0, -110), ("won", 10.0))
        self.assertEqual(bet_outcome(["won"], 10.0, None), ("won", None))
        self.assertEqual(bet_outcome(["won"], None, 150), ("won", None))
        self.assertEqual(bet_outcome(["won", "push"], 10.0, 264), ("reduced", None))
        self.assertEqual(bet_outcome(["won", "void", "won"], 10.0, 264), ("reduced", None))


class SettleTests(TempStoreCase):
    def setUp(self):
        super().setUp()
        self.history = FakeHistory({
            "201": [game_row("201", "2026-01-13", points=20, rebounds=8, assists=4),
                    game_row("201", "2026-01-15", points=25, rebounds=10, assists=5)],
            "202": [game_row("202", "2026-01-12", points=22, rebounds=9, assists=3),
                    game_row("202", "2026-01-15", points=18, rebounds=6, assists=7)],
            "203": [game_row("203", "2026-01-12", points=15), game_row("203", "2026-01-14", points=12)],
            "204": [game_row("204", "2026-01-13", points=8), game_row("204", "2026-01-15", points=12)],
        }, failures={"204": RuntimeError("NBA Stats timed out")})
        self.learning = Learning(self.store, history=self.history, clock=self.clock)

    def bets_by_label(self) -> dict:
        return {bet["note"]: bet for bet in self.learning.bets()}

    def test_settle_grades_past_legs_once_per_player(self):
        track(self.learning, [("201", "points", 22.5, "over"), ("202", "points", 20.5, "over")], [0.6, 0.55], 10, 264, note="A")
        track(self.learning, [("201", "rebounds", 10, "over"), ("202", "assists", 6.5, "over")], [0.5, 0.7], 5, 200,
              pushes=[0.12, 0.0], note="B")
        track(self.learning, [("203", "points", 15.5, "over")], [0.4], 20, -110, note="C")
        track(self.learning, [("201", "points", 30.5, "over")], [0.3], event_date="2026-01-16", note="D")
        track(self.learning, [("201", "assists", 6.5, "under")], [0.65], 10, 150, note="E")
        track(self.learning, [("204", "points", 10.5, "over")], [0.5], 5, -110, note="F")
        track(self.learning, [("202", "rebounds", 8.5, "under")], [0.8], note="G")

        result = self.learning.settle()
        self.assertEqual(Counter(self.history.calls), Counter({("201", True): 1, ("202", True): 1, ("203", True): 1, ("204", True): 1}))
        self.assertEqual((result["settled_legs"], result["updated_bets"]), (7, 5))
        self.assertEqual(len(result["warnings"]), 1)
        self.assertIn("Delta Four", result["warnings"][0])
        self.assertIn("NBA Stats timed out", result["warnings"][0])
        self.assertNotIn("Traceback", result["warnings"][0])

        bets = self.bets_by_label()
        self.assertEqual({label: (bet["status"], bet["profit"]) for label, bet in bets.items()}, {
            "A": ("lost", -10.0), "B": ("reduced", None), "C": ("void", 0.0), "D": ("pending", None),
            "E": ("won", 15.0), "F": ("pending", None), "G": ("won", None),
        })
        self.assertEqual([(leg["status"], leg["actual"]) for leg in bets["A"]["legs"]], [("won", 25), ("lost", 18)])
        self.assertEqual([(leg["status"], leg["actual"]) for leg in bets["B"]["legs"]], [("push", 10), ("won", 7)])
        self.assertEqual((bets["C"]["legs"][0]["status"], bets["C"]["legs"][0]["actual"]), ("void", None))
        self.assertEqual(bets["A"]["legs"][0]["settled_at"], "2026-01-16T18:00:00Z")
        self.assertEqual((bets["D"]["legs"][0]["status"], bets["D"]["legs"][0]["settled_at"]), ("pending", None))
        self.assertEqual(bets["F"]["legs"][0]["status"], "pending")

        record = result["record"]
        self.assertEqual(record, self.learning.track_record())
        self.assertEqual({key: record[key] for key in ("bets", "pending", "won", "lost", "void", "reduced")},
                         {"bets": 7, "pending": 2, "won": 2, "lost": 1, "void": 1, "reduced": 1})
        self.assertAlmostEqual(record["staked"], 40.0)  # A, C and E: B's profit is unknown, G has no stake
        self.assertAlmostEqual(record["profit"], 5.0)
        self.assertAlmostEqual(record["roi"], 0.125)
        self.assertEqual((record["legs_graded"], record["legs_won"]), (5, 4))
        self.assertAlmostEqual(record["expected_legs_won"], 0.6 + 0.55 + 0.7 + 0.65 + 0.8)
        self.assertAlmostEqual(record["brier"], (0.4 ** 2 + 0.55 ** 2 + 0.3 ** 2 + 0.35 ** 2 + 0.2 ** 2) / 5)

        # Settling again only retries the player whose provider call failed.
        self.history.calls.clear()
        again = self.learning.settle()
        self.assertEqual(self.history.calls, [("204", True)])
        self.assertEqual((again["settled_legs"], again["updated_bets"], len(again["warnings"])), (0, 0, 1))

        # The next day the provider recovers and Jan 16 is final: Alpha One did not play, so D is void.
        self.history.calls.clear()
        self.history.failures.clear()
        self.clock.moment = utc(2026, 1, 17, 18, 0)
        later = self.learning.settle()
        self.assertEqual(sorted(self.history.calls), [("201", True), ("204", True)])
        self.assertEqual((later["settled_legs"], later["updated_bets"], later["warnings"]), (2, 2, []))
        bets = self.bets_by_label()
        self.assertEqual((bets["D"]["status"], bets["D"]["profit"]), ("void", None))
        self.assertEqual(bets["F"]["status"], "won")
        self.assertAlmostEqual(bets["F"]["profit"], 5 * 100 / 110, places=5)
        self.assertEqual(later["record"]["pending"], 0)

    def test_only_strictly_past_games_are_settled(self):
        track(self.learning, [("201", "points", 22.5, "over")], [0.6], event_date="2026-01-16")
        track(self.learning, [("202", "points", 20.5, "over")], [0.6], event_date="2026-01-18")
        result = self.learning.settle()
        self.assertEqual(self.history.calls, [])
        self.assertEqual((result["settled_legs"], result["updated_bets"], result["warnings"]), (0, 0, []))

    def test_missing_box_score_waits_until_morning_after_late_games(self):
        self.clock.moment = utc(2026, 1, 16, 6, 30)  # 1:30 AM in New York
        late = track(self.learning, [("203", "points", 15.5, "over")], [0.4], 10, -110)
        final = track(self.learning, [("201", "points", 22.5, "over")], [0.6], 10, -110)
        early = self.learning.settle()
        self.assertEqual(early["settled_legs"], 1)
        self.assertEqual(self.learning.bet(final["id"])["status"], "won")
        self.assertEqual(self.learning.bet(late["id"])["status"], "pending")
        self.assertTrue(any("Gamma Three" in warning and "No box score yet" in warning for warning in early["warnings"]))
        self.clock.moment = utc(2026, 1, 16, 13, 0)  # 8 AM in New York
        self.assertEqual(self.learning.settle()["settled_legs"], 1)
        self.assertEqual(self.learning.bet(late["id"])["status"], "void")

    def test_dates_before_the_returned_game_log_stay_pending(self):
        old = track(self.learning, [("201", "points", 22.5, "over")], [0.6], event_date="2025-10-01")
        result = self.learning.settle()
        self.assertEqual(result["settled_legs"], 0)
        self.assertEqual(self.learning.bet(old["id"])["status"], "pending")
        self.assertIn("starts on 2026-01-13", result["warnings"][0])

    def test_settle_without_a_history_source(self):
        learning = Learning(self.store, clock=self.clock)
        track(learning, [("201", "points", 22.5, "over")], [0.6])
        result = learning.settle()
        self.assertEqual(result["settled_legs"], 0)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertEqual(learning.bets()[0]["status"], "pending")

    def test_empty_track_record(self):
        record = self.learning.track_record()
        self.assertEqual(record, {"bets": 0, "pending": 0, "won": 0, "lost": 0, "void": 0, "reduced": 0, "staked": 0.0,
                                  "profit": 0.0, "roi": None, "legs_graded": 0, "legs_won": 0, "expected_legs_won": 0.0,
                                  "brier": None})


class CalibrationMathTests(unittest.TestCase):
    def test_newton_fit_recovers_known_recalibration(self):
        rng = np.random.default_rng(11)
        p = rng.uniform(0.02, 0.98, 40_000)
        y = (rng.uniform(size=p.size) < expit(0.4 + 0.7 * logit(p))).astype(int)
        intercept, slope = fit_recalibration(p, y)
        self.assertAlmostEqual(intercept, 0.4, delta=0.05)
        self.assertAlmostEqual(slope, 0.7, delta=0.05)
        # The penalized score is zero at the returned optimum.
        x = logit(p)
        q = expit(intercept + slope * x)
        gradient = np.array([np.sum(q - y) + 20 * intercept, np.sum((q - y) * x) + 20 * (slope - 1)])
        self.assertLess(np.max(np.abs(gradient)), 1e-6)

    def test_ridge_penalty_shrinks_scarce_data_toward_identity(self):
        intercept, slope = fit_recalibration(np.full(20, 0.5), np.ones(20))
        self.assertGreater(intercept, 0.3)
        self.assertLess(intercept, 0.5)  # an unpenalized fit would diverge
        self.assertAlmostEqual(slope, 1.0, places=9)
        self.assertEqual(fit_recalibration([], []), (0.0, 1.0))

    def test_recalibrate(self):
        self.assertAlmostEqual(recalibrate(0.3, 0.0, 1.0), 0.3)
        self.assertAlmostEqual(recalibrate(0.5, 0.4, 0.7), float(expit(0.4)))
        self.assertAlmostEqual(recalibrate(0.0, 0.0, 1.0), 1e-4)  # probabilities are clipped before the logit
        np.testing.assert_allclose(recalibrate(np.array([0.2, 0.8]), 0.0, 1.0), [0.2, 0.8])

    def test_calibration_curve(self):
        curve = calibration_curve([0.05, 0.15, 0.95, 1.0, 0.0], [0, 1, 1, 1, 0], None, None)
        self.assertEqual([row["count"] for row in curve], [2, 1, 2])
        self.assertEqual([row["observed"] for row in curve], [0.0, 1.0, 1.0])
        self.assertEqual([row["predicted"] for row in curve], [0.025, 0.15, 0.975])
        self.assertEqual([row["calibrated"] for row in curve], [row["predicted"] for row in curve])
        self.assertEqual(calibration_curve([], [], 0.0, 1.0), [])

    def test_well_calibrated_predictions_stay_off(self):
        rng = np.random.default_rng(7)
        p = rng.uniform(0.05, 0.95, 4000)
        samples = pd.DataFrame({
            "date": [f"2025-{1 + index // 400:02d}-{1 + index % 28:02d}" for index in range(4000)], "player_id": "X",
            "stat": "points", "line": 10.5, "p": p, "outcome": (rng.uniform(size=4000) < p).astype(int), "source": "walk_forward",
        })
        status, params = fit_calibration(samples, "2026-01-16T18:00:00Z")
        self.assertFalse(status["active"])
        self.assertFalse(params[0])
        self.assertEqual(status["holdout"]["samples"], 1000)
        self.assertTrue(status["message"].startswith("The raw model is already as well calibrated on the most recent 1,000"))
        self.assertAlmostEqual(status["intercept"], 0.0, delta=0.15)
        self.assertAlmostEqual(status["slope"], 1.0, delta=0.15)

    def test_too_few_predictions_stay_off(self):
        samples = pd.DataFrame({"date": ["2026-01-01"] * 310, "player_id": "X", "stat": "points", "line": 10.5,
                                "p": 0.3, "outcome": 1, "source": "walk_forward"})
        status, params = fit_calibration(samples)
        self.assertEqual(params[0], False)
        self.assertIsNone(status["holdout"])
        self.assertTrue(status["message"].startswith("Needs at least 800 graded predictions (has 310); keeps the raw model for now."))


class WalkForwardTests(unittest.TestCase):
    def test_samples_use_only_games_before_the_target(self):
        frame = validate_frame(game_log("W1", 21, games=25))
        samples = walk_forward_samples(frame)
        target = frame["date"].iloc[-1]
        self.assertGreater(len(samples), 0)
        self.assertEqual(set(samples["source"]), {"walk_forward"})
        self.assertTrue(set(samples["date"]) <= set(frame["date"].iloc[15:]))
        self.assertTrue((samples["line"] % 1 == 0.5).all())
        self.assertTrue(samples["p"].between(0, 1).all())
        # Every market the box scores support, including combos and milestones (no splits here, so no threes).
        self.assertEqual(set(samples["stat"]), {"points", "rebounds", "assists", "steals", "blocks", "pts_reb_ast", "pts_reb",
                                                "pts_ast", "reb_ast", "stocks", "double_double", "triple_double"})
        self.assertTrue((samples.loc[samples["stat"].isin(["double_double", "triple_double"]), "line"] == 0.5).all())
        actual = pd.DataFrame({"date": frame["date"], **{stat: stat_values(frame, stat) for stat in set(samples["stat"])}})
        actual = actual.melt(id_vars=["date"], var_name="stat", value_name="actual")
        merged = samples.merge(actual, on=["date", "stat"])
        self.assertEqual(len(merged), len(samples))
        self.assertTrue(((merged["actual"] > merged["line"]).astype(int) == merged["outcome"]).all())

        # Changing the last game's box score and minutes moves no forecast, only that game's outcomes.
        changed = frame.copy()
        last = changed.index[-1]
        changed.loc[last, "minutes"] = 5.0
        for stat in STATS:
            changed.loc[last, stat] = 0 if frame.loc[last, stat] > 3 else 40
        moved = walk_forward_samples(changed)
        pd.testing.assert_frame_equal(moved.drop(columns="outcome"), samples.drop(columns="outcome"))
        earlier = samples["date"] != target
        pd.testing.assert_series_equal(moved.loc[earlier, "outcome"], samples.loc[earlier, "outcome"])
        self.assertFalse(moved.loc[~earlier, "outcome"].equals(samples.loc[~earlier, "outcome"]))

    def test_short_histories_produce_no_samples(self):
        samples = walk_forward_samples(validate_frame(game_log("W2", 22, games=15)))
        self.assertEqual(len(samples), 0)
        self.assertEqual(list(samples.columns), ["date", "player_id", "stat", "line", "p", "outcome", "source"])


class RetrainTests(unittest.TestCase):
    """Four players whose production climbs all season: the trailing-window model keeps under-forecasting overs."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.store = Store(Path(cls.directory.name))
        logs = {player: game_log(player, seed) for seed, player in enumerate(("P1", "P2", "P3", "P4"), start=1)}
        stale = logs["P3"].iloc[:40].assign(minutes=20.0)
        save_nba(cls.store, pd.concat([logs["P1"], logs["P2"], stale], ignore_index=True), "Older fetch")
        save_nba(cls.store, pd.concat([logs["P3"], logs["P4"]], ignore_index=True), "Newer fetch")
        cls.store.save(game_log("P9", 9, growth=-0.01), "Uploaded CSV", "csv")
        cls.logs = logs
        cls.clock = Clock(utc(2026, 1, 16, 18, 0))
        cls.learning = Learning(cls.store, clock=cls.clock)
        cls.status = cls.learning.retrain()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_game_logs_merge_nba_datasets_only(self):
        logs = nba_game_logs(self.store)
        self.assertEqual(len(logs), 4 * 72)
        self.assertFalse(logs.duplicated(["player_id", "date"]).any())
        self.assertEqual(sorted(logs["player_id"].unique()), ["P1", "P2", "P3", "P4"])
        p3 = logs[logs["player_id"] == "P3"].reset_index(drop=True)
        np.testing.assert_allclose(p3["minutes"], self.logs["P3"]["minutes"])  # the newer copy wins

    def test_drift_switches_the_correction_on(self):
        status = self.status
        self.assertTrue(status["active"])
        self.assertEqual(status["trained_at"], "2026-01-16T18:00:00Z")
        self.assertEqual((status["players"], status["ledger_samples"]), (4, 0))
        self.assertGreater(status["samples"], 4000)
        holdout = status["holdout"]
        self.assertEqual(holdout["samples"], status["samples"] // 4)
        self.assertLess(holdout["calibrated_log_loss"], holdout["raw_log_loss"] - 0.001)
        self.assertLessEqual(holdout["calibrated_brier"], holdout["raw_brier"])
        self.assertGreater(status["intercept"], 0)
        self.assertRegex(status["message"], r"^Learned from [\d,]+ graded predictions across 4 players\. On the most recent [\d,]+, "
                                            r"the correction lowered log loss from 0\.\d{3} to 0\.\d{3}, so it is switched on\.")
        self.assertIn("Overs have hit more often than forecast", status["message"])
        self.assertFalse(self.learning.training)

    def test_calibrate_over_moves_overs_up(self):
        calibrated = [self.learning.calibrate_over(p, "points") for p in (0.2, 0.3, 0.5, 0.6)]
        self.assertGreater(calibrated[2], 0.55)
        self.assertGreater(calibrated[1], 0.3)
        self.assertEqual(calibrated, sorted(calibrated))
        points = next(entry for entry in self.status["stats"] if entry["stat"] == "points")
        self.assertTrue(points["active"])
        self.assertAlmostEqual(calibrated[2], recalibrate(0.5, points["intercept"], points["slope"]), places=5)
        # Without a market the shared correction applies.
        self.assertAlmostEqual(self.learning.calibrate_over(0.5), recalibrate(0.5, self.status["intercept"], self.status["slope"]), places=5)

    def test_every_market_is_calibrated_separately(self):
        entries = {entry["stat"]: entry for entry in self.status["stats"]}
        self.assertTrue({"points", "rebounds", "assists", "steals", "blocks", "pts_reb_ast", "stocks"} <= set(entries))
        self.assertEqual(sum(entry["samples"] for entry in entries.values()), self.status["samples"])
        for stat, entry in entries.items():
            self.assertIn(entry["choice"], ("own", "shared", "none"))
            self.assertEqual(entry["active"], entry["choice"] != "none")
            self.assertTrue(entry["message"])
            if entry["holdout"]:
                self.assertTrue({"raw_log_loss", "shared_log_loss", "own_log_loss", "raw_brier"} <= set(entry["holdout"]))
            if entry["choice"] == "own":
                expected = recalibrate(0.4, entry["params"][1], entry["params"][2])
                self.assertAlmostEqual(self.learning.calibrate_over(0.4, stat), expected, places=9)
            if entry["choice"] == "none":
                self.assertIsNone(self.learning.calibrate_over(0.4, stat))
        self.assertTrue(any(entry["choice"] == "own" for entry in entries.values()))
        self.assertRegex(self.status["message"], r"Calibrated per market: \d+ of \d+ use their own correction")

    def test_calibrate_over_clips_inputs_and_stays_strictly_inside_0_1(self):
        low, high = self.learning.calibrate_over(0), self.learning.calibrate_over(1.0)
        self.assertTrue(0 < low < high < 1)
        self.assertEqual(self.learning.calibrate_over(-0.25), low)
        self.assertEqual(self.learning.calibrate_over(1.5), high)
        self.assertEqual(self.learning.calibrate_over(float("inf")), high)
        self.assertIsInstance(self.learning.calibrate_over(np.float64(0.4)), float)
        for bad in (float("nan"), None, "half"):
            self.assertIsNone(self.learning.calibrate_over(bad))
        extreme = Learning(self.store, clock=self.clock)
        extreme._params = (True, 40.0, 25.0)  # a steep map would round to exactly 0 or 1 without the final clip
        self.assertTrue(all(0 < extreme.calibrate_over(p) < 1 for p in (0, 1e-6, 0.5, 1 - 1e-6, 1)))

    def test_curve_bins(self):
        curve = self.status["curve"]
        self.assertLessEqual(len(curve), 10)
        self.assertEqual(sum(row["count"] for row in curve), self.status["samples"])
        predicted = [row["predicted"] for row in curve]
        self.assertEqual(predicted, sorted(predicted))
        for row in curve:
            for key in ("predicted", "observed", "calibrated"):
                self.assertTrue(0 <= row[key] <= 1)
        raw_gap = sum(row["count"] * abs(row["predicted"] - row["observed"]) for row in curve)
        calibrated_gap = sum(row["count"] * abs(row["calibrated"] - row["observed"]) for row in curve)
        self.assertLess(calibrated_gap, raw_gap / 2)

    def test_calibration_survives_a_restart(self):
        restarted = Learning(self.store)
        self.assertEqual(restarted.calibration_status(), self.status)
        self.assertEqual(restarted.calibrate_over(0.5), self.learning.calibrate_over(0.5))
        status = restarted.status()
        self.assertEqual(set(status), {"calibration", "memory", "record", "training"})
        self.assertTrue(status["calibration"]["active"])
        self.assertFalse(status["training"])


class SmallDataTests(TempStoreCase):
    def setUp(self):
        super().setUp()
        save_nba(self.store, game_log("P1", 5, games=20, growth=0.0), "Short history")
        self.history = FakeHistory({
            "201": [game_row("201", "2026-01-15", points=25, rebounds=10, assists=5)],
            "202": [game_row("202", "2026-01-15", points=18, rebounds=6, assists=7)],
        })
        self.learning = Learning(self.store, history=self.history, clock=self.clock)

    def test_untrained_status(self):
        status = Learning(Store(Path(self.directory.name) / "empty"), clock=self.clock).calibration_status()
        self.assertEqual({key: status[key] for key in ("active", "trained_at", "samples", "ledger_samples", "players", "holdout", "curve")},
                         {"active": False, "trained_at": None, "samples": 0, "ledger_samples": 0, "players": 0, "holdout": None, "curve": []})
        self.assertIn("graded predictions", status["message"])
        self.assertIsNone(self.learning.calibrate_over(0.6))

    def test_settled_legs_become_over_samples(self):
        track(self.learning, [("201", "points", 22.5, "over"), ("202", "assists", 6.5, "under")], [0.55, 0.7])
        track(self.learning, [("202", "points", 20.5, "over"), ("201", "rebounds", 10, "over")], [0.6, 0.5], pushes=[0.0, 0.2])
        track(self.learning, [("201", "assists", 4, "over")], [0.5], pushes=[0.2])
        self.learning.settle()
        samples = self.learning.ledger_samples()
        self.assertEqual(set(samples["source"]), {"ledger"})
        rows = {(row.player_id, row.stat): (round(row.p, 6), row.outcome, row.date) for row in samples.itertuples()}
        self.assertEqual(rows, {
            ("201", "points"): (0.55, 1, "2026-01-15"),  # over 22.5 won with 25
            ("202", "assists"): (0.3, 1, "2026-01-15"),  # under 6.5 lost with 7: the over hit
            ("202", "points"): (0.6, 0, "2026-01-15"),  # over 20.5 lost with 18
            ("201", "assists"): (0.625, 1, "2026-01-15"),  # over 4 won with 5; p is conditional on no push
        })  # the over 10 rebounds leg pushed and is not a sample

        status = self.learning.retrain()
        walk_forward = len(walk_forward_samples(nba_game_logs(self.store)))
        self.assertFalse(status["active"])
        self.assertEqual((status["samples"], status["ledger_samples"]), (walk_forward + 4, 4))
        self.assertEqual(status["players"], 3)
        self.assertIsNone(status["holdout"])
        self.assertTrue(status["message"].startswith(f"Needs at least 800 graded predictions (has {walk_forward + 4:,})"))
        self.assertIsNotNone(status["intercept"])
        self.assertIsNone(self.learning.calibrate_over(0.6))
        self.assertEqual(Learning(self.store).calibration_status(), status)

    def test_retrain_without_data(self):
        status = Learning(Store(Path(self.directory.name) / "empty"), clock=self.clock).retrain()
        self.assertEqual((status["active"], status["samples"], status["players"], status["curve"]), (False, 0, 0, []))
        self.assertTrue(status["message"].startswith("Needs at least 800 graded predictions (has 0)"))

    def test_calibration_from_an_older_model_is_not_applied(self):
        old = {"active": True, "trained_at": "2026-01-01T00:00:00Z", "samples": 5000, "message": "Learned on the old model."}
        with self.store.connect() as db:
            db.execute("""INSERT INTO learning_calibration (id, trained_at, active, intercept, slope, status)
                VALUES (1, '2026-01-01T00:00:00Z', 1, 0.5, 0.8, ?)""", (json.dumps(old),))
        learning = Learning(self.store, clock=self.clock)
        self.assertIsNone(learning.calibrate_over(0.6))
        self.assertIsNone(learning.calibrate_over(0.6, "points"))
        self.assertIsNone(learning.calibration_status()["trained_at"])
        self.assertIn("Rebuilding the calibration", learning.calibration_status()["message"])
        status = learning.retrain()
        self.assertEqual(status["version"], 2)
        self.assertEqual(Learning(self.store, clock=self.clock).calibration_status(), status)

    def test_background_retrain(self):
        worker = self.learning.retrain_in_background()
        self.assertIsNotNone(worker)
        worker.join(30)
        self.assertFalse(worker.is_alive())
        self.assertFalse(self.learning.training)
        self.assertEqual(self.learning.calibration_status()["trained_at"], "2026-01-16T18:00:00Z")
        self.assertGreater(self.learning.calibration_status()["samples"], 0)


if __name__ == "__main__":
    unittest.main()
