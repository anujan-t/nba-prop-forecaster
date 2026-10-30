"""COURTSIDE Coach beyond the slip: league questions (injuries, news, moves, coaches, styles, rankings, standings, games,
lines, scores, bios), app questions, betting math, scope, and the newsroom service and scoreboard parser behind them."""
import json
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest import mock

from backend import news
from backend.assistant import Coach, describe_leg, slip_key
from backend.knowledge import KNOWLEDGE
from backend.newsroom import Newsroom
from backend.teams import GameModel
from backend.topics import find_day, find_metric, find_teams, topics
from test_assistant import PLAYERS, CoachTestCase, FakeLearning, fake_llm
from test_teams import FakeFetch

TODAY = date(2026, 10, 28)


class MemoryStore:
    def __init__(self):
        self.saved: dict[str, dict] = {}

    def resource(self, key):
        return self.saved.get(key)

    def save_resource(self, key, payload):
        result = {**payload, "fetched_at": datetime.now(timezone.utc).isoformat()}
        self.saved[key] = result
        return result


def side(team, name, record, score=None, winner=None):
    return {"team": team, "name": name, "record": record, "score": score, "winner": winner}


ODDS = {"provider": "DraftKings", "details": "GSW -2.5", "home_spread": -2.5, "away_spread": 2.5, "total": 231.5, "favorite": "GSW",
        "home_moneyline": -135, "away_moneyline": 115, "home_spread_odds": -110, "away_spread_odds": -110, "over_odds": -110, "under_odds": -110}
BOARDS = {
    TODAY: [{"id": "1", "start": "2026-10-28T23:30Z", "state": "pre", "detail": "10/28 - 7:30 PM EDT",
             "home": side("GSW", "Golden State Warriors", "3-1"), "away": side("DEN", "Denver Nuggets", "2-2"), "odds": ODDS,
             "broadcast": "ESPN", "venue": "Chase Center", "note": "NBA Cup - Group Play"}],
    TODAY - timedelta(days=1): [{"id": "0", "start": "2026-10-27T23:00Z", "state": "post", "detail": "Final",
                                 "home": side("DEN", "Denver Nuggets", "2-2", 120, True), "away": side("BOS", "Boston Celtics", "3-1", 110, False),
                                 "odds": None, "broadcast": None, "venue": None, "note": None}],
}
INJURIES = {"teams": [
    {"team": "DEN", "team_name": "Denver Nuggets", "injuries": [
        {"player": "Jamal Murray", "player_id": "1627750", "position": "G", "team": "DEN", "status": "Out", "status_key": "out",
         "date": "2026-10-27T18:00Z", "injury": "Hamstring", "detail": "Strain", "side": "Left", "return_date": "2026-11-05",
         "short_comment": "Murray will miss at least a week.", "long_comment": None, "headshot": None}]},
    {"team": "GSW", "team_name": "Golden State Warriors", "injuries": [
        {"player": "Draymond Green", "player_id": "203110", "position": "F", "team": "GSW", "status": "Questionable",
         "status_key": "questionable", "date": "2026-10-27T19:00Z", "injury": "Ankle", "detail": None, "side": "Not Specified",
         "return_date": None, "short_comment": None, "long_comment": None, "headshot": None}]},
], "counts": {"out": 1, "questionable": 1}, "total": 2, "provider": "fixture"}
HEADLINES = {"articles": [
    {"id": "a1", "type": "Story", "kind": "injury", "headline": "Murray out at least a week with hamstring strain", "description": None,
     "published": "2026-10-27T20:00Z", "byline": None, "url": "https://example.com/a1", "image": None, "teams": ["DEN"],
     "athletes": [{"name": "Jamal Murray", "player_id": "1627750"}], "premium": False, "video": False},
    {"id": "a2", "type": "Story", "kind": "transaction", "headline": "Warriors sign guard to two-way deal", "description": None,
     "published": "2026-10-27T15:00Z", "byline": None, "url": None, "image": None, "teams": ["GSW"], "athletes": [], "premium": False,
     "video": False},
], "counts": {}, "provider": "fixture"}
MOVES = {"transactions": [
    {"date": "2026-10-27T00:00Z", "description": "Waived G John Doe.", "team": "GSW", "team_name": "Golden State Warriors"},
    {"date": "2026-10-26T00:00Z", "description": "Signed F Jane Roe to a two-way contract.", "team": "DEN", "team_name": "Denver Nuggets"},
], "provider": "fixture"}


def metric(label, value, rank, percent=False):
    return {"label": label, "value": value, "rank": rank, "percent": percent}


def profile(team, name, wins, losses, pace_rank, defense_rank):
    return {"team_id": "1", "team": team, "name": name, "games": wins + losses, "wins": wins, "losses": losses,
            "metrics": {"pace": metric("Pace (possessions per 48)", 104 - pace_rank, pace_rank),
                        "offense": metric("Offensive rating", 115.0, 2), "defense": metric("Defensive rating", 108 + defense_rank, defense_rank),
                        "three_rate": metric("Share of shots from three", 0.42, 1, True),
                        "opponent_threes": metric("Opponent threes made per game", 12 + defense_rank, defense_rank)},
            "play_types": [{"play_type": "Spotup", "label": "Spot-up", "frequency": 0.24, "frequency_rank": 1, "points_per_possession": 1.1,
                            "percentile": 0.7}], "tags": ["Plays fast"] if pace_rank == 1 else []}


STYLES = {"season": "2026-27", "team_count": 3, "provider": "fixture", "teams": [
    profile("BOS", "Boston Celtics", 3, 1, 3, 1), profile("DEN", "Denver Nuggets", 2, 2, 2, 3), profile("GSW", "Golden State Warriors", 3, 1, 1, 2)]}
COACHES = {"season": "2026-27", "provider": "fixture", "teams": [
    {"team": "DEN", "coaches": [{"name": "David Adelman", "role": "Head Coach", "head": True},
                                {"name": "Ryan Saunders", "role": "Assistant Coach", "head": False}], "error": None},
    {"team": "GSW", "coaches": [{"name": "Steve Kerr", "role": "Head Coach", "head": True}], "error": None}]}


def scoreboard(day):
    return {"date": day.isoformat(), "games": json.loads(json.dumps(BOARDS.get(day, []))), "provider": "fixture"}


class FeedPatch:
    """Patches every provider call the newsroom makes with the fixtures above."""

    def __enter__(self):
        self.patches = [mock.patch.object(news, name, side_effect=effect) for name, effect in (
            ("fetch_injuries", lambda roster=None: json.loads(json.dumps(INJURIES))),
            ("fetch_headlines", lambda roster=None, limit=60, team=None: {
                **HEADLINES, "articles": [a for a in HEADLINES["articles"] if team is None or team in a["teams"]]}),
            ("fetch_transactions", lambda limit=60: json.loads(json.dumps(MOVES))),
            ("fetch_team_styles", lambda season: json.loads(json.dumps(STYLES))),
            ("fetch_scoreboard", scoreboard),
        )]
        self.patches.append(mock.patch("backend.newsroom.Newsroom.coaches", lambda self, refresh=False: COACHES))
        for patch in self.patches:
            patch.start()
        return self

    def __exit__(self, *exc):
        for patch in self.patches:
            patch.stop()


class LeagueCoachTestCase(CoachTestCase):
    def setUp(self):
        super().setUp()
        self.feeds = FeedPatch().__enter__()
        self.addCleanup(self.feeds.__exit__)
        self.newsroom = Newsroom(MemoryStore(), lambda: PLAYERS, today=lambda: TODAY, season=lambda: "2026-27")
        bios = {"1641705": {"position": "Center", "height": "7-4", "weight": "235", "age": 22, "experience": 3, "draft_year": "2023",
                            "draft_round": "1", "draft_pick": "1", "school": "Metropolitans 92", "country": "France"}}
        self.coach = Coach(lambda: PLAYERS, self.history, self.load, learning=self.learning, today=lambda: TODAY,
                           newsroom=self.newsroom, profile=lambda player_id: bios[player_id])

    def cards(self, response):
        return [card["type"] for card in response["cards"]]


class InjuryAndNewsTests(LeagueCoachTestCase):
    def test_team_injuries_and_player_availability(self):
        team = self.say("who's injured on the nuggets?")
        self.assertIn("injuries", self.cards(team))
        self.assertIn("Jamal Murray", team["reply"])
        self.assertIn("Left hamstring strain", team["reply"])
        out = self.say("is Jamal Murray playing tonight?")
        self.assertIn("**Jamal Murray** (DEN) — Out: Left hamstring strain; estimated return Nov 5", out["reply"])
        self.assertIn("Next game: tonight at the Warriors (7:30 PM ET)", out["reply"])
        healthy = self.say("is Jokić playing tonight?")
        self.assertIn("isn't on the injury report", healthy["reply"])
        self.assertIn("Teammates listed: Jamal Murray (Out)", healthy["reply"])
        self.assertEqual(self.slip["legs"], [])

    def test_pronouns_follow_the_last_player(self):
        self.say("what's Jokić averaging?")
        follow = self.say("is he hurt?")
        self.assertIn("Nikola Jokić", follow["reply"])

    def test_tonights_injuries_and_league_report(self):
        tonight = self.say("who's out tonight?")
        self.assertIn("DEN at GSW", tonight["reply"])
        self.assertIn("Draymond Green (Questionable)", tonight["reply"])
        league = self.say("injury report")
        self.assertIn("2 players", league["reply"])

    def test_headlines_moves_and_player_news(self):
        latest = self.say("latest nba news")
        self.assertEqual(self.cards(latest), ["news"])
        self.assertIn("Murray out at least a week", latest["reply"])
        team = self.say("warriors news")
        self.assertIn("Warriors sign guard", team["reply"])
        self.assertNotIn("Murray out", team["reply"])
        moves = self.say("any trades?")
        self.assertIn("moves", self.cards(moves))
        cut = self.say("did the warriors cut anyone?")
        self.assertIn("Waived G John Doe", cut["reply"])
        self.assertNotIn("slip", cut["reply"].lower())
        player = self.say("what's going on with Jamal Murray?")
        self.assertIn("Murray out at least a week", player["reply"])

    def test_ambiguous_first_name_asks_which_player(self):
        response = self.say("tyrese injury")
        self.assertEqual(len(response["clarification"]["options"]), 2)
        chosen = self.say(response["clarification"]["options"][0]["message"])
        self.assertIn("injury report", chosen["reply"])


class TeamAndScheduleTests(LeagueCoachTestCase):
    def test_coaches_styles_rankings_and_standings(self):
        self.assertIn("David Adelman", self.say("who coaches the nuggets?")["reply"])
        style = self.say("how do the warriors play?")
        self.assertIn("team", self.cards(style))
        self.assertIn("Plays fast", style["reply"])
        fastest = self.say("which team plays the fastest?")
        self.assertEqual(fastest["cards"][0]["rows"][0]["team"], "GSW")
        threes = self.say("who gives up the most threes?")
        self.assertEqual(threes["cards"][0]["rows"][0]["team"], "DEN")
        defense = self.say("how good is the celtics defense?")
        self.assertIn("1st of 3", defense["reply"])
        standings = self.say("standings")
        self.assertEqual(self.cards(standings), ["standings"])
        self.assertIn("are **3–1**", self.say("what's the celtics record?")["reply"])

    def test_games_lines_and_scores(self):
        tonight = self.say("what games are on tonight?")
        self.assertIn("DEN at GSW", tonight["reply"])
        self.assertIn("GSW −2.5, total 231.5", tonight["reply"])
        spread = self.say("what's the spread for the nuggets game?")
        self.assertIn("the underdog", spread["reply"])
        self.assertIn("spread **+2.5**", spread["reply"])
        self.assertIn("moneyline **+115**", spread["reply"])
        result = self.say("did the nuggets win last night?")
        self.assertIn("won 120–110 vs the Celtics", result["reply"])
        nxt = self.say("when do the warriors play next?")
        self.assertIn("tonight vs the Nuggets (7:30 PM ET)", nxt["reply"])
        self.assertIn("NBA Cup", nxt["reply"])
        none = self.say("games on saturday")
        self.assertIn("next week", none["reply"])

    def test_matchup_and_team_overview(self):
        matchup = self.say("nuggets vs warriors")
        self.assertIn("Nuggets", matchup["reply"])
        overview = self.say("celtics")
        self.assertIn("**Boston Celtics** (3–1", overview["reply"])

    def test_bio_and_roster(self):
        bio = self.say("how old is wemby")
        self.assertIn("22 years old", bio["reply"])
        self.assertIn("Metropolitans 92", bio["reply"])
        roster = self.say("nuggets roster")
        self.assertIn("Jamal Murray (Out)", roster["reply"])


class AppAndBettingQuestionTests(LeagueCoachTestCase):
    def test_site_help_and_glossary(self):
        simulator = self.say("how does the simulator work?")
        self.assertIn("250,000 times", simulator["reply"])
        self.assertEqual(simulator["cards"][0], {"type": "link", "label": "Open How it works", "route": "#/how-it-works/method"})
        self.assertIn("all 22 markets", self.say("how accurate is the model?")["reply"])
        self.assertIn("52.4%", self.say("what's the vig?")["reply"])
        self.assertIn("points + rebounds + assists", self.say("what's PRA?")["reply"])
        self.assertIn("Settings › Model", self.say("how many simulations do you run?")["reply"])

    def test_odds_math(self):
        self.assertIn("**40.0%**", self.say("what does +150 mean?")["reply"])
        payout = self.say("how much do I win on a $20 bet at +350?")
        self.assertIn("$70.00 profit, $90.00 back", payout["reply"])
        self.assertIn("fractional 20/27", self.say("convert -135 to decimal")["reply"])
        self.assertIn("**+150** in American odds", self.say("what's 2.5 in american odds?")["reply"])
        self.assertEqual(self.slip["book_odds"], None)

    def test_scope_and_advice(self):
        self.assertIn("only cover the NBA", self.say("who won the super bowl?")["reply"])
        self.assertIn("didn't catch", self.say("what's the weather like?")["reply"])
        self.assertIn("Anytime", self.say("ok thanks coach!")["reply"])
        advice = self.say("who should I bet on tonight?")
        self.assertIn("don't hand out locks", advice["reply"])
        self.assertIn("games", self.cards(advice))

    def test_props_and_slip_commands_still_work_with_context(self):
        prop = self.say("Jokić over 26.5 points?")
        self.assertEqual(self.cards(prop), ["forecast"])
        self.assertIn("Next game: tonight at the Warriors", prop["reply"])
        added = self.say("add Jamal Murray over 20.5 points")
        self.assertEqual(self.legs(), [("Jamal Murray", "points", "over", 20.5)])
        self.assertIn("listed **Out**", added["reply"])
        self.say("remove leg 1")
        self.assertEqual(self.slip["legs"], [])
        self.assertIn("Your slip is empty", self.say("what are my odds?")["reply"])

    def test_league_questions_after_a_player_are_not_follow_ups(self):
        for message in ("who leads the nba in assists?", "give me your picks for rebounds", "who do you like for the over tonight?"):
            with self.subTest(message=message):
                self.say("how about curry")
                reply = self.say(message)
                self.assertNotIn("forecast", self.cards(reply))
                self.assertNotIn("Stephen Curry", reply["reply"])
                self.assertEqual(self.slip["legs"], [])


class GameMarketCoachTests(LeagueCoachTestCase):
    """Moneylines and spreads by chat: tonight's fixture is DEN at GSW, GSW −2.5 / ML −135, DEN +2.5 / ML +115."""

    def setUp(self):
        super().setUp()
        self.coach.games = self.coach.desk.games = GameModel(MemoryStore(), today=lambda: TODAY, schedule=self.newsroom.scoreboard,
                                                             fetch=FakeFetch())

    def game_legs(self):
        return [(leg["request"]["team"], leg["request"]["market"], leg["request"]["line"]) for leg in self.slip["legs"]
                if leg["request"].get("kind") == "game"]

    def test_a_moneyline_from_a_plain_sentence(self):
        added = self.say("i want to bet on the nuggets moneyline too")
        self.assertEqual(self.game_legs(), [("DEN", "moneyline", None)])
        self.assertEqual(self.slip["legs"][0]["label"], {"player": "Denver Nuggets", "team": "DEN"})
        self.assertIn("**Denver Nuggets moneyline** tonight at the Warriors", added["reply"])
        self.assertIn("by the team model", added["reply"])
        self.assertIn("the book's +115 needs 46.5%", added["reply"])
        self.assertIn("Jamal Murray", added["reply"])
        self.assertIn("parlay", self.cards(added))
        self.assertEqual(added["actions"][0]["type"], "add_leg")

    def test_a_spread_and_a_prop_in_one_message(self):
        response = self.say("add warriors -2.5 and jokic over 26.5 points")
        self.assertEqual(self.game_legs(), [("GSW", "spread", -2.5)])
        self.assertEqual(self.legs(), [("Golden State Warriors", None, None, -2.5), ("Nikola Jokić", "points", "over", 26.5)])
        parlay = next(card for card in response["cards"] if card["type"] == "parlay")
        self.assertEqual(parlay["result"]["pairs"][0]["basis"], "team_player")

    def legs(self):
        return [(leg["label"]["player"], leg["request"].get("stat"), leg["request"].get("side"), leg["request"]["line"]) for leg in self.slip["legs"]]

    def test_flip_change_and_remove_game_legs(self):
        self.say("add nuggets +2.5")
        self.assertEqual(self.game_legs(), [("DEN", "spread", 2.5)])
        self.assertIn("Golden State Warriors −2.5", self.say("flip leg 1")["reply"])
        self.assertEqual(self.game_legs(), [("GSW", "spread", -2.5)])
        self.say("change leg 1 to 4.5")
        self.assertEqual(self.game_legs(), [("GSW", "spread", -4.5)])
        self.say("add jokic over 26.5 points")
        removed = self.say("remove the warriors")
        self.assertEqual(removed["actions"][0]["description"], "Removed leg 1: Golden State Warriors −4.5")
        self.assertEqual(self.game_legs(), [])
        self.assertIn("no Warriors leg", self.say("remove the warriors")["reply"])

    def test_one_moneyline_per_game_and_contradictions_are_explained(self):
        self.say("add nuggets moneyline")
        replaced = self.say("add warriors moneyline")
        self.assertIn("Replaced leg 1", replaced["reply"])
        self.assertEqual(self.game_legs(), [("GSW", "moneyline", None)])
        middle = self.say("add nuggets +1.5")
        self.assertIn("All 2 legs hit", middle["reply"])
        parlay = next(card for card in middle["cards"] if card["type"] == "parlay")["result"]
        self.assertLess(parlay["probability"], parlay["independent_probability"] / 3)
        self.say("remove leg 2")
        conflict = self.say("add nuggets +0.5")
        self.assertIn("can't both win", conflict["reply"])

    def test_questions_leave_the_slip_alone(self):
        spread = self.say("is nuggets +2.5 a good bet?")
        self.assertIsNone(spread["slip"])
        self.assertIn("**Denver Nuggets +2.5** tonight at the Warriors (7:30 PM ET): covers", spread["reply"])
        self.assertIn("add Nuggets +2.5", spread["suggestions"])
        who = self.say("who wins nuggets vs warriors?")
        self.assertIsNone(who["slip"])
        self.assertIn("COURTSIDE's team model: **Nuggets", who["reply"])
        self.assertIsNotNone(who["cards"][0]["games"][0]["model"])
        lines = self.say("what's the spread for the warriors game?")
        self.assertIn("team model", lines["reply"])
        self.assertNotIn("prices player props, not sides", lines["reply"])

    def test_a_team_without_a_market_asks_which(self):
        asked = self.say("add the nuggets")
        options = asked["clarification"]["options"]
        self.assertEqual([option["label"] for option in options], ["Moneyline", "Spread"])
        self.say(options[1]["message"])
        self.assertEqual(self.game_legs(), [("DEN", "spread", 2.5)])

    def test_tracking_a_game_leg(self):
        self.say("add nuggets moneyline")
        self.say("track this for $10")
        self.assertEqual(self.learning.tracked[0]["legs"][0]["request"]["kind"], "game")

    def test_descriptions_and_slip_keys_cover_game_legs(self):
        self.say("add warriors -2.5 and jokic over 26.5 points")
        self.assertEqual([describe_leg(leg) for leg in self.slip["legs"]],
                         ["Golden State Warriors −2.5", "Nikola Jokić Over 26.5 points"])
        legs, odds = slip_key(self.slip)
        self.assertEqual(legs[0], ("game", "1", "spread", "GSW", -2.5))
        self.assertIsNone(odds)


class LeagueToolTests(LeagueCoachTestCase):
    def test_league_tools_return_feed_data_and_cards(self):
        seen = []
        self.coach.llm = fake_llm([
            {"content": None, "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "injury_report", "arguments": json.dumps({"team": "Nuggets"})}},
                {"id": "2", "type": "function", "function": {"name": "games", "arguments": json.dumps({"date": "today"})}},
                {"id": "3", "type": "function", "function": {"name": "site_help", "arguments": json.dumps({"question": "what is the vig"})}}]},
            {"content": "Murray is out; Denver plays at Golden State tonight."},
        ], seen)
        response = self.say("anything I should know about Denver tonight?")
        self.assertEqual(response["engine"]["mode"], "llm")
        outputs = [json.loads(message["content"]) for message in seen[1]["messages"] if message["role"] == "tool"]
        self.assertEqual(outputs[0]["entries"][0]["player"], "Jamal Murray")
        self.assertEqual(outputs[1]["games"][0]["home"]["team"], "GSW")
        self.assertEqual(outputs[2]["entries"][0]["title"], "Vig (juice)")
        self.assertEqual(self.cards(response), ["injuries", "games"])
        names = {tool["function"]["name"] for tool in seen[0]["tools"]}
        self.assertTrue({"nba_news", "injury_report", "games", "team_profile", "standings", "site_help", "odds_math"} <= names)
        self.assertIn("never answer current NBA facts from memory", seen[0]["messages"][0]["content"])

    def test_bad_team_is_reported_to_the_model(self):
        seen = []
        self.coach.llm = fake_llm([
            {"content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "team_profile", "arguments": json.dumps({"team": "Seattle"})}}]},
            {"content": "I couldn't find that team."},
        ], seen)
        self.say("tell me about seattle")
        self.assertIn("No NBA team matches", seen[1]["messages"][-1]["content"])

    def test_slip_commands_skip_the_model(self):
        self.coach.llm = fake_llm([], [])
        self.say("add Nikola Jokić over 26.5 points")
        self.assertEqual(self.legs(), [("Nikola Jokić", "points", "over", 26.5)])


class NewsroomTests(unittest.TestCase):
    def test_stale_copy_is_served_when_the_provider_fails(self):
        store = MemoryStore()
        room = Newsroom(store, lambda: PLAYERS, today=lambda: TODAY, season=lambda: "2026-27")
        with mock.patch.object(news, "fetch_transactions", return_value=MOVES):
            self.assertFalse(room.transactions()["stale"])
        store.saved["news:transactions"]["fetched_at"] = "2020-01-01T00:00:00+00:00"
        with mock.patch.object(news, "fetch_transactions", side_effect=news.NewsUnavailableError("ESPN is down.")):
            stale = room.transactions()
            self.assertTrue(stale["stale"])
            self.assertIn("ESPN is down", stale["error"])
            with self.assertRaises(news.NewsUnavailableError):
                Newsroom(MemoryStore(), lambda: PLAYERS).transactions()

    def test_next_and_last_game_scan_the_schedule(self):
        room = Newsroom(MemoryStore(), lambda: PLAYERS, today=lambda: TODAY, season=lambda: "2026-27")
        with mock.patch.object(news, "fetch_scoreboard", side_effect=scoreboard):
            self.assertEqual(room.next_game("DEN")["id"], "1")
            self.assertEqual(room.last_game("DEN")["id"], "0")
            self.assertIsNone(room.next_game("BOS"))
            self.assertEqual(room.scoreboard(TODAY)["games"][0]["tipoff"], "7:30 PM ET")

    def test_scoreboard_parser_maps_teams_lines_and_scores(self):
        raw = {"events": [
            {"id": "9", "date": "2026-10-29T00:00Z", "status": {"type": {"state": "pre", "shortDetail": "10/28 - 8:00 PM EDT"}},
             "competitions": [{"competitors": [
                 {"homeAway": "home", "team": {"id": "9", "abbreviation": "GS", "displayName": "Golden State Warriors"},
                  "records": [{"type": "total", "summary": "3-1"}], "score": "0"},
                 {"homeAway": "away", "team": {"id": "7", "abbreviation": "DEN", "displayName": "Denver Nuggets"},
                  "records": [{"type": "total", "summary": "2-2"}], "score": "0"}],
                 "odds": [{"provider": {"name": "DraftKings"}, "details": "DEN -1.5", "spread": 1.5, "overUnder": 229.5,
                           "awayTeamOdds": {"favorite": True}, "homeTeamOdds": {"favorite": False},
                           "pointSpread": {"home": {"close": {"line": "+1.5", "odds": "-108"}}, "away": {"close": {"line": "-1.5", "odds": "-112"}}},
                           "total": {"over": {"close": {"line": "o229.5", "odds": "-110"}}, "under": {"open": {"line": "u230.5", "odds": "-110"}}},
                           "moneyline": {"home": {"close": {"odds": "+105"}}, "away": {"close": {"odds": "-125"}}}}],
                 "broadcasts": [{"names": ["ESPN"]}], "notes": [{"headline": "NBA Cup - Group Play"}]}]},
            {"id": "10", "date": "2026-10-28T23:00Z", "status": {"type": {"state": "post", "shortDetail": "Final/OT"}},
             "competitions": [{"competitors": [
                 {"homeAway": "home", "team": {"id": "27", "abbreviation": "WSH"}, "score": "101", "winner": False},
                 {"homeAway": "away", "team": {"id": "3", "abbreviation": "NO"}, "score": "104", "winner": True}]}]},
        ]}
        with mock.patch.object(news, "_get", return_value=raw):
            board = news.fetch_scoreboard(TODAY)
        final, upcoming = board["games"]
        self.assertEqual((final["away"]["team"], final["away"]["score"], final["away"]["winner"]), ("NOP", 104, True))
        self.assertEqual(final["home"]["team"], "WAS")
        self.assertIsNone(final["odds"])
        self.assertEqual((upcoming["home"]["team"], upcoming["home"]["score"]), ("GSW", None))
        odds = upcoming["odds"]
        self.assertEqual((odds["home_spread"], odds["away_spread"], odds["total"], odds["favorite"]), (1.5, -1.5, 229.5, "DEN"))
        self.assertEqual((odds["home_moneyline"], odds["away_moneyline"], odds["over_odds"]), (105, -125, -110))
        self.assertEqual((upcoming["broadcast"], upcoming["note"]), ("ESPN", "NBA Cup - Group Play"))


class UnderstandingTests(unittest.TestCase):
    def test_team_mentions(self):
        codes = lambda text: [mention.code for mention in find_teams(text)]  # noqa: E731
        self.assertEqual(codes("how do the Heat play"), ["MIA"])
        self.assertEqual(codes("track this for 10 bucks"), [])
        self.assertEqual(codes("MIN at WAS tonight"), ["MIN", "WAS"])
        self.assertEqual(codes("was he good, min 30?"), [])
        self.assertEqual(codes("nets and t-wolves"), ["BKN", "MIN"])
        self.assertEqual(codes("Sixers moneyline"), ["PHI"])
        self.assertEqual(codes("golden state's next game"), ["GSW"])
        self.assertEqual(codes("IS LEBRON OUT TONIGHT WAS HE HURT"), [])
        self.assertEqual(codes("okc vs lal"), ["OKC", "LAL"])

    def test_days_metrics_and_topics(self):
        self.assertEqual(find_day("games on friday", TODAY), date(2026, 10, 30))
        self.assertEqual(find_day("scores from last night", TODAY), TODAY - timedelta(days=1))
        self.assertEqual(find_day("what happened last tuesday", TODAY), date(2026, 10, 27))
        self.assertEqual(find_metric("who gives up the most threes"), "opponent_threes")
        self.assertEqual(find_metric("which team plays the fastest"), "pace")
        self.assertEqual(topics("who's injured on the lakers")["injury"], "strong")
        self.assertEqual(topics("I lost my bet").get("scores"), "weak")
        self.assertNotIn("injury", topics("what's my return"))

    def test_knowledge_search(self):
        expected = {"how many simulations do you run": "simulation_count", "what's the vig": "vig", "is my data private": "privacy",
                    "how do I build a parlay": "parlay", "what's the difference between poisson and normal": "models",
                    "can you see live odds": "data", "what is closing line value": "clv", "where are my bets": "bets"}
        for question, entry in expected.items():
            self.assertEqual(KNOWLEDGE.best(question).id, entry, question)
        self.assertIsNone(KNOWLEDGE.best("tell me a joke"))
        self.assertIsNone(KNOWLEDGE.best("what is the weather today"))


if __name__ == "__main__":
    unittest.main()
