"""COURTSIDE Coach: a chat layer over the forecast, parlay, learning and newsroom engines.

Slip commands, props, league questions (news, injuries, games and lines, teams, coaches, standings), betting math and
questions about the app are handled by a deterministic parser wired to the same code as the UI. An optional
OpenAI-compatible model answers everything except slip commands, but it can only obtain facts by calling the tools
defined here, so every probability, price, headline and injury it reports comes from the app's own models and feeds.
"""
from __future__ import annotations

from collections.abc import Callable
import copy
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
import math
import re
from typing import Any
import uuid
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import nlu, topics
from . import news
from .desk import METRIC_KEYS, Desk, injury_what
from .engine import american_probability, forecast, profit_per_unit
from .knowledge import KNOWLEDGE
from .llm import LLMClient, LLMError
from .nba import NBAUnavailableError
from .news import NewsUnavailableError
from .parlay import MAX_LEGS, LegNotFound, fair_odds, price_parlay
from .stats import stat_values, with_stat

DEFAULTS = {"model": "auto", "lookback": 60, "expected_minutes": None, "rest_days": None, "opponent": None, "use_context": True}
STANDARD_LEG_ODDS = -110
EASTERN = ZoneInfo("America/New_York")
SUPPORT = ("If betting stops feeling like research, the National Problem Gambling Helpline is 1-800-GAMBLER (1-800-426-2537), "
           "free and confidential, 24/7.")
DISTRESS = re.compile(r"\b(chas(?:e|ing) (?:my )?loss(?:es)?|lost (?:everything|my rent|all my)|can'?t stop (?:betting|gambling)|"
                      r"addict\w*|gambling problem|in debt|borrow(?:ed)? money to bet)\b")
LLM_FALLBACK_INTENTS = {"forecast", "house", "explain", "parlay", "summary", "hit_rate"}
REMOVAL = re.compile(r"\b(?:remove|delete|drop|scratch|take (?:off|out)|get rid of|cut)\b")
CHANGE = re.compile(r"\b(?:flip|switch|swap|change|move|bump|raise|lower)\b|\bmake (?:it|that|leg|the)\b|\bset (?:leg|the line|it|that)\b")
# "How does X work?", "what's the vig?", "where do I see my bets?": questions about the app or betting terms.
HOWTO = re.compile(r"^(?:so |ok |okay |hey |and |but |quick question,? )?(?:how (?:do|does|can|should|would|is|are|accurate|many|much|often does) |"
                   r"how to |where (?:do|can|is|are|does) |what (?:is|are|does|do|was) |what'?s |whats |explain|tell me (?:about|how) |"
                   r"define |difference between|why (?:does|do|is|are) |can you explain|is there (?:a|an) |does (?:the|this|courtside|it) |"
                   r"do you |can i |can you )")
SLIP_QUESTION = re.compile(r"\b(?:my|our|the|this|these) (?:slip|parlay|legs?|ticket|bet|odds|chances?)\b|\ball (?:\d+|the|my) legs\b|"
                           r"\b(?:does|will|can|would) (?:it|this|they|that|these) (?:hit|cash|win)\b|\bwhat are (?:my|the) odds\b")
THANKS = re.compile(r"(?:(?:ok(?:ay)?|cool|nice|great|awesome|perfect|got it|sounds good|appreciate it|cheers|thx|ty|thanks?(?: you)?"
                    r"(?: so much| a lot)?)[\s!.,]*)+(?:coach)?[\s!.]*")
# Markets a bare line ("over 23.5?") can mean, and intents that never continue a conversation about one player's prop.
INFERABLE_STATS = ("points", "rebounds", "assists", "threes", "pts_reb_ast", "pts_reb", "pts_ast", "reb_ast", "steals", "blocks")
STANDALONE_INTENTS = {"teach", "forget", "undo", "clear", "help", "greet", "track", "settle", "learning", "remove", "update", "parlay"}
WINDOW = re.compile(r"\b(?:last|past|previous)\s+\d{1,2}\s+games?\b")


def strip_vocative(text: str) -> str:
    """ "Coach, who's out tonight?" -> "who's out tonight?" (the bot's name isn't a question about coaches)."""
    return re.sub(r"^\s*(?:(?:hey|hi|yo|ok|okay)\s+)?coach\b[,:!]?\s*", "", text, flags=re.I) or text


def eastern_today() -> date:
    return datetime.now(timezone.utc).astimezone(EASTERN).date()


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    percent = value * 100
    if 0 < percent < 1:
        return f"{float(f'{percent:.2g}'):g}%"
    return f"{percent:.1f}%"


def odds_text(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"+{value}" if value > 0 else f"−{abs(value)}"


def decimal_of(odds: int) -> float:
    return 1 + profit_per_unit(odds)


def american_of(decimal: float) -> int:
    return int(round((decimal - 1) * 100)) if decimal >= 2 else int(round(-100 / (decimal - 1)))


def cents(value: float) -> str:
    amount = abs(value)
    return f"${amount:.2f}" if amount >= 1 else f"{amount * 100:.1f}¢"


def slip_key(slip: dict | None) -> tuple:
    """What a slip bets on, ignoring display labels and float formatting."""
    if not slip:
        return ((), None)
    legs = tuple((leg["request"]["player_id"], leg["request"]["stat"], leg["request"]["side"], float(leg["request"]["line"]))
                 for leg in slip.get("legs", []))
    return legs, slip.get("book_odds")


def verdict(edge: float) -> str:
    return "value" if edge > 0.01 else "house" if edge < -0.01 else "fair"


def parlay_house(probability: float, legs: int, book_odds: int | None, calibrated: float | None = None) -> dict:
    """The offered parlay price (or the standard −110-per-leg price) against the model's chance that every leg wins."""
    basis = "entered" if book_odds is not None else "standard"
    if book_odds is None:
        decimal = decimal_of(STANDARD_LEG_ODDS) ** legs
        odds = american_of(decimal)
    else:
        odds, decimal = book_odds, decimal_of(book_odds)
    implied = 1 / decimal
    roi = probability * (decimal - 1) - (1 - probability)
    edge = probability - implied
    if basis == "entered":
        note = f"At {odds_text(odds)} the book needs {pct(implied)} to break even; the model gives {pct(probability)}. "
    else:
        note = (f"No book odds entered. If each of the {legs} leg{'s' if legs != 1 else ''} were −110 and priced as independent, "
                f"the parlay would pay {odds_text(odds)} ({pct(implied)} to break even); the model gives {pct(probability)}. "
                "Books price same-game parlays differently, so enter your actual odds for a real comparison. ")
    if verdict(edge) == "value":
        note += f"If the model is right, that price returns about {cents(roi)} per $1 staked on average."
    elif verdict(edge) == "house":
        note += f"If the model is right, the house keeps about {cents(roi)} of every $1 staked."
    else:
        note += "That is close to the model's fair price."
    if calibrated is not None:
        side = "above" if calibrated > implied + 0.005 else "below" if calibrated < implied - 0.005 else "right at"
        note += f" Calibrated by past results the chance is {pct(calibrated)}, {side} break-even."
    return {"basis": basis, "odds": odds, "decimal": decimal, "implied": implied, "model": probability,
            "fair_odds": fair_odds(probability), "edge": edge, "expected_roi": roi, "house_edge": -roi, "vig": None,
            "verdict": verdict(edge), "note": note}


def leg_house(win: float, loss: float, odds: int) -> dict:
    """A single-prop price; a push refunds the stake, so the comparison is conditional on the bet settling."""
    decimal, implied = decimal_of(odds), american_probability(odds)
    settled = win + loss
    model = win / settled if settled > 1e-12 else 0.0
    roi = win * profit_per_unit(odds) - loss
    edge = model - implied
    note = f"At {odds_text(odds)} you need {pct(implied)} to break even; the model gives {pct(model)}"
    note += " (excluding pushes). " if settled < 0.9999 else ". "
    note += {"value": f"If the model is right, that is about +{cents(roi)} per $1 staked.",
             "house": f"If the model is right, the house keeps about {cents(roi)} of every $1 staked.",
             "fair": "That is close to the model's fair price."}[verdict(edge)]
    return {"basis": "entered", "odds": odds, "decimal": decimal, "implied": implied, "model": model, "fair_odds": fair_odds(model),
            "edge": edge, "expected_roi": roi, "house_edge": -roi, "vig": None, "verdict": verdict(edge), "note": note}


class LocalMemory:
    """In-process memory used when the persistent learning module is unavailable (tests, degraded mode)."""

    def __init__(self):
        self.turns: list[dict] = []
        self.alias_map: dict[str, dict] = {}

    def record_turn(self, turn: dict) -> str:
        turn_id = uuid.uuid4().hex[:16]
        self.turns.append({**copy.deepcopy(turn), "turn_id": turn_id, "rating": None,
                           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")})
        return turn_id

    def last_turns(self, conversation_id: str, limit: int = 10) -> list[dict]:
        return [turn for turn in reversed(self.turns) if turn["conversation_id"] == conversation_id][:limit]

    def aliases(self) -> dict[str, dict]:
        return self.alias_map

    def use_alias(self, phrase: str) -> None:
        key = nlu.name_key(phrase)
        if key in self.alias_map:
            self.alias_map[key]["uses"] += 1

    def learn_alias(self, phrase: str, player_id: str, player_name: str, source: str) -> dict:
        key = nlu.name_key(phrase)
        if not key or key.isdigit() or key in nlu.STAT_LOOKUP:
            raise ValueError("That nickname is not usable: pick a word or two that is not a number or a stat name.")
        self.alias_map[key] = {"phrase": key, "player_id": str(player_id), "player": player_name, "source": source, "uses": 0}
        return self.alias_map[key]

    def forget_alias(self, phrase: str) -> bool:
        return self.alias_map.pop(nlu.name_key(phrase), None) is not None

    def examples(self, message: str, limit: int = 3) -> list[dict]:
        return []


@dataclass
class Turn:
    conversation_id: str
    message: str
    slip: dict
    focus: dict | None
    context: dict
    lines: list[str] = field(default_factory=list)
    cards: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    clarification: dict | None = None
    learned: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    changed: bool = False
    engine: str = "built_in"
    priced: dict | None = None
    llm_tried: bool = False

    def say(self, text: str) -> None:
        if text and text not in self.lines:
            self.lines.append(text)

    def suggest(self, *items: str) -> None:
        for item in items:
            if item and item not in self.suggestions and len(self.suggestions) < 4:
                self.suggestions.append(item)

    def act(self, kind: str, description: str, leg: int | None = None) -> None:
        self.actions.append({"type": kind, "leg": leg, "description": description})
        if kind != "track_bet":
            self.changed = True


class Coach:
    def __init__(self, roster: Callable[[], list[dict]], history: Callable[[str, bool], tuple[str, pd.DataFrame, dict]],
                 load: Callable[[str], tuple[pd.DataFrame, dict]], learning: Any = None, llm: LLMClient | None = None,
                 today: Callable[[], date] | None = None, memory: Any = None, newsroom: Any = None,
                 profile: Callable[[str], dict] | None = None):
        self.roster, self.history, self.load = roster, history, load
        self.learning, self.llm = learning, llm
        self.memory = memory or learning or LocalMemory()
        self.today = today or eastern_today
        self.newsroom = newsroom
        self.desk = Desk(newsroom, roster, profile, self.today, learning)
        self.llm_error: str | None = None

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        if self.llm is None:
            return {"mode": "built_in", "model": None, "endpoint": None, "ready": True,
                    "detail": ("Built-in coach: answers slip commands, props, parlays, news, injuries, schedules, game lines, teams, "
                               "coaches, standings, betting math and questions about the app on this computer, using the app's own "
                               "models and feeds. To add a language model for free-form conversation, set COURTSIDE_LLM_MODEL (plus "
                               "COURTSIDE_LLM_API_KEY / COURTSIDE_LLM_BASE_URL for OpenAI, Anthropic, OpenRouter, Azure OpenAI or a "
                               "local Ollama/LM Studio server) in .env and restart.")}
        ready = self.llm_error is None
        detail = (f"Language model {self.llm.model} at {self.llm.endpoint} handles questions and conversation; slip commands and every "
                  "number, headline and injury still come from the app's models, feeds and tools.")
        if not ready:
            detail += f" The last call failed ({self.llm_error}); the built-in coach is answering until it recovers."
        return {"mode": "llm", "model": self.llm.model, "endpoint": self.llm.endpoint, "ready": ready, "detail": detail}

    # ------------------------------------------------------------------ entry point
    def chat(self, message: str, conversation_id: str, slip: dict, focus: dict | None) -> dict:
        message = " ".join(message.split())
        previous = self.memory.last_turns(conversation_id, 1)
        context = dict(previous[0].get("entities") or {}) if previous else {}
        turn = Turn(conversation_id, message, copy.deepcopy(slip), copy.deepcopy(focus), context)
        slip_before = copy.deepcopy(slip)
        index = nlu.PlayerIndex(self.roster(), self.memory.aliases())
        parsed = nlu.parse(message, self.today())
        turn.intents = parsed.intents
        try:
            self.route(turn, parsed, index, previous[0] if previous else None)
        except NBAUnavailableError as exc:
            turn.say(f"NBA Stats could not be reached just now ({exc}). Your slip is unchanged; try again in a minute.")
        if DISTRESS.search(nlu.fold(message)):
            turn.say(SUPPORT)
        if turn.changed and turn.slip["legs"] and turn.priced is None:
            self.price(turn, card=True)
        reply = "\n".join(turn.lines) or "Done."
        turn_id = self.memory.record_turn({
            "conversation_id": conversation_id, "message": message, "reply": reply, "intents": turn.intents,
            "entities": turn.entities, "actions": turn.actions, "engine": turn.engine,
            "slip_before": slip_before, "slip_after": turn.slip if turn.changed else None, "tool_calls": turn.tool_calls,
        })
        return {
            "turn_id": turn_id, "reply": reply, "slip": turn.slip if turn.changed else None, "actions": turn.actions,
            "cards": turn.cards, "suggestions": turn.suggestions[:4], "clarification": turn.clarification, "learned": turn.learned,
            "engine": {"mode": turn.engine, "model": self.llm.model if turn.engine == "llm" and self.llm else None},
        }

    # ------------------------------------------------------------------ routing
    def route(self, turn: Turn, parsed: nlu.Parsed, index: nlu.PlayerIndex, previous: dict | None) -> None:
        self.learn_from_clarification(turn, index, previous)
        request = self.follow_up(turn, parsed, index)
        if request:
            parsed = nlu.parse(request, self.today())
            turn.intents = parsed.intents
        names = set(parsed.intents)
        if "teach" in names and (teach := nlu.parse_teach(parsed.text)):
            return self.teach(turn, teach, index)
        if "forget" in names:
            return self.forget(turn, parsed.text)
        if "undo" in names:
            return self.undo(turn)
        if "greet" in names and len(parsed.text.split()) <= 4:
            turn.say("Hey! I'm the COURTSIDE coach. Ask me anything about the NBA and your bets: injuries, news, tonight's games and "
                     "lines, teams and coaches, any player prop, or build and price your parlay.")
            return self.help(turn, intro=False)
        if THANKS.fullmatch(nlu.fold(parsed.text)):
            turn.say("Anytime. Ask me about a player, a team, tonight's games or your slip whenever you're ready.")
            turn.suggest("Who's out tonight?", "What games are on tonight?", "What are my odds?" if turn.slip["legs"] else "Latest NBA news")
            return
        text = strip_vocative(parsed.text)
        hits = topics.topics(text)
        team_mentions = topics.find_teams(text)
        mentions = self.player_mentions(index, text, team_mentions)
        teams = [mention.code for mention in team_mentions]
        specs = [nlu.parse_leg(segment) for segment in parsed.segments]
        has_spec = any(spec.stat or spec.side or spec.line is not None for spec in specs)
        has_prop = has_spec and (bool(mentions) or self.refers_to_focus(parsed.text, turn) or any(spec.line is not None for spec in specs))
        question = bool(names & {"forecast", "house", "hit_rate", "summary", "explain"}) or "?" in parsed.text
        slipish = self.slipish(turn, parsed, mentions, has_spec)
        if "help" in names:
            entry = KNOWLEDGE.best(text, 5.0)
            if entry is not None and entry.id not in ("coach", "navigation", "overview"):
                return self.desk.say_entry(turn, entry)
            return self.help(turn)
        if "clear" in names and (slipish or not hits):
            self.clear(turn)
            if not (mentions and has_spec and "add" in names):
                return
        if "track" in names and (slipish or not teams):
            return self.track(turn, parsed)
        if "settle" in names:
            return self.settle(turn)
        if "remove" in names and slipish:
            removal = [s for s in parsed.segments if REMOVAL.search(nlu.fold(s))] or [parsed.text]
            self.remove(turn, nlu.parse(" ".join(removal), self.today()), index)
            rest = [s for s in parsed.segments if s not in removal]
            if rest and "add" in names:
                self.add_legs(turn, nlu.parse(" and ".join(rest), self.today()), index)
            return
        if "update" in names and slipish:
            changes = [s for s in parsed.segments if CHANGE.search(nlu.fold(s))] or [parsed.text]
            if self.update(turn, nlu.parse(" ".join(changes), self.today()), index):
                rest = [s for s in parsed.segments if s not in changes]
                if rest and "add" in names:
                    self.add_legs(turn, nlu.parse(" and ".join(rest), self.today()), index)
                return
        command = "add" in names and has_spec and bool(mentions)
        if self.llm is not None and not command and self.llm_turn(turn):
            return
        folded = nlu.fold(text)
        if topics.OTHER_SPORTS.search(folded) and not teams and not mentions:
            return self.desk.say_scope(turn, "other_sport")
        if not has_prop and self.league_question(turn, parsed, text, index, mentions, teams, hits):
            return
        if not mentions and parsed.odds is None and HOWTO.search(folded) and not SLIP_QUESTION.search(folded) \
                and self.desk.site_answer(turn, text):
            return
        if "learning" in names and not mentions:
            return self.learning_status(turn)
        if mentions or self.refers_to_focus(parsed.text, turn):
            adding = "add" in names or (has_spec and not question and not names & {"parlay", "learning"})
            if adding:
                self.add_legs(turn, parsed, index)
                if parsed.odds is not None and parsed.odds_problem is None and turn.actions:
                    self.set_odds(turn, parsed.odds)
                if names & {"house", "parlay"} and turn.slip["legs"]:
                    self.price(turn, card=True, explain=True)
                return
            if "summary" in names and not any(spec.line is not None for spec in specs):
                return self.player_summary(turn, parsed, index)
            if has_spec or names & {"forecast", "hit_rate", "house"}:
                return self.leg_question(turn, parsed, index)
            return self.player_summary(turn, parsed, index)
        if parsed.odds_problem and (names & {"house", "parlay"} or parsed.odds is None and "odds" in nlu.fold(parsed.text)):
            turn.say(parsed.odds_problem)
            return
        if parsed.odds is not None and (turn.slip["legs"] or names & {"house", "parlay"}):
            if "clear" not in names:
                self.set_odds(turn, parsed.odds)
            return self.price(turn, card=True, explain=True)
        if parsed.odds is not None and self.desk.say_odds_math(turn, text, parsed.odds, parsed.stake):
            return
        if re.search(r"\b(?:no|clear|remove|drop)\s+(?:the\s+)?(?:book\s+)?odds\b", nlu.fold(parsed.text)):
            self.set_odds(turn, None)
            return self.price(turn, card=True, explain=True)
        if "explain" in names and turn.slip["legs"] and SLIP_QUESTION.search(folded) or "explain" in names and turn.slip["legs"] and not hits:
            return self.explain(turn)
        if names & {"parlay", "house"} and turn.slip["legs"] or "forecast" in names and turn.slip["legs"] and SLIP_QUESTION.search(folded):
            return self.price(turn, card=True, explain=True)
        if self.desk.site_answer(turn, text):
            return
        scope = self.desk.is_off_topic(text, bool(teams or mentions))
        if scope:
            return self.desk.say_scope(turn, scope)
        if names & {"parlay", "house"} and not turn.slip["legs"]:
            turn.say("Your slip is empty. Add a leg first, e.g. **add Nikola Jokić over 26.5 points**.")
            turn.suggest("Add Nikola Jokić over 26.5 points", "Help")
            return
        turn.say("I didn't quite catch that. Ask me about a player (**is LeBron playing tonight?**, **what's SGA averaging?**), a "
                 "team (**how do the Thunder play?**), tonight's games and lines, the news, or give me a slip command like "
                 "**add Jamal Murray over 20.5 points**.")
        self.help(turn, intro=False, short=True)

    # ------------------------------------------------------------------ league questions
    def player_mentions(self, index: nlu.PlayerIndex, text: str, teams: list[topics.TeamMention]) -> list[nlu.Mention]:
        """Player mentions, minus those that are really a team ("houston", "orlando"); a longer player name beats a team word."""
        mentions = index.find(text)
        keep = []
        for mention in mentions:
            clash = next((team for team in teams if mention.phrase == team.phrase or mention.phrase in team.phrase.split()
                          or mention.phrase in team.phrase), None)
            if clash is not None and len(mention.phrase) <= len(clash.phrase):
                continue
            keep.append(mention)
        for mention in keep:
            for team in list(teams):
                if team.phrase in mention.phrase.split() or team.phrase == mention.phrase:
                    teams.remove(team)
        return keep

    def slipish(self, turn: Turn, parsed: nlu.Parsed, mentions: list[nlu.Mention], has_spec: bool) -> bool:
        """Whether a remove/change/clear/track verb is about the slip rather than the league ("did the Lakers cut anyone?")."""
        if parsed.legs or has_spec:
            return True
        if re.search(r"\b(?:legs?|slip|parlay|ticket|it|that|this|them|last one|everything|all|book odds)\b", nlu.fold(parsed.text)):
            return True
        ids = {leg["request"]["player_id"] for leg in turn.slip["legs"]}
        return any(any(candidate["id"] in ids for candidate in mention.candidates) for mention in mentions)

    def league_question(self, turn: Turn, parsed: nlu.Parsed, text: str, index: nlu.PlayerIndex, mentions: list[nlu.Mention],
                        teams: list[str], hits: dict[str, str]) -> bool:
        if not hits and not teams:
            return False
        players: list[dict] = []
        for mention in mentions:
            candidates = self.narrow(turn, mention.candidates)
            if len(candidates) == 1:
                players.append(candidates[0])
            elif hits and not teams and not players:
                options = [{"label": f"{player['name']} · {player['team']}",
                            "message": re.sub(re.escape(mention.phrase), player["name"], nlu.fold(text), count=1)}
                           for player in candidates[:6]]
                turn.clarification = {"prompt": f"Which {mention.phrase.title()}?", "options": options}
                turn.entities["clarify"] = {"phrase": mention.phrase, "options": {o["message"]: p["id"] for o, p in zip(options, candidates)}}
                turn.say(f"More than one current player matches “{mention.phrase}”. Which one did you mean?")
                return True
        if not players and not teams and re.search(r"\b(?:he|him|his|he's)\b", nlu.fold(text)):
            player_id = turn.context.get("player_id") or (turn.focus or {}).get("player_id")
            if player_id and player_id in index.players:
                players.append(index.players[player_id])
        if players and set(hits) <= {"player_info"} and re.search(r"\b(?:tell me about|who is|who's)\b", nlu.fold(text)):
            self.desk.say_bio(turn, players[0], quiet=True)
            self.player_summary(turn, parsed, index, players[0])
            return True
        try:
            return self.desk.answer(turn, text, players, teams, hits, parsed.odds, parsed.stake,
                                    has_spec=any(nlu.parse_leg(segment).stat for segment in parsed.segments))
        except NewsUnavailableError as exc:
            turn.say(f"The news feeds couldn't be reached just now ({exc}). Try again in a minute.")
            return True

    def narrow(self, turn: Turn, candidates: list[dict]) -> list[dict]:
        if len(candidates) <= 1:
            return candidates
        preferred = {leg["request"]["player_id"] for leg in turn.slip["legs"]}
        if turn.focus:
            preferred.add(turn.focus["player_id"])
        if turn.context.get("player_id"):
            preferred.add(turn.context["player_id"])
        narrowed = [player for player in candidates if player["id"] in preferred]
        return narrowed if len(narrowed) == 1 else candidates

    def context_notes(self, turn: Turn, player: dict, schedule: bool = True) -> None:
        """Injury status and next game for a player being forecast, from the cached feeds (best-effort)."""
        if self.newsroom is None or not player.get("team"):
            return
        report, _ = self.newsroom.attempt(self.newsroom.injuries)
        entry = self.newsroom.injury_entry(player["id"], report) if report else None
        if entry:
            what = injury_what(entry)
            turn.say(f"Heads-up: {player['name']} is listed **{entry['status']}**{f' ({what})' if what else ''} on the injury report; "
                     "the forecast assumes he plays.")
        if schedule and player["team"] in topics.TEAM_NAMES:
            game = self.newsroom.next_game(player["team"])
            if game:
                turn.say(f"Next game: {self.desk.game_phrase(game, player['team'])}.")

    # ------------------------------------------------------------------ players
    def refers_to_focus(self, text: str, turn: Turn) -> bool:
        if not turn.focus:
            return False
        return bool(re.search(r"\b(?:this|that|him|his|he|the over|the under|this player|this prop|it)\b", nlu.fold(text))) \
            and bool(re.search(r"\b(?:add|over|under|line|o|u)\b", nlu.fold(text)))

    def follow_up(self, turn: Turn, parsed: nlu.Parsed, index: nlu.PlayerIndex) -> str | None:
        """The full request behind a reply that leans on the conversation: "points" after "Which stat for Scottie Barnes?",
        or "over or under 23.5?" right after a question about Stephen Curry. None when the message stands on its own."""
        awaiting = turn.context.get("awaiting")
        awaiting = awaiting if isinstance(awaiting, dict) else None
        player = index.players.get(str((awaiting or {}).get("player_id") or turn.context.get("player_id") or ""))
        if player is None:
            return None
        names = set(parsed.intents)
        text = strip_vocative(parsed.text)
        folded = nlu.fold(text)
        if names & STANDALONE_INTENTS or parsed.legs or SLIP_QUESTION.search(folded) or re.search(r"\b(?:slip|parlay|ticket|legs?)\b", folded):
            return None
        teams = topics.find_teams(text)
        if teams or self.player_mentions(index, text, teams):
            return None
        awaiting_line = bool(awaiting and awaiting.get("stat") and awaiting.get("line") is None)
        spec = nlu.parse_leg(text, bare_number=awaiting_line or bool(re.search(r"(?<![\d.])\d{1,3}\.5(?!\d)", folded)))
        if spec.unsupported or spec.problem or not (spec.stat or spec.side or spec.line is not None):
            return None
        hits = topics.topics(text)
        refers_back = bool(re.search(r"\b(?:he|him|his|he's|the over|the under|that line|same line)\b", folded))
        howto = bool(HOWTO.search(folded))
        if re.match(r"(?:and |so |ok |okay )?(?:who|whose|which)\b", folded) or "rank" in hits \
                or any(level == "strong" and (topic != "advice" or not refers_back) for topic, level in hits.items()) \
                or howto and not refers_back and (spec.line is None and spec.side is None or KNOWLEDGE.best(text, 3.5) is not None):
            return None
        either = bool(nlu.EITHER_SIDE.search(folded))
        words = len(folded.split())
        question = bool(names & {"forecast", "house", "hit_rate", "summary", "explain"}) or "?" in text or howto
        if awaiting and words <= 5:
            # A bare answer ("points", "the under") finishes what was asked; a question about it only asks.
            verb = "add" if "add" in names else "ask" if question else awaiting.get("verb") or "ask"
            stat = spec.stat or awaiting.get("stat")
            line = spec.line if spec.line is not None else awaiting.get("line")
            side = None if either else spec.side or awaiting.get("side")
        elif words <= 8 or refers_back:
            verb = "add" if "add" in names or not question else "ask"
            same_stat = spec.stat is None or spec.stat == turn.context.get("stat")
            stat = spec.stat or (turn.context.get("stat") if spec.line is None else None)
            line = spec.line if spec.line is not None else turn.context.get("line") if same_stat else None
            side = None if either else spec.side
        else:
            return None
        if stat is None and line is None:
            return None
        parts = [player["name"], side or ("over or under" if line is not None else None),
                 f"{line:g}" if line is not None else None, nlu.phrase(stat) if stat else None]
        request = " ".join(part for part in parts if part)
        if parsed.odds is not None and parsed.odds_problem is None:
            request += f" at {parsed.odds:+d}"
        if window := WINDOW.search(folded):
            request += f" in his {window.group(0)}"
        return f"add {request}" if verb == "add" else f"{request}?"

    @staticmethod
    def await_reply(turn: Turn, verb: str, player: dict, stat: str | None = None, line: float | None = None,
                    side: str | None = None) -> None:
        """Remembers a half-finished request so a short reply next turn ("points", "under", "22.5") can complete it."""
        turn.entities["player_id"] = player["id"]
        turn.entities["awaiting"] = {"verb": verb, "player_id": player["id"], "stat": stat, "line": line, "side": side}

    def resolve(self, turn: Turn, text: str, index: nlu.PlayerIndex, verb: str, spec: nlu.LegSpec | None = None) -> dict | None:
        mentions = index.find(text)
        if mentions:
            mention = mentions[0]
            candidates = mention.candidates
            if len(candidates) > 1:
                preferred = {leg["request"]["player_id"] for leg in turn.slip["legs"]}
                if turn.focus:
                    preferred.add(turn.focus["player_id"])
                if turn.context.get("player_id"):
                    preferred.add(turn.context["player_id"])
                narrowed = [player for player in candidates if player["id"] in preferred]
                if len(narrowed) == 1:
                    candidates = narrowed
            if len(candidates) > 1:
                options = []
                for player in candidates[:6]:
                    options.append({"label": f"{player['name']} · {player['team']}",
                                    "message": self.canonical(verb, player["name"], spec, text)})
                turn.clarification = {"prompt": f"Which {mention.phrase.title()}?", "options": options}
                turn.entities["clarify"] = {"phrase": mention.phrase, "options": {o["message"]: p["id"] for o, p in zip(options, candidates)}}
                turn.say(f"More than one current player matches “{mention.phrase}”. Which one did you mean?")
                return None
            player = candidates[0]
            if mention.kind == "fuzzy":
                turn.say(f"Assuming you meant **{player['name']}**.")
            if mention.kind == "alias" and mention.phrase in index.learned:
                try:
                    self.memory.use_alias(mention.phrase)
                except Exception:  # memory statistics are best-effort
                    pass
            return player
        player_id = None
        if re.search(r"\b(?:him|his|he|same (?:guy|player)|this (?:guy|player)|that (?:guy|player))\b", nlu.fold(text)):
            player_id = turn.entities.get("player_id") or turn.context.get("player_id") or (turn.focus or {}).get("player_id")
        elif turn.focus and self.refers_to_focus(text, turn):
            player_id = turn.focus["player_id"]
        if player_id and player_id in index.players:
            return index.players[player_id]
        return None

    @staticmethod
    def canonical(verb: str, name: str, spec: nlu.LegSpec | None, text: str) -> str:
        if spec and spec.stat and spec.line is not None:
            side = spec.side or "over"
            if verb == "add":
                return f"add {name} {side} {spec.line:g} {spec.stat}"
            return f"{name} {side} {spec.line:g} {spec.stat}?"
        if verb == "summary":
            return f"what's {name} averaging?"
        return f"{verb} {name}".strip()

    def learn_from_clarification(self, turn: Turn, index: nlu.PlayerIndex, previous: dict | None) -> None:
        pending = (previous or {}).get("entities", {}).get("clarify") if previous else None
        if not pending or turn.message not in pending.get("options", {}):
            return
        player_id = pending["options"][turn.message]
        player = index.players.get(player_id)
        if not player or len(pending["phrase"]) < 3:
            return
        try:
            self.memory.learn_alias(pending["phrase"], player_id, player["name"], "clarified")
        except ValueError:
            return
        index.keys["alias"][nlu.name_key(pending["phrase"])] = {player_id}
        index.learned[nlu.name_key(pending["phrase"])] = player_id
        turn.learned.append(f"Learned that “{pending['phrase']}” means {player['name']}. Say “forget {pending['phrase']}” to undo.")

    def teach(self, turn: Turn, teach: nlu.Teach, index: nlu.PlayerIndex) -> None:
        matches = index.by_name(teach.name) or [m.candidates[0] for m in index.find(teach.name) if len(m.candidates) == 1]
        if len(matches) != 1:
            turn.say(f"I couldn't match “{teach.name}” to exactly one current player. Use their full name, e.g. "
                     f"**remember {teach.phrase} means Nikola Jokić**.")
            return
        player = matches[0]
        try:
            self.memory.learn_alias(teach.phrase, player["id"], player["name"], "taught")
        except ValueError as exc:
            turn.say(str(exc))
            return
        turn.learned.append(f"Learned that “{teach.phrase}” means {player['name']}.")
        turn.say(f"Got it — “{teach.phrase}” now means **{player['name']}** ({player['team']}).")
        turn.entities["player_id"] = player["id"]
        turn.suggest(f"What's {teach.phrase} averaging?")

    def forget(self, turn: Turn, text: str) -> None:
        phrase = re.sub(r"^\s*forget\s+(?:that\s+|the nickname\s+)?", "", nlu.fold(text)).strip(" '\".?!")
        if self.memory.forget_alias(phrase):
            turn.learned.append(f"Forgot the nickname “{phrase}”.")
            turn.say(f"Forgotten: “{phrase}” is no longer a nickname.")
        else:
            turn.say(f"I don't have a learned nickname “{phrase}”.")

    # ------------------------------------------------------------------ data
    def player_data(self, player_id: str) -> tuple[str, pd.DataFrame, dict]:
        return self.history(player_id, False)

    def leg_request(self, player: dict, stat: str, line: float, side: str, focus: dict | None) -> dict:
        """A leg uses the Props view settings when the user is looking at this player, else the defaults."""
        settings = dict(DEFAULTS)
        if focus and focus.get("player_id") == player["id"]:
            settings.update({key: focus[key] for key in DEFAULTS if key in focus})
            dataset_id = focus["dataset_id"]
        else:
            dataset_id = self.player_data(player["id"])[0]
        return {"dataset_id": dataset_id, "player_id": player["id"], "stat": stat, "line": float(line), "side": side, **settings}

    def run_forecast(self, request: dict) -> dict:
        frame, info = self.load(request["dataset_id"])
        body = {key: value for key, value in request.items() if key != "side"} | {"odds_over": -110, "odds_under": -110}
        return forecast(frame, info, body)

    # ------------------------------------------------------------------ slip commands
    def add_legs(self, turn: Turn, parsed: nlu.Parsed, index: nlu.PlayerIndex) -> None:
        carried: dict = {}
        noted: set[str] = set()
        for segment in parsed.segments:
            spec = nlu.parse_leg(segment)
            if spec.unsupported:
                turn.say(f"I can't price {spec.unsupported} yet — COURTSIDE models points, rebounds, assists, threes, steals, blocks, turnovers, PRA, pts+reb, pts+ast, reb+ast, stocks, FGM/FGA, 3PA, FTM/FTA, offensive and defensive rebounds, fouls, double-doubles and triple-doubles.")
                continue
            if spec.problem:
                turn.say(spec.problem)
                continue
            player = self.resolve(turn, segment, index, "add", spec)
            if turn.clarification:
                return
            if player is None:
                player = carried.get("player")
            if player is None:
                if index.find(segment) == [] and (spec.stat or spec.line is not None):
                    turn.say("I couldn't find that player on a current NBA roster. Try their full name.")
                continue
            focus = turn.focus if turn.focus and turn.focus["player_id"] == player["id"] else None
            stat = spec.stat or (focus or {}).get("stat") or (carried.get("stat") if carried.get("player") is player else None)
            if not stat and spec.line is not None:
                # Adding never guesses a market, but keeps the one being discussed ("over 27.5" after Curry's points).
                stat = self.likely_stat(turn, player, spec.line, guess=False)[0]
            line = spec.line if spec.line is not None else (focus or {}).get("line") if (focus or {}).get("stat") == stat else None
            if not stat:
                turn.say(f"Which stat for {player['name']}? Points, rebounds, assists, threes, PRA, steals, blocks and more.")
                turn.suggest(*(f"add {player['name']} {spec.side or 'over'} {line:g} {s}" if line is not None
                               else f"what's {player['name']} averaging?" for s in ("points", "rebounds", "assists")))
                self.await_reply(turn, "add", player, line=line, side=spec.side)
                continue
            if line is None:
                turn.say(f"What line for {player['name']} {nlu.phrase(stat)}? For example **add {player['name']} over 20.5 {nlu.phrase(stat)}**.")
                turn.suggest(f"what's {player['name']} averaging?")
                self.await_reply(turn, "add", player, stat=stat, side=spec.side)
                continue
            side = spec.side
            if side is None:
                turn.clarification = {"prompt": f"{player['name']} {nlu.phrase(stat)}: over or under {line:g}?", "options": [
                    {"label": f"Over {line:g}", "message": f"add {player['name']} over {line:g} {nlu.phrase(stat)}"},
                    {"label": f"Under {line:g}", "message": f"add {player['name']} under {line:g} {nlu.phrase(stat)}"}]}
                turn.say(f"Over or under {line:g} {nlu.phrase(stat)} for {player['name']}?")
                self.await_reply(turn, "add", player, stat=stat, line=line)
                return
            carried = {"player": player, "stat": stat}
            if self.place_leg(turn, player, stat, line, side) and player["id"] not in noted:
                noted.add(player["id"])
                self.context_notes(turn, player, schedule=False)

    def place_leg(self, turn: Turn, player: dict, stat: str, line: float, side: str) -> bool:
        try:
            request = self.leg_request(player, stat, line, side, turn.focus)
            result = self.run_forecast(request)
        except (ValueError, KeyError) as exc:
            turn.say(f"Couldn't add {player['name']} {nlu.phrase(stat)}: {exc}")
            return False
        legs = turn.slip["legs"]
        label = {"player": result["player"]["name"], "team": player.get("team") or result["player"]["team"]}
        description = f"{label['player']} {side.title()} {line:g} {nlu.phrase(stat)}"
        probability = result["probabilities"][side]
        existing = next((i for i, leg in enumerate(legs) if leg["request"]["player_id"] == player["id"] and leg["request"]["stat"] == stat), None)
        if existing is not None and legs[existing]["request"] == request:
            turn.say(f"Leg {existing + 1} is already **{description}** — {pct(probability)}.")
            turn.entities.update({"player_id": player["id"], "stat": stat, "line": line, "leg": existing + 1})
            return True
        if existing is not None:
            old = legs[existing]["request"]
            legs[existing] = {"request": request, "label": label}
            turn.act("replace_leg", f"Replaced leg {existing + 1} ({old['side'].title()} {old['line']:g}) with {description}", existing + 1)
            turn.say(f"Replaced leg {existing + 1} with **{description}** — {pct(probability)}.")
            number = existing + 1
        elif len(legs) >= MAX_LEGS:
            turn.say(f"Your slip already has {MAX_LEGS} legs, the maximum. Remove one first.")
            return False
        else:
            legs.append({"request": request, "label": label})
            number = len(legs)
            turn.act("add_leg", f"Added leg {number}: {description}", number)
            turn.say(f"Added leg {number}: **{description}** — {pct(probability)} by the model.")
        turn.entities.update({"player_id": player["id"], "stat": stat, "line": line, "leg": number})
        return True

    def leg_targets(self, turn: Turn, parsed: nlu.Parsed, index: nlu.PlayerIndex) -> list[int]:
        legs = turn.slip["legs"]
        targets = []
        for number in parsed.legs:
            position = len(legs) if number == -1 else number
            if 1 <= position <= len(legs):
                targets.append(position - 1)
        if targets:
            return targets
        text = nlu.name_key(parsed.text)
        words = set(text.split())
        stat_words = {nlu.STAT_LOOKUP[w] for w in words if w in nlu.STAT_LOOKUP}
        for i, leg in enumerate(legs):
            tokens = nlu.name_key(leg["label"]["player"]).split()
            named = nlu.name_key(leg["label"]["player"]) in text or (tokens and (tokens[-1] in words or tokens[0] in words and len(tokens[0]) > 3))
            alias_hit = any(m.candidates and any(p["id"] == leg["request"]["player_id"] for p in m.candidates) for m in index.find(parsed.text))
            if (named or alias_hit) and (not stat_words or leg["request"]["stat"] in stat_words):
                targets.append(i)
        if not targets and re.search(r"\b(?:it|that|this|that leg|this leg)\b", nlu.fold(parsed.text)):
            last = turn.context.get("leg")
            if isinstance(last, int) and 1 <= last <= len(legs):
                targets.append(last - 1)
            elif legs:
                targets.append(len(legs) - 1)
        return targets

    def remove(self, turn: Turn, parsed: nlu.Parsed, index: nlu.PlayerIndex) -> None:
        legs = turn.slip["legs"]
        if not legs:
            turn.say("Your slip is already empty.")
            return
        targets = sorted(set(self.leg_targets(turn, parsed, index)), reverse=True)
        if not targets:
            turn.say("Which leg should I remove? Say **remove leg 2** or name the player.")
            turn.suggest(*(f"remove leg {i + 1}" for i in range(min(3, len(legs)))))
            return
        for i in targets:
            leg = legs.pop(i)
            request = leg["request"]
            turn.act("remove_leg", f"Removed leg {i + 1}: {leg['label']['player']} {request['side'].title()} {request['line']:g} {nlu.phrase(request['stat'])}", i + 1)
        turn.say(f"Removed {len(targets)} leg{'s' if len(targets) > 1 else ''}. {len(legs)} left." if legs else "Removed. Your slip is now empty.")
        turn.entities["leg"] = None

    def update(self, turn: Turn, parsed: nlu.Parsed, index: nlu.PlayerIndex) -> bool:
        text = nlu.fold(parsed.text)
        if parsed.odds is not None and not parsed.legs and not re.search(r"\b(?:flip|line|leg|over|under)\b", text):
            return False
        legs = turn.slip["legs"]
        targets = self.leg_targets(turn, parsed, index)
        if not targets:
            if not legs or not re.search(r"\b(?:flip|switch|swap|line|leg|the over|the under|it)\b", text):
                return False
            turn.say("Which leg should I change? Say e.g. **flip leg 2** or **change leg 1 line to 27.5**.")
            return True
        i = targets[0]
        request = legs[i]["request"]
        new_side, new_line = request["side"], request["line"]
        explicit = re.search(r"\b(?:to|make it|the|an?)\s+(over|under)\b|\b(over|under)\s+(?:instead|now)\b", text)
        if explicit:
            new_side = explicit.group(1) or explicit.group(2)
        elif re.search(r"\b(?:flip|switch|swap|other side|opposite)\b", text):
            new_side = "under" if request["side"] == "over" else "over"
        move = re.search(r"\b(?:to|at|line(?:\s+to)?|=)\s*(\d{1,3}(?:\.\d+)?)\b", text)
        step = re.search(r"\b(up|raise|higher|down|lower)\s+(?:it\s+|the line\s+)?(?:by\s+)?(\d{1,2}(?:\.\d+)?)\b", text)
        if move:
            new_line = float(move.group(1))
        elif step:
            delta = float(step.group(2))
            new_line = request["line"] + (delta if step.group(1) in ("up", "raise", "higher") else -delta)
        if new_line * 2 != int(new_line * 2) or not 0 <= new_line <= 200:
            turn.say(f"{new_line:g} isn't a valid line; use a whole or half point from 0 to 200.")
            return True
        if (new_side, new_line) == (request["side"], request["line"]):
            turn.say(f"Leg {i + 1} is already {request['side'].title()} {request['line']:g}. Say **flip leg {i + 1}** or **change leg {i + 1} line to 27.5**.")
            return True
        updated = {**request, "side": new_side, "line": float(new_line)}
        try:
            result = self.run_forecast(updated)
        except (ValueError, KeyError) as exc:
            turn.say(f"Couldn't change leg {i + 1}: {exc}")
            return True
        legs[i] = {"request": updated, "label": legs[i]["label"]}
        description = f"{legs[i]['label']['player']} {new_side.title()} {new_line:g} {nlu.phrase(request['stat'])}"
        turn.act("update_leg", f"Changed leg {i + 1} to {description}", i + 1)
        turn.say(f"Leg {i + 1} is now **{description}** — {pct(result['probabilities'][new_side])}.")
        turn.entities.update({"player_id": request["player_id"], "stat": request["stat"], "leg": i + 1})
        return True

    def set_odds(self, turn: Turn, odds: int | None) -> None:
        if turn.slip.get("book_odds") == odds:
            return
        turn.slip["book_odds"] = odds
        turn.act("set_book_odds", f"Set the parlay's book odds to {odds_text(odds)}" if odds is not None else "Cleared the book odds")

    def clear(self, turn: Turn) -> None:
        if not turn.slip["legs"] and turn.slip.get("book_odds") is None:
            turn.say("Your slip is already empty.")
            return
        count = len(turn.slip["legs"])
        turn.slip = {"legs": [], "book_odds": None}
        turn.act("clear_slip", f"Cleared {count} leg{'s' if count != 1 else ''} and the book odds")
        turn.say("Cleared your slip. Say **undo** to bring it back.")

    def undo(self, turn: Turn) -> None:
        for previous in self.memory.last_turns(turn.conversation_id, 20):
            if previous.get("slip_after") is None or previous.get("slip_before") is None:
                continue
            if slip_key(previous["slip_after"]) != slip_key(turn.slip):
                # The slip was edited outside the chat (or my last change never reached it); restoring could surprise.
                turn.say("Your slip has changed since my last edit, so there's nothing I can safely undo. Edit it directly, "
                         "or tell me what to change.")
                return
            turn.slip = copy.deepcopy(previous["slip_before"])
            turn.act("undo", "Restored the slip from before your last change")
            turn.say(f"Undone — your slip is back to {len(turn.slip['legs'])} leg{'s' if len(turn.slip['legs']) != 1 else ''}"
                     f"{' at ' + odds_text(turn.slip['book_odds']) if turn.slip.get('book_odds') is not None else ''}.")
            if not turn.slip["legs"]:
                turn.priced = {}
            return
        turn.say("There's nothing to undo in this conversation yet.")

    # ------------------------------------------------------------------ answers
    def price(self, turn: Turn, card: bool = True, explain: bool = False) -> dict | None:
        legs = turn.slip["legs"]
        if not legs:
            turn.say("Your slip is empty. Add a leg first, e.g. **add Nikola Jokić over 26.5 points**.")
            return None
        try:
            result = price_parlay([leg["request"] for leg in legs], turn.slip.get("book_odds"), self.load, calibrator=self.learning)
        except (ValueError, LegNotFound) as exc:
            turn.say(f"I couldn't price the slip: {exc}")
            return None
        turn.priced = result
        calibrated = result["calibration"]["probability"] if result.get("calibration") else None
        house = parlay_house(result["probability"], len(legs), turn.slip.get("book_odds"), calibrated)
        if card:
            turn.cards.append({"type": "parlay", "result": result, "house": house})
        n = len(legs)
        headline = f"{'All ' + str(n) + ' legs hit' if n > 1 else 'This leg hits'}: **{pct(result['probability'])}**"
        if result["fair_odds"]:
            headline += f" (fair odds {odds_text(result['fair_odds']['american'])})"
        if n > 1 and result["correlation_lift"] is not None and any(p["basis"] != "independent" for p in result["pairs"]):
            headline += f"; {pct(result['independent_probability'])} if the legs were independent"
        turn.say(headline + ".")
        if result.get("calibration") and not explain:
            turn.say(f"Calibrated by past results: {pct(result['calibration']['probability'])}.")
        if explain:
            turn.say(house["note"])
            if house["basis"] == "standard":
                turn.suggest("Book has it at +450")
        turn.suggest("Why are the legs linked?" if n > 1 else "", "Track this for $10", "Flip the last leg")
        return result

    def explain(self, turn: Turn) -> None:
        result = self.price(turn, card=True)
        if not result:
            return
        linked = [pair for pair in result["pairs"] if pair["basis"] != "independent"]
        if len(result["legs"]) < 2:
            turn.say("With one leg there is nothing to link.")
            return
        for pair in result["pairs"][:8]:
            turn.say(f"- Legs {pair['a'] + 1} & {pair['b'] + 1}: {pair['note']}")
        if not linked:
            turn.say("All legs are for players on different teams, so the model multiplies their chances.")

    def leg_question(self, turn: Turn, parsed: nlu.Parsed, index: nlu.PlayerIndex) -> None:
        spec = nlu.parse_leg(parsed.text)
        if spec.unsupported:
            turn.say(f"I can't price {spec.unsupported} yet — COURTSIDE models points, rebounds, assists, threes, steals, blocks, turnovers, PRA, pts+reb, pts+ast, reb+ast, stocks, FGM/FGA, 3PA, FTM/FTA, offensive and defensive rebounds, fouls, double-doubles and triple-doubles.")
            return
        player = self.resolve(turn, parsed.text, index, "ask", spec)
        if player is None:
            if not turn.clarification:
                turn.say("I couldn't find that player on a current NBA roster. Try their full name.")
            return
        focus = turn.focus if turn.focus and turn.focus["player_id"] == player["id"] else None
        stat = spec.stat or (focus or {}).get("stat")
        assumed = None
        if not stat and spec.line is not None:
            stat, means, inferred = self.likely_stat(turn, player, spec.line)
            if not stat:
                return self.ask_stat(turn, player, spec, means)
            assumed = means[stat] if inferred else None
        if not stat:
            return self.player_summary(turn, parsed, index, player)
        line = spec.line
        if line is None and focus and focus.get("stat") == stat:
            line = focus["line"]
        if line is None:
            return self.player_summary(turn, parsed, index, player)
        try:
            request = self.leg_request(player, stat, line, spec.side or "over", turn.focus)
            result = self.run_forecast(request)
        except (ValueError, KeyError) as exc:
            turn.say(f"Couldn't forecast {player['name']} {nlu.phrase(stat)}: {exc}")
            return
        chances = result["probabilities"]
        values = [game["value"] for game in result["history"]]
        window = re.search(r"\b(?:last|past|previous)\s+(\d{1,2})\s+games?\b", nlu.fold(parsed.text))
        if window:
            frame, _ = self.load(request["dataset_id"])
            rows = with_stat(frame[frame["player_id"] == player["id"]].sort_values("date"), stat).tail(int(window.group(1)))
            values = rows[stat].astype(int).tolist()
        over_hits = sum(v > line for v in values)
        under_hits = sum(v < line for v in values)
        settled = chances["over"] + chances["under"]
        conditional_over = chances["over"] / settled if settled > 1e-12 else 0.0
        calibrated = None
        if self.learning is not None:
            q = self.learning.calibrate_over(conditional_over, stat)
            if q is not None:
                calibrated = {"over": q, "under": 1 - q}
        side = spec.side
        house = None
        if parsed.odds is not None and parsed.odds_problem is None:
            pick = side or "over"
            house = leg_house(chances[pick], chances["under" if pick == "over" else "over"], parsed.odds)
        turn.cards.append({
            "type": "forecast", "player": result["player"], "stat": stat, "line": line, "model": request["model"],
            "resolved_model": result["resolved_model"], "sample_size": result["sample_size"], "mean": result["summary"]["mean"],
            "interval": result["interval"], "over": chances["over"], "under": chances["under"], "push": chances["push"],
            "hit_rate": {"over": over_hits, "under": under_hits, "games": len(values)},
            "fair": {"over": fair_odds(conditional_over), "under": fair_odds(1 - conditional_over)},
            "calibrated": calibrated, "house": house, "side": side,
        })
        name = result["player"]["name"]
        if assumed is not None:
            turn.say(f"Assuming you mean **{nlu.phrase(stat)}** (his recent average is {assumed:.1f}).")
        if side:
            turn.say(f"**{name} {side} {line:g} {nlu.phrase(stat)}: {pct(chances[side])}** by the model "
                     f"({'under' if side == 'over' else 'over'} {pct(chances['under' if side == 'over' else 'over'])}).")
        else:
            lean = "over" if chances["over"] > chances["under"] else "under"
            turn.say(f"**{name} {line:g} {nlu.phrase(stat)}:** over {pct(chances['over'])}, under {pct(chances['under'])}"
                     + (", close to a coin flip." if abs(chances["over"] - chances["under"]) < 0.04 else f", so the model leans **{lean}**."))
        if chances["push"] > 1e-9:
            turn.say(f"Push (exactly {line:g}): {pct(chances['push'])}.")
        span = f"last {len(values)} games"
        turn.say(f"Over {line:g} in {over_hits} of the {span} ({pct(over_hits / len(values)) if values else 'n/a'}); "
                 f"model mean {result['summary']['mean']:.1f}, 80% range {result['interval']['low']}–{result['interval']['high']}.")
        if calibrated:
            turn.say(f"Calibrated by past results: over {pct(calibrated['over'])}, under {pct(calibrated['under'])} (if it settles).")
        if house:
            turn.say(house["note"])
        self.context_notes(turn, player)
        turn.entities.update({"player_id": player["id"], "stat": stat, "line": line})
        pick = side or ("over" if chances["over"] >= chances["under"] else "under")
        turn.suggest(f"add {name} {pick} {line:g} {nlu.phrase(stat)}", f"what's {name} averaging?")

    def likely_stat(self, turn: Turn, player: dict, line: float, guess: bool = True) -> tuple[str | None, dict[str, float], bool]:
        """(market, recent averages, assumed) for a line given without a stat. The conversation's market wins if the line
        suits it; otherwise, when ``guess`` is set, the market whose average is clearly nearest the line (23.5 for a 25-point
        scorer is points, and assumed is True). No market when two fit about equally well, e.g. 3.5 for a guard with 3.7
        threes and 3.3 rebounds; the averages returned then are the closest candidates."""
        try:
            _, frame, _ = self.player_data(player["id"])
        except ValueError:
            return None, {}, False
        recent = frame[frame["player_id"] == player["id"]].sort_values("date").tail(10)
        current = turn.context.get("stat") if turn.context.get("player_id") == player["id"] else None
        means = {}
        for stat in INFERABLE_STATS + ((current,) if current in nlu.STATS and current not in INFERABLE_STATS else ()):
            values = stat_values(recent, stat).dropna()
            if len(values) >= 3:
                means[stat] = float(values.mean())
        distance = {stat: abs(line - mean) / max(mean, 1.0) for stat, mean in means.items()}
        ranked = sorted(distance, key=distance.get)
        if current in distance and distance[current] <= 0.35:
            return current, means, False
        if guess and ranked:
            best, runner = distance[ranked[0]], distance[ranked[1]] if len(ranked) > 1 else math.inf
            if best <= 0.35 and runner >= max(2 * best, best + 0.05):
                return ranked[0], means, True
        close = [stat for stat in ranked[:3] if distance[stat] <= 0.5] or ranked[:3]
        return None, {stat: means[stat] for stat in close}, False

    def ask_stat(self, turn: Turn, player: dict, spec: nlu.LegSpec, options: dict[str, float]) -> None:
        if options:
            named = [f"{nlu.phrase(stat)} ({mean:.1f})" for stat, mean in options.items()]
            choices = f"{', '.join(named[:-1])} or {named[-1]}" if len(named) > 1 else named[0]
            turn.say(f"Which stat for {player['name']} at {spec.line:g}? Nearest his recent averages: {choices}.")
        else:
            turn.say(f"Which stat for {player['name']} at {spec.line:g}? Points, rebounds, assists, threes, PRA, steals, blocks and more.")
        turn.suggest(*(" ".join(part for part in (player["name"], spec.side, f"{spec.line:g}", nlu.phrase(stat)) if part) + "?"
                       for stat in (options or ("points", "rebounds", "assists"))))
        self.await_reply(turn, "ask", player, line=spec.line, side=spec.side)

    def player_summary(self, turn: Turn, parsed: nlu.Parsed, index: nlu.PlayerIndex, player: dict | None = None) -> None:
        if player is None:
            player = self.resolve(turn, parsed.text, index, "summary")
        if player is None:
            if not turn.clarification:
                turn.say("I couldn't find that player on a current NBA roster. Try their full name.")
            return
        try:
            dataset_id, frame, _ = self.player_data(player["id"])
        except ValueError as exc:
            turn.say(f"No recent games for {player['name']}: {exc}")
            return
        window = re.search(r"\b(?:last|past|previous)\s+(\d{1,2})\s+games?\b", nlu.fold(parsed.text))
        count = int(window.group(1)) if window else 10
        rows = frame[frame["player_id"] == player["id"]].sort_values("date")
        recent = rows.tail(count)
        if recent.empty:
            turn.say(f"No completed games for {player['name']} in the recent window.")
            return
        averages = {column: round(float(recent[column].mean()), 1) for column in ("points", "rebounds", "assists", "steals", "blocks", "minutes")}
        threes = pd.to_numeric(recent["three_pointers_made"], errors="coerce") if "three_pointers_made" in recent else None
        if threes is not None and threes.notna().any():
            averages["threes"] = round(float(threes.mean()), 1)
        last = [{"date": row.date, "opponent": row.opponent, "minutes": float(row.minutes), "points": int(row.points),
                 "rebounds": int(row.rebounds), "assists": int(row.assists), "steals": int(row.steals), "blocks": int(row.blocks)}
                for row in recent.iloc[::-1].head(5).itertuples()]
        name = str(rows.iloc[-1]["player_name"])
        team = str(rows.iloc[-1]["team"])
        turn.cards.append({"type": "player", "player": {"id": player["id"], "name": name, "team": team},
                           "games": len(recent), "averages": averages, "last_games": last})
        pra = averages["points"] + averages["rebounds"] + averages["assists"]
        turn.say(f"**{name}** over his last {len(recent)} games: {averages['points']} pts, {averages['rebounds']} reb, "
                 f"{averages['assists']} ast ({pra:.1f} PRA)" + (f", {averages['threes']} threes" if "threes" in averages else "")
                 + f", {averages['steals']} stl, {averages['blocks']} blk in {averages['minutes']} min.")
        turn.entities["player_id"] = player["id"]
        turn.suggest(f"{name} over {math.floor(averages['points']) + 0.5:g} points?", f"add {name} over {math.floor(averages['points']) - 0.5:g} points")

    # ------------------------------------------------------------------ tracking and learning
    def track(self, turn: Turn, parsed: nlu.Parsed) -> None:
        if self.learning is None:
            turn.say("Bet tracking isn't available right now.")
            return
        legs = turn.slip["legs"]
        if not legs:
            turn.say("Your slip is empty — add legs first, then say **track this for $10**.")
            return
        if parsed.odds is not None and parsed.odds_problem is None:
            self.set_odds(turn, parsed.odds)
        result = self.price(turn, card=False)
        if result is None:
            return
        if parsed.stake is not None and not 0 < parsed.stake <= 1_000_000:
            turn.say("The stake must be more than 0 and at most 1,000,000.")
            return
        try:
            bet = self.learning.track(legs, turn.slip.get("book_odds"), parsed.stake, parsed.event_date or self.today(), result, None)
        except ValueError as exc:
            turn.say(f"Couldn't track that bet: {exc}")
            return
        turn.cards.append({"type": "bet", "bet": bet})
        turn.actions.append({"type": "track_bet", "leg": None, "description": f"Tracked a {len(legs)}-leg bet for {bet['event_date']}"})
        stake = f" for {parsed.stake:g} units" if parsed.stake is not None else ""
        odds = f" at {odds_text(bet['book_odds'])}" if bet.get("book_odds") is not None else ""
        turn.say(f"Tracked your {len(legs)}-leg parlay{odds}{stake} for games on **{bet['event_date']}**. "
                 "Say **grade my bets** after the games and I'll settle it from the box scores — every graded leg also teaches the calibration.")
        turn.suggest("Grade my bets", "Are you getting smarter?")

    def settle(self, turn: Turn) -> None:
        if self.learning is None:
            turn.say("Bet tracking isn't available right now.")
            return
        result = self.learning.settle()
        record = result["record"]
        turn.cards.append({"type": "track_record", "record": record})
        if result["settled_legs"]:
            turn.say(f"Graded {result['settled_legs']} leg{'s' if result['settled_legs'] != 1 else ''} across {result['updated_bets']} bet{'s' if result['updated_bets'] != 1 else ''}.")
        else:
            turn.say("No finished games to grade yet — legs are graded the day after their game date.")
        if record["bets"]:
            line = f"Record: {record['won']}–{record['lost']}" + (f" ({record['void']} void" + (f", {record['reduced']} reduced)" if record["reduced"] else ")") if record["void"] or record["reduced"] else "")
            line += f", {record['pending']} pending."
            if record["roi"] is not None:
                line += f" ROI {record['roi'] * 100:+.1f}% on {record['staked']:g} staked."
            turn.say(line)
            if record["legs_graded"]:
                turn.say(f"Legs: {record['legs_won']} of {record['legs_graded']} won vs {record['expected_legs_won']:.1f} expected by the model.")
        else:
            turn.say("You haven't tracked any bets yet. Build a slip and say **track this for $10**.")
        for warning in result["warnings"][:3]:
            turn.say(f"- {warning}")

    def learning_status(self, turn: Turn) -> None:
        if self.learning is None:
            turn.say("The learning loop isn't available right now.")
            return
        status = self.learning.status()
        turn.cards.append({"type": "learning", "status": status})
        calibration, memory = status["calibration"], status["memory"]
        turn.say("I get sharper in three ways: I grade tracked bets against box scores, learn a calibration correction from graded "
                 "predictions (it only switches on if it beats the raw model on recent games it hasn't seen), and remember your nicknames and feedback.")
        turn.say(calibration["message"])
        turn.say(f"Memory: {len(memory['aliases'])} learned nickname{'s' if len(memory['aliases']) != 1 else ''}, "
                 f"{memory['liked']} 👍 and {memory['disliked']} 👎 on {memory['turns']} replies.")
        turn.suggest("Grade my bets", "remember joker means Nikola Jokić")

    def help(self, turn: Turn, intro: bool = True, short: bool = False) -> None:
        if intro:
            turn.say("Ask me anything about the NBA, your bets or this app. For example:")
        if not short:
            for line in ("- **League & news:** “who's out tonight?”, “is LeBron playing?”, “latest Celtics news”, “any trades?”, "
                         "“who coaches the Warriors?”, “how do the Thunder play?”, “which team plays fastest?”, “standings”",
                         "- **Games & lines:** “what games are on tonight?”, “what's the spread for the Lakers game?”, “did the Knicks win?”",
                         "- **Props & research:** “Jokić over 26.5 points?”, “what's SGA averaging?”, “how often has Wemby had 4+ blocks in his last 10?”",
                         "- **Build & price the slip:** “add Jokić over 26.5 points and Murray 21+ points”, “flip leg 2”, “book has it at +450 — good price?”",
                         "- **Track & learn:** “track this for $10 at +450”, “grade my bets”, “are you getting smarter?”",
                         "- **The app & betting basics:** “how does the simulator work?”, “how accurate is the model?”, “what does +150 mean?”, “what's the vig?”"):
                turn.say(line)
        turn.suggest("Who's out tonight?", "What games are on tonight?", "Add Nikola Jokić over 26.5 points", "How accurate is the model?")

    # ------------------------------------------------------------------ language model
    def llm_turn(self, turn: Turn) -> bool:
        """Lets the configured model answer with tool calls; returns False (built-in fallback) on any failure."""
        assert self.llm is not None
        if turn.llm_tried:
            return False
        turn.llm_tried = True
        index = nlu.PlayerIndex(self.roster(), self.memory.aliases())
        messages = [{"role": "system", "content": self.system_prompt(turn)}]
        for example in self.memory.examples(turn.message, 3):
            messages.append({"role": "user", "content": example["message"]})
            messages.append({"role": "assistant", "content": example["reply"][:600]})
        for previous in reversed(self.memory.last_turns(turn.conversation_id, 4)):
            messages.append({"role": "user", "content": previous["message"]})
            messages.append({"role": "assistant", "content": previous["reply"][:600]})
        messages.append({"role": "user", "content": turn.message})
        try:
            for _ in range(6):
                reply = self.llm.complete(messages, TOOLS)
                calls = reply.get("tool_calls") or []
                if not calls:
                    text = (reply.get("content") or "").strip()
                    if not text:
                        raise LLMError("The language model returned an empty reply.")
                    turn.lines.append(text[:2000])
                    turn.engine = "llm"
                    self.llm_error = None
                    return True
                messages.append({"role": "assistant", "content": reply.get("content") or "", "tool_calls": calls})
                for call in calls[:6]:
                    name = call.get("function", {}).get("name", "")
                    try:
                        arguments = json.loads(call.get("function", {}).get("arguments") or "{}")
                        if not isinstance(arguments, dict):
                            raise ValueError("Tool arguments must be a JSON object.")
                        output = self.tool(turn, index, name, arguments)
                    except (ValueError, KeyError, TypeError, LegNotFound, NewsUnavailableError, NBAUnavailableError) as exc:
                        output = {"error": str(exc)}
                    turn.tool_calls.append({"name": name, "arguments": call.get("function", {}).get("arguments", "")[:500]})
                    messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps(output, default=str)[:6000]})
            raise LLMError("The language model did not finish within the tool-call limit.")
        except LLMError as exc:
            self.llm_error = str(exc)
            turn.tool_calls.append({"name": "llm_error", "arguments": str(exc)[:200]})
            if turn.actions:
                turn.engine = "llm"
                turn.say("The language model stopped mid-answer; the slip changes listed above were applied.")
                return True
            return False

    def system_prompt(self, turn: Turn) -> str:
        slip = [f"{i + 1}. {leg['label']['player']} {leg['request']['side']} {leg['request']['line']:g} {nlu.phrase(leg['request']['stat'])}"
                for i, leg in enumerate(turn.slip["legs"])]
        focus = turn.focus
        return (
            "You are COURTSIDE Coach, the assistant inside COURTSIDE, a local NBA player-prop research app. Pages: Players (Props "
            "forecasts, Game log, Career, Lines notebook), Parlay, My Bets, News (Headlines, Injury report, Teams & coaches, Roster "
            "moves, Games with betting lines), Coach; menus How it works (Method, Accuracy) and Settings (Appearance, Model, Data, Coach). "
            "Answer any question about NBA betting, the NBA (news, injuries, schedules, scores, game lines, teams, coaches, playing "
            "styles, standings, rosters, players) or the app, using the tools. "
            "Rules: (1) Every fact and number (probabilities, odds, stats, injuries, headlines, schedules, lines, records) must come "
            "from a tool result in this conversation; never answer current NBA facts from memory, which may be outdated. If the tools "
            "have nothing, say so. (2) Use the slip tools only when the user asks to change the parlay, and confirm what changed. "
            "(3) This is research, not betting advice: never promise outcomes or give 'locks', never encourage betting, chasing "
            "losses or bigger stakes; if the user seems distressed, mention 1-800-GAMBLER. (4) If a question has nothing to do with "
            "basketball, betting or the app, decline in one sentence and say what you can help with. (5) Supported prop markets: "
            "points, rebounds, assists, threes, steals, blocks, turnovers, PRA and the other combos, stocks, shooting makes and "
            "attempts, offensive and defensive rebounds, fouls, double-doubles and triple-doubles. (6) Be concise: at most 150 words, "
            "plain text with **bold** and '- ' bullets only; the app shows cards for tool results, so summarize rather than list everything. "
            f"Today (US Eastern): {self.today().isoformat()}. Current slip: {('; '.join(slip)) or 'empty'}; book odds: "
            f"{odds_text(turn.slip.get('book_odds')) if turn.slip.get('book_odds') is not None else 'none'}. "
            f"Props view focus: {focus['player'] + ' ' + focus['stat'] + ' ' + format(focus['line'], 'g') if focus else 'none'}."
        )

    def tool(self, turn: Turn, index: nlu.PlayerIndex, name: str, args: dict) -> dict:
        def player_named(text: str) -> dict:
            matches = index.by_name(str(text)) or [m.candidates[0] for m in index.find(str(text)) if len(m.candidates) == 1]
            if len(matches) != 1:
                raise ValueError(f"No single current player matches {text!r}; ask the user to clarify.")
            return matches[0]

        def stat_of(value) -> str:
            stat = str(value).lower() if str(value).lower() in nlu.STATS else nlu.STAT_LOOKUP.get(nlu.stat_phrases(nlu.fold(str(value))).strip())
            if not stat:
                raise ValueError("Stat must be one of: " + ", ".join(nlu.STATS) + ".")
            return stat

        def line_of(value) -> float:
            line = float(value)
            if line * 2 != int(line * 2) or not 0 <= line <= 200:
                raise ValueError("Line must be a whole or half point from 0 to 200.")
            return line

        def side_of(value) -> str:
            if value not in ("over", "under"):
                raise ValueError("Side must be over or under.")
            return value

        def team_of(value) -> str:
            raw = str(value or "").strip()
            code = topics.CODE_ALIASES.get(raw.upper(), raw.upper())
            if code in topics.TEAM_NAMES:
                return code
            found = topics.find_teams(raw) or topics.find_teams(raw.lower())
            if not found:
                raise ValueError(f"No NBA team matches {raw!r}.")
            return found[0].code

        def day_of(value) -> date:
            if not value:
                return self.today()
            day = topics.find_day(str(value), self.today())
            if day is None:
                raise ValueError("Use today, tomorrow, yesterday, a weekday or YYYY-MM-DD.")
            return day

        def card(item: dict) -> None:
            if sum(1 for existing in turn.cards if existing.get("type") not in ("forecast", "parlay")) < 3:
                turn.cards.append(item)

        if name in LEAGUE_TOOLS and self.newsroom is None and name not in ("site_help", "model_accuracy", "odds_math", "player_profile", "team_roster"):
            raise ValueError("The newsroom feeds aren't connected.")
        if name == "nba_news":
            player = player_named(args["player"]) if args.get("player") else None
            team = team_of(args["team"]) if args.get("team") else None
            data = self.desk.data_headlines(team=team, player=player, kind=args.get("kind"), query=args.get("query"))
            if data["articles"]:
                card(self.desk.news_card(f"{player['name'] if player else self.desk.team_name(team) if team else 'NBA'} headlines", data,
                                         f"#/news/teams/{team.lower()}" if team else "#/news"))
            return data
        if name == "injury_report":
            if args.get("player"):
                player = player_named(args["player"])
                data = self.desk.data_injuries(player=player)
                entries = ([data["entry"]] if data["entry"] else []) + [m for m in data["teammates"] if m["status_key"] != "probable"]
                card(self.desk.injury_card(f"{player['name']} injury report", entries, data))
                return data
            if args.get("team"):
                team = team_of(args["team"])
                data = self.desk.data_injuries(team=team)
                card(self.desk.injury_card(f"{self.desk.team_name(team)} injury report", data["entries"], data, f"#/news/teams/{team.lower()}"))
                return data
            if args.get("date"):
                day = day_of(args["date"])
                data = self.desk.data_injuries(day=day)
                card(self.desk.injury_card(f"Injuries for {day.isoformat()} games", data["entries"], data))
                return data
            data = self.desk.data_injuries()
            card(self.desk.injury_card("Latest injury updates", data["latest"], data))
            return data
        if name == "roster_moves":
            team = team_of(args["team"]) if args.get("team") else None
            player = player_named(args["player"]) if args.get("player") else None
            return self.desk.data_moves(team, player)
        if name == "games":
            team = team_of(args["team"]) if args.get("team") else None
            if team and not args.get("date"):
                game = self.newsroom.next_game(team)
                if game:
                    card(self.desk.games_card("Next game", date.fromisoformat(game["date"]) if game.get("date") else self.today(), [game]))
                return {"team": team, "next_game": game and self.desk.compact_game(game)}
            day = day_of(args.get("date"))
            data = self.desk.data_games(day, team)
            if data["games"]:
                card(self.desk.games_card(f"Games on {day.isoformat()}", day, data["games"]))
            return data
        if name == "team_profile":
            data = self.desk.data_team(team_of(args["team"]))
            card(self.desk.team_card(data))
            return data
        if name == "coaching_staff":
            team = team_of(args["team"])
            staffs = self.newsroom.coaches()
            staff = next((item for item in staffs.get("teams", []) if item["team"] == team), None)
            return {"team": team, "season": staffs.get("season"), "coaches": (staff or {}).get("coaches", [])}
        if name == "team_rankings":
            metric = str(args.get("metric"))
            if metric not in METRIC_KEYS:
                raise ValueError("Metric must be one of: " + ", ".join(METRIC_KEYS) + ".")
            return self.desk.data_rankings(metric, args.get("order") != "worst")
        if name == "standings":
            conference = args.get("conference")
            return self.desk.data_standings(conference if conference in ("East", "West") else None)
        if name == "team_roster":
            team = team_of(args["team"])
            report, _ = self.newsroom.attempt(self.newsroom.injuries) if self.newsroom else (None, None)
            status = {entry.get("player_id"): entry["status"] for item in (report or {}).get("teams", []) for entry in item["injuries"]}
            return {"team": team, "players": [{"name": p["name"], "status": status.get(p["id"])} for p in self.desk.players_on(team)]}
        if name == "player_profile":
            player = player_named(args["player"])
            out = self.desk.data_bio(player)
            if self.newsroom is not None:
                report, _ = self.newsroom.attempt(self.newsroom.injuries)
                out["injury"] = self.newsroom.injury_entry(player["id"], report)
                game = self.newsroom.next_game(player["team"]) if player.get("team") in topics.TEAM_NAMES else None
                out["next_game"] = game and self.desk.compact_game(game)
            return out
        if name == "site_help":
            return self.desk.data_site_help(str(args.get("question", "")))
        if name == "model_accuracy":
            return self.desk.data_accuracy(stat_of(args["stat"]) if args.get("stat") else None)
        if name == "odds_math":
            odds = int(args["odds"])
            american_probability(odds)
            stake = float(args["stake"]) if args.get("stake") is not None else None
            if stake is not None and not 0 < stake <= 1_000_000:
                raise ValueError("Stake must be between 0 and 1,000,000.")
            return self.desk.data_odds(odds, stake)
        if name == "get_slip":
            result = self.price(turn, card=False) if turn.slip["legs"] else None
            return {"legs": [{"leg": i + 1, **leg["label"], **{k: leg["request"][k] for k in ("stat", "side", "line")}} for i, leg in enumerate(turn.slip["legs"])],
                    "book_odds": turn.slip.get("book_odds"),
                    "probability": result and result["probability"], "independent_probability": result and result["independent_probability"]}
        if name == "find_player":
            matches = index.by_name(str(args.get("name", ""))) or [p for m in index.find(str(args.get("name", ""))) for p in m.candidates]
            return {"players": [{"id": p["id"], "name": p["name"], "team": p["team"]} for p in matches[:8]]}
        if name == "forecast_prop":
            player, stat, line = player_named(args["player"]), stat_of(args["stat"]), line_of(args["line"])
            result = self.run_forecast(self.leg_request(player, stat, line, "over", turn.focus))
            chances = result["probabilities"]
            out = {"player": result["player"]["name"], "stat": stat, "line": line, "over": chances["over"], "under": chances["under"],
                   "push": chances["push"], "mean": result["summary"]["mean"], "interval_80": [result["interval"]["low"], result["interval"]["high"]],
                   "games": result["sample_size"], "model": result["resolved_model"]}
            if args.get("odds") is not None:
                side = side_of(args.get("side") or "over")
                out["house"] = leg_house(chances[side], chances["under" if side == "over" else "over"], int(args["odds"]))
            return out
        if name == "player_summary":
            player = player_named(args["player"])
            before = len(turn.cards)
            self.player_summary(turn, nlu.parse(f"what's {player['name']} averaging", self.today()), index, player)
            return turn.cards[-1] if len(turn.cards) > before else {"error": "No recent games."}
        if name == "add_leg":
            player = player_named(args["player"])
            ok = self.place_leg(turn, player, stat_of(args["stat"]), line_of(args["line"]), side_of(args["side"]))
            return {"ok": ok, "slip_legs": len(turn.slip["legs"]), "message": turn.lines[-1] if turn.lines else ""}
        if name == "remove_leg":
            number = int(args["leg"])
            if not 1 <= number <= len(turn.slip["legs"]):
                raise ValueError(f"There is no leg {number}.")
            self.remove(turn, nlu.parse(f"remove leg {number}", self.today()), index)
            return {"ok": True, "slip_legs": len(turn.slip["legs"])}
        if name == "update_leg":
            number = int(args["leg"])
            if not 1 <= number <= len(turn.slip["legs"]):
                raise ValueError(f"There is no leg {number}.")
            words = [f"leg {number}"]
            if args.get("side"):
                words.append(f"to the {side_of(args['side'])}")
            if args.get("line") is not None:
                words.append(f"line to {line_of(args['line']):g}")
            self.update(turn, nlu.parse("change " + " ".join(words), self.today()), index)
            return {"ok": True, "leg": turn.slip["legs"][number - 1]["request"] | {"label": turn.slip["legs"][number - 1]["label"]}}
        if name == "set_book_odds":
            odds = args.get("odds")
            if odds is not None:
                odds = int(odds)
                american_probability(odds)
            self.set_odds(turn, odds)
            return {"ok": True, "book_odds": odds}
        if name == "clear_slip":
            self.clear(turn)
            return {"ok": True}
        if name == "price_slip":
            result = self.price(turn, card=True)
            if result is None:
                return {"error": "The slip is empty or could not be priced."}
            house = turn.cards[-1]["house"]
            return {"probability": result["probability"], "independent_probability": result["independent_probability"],
                    "fair_odds": result["fair_odds"], "house": house, "pairs": [p["note"] for p in result["pairs"]][:10],
                    "calibrated": result.get("calibration") and result["calibration"]["probability"]}
        if name == "track_record":
            if self.learning is None:
                return {"error": "Tracking is unavailable."}
            record = self.learning.track_record()
            turn.cards.append({"type": "track_record", "record": record})
            return record
        if name == "learning_status":
            if self.learning is None:
                return {"error": "Learning is unavailable."}
            status = self.learning.status()
            turn.cards.append({"type": "learning", "status": status})
            return {"calibration": status["calibration"]["message"], "aliases": len(status["memory"]["aliases"]), "record": status["record"]}
        raise ValueError(f"Unknown tool {name!r}.")


def _tool(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": required or [], "additionalProperties": False}}}


_PLAYER = {"type": "string", "description": "Player name as the user wrote it or the full name"}
_STAT = {"type": "string", "enum": list(nlu.STATS)}
_LINE = {"type": "number", "description": "Whole or half point, e.g. 26.5"}
_SIDE = {"type": "string", "enum": ["over", "under"]}
_TEAM = {"type": "string", "description": "NBA team name, nickname, city or abbreviation"}
_DATE = {"type": "string", "description": "today, tomorrow, yesterday, a weekday or YYYY-MM-DD (US Eastern)"}
TOOLS = [
    _tool("get_slip", "Current parlay slip legs, book odds and the model's joint probability."),
    _tool("find_player", "Look up current NBA players by name or nickname.", {"name": _PLAYER}, ["name"]),
    _tool("forecast_prop", "Model probabilities for one player prop; include odds (American) and side to compare a price.",
          {"player": _PLAYER, "stat": _STAT, "line": _LINE, "side": _SIDE, "odds": {"type": "integer"}}, ["player", "stat", "line"]),
    _tool("player_summary", "Recent averages and last five games for a player.", {"player": _PLAYER}, ["player"]),
    _tool("add_leg", "Add a leg to the slip (replaces the same player and stat if present).",
          {"player": _PLAYER, "stat": _STAT, "line": _LINE, "side": _SIDE}, ["player", "stat", "line", "side"]),
    _tool("remove_leg", "Remove a leg by its 1-based number.", {"leg": {"type": "integer"}}, ["leg"]),
    _tool("update_leg", "Change a leg's side and/or line.", {"leg": {"type": "integer"}, "side": _SIDE, "line": _LINE}, ["leg"]),
    _tool("set_book_odds", "Set (or clear with null) the American odds the book offers for the whole parlay.",
          {"odds": {"type": ["integer", "null"]}}, ["odds"]),
    _tool("clear_slip", "Remove every leg and the book odds."),
    _tool("price_slip", "Price the slip: joint probability, fair odds, comparison with the book's price, leg links."),
    _tool("track_record", "Results of the user's tracked bets."),
    _tool("learning_status", "What the app has learned: calibration, nicknames, feedback."),
    _tool("nba_news", "Latest NBA headlines, optionally for one team or player, one kind, or containing keywords.",
          {"team": _TEAM, "player": _PLAYER, "kind": {"type": "string", "enum": list(news.KINDS)}, "query": {"type": "string"}}),
    _tool("injury_report", "Injury report: one player's status with his next game, one team's list, every listed player in a "
          "date's games, or (no arguments) the latest league updates.", {"player": _PLAYER, "team": _TEAM, "date": _DATE}),
    _tool("roster_moves", "Recent signings, trades, waivers and other transactions.", {"team": _TEAM, "player": _PLAYER}),
    _tool("games", "Games on a date (start time, status, scores, spread, total, moneyline), or a team's next game when only "
          "team is given.", {"date": _DATE, "team": _TEAM}),
    _tool("team_profile", "A team's record and seed, head coach, playing style, key league ranks, play types, injuries and next game.",
          {"team": _TEAM}, ["team"]),
    _tool("coaching_staff", "A team's head coach and assistant coaches.", {"team": _TEAM}, ["team"]),
    _tool("team_rankings", "The league's top (best) or bottom (worst) five teams in a metric.",
          {"metric": {"type": "string", "enum": list(METRIC_KEYS)}, "order": {"type": "string", "enum": ["best", "worst"]}}, ["metric"]),
    _tool("standings", "Standings with record, seed and games back, for one conference or both.",
          {"conference": {"type": "string", "enum": ["East", "West"]}}),
    _tool("team_roster", "Current players on a team with any injury status.", {"team": _TEAM}, ["team"]),
    _tool("player_profile", "A player's bio (team, position, height, weight, age, experience, draft, school), injury status and next game.",
          {"player": _PLAYER}, ["player"]),
    _tool("site_help", "How COURTSIDE works: pages, features, the game simulator, accuracy, calibration, data sources, settings, "
          "privacy and betting terms (vig, EV, spreads, parlays, pushes).", {"question": {"type": "string"}}, ["question"]),
    _tool("model_accuracy", "Walk-forward accuracy of the model overall or for one market, and the current calibration status.",
          {"stat": _STAT}),
    _tool("odds_math", "Convert American odds to decimal, fractional and implied probability, with the payout for a stake.",
          {"odds": {"type": "integer"}, "stake": {"type": "number"}}, ["odds"]),
]
LEAGUE_TOOLS = {"nba_news", "injury_report", "roster_moves", "games", "team_profile", "coaching_staff", "team_rankings", "standings",
                "team_roster", "player_profile", "site_help", "model_accuracy", "odds_math"}
