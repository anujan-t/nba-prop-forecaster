"""COURTSIDE Coach's league desk: answers about injuries, news, roster moves, coaches, playing styles, rankings,
standings, schedules, scores, game lines, rosters, player bios, odds math and the app itself.

Each `data_*` method returns a compact dict (also used as a language-model tool result); the matching `say_*` method
turns it into reply lines and a card for the built-in coach. Facts come only from the newsroom feeds, the roster, the
player profile provider and the knowledge base; nothing is invented.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
import re
from typing import Any

from . import news
from .knowledge import KNOWLEDGE, WALK_FORWARD, Entry
from .newsroom import EASTERN, game_day, tipoff
from .nlu import fold
from .stats import label as stat_label
from .topics import (NICKNAME, TEAM_NAMES, find_day, find_metric, off_topic, rank_direction)

SERIOUS = ("out", "suspended", "doubtful", "questionable", "day_to_day")
METRIC_KEYS = {key: (label, higher, percent) for key, _, label, higher, percent in news.METRICS}
KEY_METRICS = ("pace", "offense", "defense", "net", "three_rate", "opponent_points", "opponent_threes", "forced_turnovers")
SUGGEST_NEWS = ("Who's out tonight?", "Latest NBA news", "What games are on tonight?", "Any trades?")


def ordinal(number: int) -> str:
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def odds_text(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"+{value}" if value > 0 else f"−{abs(value)}"


def line_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value == 0:
        return "PK"
    return f"+{value:g}" if value > 0 else f"−{abs(value):g}"


def ago(stamp: str | None, now: datetime | None = None) -> str:
    try:
        moment = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    minutes = int(((now or datetime.now(timezone.utc)) - moment).total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    if minutes < 48 * 60:
        return f"{minutes // 60}h ago"
    local = moment.astimezone(EASTERN)
    return local.strftime("%b ") + str(local.day)


def day_phrase(day: date, today: date) -> str:
    delta = (day - today).days
    if delta == 0:
        return "tonight"
    if delta == 1:
        return "tomorrow"
    if delta == -1:
        return "last night"
    if 1 < delta < 7:
        return day.strftime("%A")
    return day.strftime("%a, %b ") + str(day.day)


def money(value: float) -> str:
    return f"${value:,.2f}"


def american_to_decimal(odds: int) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def implied(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def fraction(odds: int) -> str:
    """Exact fractional odds: +150 -> 3/2, -135 -> 20/27."""
    value = Fraction(odds, 100) if odds > 0 else Fraction(100, abs(odds))
    return f"{value.numerator}/{value.denominator}"


def status_label(entry: dict) -> str:
    return entry.get("status") or "Listed"


def injury_what(entry: dict) -> str:
    """ "Right knee sprain", "illness", "personal reasons"; ESPN's "Not Specified" placeholders are dropped."""
    def given(value) -> str | None:
        return value if value and str(value).strip().lower() not in ("not specified", "unknown", "other") else None

    injury = given(entry.get("injury"))
    if injury and injury.lower() == "personal":
        return "personal reasons"
    parts = [given(entry.get("side")), injury, (given(entry.get("detail")) or "").lower() or None]
    text = " ".join(part for part in parts if part).strip()
    return text[:1].upper() + text[1:].lower() if text else ""


def short_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        day = date.fromisoformat(value[:10])
    except ValueError:
        return None
    return day.strftime("%b ") + str(day.day) + ("" if day.year == datetime.now().year else f", {day.year}")


class Desk:
    def __init__(self, newsroom: Any, roster: Callable[[], list[dict]], profile: Callable[[str], dict] | None = None,
                 today: Callable[[], date] | None = None, learning: Any = None):
        self.newsroom, self.roster, self.profile, self.learning = newsroom, roster, profile, learning
        self.today = today or (lambda: newsroom.today() if newsroom is not None else date.today())

    # ------------------------------------------------------------------ helpers
    def available(self, turn) -> bool:
        if self.newsroom is None:
            turn.say("The newsroom isn't connected right now, so I can't check news, injuries or the schedule.")
            return False
        return True

    def feed(self, turn, call: Callable[[], dict], what: str) -> dict | None:
        try:
            data = call()
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load {what}: {exc}")
            return None
        if data.get("stale"):
            turn.say(f"_(The {what} provider didn't answer; this is the copy saved {ago(data.get('fetched_at'))}.)_")
        return data

    def team_name(self, code: str) -> str:
        return next((team["name"] for team in news.nba_teams() if team["abbreviation"] == code), code)

    def nickname(self, code: str) -> str:
        return NICKNAME.get(code, code)

    def players_on(self, code: str) -> list[dict]:
        try:
            return [player for player in self.roster() if player.get("team") == code]
        except Exception:  # roster lookups are best-effort here
            return []

    @staticmethod
    def side(game: dict, code: str) -> tuple[dict, dict, bool]:
        """(team side, opponent side, is home) for one team in a game."""
        home = game["home"]["team"] == code
        return (game["home"], game["away"], True) if home else (game["away"], game["home"], False)

    def game_phrase(self, game: dict, code: str | None = None) -> str:
        today = self.today()
        day = game_day(game) or today
        when = day_phrase(day, today)
        time = game.get("tipoff") or tipoff(game)
        if code:
            mine, other, home = self.side(game, code)
            text = f"{when} {'vs' if home else 'at'} the {self.nickname(other['team'])}"
        else:
            text = f"{game['away']['team']} at {game['home']['team']} {when}"
        if game["state"] == "pre" and time:
            text += f" ({time})"
        elif game["state"] == "in":
            text += f" (live: {game['away']['team']} {game['away']['score']}–{game['home']['score']} {game['home']['team']}, {game.get('detail') or ''})".replace(", )", ")")
        return text

    def lines_phrase(self, game: dict) -> str | None:
        odds = game.get("odds")
        if not odds:
            return None
        parts = []
        if odds.get("home_spread") is not None:
            favorite = odds.get("favorite") or (game["home"]["team"] if odds["home_spread"] < 0 else game["away"]["team"])
            spread = odds["home_spread"] if favorite == game["home"]["team"] else odds.get("away_spread")
            parts.append(f"{favorite} {line_text(spread)}" if spread else "pick'em")
        if odds.get("total") is not None:
            parts.append(f"total {odds['total']:g}")
        if odds.get("away_moneyline") is not None and odds.get("home_moneyline") is not None:
            parts.append(f"moneyline {game['away']['team']} {odds_text(odds['away_moneyline'])} / {game['home']['team']} {odds_text(odds['home_moneyline'])}")
        provider = f" ({odds['provider']})" if odds.get("provider") else ""
        return ", ".join(parts) + provider if parts else None

    @staticmethod
    def compact_game(game: dict) -> dict:
        return {key: game.get(key) for key in ("id", "start", "tipoff", "state", "detail", "home", "away", "odds", "broadcast", "note", "date")}

    def games_card(self, title: str, day: date, games: list[dict]) -> dict:
        return {"type": "games", "title": title, "date": day.isoformat(), "games": [self.compact_game(game) for game in games],
                "route": "#/news/games"}

    # ------------------------------------------------------------------ dispatcher
    def answer(self, turn, text: str, players: list[dict], teams: list[str], hits: dict[str, str], odds: int | None = None,
               stake: float | None = None, has_spec: bool = False) -> bool:
        """Answers a league or app question; False when the message isn't one (the caller tries other handlers)."""
        named = bool(players or teams)
        active = {name for name, level in hits.items() if level == "strong" or named}
        t = fold(text)
        if "odds_math" in active and self.say_odds_math(turn, text, odds, stake):
            return True
        if "advice" in active and not has_spec:
            return self.say_advice(turn)
        if self.newsroom is None and active & {"injury", "lines", "scores", "schedule", "coaching", "standings", "roster", "moves", "news", "style", "rank"}:
            self.available(turn)
            return True
        day = find_day(text, self.today())
        if "injury" in active:
            if players:
                for player in players[:3]:
                    self.say_availability(turn, player)
                return True
            if teams:
                for code in teams[:2]:
                    self.say_team_injuries(turn, code)
                return True
            return self.say_injuries_on(turn, day or self.today()) if day or re.search(r"\b(?:tonight|today)\b", t) else self.say_league_injuries(turn)
        if "lines" in active and not has_spec and (teams or players or "lines" in hits and hits["lines"] == "strong"):
            if teams or players:
                return self.say_team_lines(turn, teams or [players[0]["team"]], day)
            return self.say_games(turn, day or self.today(), lines=True)
        if "scores" in active and not has_spec:
            if teams:
                return self.say_team_result(turn, teams[0], day)
            if not players:
                return self.say_scores(turn, day or self.today() - timedelta(days=1))
        if "schedule" in active and not has_spec:
            if teams:
                return self.say_team_schedule(turn, teams, day)
            if players:
                player = players[0]
                self.say_team_schedule(turn, [player["team"]], day, player=player)
                return True
            if hits.get("schedule") == "strong" or day:
                return self.say_games(turn, day or self.today())
        if "coaching" in active and (teams or players or hits.get("coaching") == "strong"):
            if teams or players:
                for code in (teams or [players[0]["team"]])[:2]:
                    self.say_coaches(turn, code)
                return True
            return self.say_head_coaches(turn)
        if "standings" in active:
            if teams:
                for code in teams[:2]:
                    self.say_team_record(turn, code)
                return True
            east, west = bool(re.search(r"\beast\w*\b", t)), bool(re.search(r"\bwest\w*\b", t))
            return self.say_standings(turn, "East" if east and not west else "West" if west and not east else None)
        if "roster" in active and (teams or players):
            return self.say_roster(turn, teams[0] if teams else players[0]["team"])
        if "player_info" in active and players:
            return self.say_bio(turn, players[0])
        if "moves" in active and (hits.get("moves") == "strong" or named):
            return self.say_moves(turn, teams[0] if teams else None, players[0] if players else None)
        if "news" in active and (hits.get("news") == "strong" or named):
            if players:
                return self.say_player_news(turn, players[0])
            return self.say_headlines(turn, teams[0] if teams else None)
        metric = find_metric(text)
        if ("rank" in hits or "style" in hits) and metric and not teams and not players:
            return self.say_rankings(turn, metric, text)
        if teams and not players and not has_spec:
            if len(teams) >= 2:
                return self.say_matchup(turn, teams[0], teams[1])
            if "style" in active and metric and metric in METRIC_KEYS:
                return self.say_team_metric(turn, teams[0], metric)
            return self.say_team(turn, teams[0])
        return False

    # ------------------------------------------------------------------ injuries
    def data_injuries(self, team: str | None = None, player: dict | None = None, day: date | None = None) -> dict:
        report = self.newsroom.injuries()
        teams = report.get("teams", [])
        if player is not None:
            entry = self.newsroom.injury_entry(player["id"], report)
            teammates = next(([e for e in item["injuries"] if e.get("player_id") != player["id"]] for item in teams if item["team"] == player.get("team")), [])
            game = self.newsroom.next_game(player["team"]) if player.get("team") in TEAM_NAMES else None
            return {"player": player["name"], "team": player.get("team"), "listed": entry is not None, "entry": entry,
                    "teammates": teammates, "next_game": game and self.compact_game(game),
                    "fetched_at": report.get("fetched_at"), "stale": report.get("stale")}
        if team is not None:
            entries = next((item["injuries"] for item in teams if item["team"] == team), [])
            return {"team": team, "entries": entries, "fetched_at": report.get("fetched_at"), "stale": report.get("stale")}
        if day is not None:
            board = self.newsroom.scoreboard(day)
            playing = [code for game in board["games"] for code in (game["away"]["team"], game["home"]["team"])]
            entries = [entry for code in playing for item in teams if item["team"] == code for entry in item["injuries"]
                       if entry["status_key"] in SERIOUS]
            return {"date": day.isoformat(), "games": [self.compact_game(game) for game in board["games"]], "entries": entries,
                    "fetched_at": report.get("fetched_at"), "stale": report.get("stale")}
        entries = sorted((entry for item in teams for entry in item["injuries"]), key=lambda entry: entry.get("date") or "", reverse=True)
        return {"counts": report.get("counts", {}), "total": report.get("total", len(entries)), "latest": entries[:12],
                "fetched_at": report.get("fetched_at"), "stale": report.get("stale")}

    def injury_sentence(self, entry: dict, with_team: bool = False) -> str:
        what = injury_what(entry)
        back = short_date(entry.get("return_date"))
        text = f"**{entry['player']}**" + (f" ({entry['team']})" if with_team else "") + f" — {status_label(entry)}"
        if what:
            text += f": {what}"
        if back and entry["status_key"] in ("out", "suspended"):
            text += f"; estimated return {back}"
        return text

    def injury_card(self, title: str, entries: list[dict], data: dict, route: str = "#/news/injuries") -> dict:
        keep = ("player", "player_id", "team", "position", "status", "status_key", "injury", "detail", "side", "return_date", "short_comment", "date")
        return {"type": "injuries", "title": title, "entries": [{key: entry.get(key) for key in keep} for entry in entries[:30]],
                "total": len(entries), "fetched_at": data.get("fetched_at"), "stale": bool(data.get("stale")), "route": route}

    def say_availability(self, turn, player: dict) -> bool:
        data = self.feed(turn, lambda: self.data_injuries(player=player), "injury report")
        if data is None:
            return True
        entry, game = data["entry"], data["next_game"]
        when = f" for {self.game_phrase(game, player['team'])}" if game else ""
        team = f" ({player['team']})" if player.get("team") in TEAM_NAMES else ""
        if entry:
            turn.say(f"{self.injury_sentence(entry, with_team=bool(team))} on the latest injury report.")
            if game:
                turn.say(f"Next game: {self.game_phrase(game, player['team'])}.")
            if entry.get("short_comment"):
                turn.say(entry["short_comment"])
            if entry["status_key"] in ("out", "suspended"):
                turn.say("Any prop on him would likely be voided by the book if he doesn't play; check before you bet.")
            else:
                turn.say("The forecast assumes he plays; if he's limited, lower his expected minutes in Props.")
        else:
            turn.say(f"**{player['name']}**{team} isn't on the injury report, so he's expected to be available{when}.")
        out = [mate for mate in data["teammates"] if mate["status_key"] in SERIOUS]
        if out:
            turn.say("Teammates listed: " + "; ".join(f"{mate['player']} ({status_label(mate)})" for mate in out[:6]) + ".")
        turn.say(f"_Report updated {ago(data.get('fetched_at'))}._")
        turn.cards.append(self.injury_card(f"{player['name']} · {player.get('team') or ''} injury report".strip(),
                                           ([entry] if entry else []) + out, data, f"#/news/teams/{(player.get('team') or '').lower()}"
                                           if player.get("team") in TEAM_NAMES else "#/news/injuries"))
        turn.entities["player_id"] = player["id"]
        turn.suggest(f"What's {player['name']} averaging?", f"Latest news on {player['name']}")
        return True

    def say_team_injuries(self, turn, code: str) -> bool:
        data = self.feed(turn, lambda: self.data_injuries(team=code), "injury report")
        if data is None:
            return True
        entries = data["entries"]
        name = self.team_name(code)
        if not entries:
            turn.say(f"The **{name}** have nobody on the injury report.")
        else:
            turn.say(f"**{name}** injury report ({len(entries)} listed):")
            for entry in entries[:8]:
                turn.say("- " + self.injury_sentence(entry))
            if len(entries) > 8:
                turn.say(f"- …and {len(entries) - 8} more in the card below.")
        game = self.newsroom.next_game(code)
        if game:
            turn.say(f"Next game: {self.game_phrase(game, code)}.")
        turn.cards.append(self.injury_card(f"{name} injury report", entries, data, f"#/news/teams/{code.lower()}"))
        turn.suggest(f"How do the {self.nickname(code)} play?", f"{self.nickname(code)} news", "Who's out tonight?")
        return True

    def say_injuries_on(self, turn, day: date) -> bool:
        data = self.feed(turn, lambda: self.data_injuries(day=day), "injury report")
        if data is None:
            return True
        when = day_phrase(day, self.today())
        if not data["games"]:
            turn.say(f"There are no NBA games {when}.")
            return True
        entries = data["entries"]
        by_team: dict[str, list[dict]] = {}
        for entry in entries:
            by_team.setdefault(entry["team"], []).append(entry)
        turn.say(f"Injuries for {when}'s {len(data['games'])} game{'s' if len(data['games']) != 1 else ''}:")
        for game in data["games"]:
            parts = []
            for side in (game["away"], game["home"]):
                listed = by_team.get(side["team"], [])
                names = ", ".join(f"{entry['player']} ({status_label(entry)})" for entry in listed[:5])
                parts.append(f"{side['team']}: {names or 'none listed'}" + (f" +{len(listed) - 5} more" if len(listed) > 5 else ""))
            turn.say(f"- **{game['away']['team']} at {game['home']['team']}** — " + " · ".join(parts))
        turn.say(f"_Report updated {ago(data.get('fetched_at'))}. Statuses are as reported, not official league filings._")
        turn.cards.append(self.injury_card(f"Injuries for {when}'s games", entries, data))
        turn.suggest("What are the lines tonight?", "Latest NBA news")
        return True

    def say_league_injuries(self, turn) -> bool:
        data = self.feed(turn, lambda: self.data_injuries(), "injury report")
        if data is None:
            return True
        counts = data["counts"]
        summary = ", ".join(f"{count} {key.replace('_', '-')}" for key, count in sorted(counts.items(), key=lambda item: -item[1]))
        turn.say(f"The league injury report lists **{data['total']} players** ({summary}). Most recent updates:")
        for entry in data["latest"][:6]:
            turn.say("- " + self.injury_sentence(entry, with_team=True))
        turn.cards.append(self.injury_card("Latest injury updates", data["latest"], data))
        turn.suggest("Who's out tonight?", "Lakers injuries", "Is LeBron James playing tonight?")
        return True

    # ------------------------------------------------------------------ schedule, scores and lines
    def data_games(self, day: date, team: str | None = None) -> dict:
        board = self.newsroom.scoreboard(day)
        games = [game for game in board["games"] if team is None or team in (game["home"]["team"], game["away"]["team"])]
        return {"date": day.isoformat(), "games": [self.compact_game(game) for game in games], "stale": board.get("stale"),
                "fetched_at": board.get("fetched_at")}

    def next_slate(self, start: date, days: int = 7) -> tuple[date, list[dict]]:
        for offset in range(days + 1):
            day = start + timedelta(days=offset)
            board = self.newsroom.scoreboard(day)
            if board["games"]:
                return day, board["games"]
        return start, []

    def game_line(self, game: dict, show_day: bool = True) -> str:
        today = self.today()
        if game["state"] == "post":
            away, home = game["away"], game["home"]
            return f"**{away['team']} {away['score']}, {home['team']} {home['score']}** ({game.get('detail') or 'Final'})"
        if game["state"] == "in":
            return f"**{game['away']['team']} {game['away']['score']}, {game['home']['team']} {game['home']['score']}** — live, {game.get('detail') or ''}".rstrip(", ")
        when = game.get("tipoff") or ""
        day = game_day(game)
        if show_day and day and day != today:
            when = f"{day_phrase(day, today)} {when}".strip()
        lines = self.lines_phrase(game)
        text = f"**{game['away']['team']} at {game['home']['team']}** — {when}"
        if lines:
            text += f" · {lines}"
        if game.get("broadcast") and "League Pass" not in game["broadcast"].split(",")[0]:
            text += f" · {game['broadcast'].split(',')[0]}"
        return text

    def say_games(self, turn, day: date, lines: bool = False) -> bool:
        try:
            board = self.newsroom.scoreboard(day)
            games = board["games"]
            if not games and day >= self.today():
                day, games = self.next_slate(day)
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load the schedule: {exc}")
            return True
        when = day_phrase(day, self.today())
        if not games:
            turn.say("I don't see any NBA games in the next week.")
            return True
        label = "game" if len(games) == 1 else "games"
        intro = f"{len(games)} {label} {when}" if when in ("tonight", "tomorrow", "last night") else f"{len(games)} {label} on {when}"
        turn.say(intro[0].upper() + intro[1:] + (" with the current lines:" if lines else ":"))
        for game in games[:15]:
            turn.say("- " + self.game_line(game, show_day=False))
        if lines and not any(game.get("odds") for game in games):
            turn.say("No lines are posted for these games yet; books usually post them a day or two ahead.")
        turn.say("_Game lines are the sportsbook ESPN lists, for reference; the model prices player props, not game sides._" if lines else "")
        turn.cards.append(self.games_card(f"Games {when}" if when in ("tonight", "tomorrow", "last night") else f"Games on {when}", day, games))
        turn.suggest("Who's out tonight?" if day == self.today() else "", "What's the spread for the first game?" if not lines else "", "Latest NBA news")
        return True

    def say_scores(self, turn, day: date) -> bool:
        try:
            board = self.newsroom.scoreboard(day)
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load the scores: {exc}")
            return True
        games = board["games"]
        when = day_phrase(day, self.today())
        if not games:
            turn.say(f"There were no NBA games {when}." if day < self.today() else f"No NBA games {when}.")
            return True
        finished = [game for game in games if game["state"] != "pre"]
        if not finished:
            return self.say_games(turn, day)
        turn.say(f"Scores {when}:" if when in ("tonight", "last night") else f"Scores from {when}:")
        for game in finished:
            turn.say("- " + self.game_line(game))
        turn.cards.append(self.games_card(f"Scores {when}", day, finished))
        turn.suggest("Grade my bets", "Latest NBA news")
        return True

    def say_team_result(self, turn, code: str, day: date | None) -> bool:
        try:
            if day is not None:
                game = next((g for g in self.newsroom.scoreboard(day)["games"] if code in (g["home"]["team"], g["away"]["team"])), None)
            else:
                game = self.newsroom.last_game(code)
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load the scores: {exc}")
            return True
        name = self.team_name(code)
        if game is None:
            turn.say(f"I don't see a recent {name} game{' on that day' if day else ' in the last week'}.")
            return True
        if game["state"] == "pre":
            turn.say(f"The {self.nickname(code)} haven't played yet: {self.game_phrase(game, code)}.")
        else:
            mine, other, home = self.side(game, code)
            result = "won" if (mine.get("winner") or (mine["score"] or 0) > (other["score"] or 0)) else "lost"
            verb = "lead" if game["state"] == "in" and (mine["score"] or 0) > (other["score"] or 0) else "trail" if game["state"] == "in" else result
            day_text = day_phrase(game_day(game) or self.today(), self.today())
            if game["state"] == "in":
                turn.say(f"The **{self.nickname(code)}** {verb} the {self.nickname(other['team'])} {mine['score']}–{other['score']} ({game.get('detail')}).")
            else:
                turn.say(f"The **{self.nickname(code)}** {result} {mine['score']}–{other['score']} {'vs' if home else 'at'} the "
                         f"{self.nickname(other['team'])} {day_text} ({game.get('detail') or 'Final'}); they're now {mine.get('record') or 'n/a'}.")
        turn.cards.append(self.games_card(f"{name}", game_day(game) or self.today(), [game]))
        nxt = self.newsroom.next_game(code)
        if nxt and nxt["id"] != game["id"]:
            turn.say(f"Next: {self.game_phrase(nxt, code)}.")
        turn.suggest(f"{self.nickname(code)} news", f"{self.nickname(code)} injuries")
        return True

    def find_meeting(self, a: str, b: str) -> dict | None:
        for game in self.newsroom.team_games(a, upcoming=True):
            if b in (game["home"]["team"], game["away"]["team"]):
                return game
        return None

    def say_team_schedule(self, turn, teams: list[str], day: date | None, player: dict | None = None) -> bool:
        code = teams[0]
        try:
            if len(teams) >= 2:
                game = self.find_meeting(teams[0], teams[1])
                if game is None:
                    turn.say(f"The {self.nickname(teams[0])} and {self.nickname(teams[1])} don't meet in the next {10} days.")
                    return self.say_matchup(turn, teams[0], teams[1])
            elif day is not None:
                game = next((g for g in self.newsroom.scoreboard(day)["games"] if code in (g["home"]["team"], g["away"]["team"])), None)
                if game is None:
                    turn.say(f"The **{self.team_name(code)}** don't play {day_phrase(day, self.today())}.")
                    game = self.newsroom.next_game(code)
                    if game is None:
                        return True
            else:
                game = self.newsroom.next_game(code)
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load the schedule: {exc}")
            return True
        if game is None:
            turn.say(f"I don't see a {self.team_name(code)} game in the next 10 days.")
            return True
        subject = f"**{player['name']}** and the {self.nickname(code)}" if player else f"The **{self.team_name(code)}**"
        verb = "are playing" if game["state"] == "in" else "play"
        turn.say(f"{subject} {verb} {self.game_phrase(game, code)}" + (f" on {game['broadcast'].split(',')[0]}" if game.get("broadcast") and "League Pass" not in game["broadcast"].split(",")[0] else "") + ".")
        lines = self.lines_phrase(game)
        if lines:
            turn.say(f"Lines: {lines}.")
        if game.get("note"):
            turn.say(f"{game['note']}.")
        turn.cards.append(self.games_card("Next game", game_day(game) or self.today(), [game]))
        if player:
            turn.suggest(f"Is {player['name']} playing tonight?", f"What's {player['name']} averaging?")
        else:
            turn.suggest(f"{self.nickname(code)} injuries", f"How do the {self.nickname(code)} play?")
        return True

    def say_team_lines(self, turn, teams: list[str], day: date | None) -> bool:
        code = teams[0]
        try:
            if len(teams) >= 2:
                game = self.find_meeting(teams[0], teams[1])
            elif day is not None:
                game = next((g for g in self.newsroom.scoreboard(day)["games"] if code in (g["home"]["team"], g["away"]["team"])), None)
            else:
                game = self.newsroom.next_game(code)
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load the lines: {exc}")
            return True
        if game is None:
            turn.say(f"I don't see an upcoming {self.team_name(code)} game{' on that day' if day else ''}, so there's no line to show.")
            return True
        odds = game.get("odds")
        if not odds:
            turn.say(f"No line is posted yet for {self.game_phrase(game, code)}; books usually post them a day or two ahead.")
        else:
            mine, other, home = self.side(game, code)
            spread = odds["home_spread"] if home else odds.get("away_spread")
            ml = odds["home_moneyline"] if home else odds.get("away_moneyline")
            parts = []
            if spread is not None:
                parts.append(f"spread **{line_text(spread)}**" + (f" ({odds_text(odds['home_spread_odds' if home else 'away_spread_odds'])})" if odds.get('home_spread_odds' if home else 'away_spread_odds') is not None else ""))
            if ml is not None:
                parts.append(f"moneyline **{odds_text(ml)}** ({implied(ml):.0%} implied)")
            if odds.get("total") is not None:
                parts.append(f"total **{odds['total']:g}**")
            role = "favored" if odds.get("favorite") == code else "the underdog" if odds.get("favorite") else "even"
            turn.say(f"The **{self.nickname(code)}** are {role} {self.game_phrase(game, code)}: " + ", ".join(parts) + f" ({odds.get('provider') or 'listed book'}).")
            turn.say("_Game lines are shown for reference. COURTSIDE's model prices player props, not sides or totals._")
        turn.cards.append(self.games_card("Game lines", game_day(game) or self.today(), [game]))
        turn.suggest(f"{self.nickname(code)} injuries", "What are the lines tonight?")
        return True

    # ------------------------------------------------------------------ news and moves
    def data_headlines(self, team: str | None = None, player: dict | None = None, kind: str | None = None, query: str | None = None, limit: int = 6) -> dict:
        if player is not None:
            data = self.newsroom.player_news(player)
            articles = data["headlines"]
        else:
            data = self.newsroom.team_headlines(team) if team else self.newsroom.headlines()
            articles = [item for item in data.get("articles", []) if not item.get("video")]
        if kind:
            articles = [item for item in articles if item.get("kind") == kind]
        if query:
            words = [word for word in re.findall(r"[a-z0-9]+", fold(query)) if len(word) > 2]
            articles = [item for item in articles if any(word in fold(f"{item['headline']} {item.get('description') or ''}") for word in words)]
        keep = ("id", "headline", "description", "published", "url", "kind", "teams", "byline")
        return {"articles": [{key: item.get(key) for key in keep} for item in articles[:limit]], "team": team,
                "player": player and player["name"], "fetched_at": data.get("fetched_at") or data.get("report_fetched_at"),
                "stale": data.get("stale")}

    def news_card(self, title: str, data: dict, route: str) -> dict:
        return {"type": "news", "title": title, "articles": data["articles"], "fetched_at": data.get("fetched_at"),
                "stale": bool(data.get("stale")), "route": route}

    def say_headlines(self, turn, team: str | None = None, kind: str | None = None) -> bool:
        data = self.feed(turn, lambda: self.data_headlines(team=team, kind=kind), "headlines")
        if data is None:
            return True
        name = self.team_name(team) if team else "NBA"
        if not data["articles"]:
            turn.say(f"No recent {name} headlines.")
            return True
        turn.say(f"Latest {name} headlines:")
        for item in data["articles"][:5]:
            turn.say(f"- **{item['headline']}**" + (f" ({ago(item.get('published'))})" if item.get("published") else ""))
        turn.cards.append(self.news_card(f"{name} headlines", data, f"#/news/teams/{team.lower()}" if team else "#/news"))
        if team:
            turn.suggest(f"{self.nickname(team)} injuries", f"{self.nickname(team)} next game", f"How do the {self.nickname(team)} play?")
        else:
            turn.suggest("Who's out tonight?", "Any trades?", "What games are on tonight?")
        return True

    def say_player_news(self, turn, player: dict) -> bool:
        data = self.feed(turn, lambda: self.data_headlines(player=player), "news")
        if data is None:
            return True
        self.say_availability(turn, player)
        if data["articles"]:
            turn.say(f"Recent headlines mentioning {player['name']}:")
            for item in data["articles"][:4]:
                turn.say(f"- **{item['headline']}**" + (f" ({ago(item.get('published'))})" if item.get("published") else ""))
            turn.cards.append(self.news_card(f"{player['name']} in the news", data, f"#/players/{player['id']}/props"))
        else:
            turn.say(f"No recent headlines mention {player['name']}.")
        return True

    def data_moves(self, team: str | None = None, player: dict | None = None, limit: int = 10) -> dict:
        data = self.newsroom.transactions()
        rows = data.get("transactions", [])
        if team:
            rows = [row for row in rows if row.get("team") == team]
        if player is not None:
            last = fold(player["name"]).split()[-1]
            rows = [row for row in rows if last in fold(row["description"])]
        return {"transactions": rows[:limit], "team": team, "fetched_at": data.get("fetched_at"), "stale": data.get("stale")}

    def say_moves(self, turn, team: str | None, player: dict | None) -> bool:
        data = self.feed(turn, lambda: self.data_moves(team, player), "roster moves")
        if data is None:
            return True
        rows = data["transactions"]
        subject = player["name"] if player else self.team_name(team) if team else "the league"
        if not rows:
            turn.say(f"No recent roster moves for {subject} in ESPN's transactions feed.")
            if team:
                self.say_headlines(turn, team, kind="transaction")
            return True
        turn.say(f"Recent roster moves for {subject}:")
        for row in rows[:8]:
            turn.say(f"- {short_date(row.get('date')) or ''} · {row.get('team') or ''}: {row['description']}".replace(" ·  :", ":"))
        turn.cards.append({"type": "moves", "title": f"Roster moves · {subject}", "transactions": rows,
                           "route": f"#/news/teams/{team.lower()}" if team else "#/news/moves", "fetched_at": data.get("fetched_at"),
                           "stale": bool(data.get("stale"))})
        turn.suggest("Latest NBA news", "Who's out tonight?")
        return True

    # ------------------------------------------------------------------ teams
    def data_team(self, code: str) -> dict:
        profile = self.newsroom.team_profile(code)
        coach = self.newsroom.head_coach(code, fetch=False)
        report, _ = self.newsroom.attempt(self.newsroom.injuries)
        out = [entry for item in (report or {}).get("teams", []) if item["team"] == code for entry in item["injuries"]
               if entry["status_key"] in SERIOUS]
        standings, _ = self.newsroom.attempt(self.newsroom.standings)
        row = next((item for item in (standings or {}).get("teams", []) if item["team"] == code), None)
        game = self.newsroom.next_game(code)
        metrics = []
        if profile:
            for key in KEY_METRICS:
                if key in profile["metrics"]:
                    metrics.append({"key": key, **profile["metrics"][key]})
        return {"team": code, "name": self.team_name(code), "season": profile and profile["season"],
                "record": row and {"wins": row["wins"], "losses": row["losses"], "seed": row.get("seed"), "conference": row["conference"],
                                   "games_back": row.get("games_back")},
                "head_coach": coach, "summary": profile and profile["summary"], "tags": (profile or {}).get("tags", []),
                "metrics": metrics, "play_types": [{key: item[key] for key in ("label", "frequency", "points_per_possession", "frequency_rank")}
                                                   for item in (profile or {}).get("play_types", [])[:4]],
                "injuries": [{key: entry.get(key) for key in ("player", "status", "status_key", "injury")} for entry in out],
                "next_game": game and self.compact_game(game), "team_count": profile and profile["team_count"]}

    def team_card(self, data: dict) -> dict:
        return {"type": "team", **{key: data[key] for key in ("team", "name", "season", "record", "head_coach", "summary", "tags",
                                                              "metrics", "play_types", "injuries", "next_game", "team_count")},
                "route": f"#/news/teams/{data['team'].lower()}"}

    def say_team(self, turn, code: str) -> bool:
        try:
            data = self.data_team(code)
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load the {self.team_name(code)} profile: {exc}")
            return True
        record = data["record"]
        head = f"**{data['name']}**"
        if record:
            head += f" ({record['wins']}–{record['losses']}, {ordinal(record['seed'])} in the {record['conference']})" if record.get("seed") else f" ({record['wins']}–{record['losses']})"
        if data["head_coach"]:
            head += f", coached by {data['head_coach']}"
        turn.say(head + ".")
        if data["summary"]:
            turn.say(f"Style: {data['summary']}")
        if data["tags"]:
            turn.say("Identity: " + ", ".join(data["tags"][:5]) + ".")
        if data["next_game"]:
            lines = self.lines_phrase(data["next_game"])
            turn.say(f"Next: {self.game_phrase(data['next_game'], code)}" + (f" — {lines}" if lines else "") + ".")
        if data["injuries"]:
            turn.say("Injuries: " + ", ".join(f"{entry['player']} ({entry['status']})" for entry in data["injuries"][:6]) + ".")
        turn.cards.append(self.team_card(data))
        nick = self.nickname(code)
        turn.suggest(f"{nick} news", f"Who coaches the {nick}?", f"{nick} roster", f"What's the spread for the {nick} game?")
        return True

    def say_team_metric(self, turn, code: str, metric: str) -> bool:
        try:
            profile = self.newsroom.team_profile(code)
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load team stats: {exc}")
            return True
        if not profile or metric not in profile["metrics"]:
            return self.say_team(turn, code)
        item = profile["metrics"][metric]
        value = f"{item['value']:.1%}" if item["percent"] else f"{item['value']:.1f}"
        turn.say(f"The **{self.team_name(code)}** rank **{ordinal(item['rank'])} of {profile['team_count']}** in {item['label'].lower()} "
                 f"({value}) this season ({profile['season']}).")
        if profile["summary"]:
            turn.say(f"Overall: {profile['summary']}")
        turn.cards.append(self.team_card(self.data_team(code)))
        return True

    def say_matchup(self, turn, a: str, b: str) -> bool:
        try:
            first, second = self.data_team(a), self.data_team(b)
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load team stats: {exc}")
            return True
        game = self.find_meeting(a, b)
        if game:
            lines = self.lines_phrase(game)
            turn.say(f"**{self.nickname(game['away']['team'])} at {self.nickname(game['home']['team'])}**, {self.game_phrase(game)}" + (f" — {lines}" if lines else "") + ".")
        by_key = lambda data: {item["key"]: item for item in data["metrics"]}  # noqa: E731
        ma, mb = by_key(first), by_key(second)
        for key in ("pace", "offense", "defense", "three_rate", "opponent_threes"):
            if key in ma and key in mb:
                label = ma[key]["label"]
                turn.say(f"- {label}: {a} {ordinal(ma[key]['rank'])} · {b} {ordinal(mb[key]['rank'])}")
        if "pace" in ma and "pace" in mb:
            fast = (ma["pace"]["rank"] + mb["pace"]["rank"]) / 2
            if fast <= 10:
                turn.say("Two fast teams: more possessions usually means more counting stats for both sides.")
            elif fast >= 21:
                turn.say("Two slow teams: fewer possessions, which trims volume for everyone.")
        turn.cards.append({"type": "matchup", "a": self.team_card(first), "b": self.team_card(second), "game": game and self.compact_game(game)})
        turn.suggest(f"{self.nickname(a)} injuries", f"{self.nickname(b)} injuries")
        return True

    def say_coaches(self, turn, code: str) -> bool:
        try:
            staffs = self.newsroom.coaches()
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load coaching staffs: {exc}")
            return True
        staff = next((item for item in staffs.get("teams", []) if item["team"] == code), None)
        coaches = (staff or {}).get("coaches", [])
        name = self.team_name(code)
        if not coaches:
            turn.say(f"NBA Stats didn't return a coaching staff for the {name}.")
            return True
        head = next((coach for coach in coaches if coach["head"]), None)
        assistants = [coach["name"] for coach in coaches if not coach["head"]]
        text = f"**{head['name']}** is the head coach of the {name}" if head else f"The {name} staff"
        if assistants:
            text += f"; the staff includes {', '.join(assistants[:6])}" + (f" and {len(assistants) - 6} more" if len(assistants) > 6 else "")
        turn.say(text + f" ({staffs.get('season')}).")
        try:
            profile = self.newsroom.team_profile(code)
        except news.NewsUnavailableError:
            profile = None
        if profile and profile.get("summary"):
            turn.say(f"How they play: {profile['summary']}")
        turn.cards.append({"type": "coaches", "team": code, "name": name, "season": staffs.get("season"), "coaches": coaches,
                           "route": f"#/news/teams/{code.lower()}"})
        turn.suggest(f"How do the {self.nickname(code)} play?", f"{self.nickname(code)} news")
        return True

    def say_head_coaches(self, turn) -> bool:
        try:
            staffs = self.newsroom.coaches()
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load coaching staffs: {exc}")
            return True
        rows = []
        for item in sorted(staffs.get("teams", []), key=lambda item: item["team"]):
            head = next((coach["name"] for coach in item.get("coaches", []) if coach["head"]), None)
            if head:
                rows.append(f"{item['team']} {head}")
        turn.say(f"Head coaches ({staffs.get('season')}): " + "; ".join(rows) + ".")
        turn.suggest("Who coaches the Lakers?", "How do the Celtics play?")
        return True

    # ------------------------------------------------------------------ league tables
    def data_rankings(self, metric: str, best_first: bool = True, limit: int = 5) -> dict:
        data = self.newsroom.rankings(metric, worst=not best_first, limit=limit)
        return {key: data[key] for key in ("season", "metric", "label", "percent", "rows", "team_count", "stale")}

    def say_rankings(self, turn, metric: str, text: str) -> bool:
        label, higher, _ = METRIC_KEYS[metric]
        best_first = rank_direction(text, metric, higher)
        try:
            data = self.data_rankings(metric, best_first)
        except (news.NewsUnavailableError, ValueError) as exc:
            turn.say(f"I couldn't load team rankings: {exc}")
            return True
        direction = "top" if best_first else "bottom"
        turn.say(f"{label}, {data['season']} — {direction} five:")
        for row in data["rows"]:
            value = f"{row['value']:.1%}" if row["percent"] else f"{row['value']:.1f}"
            turn.say(f"- {ordinal(row['rank'])}. **{row['name']}** — {value}")
        if metric.startswith("opponent") or metric == "defense":
            turn.say("Rank 1 = allows the least. Weak defenses are friendlier matchups for player overs.")
        turn.cards.append({"type": "ranking", "title": f"{label} · {direction} 5", "metric": metric, "label": label,
                           "percent": data["percent"], "season": data["season"], "rows": data["rows"], "team_count": data["team_count"],
                           "route": "#/news/teams"})
        turn.suggest("Which team plays the fastest?", "Who gives up the most threes?", "Worst defense in the league?")
        return True

    def data_standings(self, conference: str | None = None) -> dict:
        data = self.newsroom.standings()
        rows = [row for row in data["teams"] if conference is None or row["conference"] == conference]
        return {"season": data["season"], "teams": rows, "stale": data.get("stale")}

    def say_standings(self, turn, conference: str | None) -> bool:
        try:
            data = self.data_standings(conference)
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load the standings: {exc}")
            return True
        for conf in ([conference] if conference else ["East", "West"]):
            rows = [row for row in data["teams"] if row["conference"] == conf]
            top = ", ".join(f"{row['seed']}. {row['team']} {row['wins']}–{row['losses']}" for row in rows[:6])
            turn.say(f"**{conf}**: {top}.")
        turn.cards.append({"type": "standings", "season": data["season"], "teams": data["teams"], "route": "#/news/teams"})
        turn.suggest("Best record in the league?", "What games are on tonight?")
        return True

    def say_team_record(self, turn, code: str) -> bool:
        try:
            data = self.data_standings()
        except news.NewsUnavailableError as exc:
            turn.say(f"I couldn't load the standings: {exc}")
            return True
        row = next((item for item in data["teams"] if item["team"] == code), None)
        if row is None:
            turn.say(f"No record for the {self.team_name(code)} yet.")
            return True
        gap = row.get("games_back") or 0
        back = "leading" if not gap else f"{gap:g} game{'s' if gap != 1 else ''} back"
        turn.say(f"The **{self.team_name(code)}** are **{row['wins']}–{row['losses']}**, {ordinal(row['seed'])} in the {row['conference']} ({back}).")
        return True

    # ------------------------------------------------------------------ players
    def say_roster(self, turn, code: str) -> bool:
        players = self.players_on(code)
        name = self.team_name(code)
        if not players:
            turn.say(f"I couldn't load the {name} roster.")
            return True
        report, _ = self.newsroom.attempt(self.newsroom.injuries) if self.newsroom else (None, None)
        status = {entry.get("player_id"): entry["status"] for item in (report or {}).get("teams", []) for entry in item["injuries"]}
        names = [player["name"] + (f" ({status[player['id']]})" if player["id"] in status else "") for player in players]
        turn.say(f"**{name}** roster ({len(players)} players): " + ", ".join(names) + ".")
        turn.say("Starting lineups aren't in the feeds; check injury statuses before tip-off.")
        turn.cards.append({"type": "link", "label": f"Open {name}", "route": f"#/news/teams/{code.lower()}"})
        return True

    def data_bio(self, player: dict) -> dict:
        bio = self.profile(player["id"]) if self.profile else {}
        keep = ("position", "jersey", "height", "weight", "age", "birth_date", "experience", "country", "school", "draft_year", "draft_round", "draft_pick")
        return {"id": player["id"], "name": player["name"], "team": player.get("team"), "team_name": player.get("team_name"),
                **{key: bio.get(key) for key in keep}}

    def say_bio(self, turn, player: dict, quiet: bool = False) -> bool:
        try:
            data = self.data_bio(player)
        except Exception as exc:  # the profile provider is NBA Stats; any failure is reported, not raised
            if not quiet:
                turn.say(f"I couldn't load {player['name']}'s profile: {exc}")
            return True
        parts = [part for part in (data.get("team_name") or data.get("team"), data.get("position"),
                                   data.get("height") and f"{data['height'].replace('-', ' ft ')} in",
                                   data.get("weight") and f"{data['weight']} lb",
                                   data.get("age") and f"{data['age']} years old") if part]
        turn.say(f"**{data['name']}** — " + " · ".join(parts) + ".")
        extra = []
        if data.get("experience") is not None:
            extra.append("rookie" if data["experience"] == 0 else f"{data['experience']} season{'s' if data['experience'] != 1 else ''} of experience")
        if data.get("draft_year") and str(data["draft_year"]).isdigit():
            extra.append(f"drafted {data['draft_year']}, round {data.get('draft_round')}, pick {data.get('draft_pick')}")
        elif data.get("draft_year"):
            extra.append("undrafted")
        if data.get("school"):
            extra.append(f"from {data['school']}")
        if data.get("country") and data["country"] != "USA":
            extra.append(data["country"])
        if extra:
            sentence = ", ".join(extra)
            turn.say(sentence[0].upper() + sentence[1:] + ".")
        turn.cards.append({"type": "link", "label": f"Open {player['name']}", "route": f"#/players/{player['id']}/career"})
        turn.entities["player_id"] = player["id"]
        turn.suggest(f"What's {player['name']} averaging?", f"Is {player['name']} playing tonight?")
        return True

    # ------------------------------------------------------------------ betting math and advice
    def data_odds(self, odds: int, stake: float | None = None) -> dict:
        decimal = american_to_decimal(odds)
        out = {"american": odds, "decimal": round(decimal, 3), "fractional": fraction(odds), "implied": implied(odds)}
        if stake:
            out.update({"stake": stake, "profit": round(stake * (decimal - 1), 2), "return": round(stake * decimal, 2)})
        return out

    def say_odds_math(self, turn, text: str, odds: int | None, stake: float | None) -> bool:
        source = None
        if odds is None:
            t = fold(text)
            match = re.search(r"(?<![\d/.])(\d{1,3})\s*/\s*(\d{1,3})(?![\d/])", t)
            decimal = None
            if match and int(match.group(2)) > 0:
                decimal, source = 1 + int(match.group(1)) / int(match.group(2)), f"Fractional {match.group(1)}/{match.group(2)}"
            else:
                match = re.search(r"(?<![\d.$])(\d{1,3}(?:\.\d{1,3})?)(?![\d.%])", t)
                if match and re.search(r"\b(?:decimal|american|odds)\b", t):
                    decimal, source = float(match.group(1)), f"Decimal {float(match.group(1)):.2f}"
            if decimal is None or not 1.01 <= decimal <= 101:
                return False
            odds = int(round((decimal - 1) * 100)) if decimal >= 2 else int(round(-100 / (decimal - 1)))
        data = self.data_odds(odds, stake)
        if source:
            turn.say(f"{source} = **{odds_text(odds)}** in American odds.")
        win = 100 * (data["decimal"] - 1)
        turn.say(f"**{odds_text(odds)}** = decimal {data['decimal']:.2f} = fractional {data['fractional']}; it needs **{data['implied']:.1%}** to break even.")
        if odds > 0:
            turn.say(f"A $100 bet wins {money(win)} profit.")
        else:
            turn.say(f"You risk {money(abs(odds))} to win $100.")
        if stake:
            turn.say(f"On a {money(stake)} stake: {money(data['profit'])} profit, {money(data['return'])} back in total.")
        turn.suggest("What's the vig?", "What's EV?")
        return True

    def say_advice(self, turn) -> bool:
        turn.say("I don't hand out locks: no bet is a sure thing, and I won't tell you what to wager. What I can do is price any prop "
                 "you're looking at against the book's odds, e.g. **Tatum over 27.5 points at −115**, and show you what the model, "
                 "the injury report and the matchup say.")
        if self.newsroom is not None:
            try:
                day, games = self.next_slate(self.today(), 2)
                if games:
                    when = day_phrase(day, self.today())
                    turn.say(f"{len(games)} game{'s' if len(games) != 1 else ''} {when}; ask about any player in them.")
                    turn.cards.append(self.games_card(f"Games {when}", day, games))
            except news.NewsUnavailableError:
                pass
        turn.suggest("Who's out tonight?", "Jokić over 26.5 points?", "What's EV?")
        return True

    # ------------------------------------------------------------------ the app and the scope
    def say_entry(self, turn, entry: Entry) -> None:
        turn.say(entry.text)
        if entry.id == "accuracy" and self.learning is not None:
            try:
                turn.say(self.learning.status()["calibration"]["message"])
            except Exception:  # status is a bonus
                pass
        if entry.route:
            turn.cards.append({"type": "link", "label": entry.route_label or "Open", "route": entry.route})
        follow = {"simulator": "How accurate is the model?", "accuracy": "How does calibration work?", "calibration": "Are you getting smarter?",
                  "parlay": "Why are my legs linked?", "american_odds": "What does +250 mean?", "vig": "What's EV?",
                  "expected_value": "What's the vig?", "news_tab": "Who's out tonight?", "markets": "What's PRA?"}
        turn.suggest(follow.get(entry.id, ""), "What can you do?")

    def site_answer(self, turn, text: str, threshold: float = 3.5) -> bool:
        entry = KNOWLEDGE.best(text, threshold)
        if entry is None:
            return False
        self.say_entry(turn, entry)
        return True

    def data_site_help(self, question: str) -> dict:
        found = KNOWLEDGE.search(question, 2)
        return {"entries": [{"title": entry.title, "text": entry.text, "route": entry.route} for entry, score in found if score >= 2.0]}

    def data_accuracy(self, stat: str | None = None) -> dict:
        out: dict = {"walk_forward": {key: WALK_FORWARD[key] for key in ("players", "games", "predictions", "period", "overall", "joint_pairs")}}
        if stat and stat in WALK_FORWARD["markets"]:
            old, new = WALK_FORWARD["markets"][stat]
            out["market"] = {"stat": stat, "label": stat_label(stat), "old_log_loss": old, "simulator_log_loss": new}
        if self.learning is not None:
            try:
                calibration = self.learning.status()["calibration"]
                out["calibration"] = calibration["message"]
                if stat:
                    entry = next((item for item in calibration.get("stats") or [] if item["stat"] == stat), None)
                    if entry:
                        out["market_calibration"] = {key: entry.get(key) for key in ("choice", "samples", "message")}
            except Exception:  # learning status is optional context
                pass
        return out

    def say_scope(self, turn, kind: str) -> None:
        if kind == "other_sport":
            turn.say("I only cover the NBA: COURTSIDE's data, models and news are basketball-only.")
        else:
            turn.say("I didn't catch that as an NBA or betting question. I'm built for NBA betting research and this app, so ask me "
                     "about players, props, parlays, injuries, news, schedules, lines, teams or how the model works.")
        turn.suggest("What can you do?", "Who's out tonight?", "Latest NBA news", "How accurate is the model?")

    def is_off_topic(self, text: str, named: bool) -> str | None:
        return None if named else off_topic(text)
