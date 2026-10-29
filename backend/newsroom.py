"""The newsroom service: cached NBA feeds (headlines, injuries, roster moves, coaching staffs, team styles and the
scoreboard with game lines) shared by the News API and COURTSIDE Coach.

Every feed is cached in the local store with its own freshness window. When a provider fails, the last saved copy is
returned and marked stale; only when nothing was ever saved does a call raise NewsUnavailableError.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from threading import Lock
from typing import Any, Callable
from zoneinfo import ZoneInfo

from . import news
from .league import current_season, previous_season
from .nba import NBAUnavailableError, normalize_name

EASTERN = ZoneInfo("America/New_York")
TTL = {"headlines": 600, "injuries": 900, "transactions": 1800, "styles": 6 * 3600, "coaches": 24 * 3600}
LOOKAHEAD_DAYS = 10
LOOKBACK_DAYS = 7


def eastern_today() -> date:
    return datetime.now(timezone.utc).astimezone(EASTERN).date()


def fresh(cached: dict | None, seconds: float) -> bool:
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(cached["fetched_at"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return False
    return timedelta(0) <= age < timedelta(seconds=seconds)


def tipoff(game: dict) -> str | None:
    """Start time in US Eastern, e.g. "8:00 PM ET"."""
    try:
        start = datetime.fromisoformat(str(game["start"]).replace("Z", "+00:00")).astimezone(EASTERN)
    except (KeyError, TypeError, ValueError):
        return None
    return start.strftime("%I:%M %p ET").lstrip("0")


def game_day(game: dict) -> date | None:
    try:
        return datetime.fromisoformat(str(game["start"]).replace("Z", "+00:00")).astimezone(EASTERN).date()
    except (KeyError, TypeError, ValueError):
        return None


class Newsroom:
    def __init__(self, store: Any, roster: Callable[[], list[dict]], today: Callable[[], date] | None = None,
                 season: Callable[[], str] = current_season):
        self.store, self.roster = store, roster
        self.today = today or eastern_today
        self.season = season
        self._locks: dict[str, Lock] = {}
        self._guard = Lock()

    # ------------------------------------------------------------------ cache
    def cached(self, key: str, fetcher: Callable[[], dict], ttl: float | Callable[[dict | None], float], refresh: bool = False) -> dict:
        """A fresh copy, a new fetch, or the last saved copy marked stale when the provider fails."""
        with self._guard:
            lock = self._locks.setdefault(key, Lock())
        with lock:
            saved = self.store.resource(key)
            seconds = ttl(saved) if callable(ttl) else ttl
            if saved is not None and not refresh and fresh(saved, seconds):
                return {**saved, "stale": False, "error": None}
            try:
                return {**self.store.save_resource(key, fetcher()), "stale": False, "error": None}
            except (news.NewsUnavailableError, NBAUnavailableError) as exc:
                if saved is not None:
                    return {**saved, "stale": True, "error": f"{exc} Showing the last saved copy."}
                raise news.NewsUnavailableError(str(exc)) from exc

    @staticmethod
    def attempt(call: Callable[[], Any]) -> tuple[Any, str | None]:
        try:
            return call(), None
        except news.NewsUnavailableError as exc:
            return None, str(exc)

    def players(self) -> list[dict]:
        try:
            return self.roster()
        except NBAUnavailableError:
            return []

    # ------------------------------------------------------------------ feeds
    def headlines(self, refresh: bool = False) -> dict:
        return self.cached("news:headlines", lambda: news.fetch_headlines(self.players()), TTL["headlines"], refresh)

    def team_headlines(self, team: str, refresh: bool = False) -> dict:
        return self.cached(f"news:headlines:{team}", lambda: news.fetch_headlines(self.players(), 30, team=team), TTL["headlines"], refresh)

    def injuries(self, refresh: bool = False) -> dict:
        return self.cached("news:injuries", lambda: news.fetch_injuries(self.players()), TTL["injuries"], refresh)

    def transactions(self, refresh: bool = False) -> dict:
        return self.cached("news:transactions", news.fetch_transactions, TTL["transactions"], refresh)

    def style_season(self, which: str) -> str:
        return self.season() if which == "current" else previous_season(self.season())

    def styles(self, which: str = "current", refresh: bool = False) -> dict:
        season = self.style_season(which)
        return self.cached(f"news:styles:{season}", lambda: news.fetch_team_styles(season), TTL["styles"], refresh)

    def coaches(self, refresh: bool = False) -> dict:
        season = self.season()

        def fetch() -> dict:
            def staff(team: dict) -> dict:
                try:
                    return {**news.fetch_coaches(team["id"], season), "team": team["abbreviation"], "error": None}
                except NBAUnavailableError as exc:
                    return {"team_id": team["id"], "team": team["abbreviation"], "season": season, "coaches": [], "error": str(exc)}

            with ThreadPoolExecutor(max_workers=4) as pool:
                staffs = list(pool.map(staff, news.nba_teams()))
            if all(item["error"] for item in staffs):
                raise NBAUnavailableError("NBA Stats returned no coaching staffs. Try again later.")
            return {"season": season, "teams": staffs, "provider": "NBA Stats / CommonTeamRoster coaching staffs"}

        return self.cached(f"news:coaches:{season}", fetch, TTL["coaches"], refresh)

    def scoreboard(self, day: date, refresh: bool = False) -> dict:
        today = self.today()

        def ttl(saved: dict | None) -> float:
            games = (saved or {}).get("games") or []
            if any(game.get("state") == "in" for game in games):
                return 60
            if day < today:
                return 6 * 3600 if games and all(game.get("state") == "post" for game in games) else 300
            return 120 if day == today else 900

        board = self.cached(f"news:scoreboard:{day.isoformat()}", lambda: news.fetch_scoreboard(day), ttl, refresh)
        for game in board.get("games", []):
            game["tipoff"] = tipoff(game)
        return board

    def warm(self) -> None:
        """Fill the caches in the background so the first questions are answered quickly; failures are ignored."""
        for call in (self.injuries, self.headlines, self.transactions, lambda: self.scoreboard(self.today()),
                     lambda: self.scoreboard(self.today() + timedelta(days=1)), self.styles, self.coaches):
            self.attempt(call)

    # ------------------------------------------------------------------ derived views
    def team_games(self, team: str, upcoming: bool = True, days: int | None = None) -> list[dict]:
        """The team's games scanning forward from today (upcoming) or back from today (completed), nearest first."""
        today = self.today()
        days = days if days is not None else (LOOKAHEAD_DAYS if upcoming else LOOKBACK_DAYS)
        found = []
        for offset in range(days + 1):
            day = today + timedelta(days=offset if upcoming else -offset)
            board, _ = self.attempt(lambda day=day: self.scoreboard(day))
            for game in (board or {}).get("games", []):
                if team in (game["home"]["team"], game["away"]["team"]):
                    if upcoming and game["state"] != "post" or not upcoming and game["state"] == "post":
                        found.append({**game, "date": day.isoformat()})
            if found:
                return found
        return found

    def next_game(self, team: str) -> dict | None:
        games = self.team_games(team, upcoming=True)
        return games[0] if games else None

    def last_game(self, team: str) -> dict | None:
        games = self.team_games(team, upcoming=False)
        return games[0] if games else None

    def team_profile(self, team: str, which: str = "current") -> dict | None:
        styles = self.styles(which)
        profile = next((item for item in styles["teams"] if item["team"] == team), None)
        if profile is None:
            return None
        return {**profile, "season": styles["season"], "summary": news.style_sentence(profile), "team_count": styles["team_count"],
                "fetched_at": styles.get("fetched_at"), "stale": styles.get("stale")}

    def head_coach(self, team: str, fetch: bool = True) -> str | None:
        """The head coach; with fetch=False only a saved copy is read (the 30-team staff fetch can take a while)."""
        staffs, _ = self.attempt(self.coaches) if fetch else (self.store.resource(f"news:coaches:{self.season()}"), None)
        staff = next((item for item in (staffs or {}).get("teams", []) if item["team"] == team), None)
        return next((coach["name"] for coach in (staff or {}).get("coaches", []) if coach["head"]), None)

    def standings(self, which: str = "current") -> dict:
        styles = self.styles(which)
        rows = []
        for team in styles["teams"]:
            played = team["wins"] + team["losses"]
            rows.append({"team": team["team"], "name": team["name"], "conference": news.CONFERENCES.get(team["team"]),
                         "wins": team["wins"], "losses": team["losses"], "pct": team["wins"] / played if played else 0.0})
        rows.sort(key=lambda row: (-row["pct"], -row["wins"], row["name"]))
        for conference in ("East", "West"):
            members = [row for row in rows if row["conference"] == conference]
            if members:
                leader = members[0]
                for seed, row in enumerate(members, start=1):
                    row["seed"] = seed
                    row["games_back"] = ((leader["wins"] - row["wins"]) + (row["losses"] - leader["losses"])) / 2
        return {"season": styles["season"], "teams": rows, "fetched_at": styles.get("fetched_at"), "stale": styles.get("stale")}

    def rankings(self, metric: str, which: str = "current", worst: bool = False, limit: int = 5) -> dict:
        styles = self.styles(which)
        rows = [{"team": team["team"], "name": team["name"], **team["metrics"][metric]}
                for team in styles["teams"] if metric in team.get("metrics", {})]
        if not rows:
            raise ValueError(f"No team ranks for {metric!r} this season yet.")
        rows.sort(key=lambda row: row["rank"], reverse=worst)
        return {"season": styles["season"], "metric": metric, "label": rows[0]["label"], "percent": rows[0]["percent"],
                "rows": rows[:limit], "team_count": styles["team_count"], "stale": styles.get("stale")}

    def team_detail(self, code: str, refresh: bool = False) -> dict:
        team = next((item for item in news.nba_teams() if item["abbreviation"] == code), None)
        if team is None:
            raise KeyError(code)
        profiles, errors = {}, []
        for which in ("current", "previous"):
            profile, error = self.attempt(lambda which=which: self.styles(which, refresh))
            if profile:
                found = next((item for item in profile["teams"] if item["team"] == code), None)
                if found:
                    profiles[which] = {**found, "season": profile["season"], "summary": news.style_sentence(found),
                                       "team_count": profile["team_count"], "fetched_at": profile.get("fetched_at"),
                                       "stale": profile.get("stale")}
            elif error:
                errors.append(error)
        staffs, staff_error = self.attempt(lambda: self.coaches(refresh))
        staff = next((item for item in (staffs or {}).get("teams", []) if item["team"] == code), None)
        report, injury_error = self.attempt(lambda: self.injuries(refresh))
        feed, headline_error = self.attempt(lambda: self.team_headlines(code, refresh))
        if feed is None:
            league, _ = self.attempt(lambda: self.headlines(refresh))
            feed = {"articles": [item for item in (league or {}).get("articles", []) if code in item["teams"]]}
        moves, move_error = self.attempt(lambda: self.transactions(refresh))
        return {
            "team": team, "profiles": profiles, "conference": news.CONFERENCES.get(code),
            "coaches": staff["coaches"] if staff else [], "coaches_season": (staffs or {}).get("season"),
            "injuries": next((item["injuries"] for item in (report or {}).get("teams", []) if item["team"] == code), []),
            "headlines": [item for item in feed.get("articles", []) if not item.get("video")][:12],
            "transactions": [item for item in (moves or {}).get("transactions", []) if item["team"] == code][:15],
            "next_game": self.next_game(code),
            "errors": [error for error in (*errors, staff_error, injury_error, headline_error, move_error) if error],
        }

    def injury_entry(self, player_id: str, report: dict | None = None) -> dict | None:
        for team in (report or {}).get("teams", []):
            for entry in team["injuries"]:
                if entry.get("player_id") == player_id:
                    return entry
        return None

    def player_news(self, player: dict, refresh: bool = False) -> dict:
        player_id = player["id"]
        report, injury_error = self.attempt(lambda: self.injuries(refresh))
        entry = self.injury_entry(player_id, report)
        team_entries = next(([item for item in team["injuries"] if item.get("player_id") != player_id]
                             for team in (report or {}).get("teams", []) if team["team"] == player.get("team")), [])
        articles: list[dict] = []
        for call in (lambda: self.headlines(refresh), lambda: self.team_headlines(player["team"], refresh) if player.get("team") in news.ESPN_IDS else None):
            feed, _ = self.attempt(call)
            articles += (feed or {}).get("articles", [])
        name_key = normalize_name(player["name"])
        seen, mentions = set(), []
        for item in sorted(articles, key=lambda item: item.get("published") or "", reverse=True):
            if item["id"] in seen:
                continue
            if any(athlete.get("player_id") == player_id or normalize_name(athlete["name"]) == name_key for athlete in item["athletes"]) \
                    or name_key in normalize_name(item["headline"]):
                seen.add(item["id"])
                mentions.append(item)
        return {"player_id": player_id, "team": player.get("team"), "injury": entry, "teammates": team_entries,
                "headlines": mentions[:6], "report_fetched_at": (report or {}).get("fetched_at"),
                "stale": bool((report or {}).get("stale")), "error": injury_error}
