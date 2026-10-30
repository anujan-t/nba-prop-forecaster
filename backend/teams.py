"""COURTSIDE's team model for moneylines and point spreads.

Every NBA game this season and last feeds one rating per team. A game's home margin is modeled as home-court advantage
plus the home team's rating minus the away team's, less a back-to-back penalty for a team that played the day before,
fitted by ridge regression with recent games weighted most (half-life 120 days; last season's games count 0.3 as much).
The margin is Normal(mean, 13 points) on whole points with no ties (overtime decides), which prices moneylines, spreads
and pushes, and gives exact joint chances for two legs on the same game. These constants were chosen by walk-forward
tests on 2023-24 to 2025-26, where every game was predicted only from games before it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import math
from threading import Lock
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from requests.exceptions import RequestException
from scipy.stats import norm

from .engine import american_probability, profit_per_unit
from .league import current_season, previous_season
from .nba import NBAUnavailableError

EASTERN = ZoneInfo("America/New_York")
HALF_LIFE_DAYS = 120.0
CARRYOVER = 0.3  # weight per season back
RIDGE = 2.0
SIGMA = 13.0  # points; residual spread of game margins around the model's mean
MAX_MARGIN = 80
MAX_REST = 4
MIN_GAMES = 100  # completed games (both seasons) before the model prices anything
EARLY_GAMES = 10  # until both teams have played this many games, ratings lean on last season
MAX_SPREAD = 60.0
MARKETS = ("moneyline", "spread")
TTL = {"current": 3600, "previous": 7 * 24 * 3600}
PROVIDER = "NBA Stats / LeagueGameLog (team results)"
EARLY_NOTE = ("Early season: these teams have played fewer than 10 games, so the ratings still lean on last season and "
              "don't know about offseason trades, signings or injuries. Treat big gaps with the book with suspicion.")


class GameModelUnavailable(RuntimeError):
    """No team results could be loaded, so moneylines and spreads cannot be priced."""


def eastern_today() -> date:
    return datetime.now(timezone.utc).astimezone(EASTERN).date()


def season_of(day: date) -> str:
    return current_season(day)


# ---------- data ----------

def pair_games(rows: list[dict], season: str) -> list[dict]:
    """One record per completed game from NBA Stats' one-row-per-team log ("BOS vs. NYK" is the home row)."""
    sides: dict[str, dict[str, dict]] = {}
    for row in rows:
        matchup = str(row.get("MATCHUP") or "")
        side = "home" if " vs. " in matchup else "away" if " @ " in matchup else None
        points = row.get("PTS")
        if side is None or points is None or (isinstance(points, float) and math.isnan(points)):
            continue
        sides.setdefault(str(row["GAME_ID"]), {})[side] = row
    games = []
    for game_id, pair in sides.items():
        if "home" not in pair or "away" not in pair:
            continue
        home, away = pair["home"], pair["away"]
        games.append({"game_id": game_id, "date": str(home["GAME_DATE"])[:10], "season": season,
                      "home": str(home["TEAM_ABBREVIATION"]), "away": str(away["TEAM_ABBREVIATION"]),
                      "home_points": int(home["PTS"]), "away_points": int(away["PTS"])})
    games.sort(key=lambda game: (game["date"], game["game_id"]))
    return games


def fetch_team_games(season: str) -> dict:
    """Every completed regular-season game of one season."""
    from nba_api.stats.endpoints import leaguegamelog

    try:
        frame = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation="T",
                                            season_type_all_star="Regular Season", timeout=30).get_data_frames()[0]
    except RequestException as exc:
        raise NBAUnavailableError(f"NBA Stats could not return the {season} team results. The request timed out or was blocked; try again later.") from exc
    except (ValueError, KeyError, IndexError) as exc:
        raise NBAUnavailableError(f"NBA Stats returned an unexpected {season} team results response.") from exc
    return {"season": season, "games": pair_games(frame.to_dict("records"), season), "provider": PROVIDER}


# ---------- ratings ----------

@dataclass
class Ratings:
    as_of: date
    season: str
    teams: dict[str, float]
    hca: float
    b2b: float
    games: int
    current_games: int
    last_played: dict[str, date] = field(default_factory=dict)
    played: dict[str, int] = field(default_factory=dict)  # games this season per team

    def rating(self, team: str) -> float:
        return self.teams.get(team, 0.0)

    def early(self, *teams: str) -> bool:
        return min(self.played.get(team, 0) for team in teams) < EARLY_GAMES


def rest_days(games: list[dict]) -> dict[tuple[str, str], int]:
    """Days since each team's previous game in the same season, capped at MAX_REST (first game: MAX_REST)."""
    last: dict[tuple[str, str], date] = {}
    rest: dict[tuple[str, str], int] = {}
    for game in games:
        day = date.fromisoformat(game["date"])
        for team in (game["home"], game["away"]):
            previous = last.get((team, game["season"]))
            rest[(game["game_id"], team)] = MAX_REST if previous is None else min(MAX_REST, (day - previous).days)
            last[(team, game["season"])] = day
    return rest


def fit_ratings(games: list[dict], as_of: date) -> Ratings:
    """Ratings from games played before ``as_of`` (US Eastern date)."""
    season = season_of(as_of)
    lag = {season: 0, previous_season(season): 1}
    past = [game for game in games if game["date"] < as_of.isoformat() and game["season"] in lag]
    if len(past) < MIN_GAMES:
        raise GameModelUnavailable(f"Only {len(past)} completed games are available before {as_of.isoformat()}; the team model needs {MIN_GAMES}.")
    rest = rest_days(sorted(past, key=lambda game: (game["date"], game["game_id"])))
    teams = sorted({game["home"] for game in past} | {game["away"] for game in past})
    index = {team: position for position, team in enumerate(teams)}
    size = len(teams)
    rows = np.arange(len(past))
    design = np.zeros((len(past), size + 2))
    design[rows, [index[game["home"]] for game in past]] = 1.0
    design[rows, [index[game["away"]] for game in past]] = -1.0
    design[:, size] = 1.0
    design[:, size + 1] = [float(rest[(game["game_id"], game["away"])] == 1) - float(rest[(game["game_id"], game["home"])] == 1)
                           for game in past]
    margin = np.array([game["home_points"] - game["away_points"] for game in past], dtype=float)
    age = np.array([(as_of - date.fromisoformat(game["date"])).days for game in past], dtype=float)
    weight = np.power(0.5, age / HALF_LIFE_DAYS) * np.power(CARRYOVER, [lag[game["season"]] for game in past])
    penalty = np.full(size + 2, RIDGE)
    penalty[size:] = 1e-3
    beta = np.linalg.solve(design.T @ (design * weight[:, None]) + np.diag(penalty), design.T @ (weight * margin))
    last_played: dict[str, date] = {}
    played: dict[str, int] = {}
    for game in past:
        for team in (game["home"], game["away"]):
            last_played[team] = max(last_played.get(team, date.min), date.fromisoformat(game["date"]))
            if game["season"] == season:
                played[team] = played.get(team, 0) + 1
    return Ratings(as_of=as_of, season=season, teams={team: float(beta[index[team]]) for team in teams}, hca=float(beta[size]),
                   b2b=float(beta[size + 1]), games=len(past), current_games=sum(game["season"] == season for game in past),
                   last_played=last_played, played=played)


def expected_margin(ratings: Ratings, home: str, away: str, home_b2b: bool = False, away_b2b: bool = False) -> float:
    """The home team's expected margin."""
    return ratings.hca + ratings.rating(home) - ratings.rating(away) + ratings.b2b * (float(away_b2b) - float(home_b2b))


# ---------- probabilities ----------

MARGINS = np.arange(-MAX_MARGIN, MAX_MARGIN + 1)


def margin_pmf(mean: float, sigma: float = SIGMA) -> np.ndarray:
    """P(home margin = k) for k in MARGINS. A tied regulation score goes to overtime, so the zero mass is shared out
    to ±1 so that P(home wins) = Φ(mean / σ)."""
    upper = norm.cdf((MARGINS + 0.5 - mean) / sigma)
    lower = norm.cdf((MARGINS - 0.5 - mean) / sigma)
    pmf = upper - lower
    pmf[0] += lower[0]
    pmf[-1] += 1 - upper[-1]
    zero = MAX_MARGIN
    tie, pmf[zero] = pmf[zero], 0.0
    if tie > 0:
        above = pmf[zero + 1:].sum()
        share = float(np.clip((norm.cdf(mean / sigma) - above) / tie, 0.0, 1.0))
        pmf[zero + 1] += tie * share
        pmf[zero - 1] += tie * (1 - share)
    return pmf / pmf.sum()


def team_margins(home: bool) -> np.ndarray:
    """The backed team's margin for each home margin in MARGINS."""
    return MARGINS if home else -MARGINS


def leg_outcomes(leg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(wins, pushes) masks over MARGINS for a moneyline or spread leg."""
    margin = team_margins(bool(leg["home"]))
    if leg["market"] == "moneyline":
        return margin > 0, np.zeros(len(MARGINS), dtype=bool)
    line = float(leg["line"])
    return margin + line > 0, margin + line == 0


def joint(pmf: np.ndarray, legs: list[dict], pushes_count: bool = False) -> float:
    """P(every leg wins), or with pushes_count P(no leg loses), for legs on the same game."""
    mask = np.ones(len(MARGINS), dtype=bool)
    for leg in legs:
        wins, pushes = leg_outcomes(leg)
        mask &= (wins | pushes) if pushes_count else wins
    return float(pmf[mask].sum())


def fair_line(team_margin: float) -> float:
    """The model's spread for a team, to the nearest half point (negative when favored)."""
    return round(-team_margin * 2) / 2 + 0.0


def implied(odds: int | None) -> float | None:
    if odds is None:
        return None
    try:
        return american_probability(int(odds))
    except (TypeError, ValueError):
        return None


def two_way(own: int | None, other: int | None) -> dict | None:
    """Break-even and no-vig probabilities for one side of a two-way market."""
    mine, theirs = implied(own), implied(other)
    if mine is None:
        return None
    result = {"odds": int(own), "implied": mine, "no_vig": None, "vig": None}
    if theirs is not None and mine + theirs > 0:
        result.update(no_vig=mine / (mine + theirs), vig=mine + theirs - 1)
    return result


def price_side(probability: float, push: float, book: dict | None) -> dict | None:
    """Model versus the book's price for one side: edge over break-even and expected return per unit (pushes refund)."""
    if not book:
        return None
    loss = max(0.0, 1.0 - probability - push)
    roi = probability * profit_per_unit(book["odds"]) - loss
    settled = probability + loss
    edge = (probability / settled if settled > 1e-12 else 0.0) - book["implied"]
    return {**book, "model": probability, "edge": edge, "expected_roi": roi,
            "verdict": "value" if edge > 0.01 else "house" if edge < -0.01 else "fair"}


def fair_odds(probability: float) -> dict | None:
    if not 0 < probability < 1:
        return None
    american = -100 * probability / (1 - probability) if probability > 0.5 else 100 * (1 - probability) / probability
    return {"decimal": 1 / probability, "american": int(round(american))}


# ---------- service ----------

class GameModel:
    """Cached team results, fitted ratings and priced moneyline/spread legs for the app, Coach and the parlay."""

    def __init__(self, store: Any, today: Callable[[], date] | None = None, schedule: Callable[[date], dict] | None = None,
                 fetch: Callable[[str], dict] | None = None):
        self.store, self.fetch = store, fetch or fetch_team_games
        self.today = today or eastern_today
        self.schedule = schedule
        self._lock = Lock()
        self._ratings: dict[tuple, Ratings] = {}
        self._backtest: dict[tuple, dict] = {}

    # ------------------------------------------------------------------ data
    def season_games(self, season: str, which: str, refresh: bool = False) -> dict:
        key = f"teams:games:{season}"
        saved = self.store.resource(key)
        if saved is not None and not refresh and self._fresh(saved, TTL[which]):
            return {**saved, "stale": False, "error": None}
        try:
            return {**self.store.save_resource(key, self.fetch(season)), "stale": False, "error": None}
        except NBAUnavailableError as exc:
            if saved is not None:
                return {**saved, "stale": True, "error": f"{exc} Using the last saved results."}
            raise

    @staticmethod
    def _fresh(saved: dict, seconds: float) -> bool:
        try:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(saved["fetched_at"])
        except (KeyError, TypeError, ValueError):
            return False
        return timedelta(0) <= age < timedelta(seconds=seconds)

    def games(self, refresh: bool = False) -> tuple[list[dict], dict]:
        """Completed games of this season and last, plus where they came from."""
        season = season_of(self.today())
        games, errors, stamps, stale = [], [], [], False
        for which, label in (("current", season), ("previous", previous_season(season))):
            try:
                payload = self.season_games(label, which, refresh)
            except NBAUnavailableError as exc:
                errors.append(str(exc))
                continue
            games += payload.get("games", [])
            stamps.append(str(payload.get("fetched_at")))
            stale = stale or bool(payload.get("stale"))
            if payload.get("error"):
                errors.append(payload["error"])
        if not games:
            raise GameModelUnavailable(" ".join(errors) or "No team results are available yet.")
        games.sort(key=lambda game: (game["date"], game["game_id"]))
        return games, {"stamps": tuple(stamps), "stale": stale, "errors": errors, "provider": PROVIDER}

    def ratings(self, as_of: date | None = None) -> tuple[Ratings, dict]:
        as_of = min(as_of or self.today(), self.today())
        games, meta = self.games()
        key = (as_of, meta["stamps"])
        with self._lock:
            cached = self._ratings.get(key)
        if cached is None:
            cached = fit_ratings(games, as_of)
            with self._lock:
                if len(self._ratings) > 64:
                    self._ratings.clear()
                self._ratings[key] = cached
        return cached, meta

    def team_margins(self, team: str) -> dict[str, float]:
        """The team's final margin on each date it played this season or last."""
        try:
            games, _ = self.games()
        except GameModelUnavailable:
            return {}
        margins = {}
        for game in games:
            if team == game["home"]:
                margins[game["date"]] = float(game["home_points"] - game["away_points"])
            elif team == game["away"]:
                margins[game["date"]] = float(game["away_points"] - game["home_points"])
        return margins

    # ------------------------------------------------------------------ schedule
    def scoreboard_game(self, game_id: str, day: date) -> dict | None:
        if self.schedule is None:
            return None
        try:
            board = self.schedule(day)
        except Exception:  # noqa: BLE001 - the schedule only enriches a price; the model works without it
            return None
        return next((game for game in board.get("games", []) if str(game.get("id")) == str(game_id)), None)

    def played_day_before(self, team: str, day: date, ratings: Ratings) -> bool:
        """Whether the team plays (or played) the day before: from the schedule when it can be read, else results."""
        before = day - timedelta(days=1)
        if self.schedule is not None:
            try:
                board = self.schedule(before)
                return any(team in (game["home"]["team"], game["away"]["team"]) for game in board.get("games", []))
            except Exception:  # noqa: BLE001 - fall back to completed results
                pass
        return ratings.last_played.get(team) == before

    # ------------------------------------------------------------------ pricing
    def matchup(self, home: str, away: str, day: date) -> dict:
        ratings, meta = self.ratings(day)
        home_b2b = self.played_day_before(home, day, ratings)
        away_b2b = self.played_day_before(away, day, ratings)
        mean = expected_margin(ratings, home, away, home_b2b, away_b2b)
        pmf = margin_pmf(mean)
        home_win = float(pmf[MARGINS > 0].sum())
        return {
            "home": home, "away": away, "date": day.isoformat(), "home_margin": mean, "sigma": SIGMA,
            "home_win": home_win, "away_win": 1 - home_win, "fair_spread": {"home": fair_line(mean), "away": fair_line(-mean)},
            "home_b2b": home_b2b, "away_b2b": away_b2b, "ratings": {"home": ratings.rating(home), "away": ratings.rating(away)},
            "hca": ratings.hca, "b2b_penalty": ratings.b2b, "as_of": ratings.as_of.isoformat(), "games": ratings.games,
            "played": {"home": ratings.played.get(home, 0), "away": ratings.played.get(away, 0)},
            "early_season": ratings.early(home, away), "stale": meta["stale"], "pmf": pmf,
        }

    def leg(self, leg: dict) -> dict:
        """A moneyline or spread leg priced by the model, with the book's current line when the game still has one."""
        day = leg["date"] if isinstance(leg["date"], date) else date.fromisoformat(str(leg["date"]))
        home, away = (leg["team"], leg["opponent"]) if leg["home"] else (leg["opponent"], leg["team"])
        game = self.matchup(home, away, day)
        wins, pushes = leg_outcomes(leg)
        probability, push = float(game["pmf"][wins].sum()), float(game["pmf"][pushes].sum())
        team_margin = game["home_margin"] if leg["home"] else -game["home_margin"]
        board = self.scoreboard_game(leg["game_id"], day)
        book = self.book_side(board, leg) if board else None
        warnings = []
        if board and board.get("state") in ("in", "post"):
            warnings.append(f"{leg['team']} vs {leg['opponent']} has {'started' if board['state'] == 'in' else 'finished'}; "
                            "this is the model's pregame chance.")
        if game["early_season"]:
            warnings.append(EARLY_NOTE)
        return {
            "kind": "game", "game_id": str(leg["game_id"]), "date": day.isoformat(), "team": leg["team"], "opponent": leg["opponent"],
            "home": bool(leg["home"]), "market": leg["market"], "line": None if leg["market"] == "moneyline" else float(leg["line"]),
            "probability": probability, "push": push, "fair_odds": fair_odds(probability),
            "win_probability": game["home_win"] if leg["home"] else game["away_win"], "margin": team_margin, "sigma": SIGMA,
            "fair_spread": fair_line(team_margin), "back_to_back": game["home_b2b"] if leg["home"] else game["away_b2b"],
            "opponent_back_to_back": game["away_b2b"] if leg["home"] else game["home_b2b"],
            "rating": game["ratings"]["home" if leg["home"] else "away"], "opponent_rating": game["ratings"]["away" if leg["home"] else "home"],
            "home_court": game["hca"], "games": game["games"], "as_of": game["as_of"], "early_season": game["early_season"],
            "book": price_side(probability, push, book), "state": (board or {}).get("state"), "warnings": warnings, "pmf": game["pmf"],
        }

    @staticmethod
    def book_side(game: dict, leg: dict) -> dict | None:
        """The listed price for this side, if the leg is on the book's market and line."""
        odds = game.get("odds") or {}
        mine, theirs = ("home", "away") if leg["home"] else ("away", "home")
        if leg["market"] == "moneyline":
            book = two_way(odds.get(f"{mine}_moneyline"), odds.get(f"{theirs}_moneyline"))
            return {**book, "line": None, "provider": odds.get("provider")} if book else None
        line = odds.get(f"{mine}_spread")
        if line is None or float(line) != float(leg["line"]):
            return None
        book = two_way(odds.get(f"{mine}_spread_odds") or -110, odds.get(f"{theirs}_spread_odds") or -110)
        return {**book, "line": float(line), "provider": odds.get("provider")} if book else None

    def annotate(self, game: dict) -> dict:
        """The model's view of a scoreboard game: win chances, fair spread and each side of the listed lines."""
        start = str(game.get("start") or "")
        try:
            day = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(EASTERN).date()
        except ValueError:
            day = self.today()
        home, away = game["home"]["team"], game["away"]["team"]
        model = self.matchup(home, away, day)
        pmf = model.pop("pmf")
        odds = game.get("odds") or {}
        markets: dict[str, dict] = {"moneyline": {}, "spread": {}}
        for side, team, other, at_home in (("home", home, away, True), ("away", away, home, False)):
            base = {"kind": "game", "game_id": str(game.get("id")), "date": day.isoformat(), "team": team, "opponent": other, "home": at_home}
            moneyline = {**base, "market": "moneyline", "line": None}
            wins, _ = leg_outcomes(moneyline)
            probability = float(pmf[wins].sum())
            markets["moneyline"][side] = {"probability": probability, "push": 0.0, "fair_odds": fair_odds(probability),
                                          "book": price_side(probability, 0.0, self.book_side(game, moneyline)) if odds else None}
            line = odds.get(f"{side}_spread")
            if line is not None:
                spread = {**base, "market": "spread", "line": float(line)}
                wins, pushes = leg_outcomes(spread)
                probability, push = float(pmf[wins].sum()), float(pmf[pushes].sum())
                markets["spread"][side] = {"line": float(line), "probability": probability, "push": push,
                                           "fair_odds": fair_odds(probability),
                                           "book": price_side(probability, push, self.book_side(game, spread))}
        return {**model, "markets": markets}

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        ratings, meta = self.ratings()
        table = sorted(({"team": team, "rating": value} for team, value in ratings.teams.items()), key=lambda row: -row["rating"])
        for rank, row in enumerate(table, start=1):
            row["rank"] = rank
        return {
            "as_of": ratings.as_of.isoformat(), "season": ratings.season, "games": ratings.games, "current_games": ratings.current_games,
            "home_court": ratings.hca, "back_to_back": ratings.b2b, "sigma": SIGMA, "half_life_days": HALF_LIFE_DAYS,
            "carryover": CARRYOVER, "ratings": table, "stale": meta["stale"], "errors": meta["errors"], "provider": meta["provider"],
            "backtest": self.backtest(),
        }

    def backtest(self) -> dict | None:
        """Walk-forward check on last season: each game predicted only from games before its date."""
        try:
            games, meta = self.games()
        except GameModelUnavailable:
            return None
        season = previous_season(season_of(self.today()))
        key = (season, meta["stamps"])
        with self._lock:
            if key in self._backtest:
                return self._backtest[key]
        try:
            earlier = self.season_games(previous_season(season), "previous")["games"]
        except NBAUnavailableError:
            earlier = []
        history = sorted(earlier + games, key=lambda game: (game["date"], game["game_id"]))
        target = [game for game in history if game["season"] == season]
        rest = rest_days(history)
        predictions, outcomes, margins, means = [], [], [], []
        for day in sorted({game["date"] for game in target}):
            try:
                ratings = fit_ratings(history, date.fromisoformat(day))
            except GameModelUnavailable:
                continue
            for game in (item for item in target if item["date"] == day):
                mean = expected_margin(ratings, game["home"], game["away"], rest[(game["game_id"], game["home"])] == 1,
                                       rest[(game["game_id"], game["away"])] == 1)
                predictions.append(float(norm.cdf(mean / SIGMA)))
                margin = game["home_points"] - game["away_points"]
                outcomes.append(float(margin > 0))
                margins.append(margin)
                means.append(mean)
        if not predictions:
            return None
        p = np.clip(np.array(predictions), 1e-6, 1 - 1e-6)
        y = np.array(outcomes)
        base = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        bins = []
        for low, high in ((0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)):
            inside = (p > low) & (p <= high)
            if inside.any():
                bins.append({"low": low, "high": high, "predicted": float(p[inside].mean()), "observed": float(y[inside].mean()),
                             "count": int(inside.sum())})
        result = {
            "season": season, "games": len(p), "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2)), "accuracy": float(np.mean((p > 0.5) == (y == 1))),
            "margin_mae": float(np.mean(np.abs(np.array(margins) - np.array(means)))),
            "home_win_rate": base, "baseline_log_loss": float(-(base * math.log(base) + (1 - base) * math.log(1 - base))),
            "bins": bins,
        }
        with self._lock:
            self._backtest[key] = result
        return result
