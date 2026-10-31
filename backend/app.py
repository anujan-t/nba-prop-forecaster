from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
from threading import Lock, RLock, Thread
import time
from typing import Annotated, Literal, Union
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator, model_validator
import pandas as pd
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import news, simulation
from .data import Store, demo_frame, nba_cache_fresh, nba_cache_id, player_list, read_csv, validate_frame
from .engine import american_probability, backtest, forecast
from .nba import NBAUnavailableError, NoGamesError, import_game_logs, normalize_name, search_players, static_player_ids
from .league import current_season, fetch_career, fetch_profile, fetch_roster, previous_season, summarize_history
from .parlay import MAX_LEGS, LegNotFound, price_parlay
from .assistant import Coach
from .learning import Learning
from .newsroom import Newsroom
from .llm import LLMClient, StoreCache
from .stats import PROP_STATS, catalog, label as stat_label, validate_line as validate_market_line
from .teams import MAX_SPREAD, GameModel, GameModelUnavailable

ROOT = Path(__file__).resolve().parent.parent
Stat = Literal[PROP_STATS]  # type: ignore[valid-type]
Model = Literal["auto", "empirical", "normal", "poisson", "negative_binomial", "simulation"]
DEFAULT_LOOKBACK = 60
GameDate = date


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ForecastRequest(StrictModel):
    dataset_id: str = Field(default="demo", min_length=1, max_length=80)
    player_id: str = Field(min_length=1, max_length=80)
    stat: Stat = "points"
    line: float = Field(default=26.5, ge=0, le=200)
    model: Model = "auto"
    lookback: int = Field(default=DEFAULT_LOOKBACK, ge=10, le=82)
    odds_over: int = -110
    odds_under: int = -110
    expected_minutes: float | None = Field(default=None, ge=1, le=60)
    rest_days: int | None = Field(default=None, ge=0, le=30)
    opponent: str | None = Field(default=None, pattern=r"^[A-Za-z0-9]{2,5}$")
    use_context: bool = True
    simulations: int | None = Field(default=None, ge=simulation.MIN_SIMULATIONS, le=simulation.MAX_SIMULATIONS)

    @field_validator("line")
    @classmethod
    def validate_line(cls, value: float) -> float:
        if value * 2 != int(value * 2):
            raise ValueError("Use a whole-number or half-point line.")
        return value

    @field_validator("odds_over", "odds_under")
    @classmethod
    def validate_odds(cls, value: int) -> int:
        american_probability(value)
        return value

    @model_validator(mode="after")
    def validate_market(self):
        validate_market_line(self.stat, self.line)
        return self


class BacktestRequest(StrictModel):
    dataset_id: str = Field(default="demo", min_length=1, max_length=80)
    player_id: str | None = Field(default=None, max_length=80)
    stat: Stat = "points"
    model: Model = "auto"
    lookback: int = Field(default=DEFAULT_LOOKBACK, ge=10, le=82)
    line: float | None = Field(default=None, ge=0, le=200)
    use_context: bool = True

    @field_validator("line")
    @classmethod
    def validate_line(cls, value: float | None) -> float | None:
        if value is not None:
            return ForecastRequest.validate_line(value)
        return value

    @model_validator(mode="after")
    def validate_market(self):
        if self.line is not None:
            validate_market_line(self.stat, self.line)
        return self


class NBARequest(StrictModel):
    player_ids: list[str] = Field(min_length=1, max_length=5)
    season: str = Field(default="2025-26", pattern=r"^\d{4}-\d{2}$")
    season_type: Literal["Regular Season", "Playoffs"] = "Regular Season"
    refresh: bool = False
    include_previous: bool = False

    @field_validator("season")
    @classmethod
    def validate_season(cls, value: str) -> str:
        start, end = value.split("-")
        if int(end) != (int(start) + 1) % 100 or not 1946 <= int(start) <= date.today().year:
            raise ValueError("Use a valid NBA season such as 2025-26.")
        return value

    @field_validator("player_ids")
    @classmethod
    def validate_ids(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values) or any(not v.isdigit() or len(v) > 10 for v in values):
            raise ValueError("Supply unique numeric NBA player IDs.")
        return values


class PlayerHistoryRequest(StrictModel):
    player_id: str = Field(pattern=r"^\d{1,10}$")
    window: str = "recent"
    season_type: Literal["Regular Season", "Playoffs"] = "Regular Season"
    refresh: bool = False

    @field_validator("window")
    @classmethod
    def validate_window(cls, value: str) -> str:
        if value != "recent":
            if len(value) != 7 or value[4] != "-" or not value[:4].isdigit() or not value[5:].isdigit():
                raise ValueError("Use 'recent' or an NBA season such as 2025-26.")
            NBARequest.validate_season(value)
        return value


class ParlayLeg(StrictModel):
    kind: Literal["player"] = "player"
    dataset_id: str = Field(min_length=1, max_length=80)
    player_id: str = Field(min_length=1, max_length=80)
    stat: Stat
    line: float = Field(ge=0, le=200)
    side: Literal["over", "under"]
    model: Model = "auto"
    lookback: int = Field(default=DEFAULT_LOOKBACK, ge=10, le=82)
    expected_minutes: float | None = Field(default=None, ge=1, le=60)
    rest_days: int | None = Field(default=None, ge=0, le=30)
    opponent: str | None = Field(default=None, pattern=r"^[A-Za-z0-9]{2,5}$")
    use_context: bool = True

    @field_validator("line")
    @classmethod
    def validate_line(cls, value: float) -> float:
        return ForecastRequest.validate_line(value)

    @model_validator(mode="after")
    def validate_market(self):
        validate_market_line(self.stat, self.line)
        return self


class GameLeg(StrictModel):
    """A moneyline or point spread on one team in one game (the game is ESPN's event on that US Eastern date)."""
    kind: Literal["game"]
    game_id: str = Field(pattern=r"^\d{1,20}$")
    date: GameDate
    team: str = Field(pattern=r"^[A-Z]{2,4}$")
    opponent: str = Field(pattern=r"^[A-Z]{2,4}$")
    home: bool
    market: Literal["moneyline", "spread"]
    line: float | None = Field(default=None, ge=-MAX_SPREAD, le=MAX_SPREAD)

    @model_validator(mode="after")
    def validate_market(self):
        if self.team == self.opponent:
            raise ValueError("A team can't play itself; check the opponent.")
        if self.market == "moneyline" and self.line is not None:
            raise ValueError("A moneyline has no line; use a spread for a points handicap.")
        if self.market == "spread":
            if self.line is None:
                raise ValueError("A spread needs a line, such as -5.5 or +3.")
            if self.line * 2 != int(self.line * 2):
                raise ValueError("Use a whole-number or half-point spread.")
        return self


def leg_kind(value) -> str:
    """Legs without a kind are player props, so slips saved before moneylines and spreads still load."""
    kind = value.get("kind") if isinstance(value, dict) else getattr(value, "kind", None)
    return kind or "player"


AnyLeg = Annotated[Union[Annotated[ParlayLeg, Tag("player")], Annotated[GameLeg, Tag("game")]], Discriminator(leg_kind)]


def leg_key(leg: ParlayLeg | GameLeg) -> tuple:
    return ("game", leg.game_id, leg.market) if isinstance(leg, GameLeg) else ("player", leg.player_id, leg.stat)


class ParlayRequest(StrictModel):
    legs: list[AnyLeg] = Field(min_length=1, max_length=MAX_LEGS)
    book_odds: int | None = None
    simulations: int | None = Field(default=None, ge=simulation.MIN_SIMULATIONS, le=simulation.MAX_SIMULATIONS)

    @field_validator("book_odds")
    @classmethod
    def validate_odds(cls, value: int | None) -> int | None:
        return value if value is None else ForecastRequest.validate_odds(value)


class LegLabel(StrictModel):
    player: str = Field(min_length=1, max_length=120)  # the player, or the team for a moneyline or spread
    team: str = Field(default="", max_length=60)


class SlipLeg(StrictModel):
    request: AnyLeg
    label: LegLabel


class Slip(StrictModel):
    legs: list[SlipLeg] = Field(default_factory=list, max_length=MAX_LEGS)
    book_odds: int | None = None

    @field_validator("book_odds")
    @classmethod
    def validate_odds(cls, value: int | None) -> int | None:
        return value if value is None else ForecastRequest.validate_odds(value)

    @field_validator("legs")
    @classmethod
    def unique_legs(cls, legs: list[SlipLeg]) -> list[SlipLeg]:
        keys = [leg_key(leg.request) for leg in legs]
        if len(set(keys)) != len(keys):
            raise ValueError("Each player and stat, and each game's moneyline and spread, may appear only once in the slip.")
        return legs


class Focus(StrictModel):
    dataset_id: str = Field(min_length=1, max_length=80)
    player_id: str = Field(min_length=1, max_length=80)
    player: str = Field(min_length=1, max_length=120)
    team: str = Field(default="", max_length=60)
    stat: Stat
    line: float = Field(ge=0, le=200)
    model: Model = "auto"
    lookback: int = Field(default=DEFAULT_LOOKBACK, ge=10, le=82)
    expected_minutes: float | None = Field(default=None, ge=1, le=60)
    rest_days: int | None = Field(default=None, ge=0, le=30)
    opponent: str | None = Field(default=None, pattern=r"^[A-Za-z0-9]{2,5}$")
    use_context: bool = True

    @field_validator("line")
    @classmethod
    def validate_line(cls, value: float) -> float:
        return ForecastRequest.validate_line(value)


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=1000)
    conversation_id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")
    slip: Slip = Field(default_factory=Slip)
    focus: Focus | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip() or any(ord(char) < 32 and char not in "\n\t" for char in value):
            raise ValueError("Write a message without control characters.")
        return value


class FeedbackRequest(StrictModel):
    turn_id: str = Field(pattern=r"^[A-Za-z0-9_-]{6,40}$")
    rating: Literal[1, -1]
    note: str | None = Field(default=None, max_length=500)


class TrackBetRequest(StrictModel):
    slip: Slip
    stake: float | None = Field(default=None, gt=0, le=1_000_000)
    event_date: date | None = None
    note: str | None = Field(default=None, max_length=200)

    @field_validator("slip")
    @classmethod
    def has_legs(cls, slip: Slip) -> Slip:
        if not slip.legs:
            raise ValueError("Add at least one leg before tracking a bet.")
        return slip


class LineRequest(StrictModel):
    dataset_id: str = Field(min_length=1, max_length=80)
    player_id: str = Field(min_length=1, max_length=80)
    stat: Stat
    event_date: date
    book: str = Field(default="Manual entry", min_length=1, max_length=60)
    line: float = Field(ge=0, le=200)
    odds_over: int = -110
    odds_under: int = -110

    @field_validator("line")
    @classmethod
    def validate_line(cls, value: float) -> float:
        return ForecastRequest.validate_line(value)

    @field_validator("odds_over", "odds_under")
    @classmethod
    def validate_odds(cls, value: int) -> int:
        return ForecastRequest.validate_odds(value)

    @field_validator("book")
    @classmethod
    def validate_book(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("Book name cannot be blank or contain control characters.")
        return value


def create_app(data_dir: Path | None = None, background: bool = False) -> FastAPI:
    app = FastAPI(title="Courtside / NBA Prop Forecaster", version="1.3.0")
    store = Store(data_dir or Path(os.environ.get("NBA_DATA_DIR", ROOT / "runtime")))
    app.state.store = store
    nba_lock = RLock()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

    @app.middleware("http")
    async def protect_local_writes(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin:
                parsed = urlparse(origin)
                allowed = {str(request.base_url).rstrip("/"), "http://127.0.0.1:5173", "http://localhost:5173"}
                if parsed.hostname not in ("127.0.0.1", "localhost") or origin not in allowed:
                    return JSONResponse(status_code=403, content={"detail": "Cross-origin writes are disabled for this local application."})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    def load(dataset_id: str):
        try:
            return store.load(dataset_id)
        except KeyError as exc:
            raise HTTPException(404, "Dataset not found.") from exc

    def check_player(dataset_id: str, player_id: str):
        frame, info = load(dataset_id)
        if not frame["player_id"].eq(player_id).any():
            raise HTTPException(404, "Player not found in this dataset.")
        return frame, info

    def cached_provider(key: str, fetcher, refresh: bool = False):
        with nba_lock:
            cached = store.resource(key)
            if cached is not None and not refresh and nba_cache_fresh(cached):
                return cached
            return store.save_resource(key, fetcher())

    def current_roster(refresh: bool = False):
        return cached_provider("roster:" + current_season(), fetch_roster, refresh)

    def roster_or_saved() -> dict:
        """Roster membership for lookups: the last saved roster is acceptable if NBA Stats is briefly unreachable."""
        try:
            return current_roster()
        except NBAUnavailableError:
            saved = store.resource("roster:" + current_season())
            if saved is None:
                raise
            return saved

    def current_player(player_id: str):
        if not player_id.isdigit() or len(player_id) > 10:
            raise HTTPException(422, "Use a numeric NBA player ID.")
        roster = roster_or_saved()
        player = next((player for player in roster["players"] if player["id"] == player_id), None)
        if player is None:
            raise HTTPException(404, "Player is not listed on NBA's current roster. Refresh the league directory if a recent signing is missing.")
        return player

    try:
        language_model = LLMClient.from_env()
    except ValueError as exc:
        language_model = None
        logging.getLogger("courtside").warning("Language model disabled: %s", exc)
    if language_model is not None:
        # Identical requests (same question, same tool results) reuse the saved answer for six hours: fewer free-tier calls.
        language_model.cache, language_model.cache_ttl = StoreCache(store), 6 * 3600.0
    def roster_players() -> list[dict]:
        try:
            return roster_or_saved()["players"]
        except NBAUnavailableError:
            return []

    newsroom = Newsroom(store, roster_players)
    app.state.newsroom = newsroom
    team_model = GameModel(store, today=newsroom.today, schedule=newsroom.scoreboard)
    app.state.team_model = team_model
    learning = Learning(store, history=lambda player_id, refresh: history_frame(player_id, refresh), scores=newsroom.scoreboard)
    app.state.learning = learning
    if background:
        Thread(target=newsroom.warm, name="courtside-news-warmup", daemon=True).start()
    coach = Coach(lambda: roster_or_saved()["players"], lambda player_id, refresh: history_frame(player_id, refresh),
                  store.load, learning=learning, llm=language_model, newsroom=newsroom,
                  profile=lambda player_id: cached_provider("profile:" + player_id, lambda: fetch_profile(player_id)),
                  games=team_model)
    app.state.coach = coach
    retrain_lock = Lock()
    last_retrain = {"at": float("-inf")}

    def retrain_now() -> None:
        with retrain_lock:
            last_retrain["at"] = time.monotonic()
            try:
                learning.retrain()
            except Exception:  # a failed background refit must never take the app down
                logging.getLogger("courtside").exception("Calibration retrain failed.")

    def schedule_retrain(force: bool = False) -> None:
        """Background refit when new games or graded bets arrive (at most every 10 minutes)."""
        if not background or retrain_lock.locked() or (not force and time.monotonic() - last_retrain["at"] < 600):
            return
        Thread(target=retrain_now, name="courtside-retrain", daemon=True).start()

    if background:
        trained = learning.calibration_status().get("trained_at")
        if not trained or datetime.now(timezone.utc) - datetime.fromisoformat(trained.replace("Z", "+00:00")) > timedelta(hours=6):
            schedule_retrain(force=True)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/datasets")
    def datasets():
        return {"datasets": store.datasets(), "stats": catalog(),
                "simulation": {"default": simulation.DEFAULT_SIMULATIONS, "choices": list(simulation.SIMULATION_CHOICES),
                               "minimum": simulation.MIN_SIMULATIONS, "maximum": simulation.MAX_SIMULATIONS}}

    @app.get("/api/players")
    def players(dataset_id: str = "demo"):
        frame, _ = load(dataset_id)
        return {"players": player_list(frame)}

    def saved_injury(player_id: str) -> dict | None:
        """The player's entry on the last saved injury report (no network call), with the report's age."""
        report = store.resource("news:injuries")
        if not report:
            return None
        for team in report.get("teams", []):
            for entry in team.get("injuries", []):
                if entry.get("player_id") == player_id:
                    return {**entry, "report_fetched_at": report.get("fetched_at")}
        return None

    def injury_warning(entry: dict) -> str:
        what = " ".join(filter(None, [entry.get("side"), entry.get("injury"), (entry.get("detail") or "").lower() or None]))
        comment = f" {entry['short_comment']}" if entry.get("short_comment") else ""
        return (f"{entry['player']} is listed {entry['status'].upper()} on the latest injury report"
                f"{f' ({what})' if what else ''}.{comment} The forecast assumes he plays.")

    def calibrated_probabilities(result: dict, model: str) -> dict | None:
        """The market's learned correction applied to an Auto forecast (it was learned on Auto's graded predictions)."""
        if model != "auto":
            return None
        over, under, push = (result["probabilities"][key] for key in ("over", "under", "push"))
        settled = over + under
        if settled <= 1e-12:
            return None
        q = learning.calibrate_over(over / settled, result["stat"])
        if q is None:
            return None
        entry = learning.calibration_for(result["stat"]) or {}
        choice = entry.get("choice") or "shared"
        source = (f"its own {stat_label(result['stat']).lower()} correction" if choice == "own"
                  else "the shared correction across markets")
        return {"over": q * settled, "under": (1 - q) * settled, "push": push, "choice": choice,
                "samples": int(entry.get("samples") or learning.calibration_status().get("samples") or 0),
                "note": f"Adjusted by {source}, learned from graded predictions and switched on because it beat the raw model on recent games."}

    @app.post("/api/forecast")
    def run_forecast(body: ForecastRequest):
        frame, info = check_player(body.dataset_id, body.player_id)
        try:
            result = forecast(frame, info, body.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        result["calibrated"] = calibrated_probabilities(result, body.model)
        injury = saved_injury(body.player_id) if info.get("source") == "nba_api" else None
        result["injury"] = injury
        if injury:
            result["warnings"].insert(0, injury_warning(injury))
        return result

    @app.post("/api/backtest")
    def run_backtest(body: BacktestRequest):
        frame, _ = load(body.dataset_id)
        try:
            return backtest(frame, body.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/parlay")
    def run_parlay(body: ParlayRequest):
        try:
            result = price_parlay([leg.model_dump(mode="json") for leg in body.legs], body.book_odds, store.load, calibrator=learning,
                                  simulations=body.simulations, team_model=team_model)
        except LegNotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        flagged = []
        for leg in result["legs"]:
            if leg.get("kind") == "game":
                continue
            injury = saved_injury(leg["player"]["id"])
            leg["injury"] = injury
            if injury and injury["player_id"] not in flagged:
                flagged.append(injury["player_id"])
                result["warnings"].insert(0, f"Leg {leg['index'] + 1}: " + injury_warning(injury))
        return result

    @app.get("/api/games/model")
    def games_model():
        """Team ratings, home court, the back-to-back penalty and the walk-forward check behind moneylines and spreads."""
        try:
            return team_model.status()
        except (GameModelUnavailable, NBAUnavailableError) as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.get("/api/assistant/status")
    def assistant_status():
        return coach.status()

    @app.post("/api/assistant/chat")
    def assistant_chat(body: ChatRequest):
        slip = body.slip.model_dump(mode="json")
        try:
            return coach.chat(body.message, body.conversation_id, slip, body.focus.model_dump() if body.focus else None)
        except HTTPException:
            raise
        except NBAUnavailableError as exc:
            raise HTTPException(502, f"{exc} Your slip is unchanged.") from exc

    @app.post("/api/assistant/feedback")
    def assistant_feedback(body: FeedbackRequest):
        try:
            return learning.feedback(body.turn_id, body.rating, body.note)
        except KeyError as exc:
            raise HTTPException(404, "That reply was not found; feedback was not saved.") from exc

    @app.get("/api/ledger")
    def ledger():
        return {"bets": learning.bets(), "record": learning.track_record()}

    @app.post("/api/ledger", status_code=201)
    def track_bet(body: TrackBetRequest):
        slip = body.slip.model_dump(mode="json")
        try:
            priced = price_parlay([leg["request"] for leg in slip["legs"]], slip["book_odds"], store.load, team_model=team_model)
            return learning.track(slip["legs"], slip["book_odds"], body.stake, body.event_date, priced, body.note)
        except LegNotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/ledger/settle")
    def settle_bets():
        result = learning.settle()
        if result["settled_legs"]:
            schedule_retrain(force=True)
        return result

    @app.delete("/api/ledger/{bet_id}", status_code=204)
    def delete_bet(bet_id: str):
        if not learning.delete_bet(bet_id):
            raise HTTPException(404, "Tracked bet not found.")
        return Response(status_code=204)

    @app.get("/api/learning")
    def learning_status():
        return learning.status()

    @app.post("/api/learning/retrain")
    def retrain():
        if retrain_lock.locked():
            return learning.status()
        retrain_now()
        return learning.status()

    @app.delete("/api/learning/aliases/{phrase}", status_code=204)
    def forget_alias(phrase: str):
        if not learning.forget_alias(phrase):
            raise HTTPException(404, "That nickname is not in memory.")
        return Response(status_code=204)

    @app.get("/api/datasets/template")
    def template():
        frame = demo_frame().head(20).drop(columns=["rest_days"])
        return Response(frame.to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="synthetic-template.csv"'})

    @app.get("/api/datasets/{dataset_id}/export")
    def export(dataset_id: str):
        frame, _ = load(dataset_id)
        return Response(frame.to_csv(index=False), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="player-games.csv"'})

    @app.post("/api/datasets/import", status_code=201)
    async def import_csv(file: UploadFile = File(...), name: str = Form(default="Imported CSV", min_length=1, max_length=80)):
        payload = await file.read(5 * 1024 * 1024 + 1)
        if len(payload) > 5 * 1024 * 1024:
            raise HTTPException(413, "CSV exceeds the 5 MB upload limit.")
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(422, "Save the CSV with UTF-8 encoding.") from exc
        name = name.strip()
        if not name:
            raise HTTPException(422, "Dataset name cannot be blank.")
        try:
            frame = read_csv(text)
            return store.save(frame, name, "csv")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/nba/players")
    def nba_players(search: str = Query(default="", max_length=80), include_historical: bool = False):
        if include_historical:
            return {"players": search_players(search)}
        try:
            roster = current_roster()
            needle = normalize_name(search.strip())
            matches = [p for p in roster["players"] if needle in normalize_name(p["name"])]
            return {**roster, "players": matches, "matched_players": len(matches)}
        except NBAUnavailableError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.get("/api/nba/roster")
    def nba_roster(refresh: bool = False):
        try:
            return current_roster(refresh)
        except NBAUnavailableError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.get("/api/nba/players/{player_id}/profile")
    def nba_profile(player_id: str, refresh: bool = False):
        try:
            current_player(player_id)
            return cached_provider("profile:" + player_id, lambda: fetch_profile(player_id), refresh)
        except NBAUnavailableError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.get("/api/nba/players/{player_id}/career")
    def nba_career(player_id: str, refresh: bool = False):
        try:
            current_player(player_id)
            return cached_provider("career:" + player_id, lambda: fetch_career(player_id), refresh)
        except NBAUnavailableError as exc:
            raise HTTPException(502, str(exc)) from exc

    def player_history(body: PlayerHistoryRequest) -> dict:
        """Recent or single-season game logs for one current player, cached for six hours (raises provider errors)."""
        player = current_player(body.player_id)
        latest = current_season() if body.window == "recent" else body.window
        seasons = [previous_season(latest), latest] if body.window == "recent" else [latest]
        query = {"player_ids": [body.player_id], "season": latest, "season_type": body.season_type}
        if body.window == "recent":
            query["include_previous"] = True
        empty_key = "empty-history:" + nba_cache_id(query)
        with nba_lock:
            cached = store.cached_nba(query)
            if cached is not None and cached.get("details_version") == 1 and not body.refresh and nba_cache_fresh(cached):
                frame, info = store.load(cached["id"])
                cache_hit = True
                warnings = info.get("history_warnings", [])
                queried = info["queried_seasons"]
            else:
                no_games = store.resource(empty_key)
                if no_games is not None and not body.refresh and nba_cache_fresh(no_games):
                    return {**no_games, "cache_hit": True}
                frames, warnings, queried = [], [], []
                for season in seasons:
                    if player.get("from_year") is not None and int(season[:4]) < player["from_year"]:
                        warnings.append(f"{season} is before this player's listed NBA debut; no earlier NBA results are assumed.")
                        continue
                    queried.append(season)
                    try:
                        frames.append(import_game_logs([body.player_id], season, body.season_type, player_names={body.player_id: player["name"]}))
                    except NoGamesError as exc:
                        warnings.append(str(exc))
                if not frames:
                    return store.save_resource(empty_key, {
                        "player_id": body.player_id, "status": "empty", "history": None, "summary": None,
                        "splits": [], "games": [], "queried_seasons": queried, "cache_hit": False,
                        "warnings": [*warnings, "No completed NBA games with positive minutes in this window. The player remains accessible; no forecast is fabricated."],
                    })
                frame = validate_frame(pd.concat(frames, ignore_index=True))
                if len(frame) < 10:
                    warnings.append(f"Only {len(frame)} completed games are available. Profile and game details are shown, but a forecast needs at least 10.")
                if len(queried) > 1:
                    warnings.append("Recent history pools the current and previous NBA seasons; these are not current-season-only averages. Season-specific splits are shown separately.")
                info = store.save(
                    frame, f"{player['name']} / {body.window if body.window != 'recent' else 'Recent NBA history'} / {body.season_type}",
                    "nba_api", nba_query=query,
                    attributes={"details_version": 1, "queried_seasons": queried, "history_warnings": warnings},
                )
                with store.connect() as db:
                    db.execute("DELETE FROM provider_cache WHERE key = ?", (empty_key,))
                cache_hit = False
                schedule_retrain()
            summary, splits = summarize_history(frame)
            return {
                "player_id": body.player_id, "status": "ready" if len(frame) >= 10 else "limited",
                "history": info, "summary": summary, "splits": splits,
                "games": json.loads(frame.to_json(orient="records", force_ascii=False)),
                "queried_seasons": queried, "warnings": warnings, "cache_hit": cache_hit,
                "fetched_at": info["fetched_at"],
            }

    @app.post("/api/nba/player-history")
    def nba_player_history(body: PlayerHistoryRequest):
        try:
            return player_history(body)
        except NBAUnavailableError as exc:
            raise HTTPException(502, f"{exc} Previously saved history is unchanged.") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def history_frame(player_id: str, refresh: bool = False) -> tuple[str, pd.DataFrame, dict]:
        """(dataset_id, frame, info) for a player's recent regular-season logs, for the coach and the bet grader."""
        try:
            payload = player_history(PlayerHistoryRequest(player_id=player_id, refresh=refresh))
        except HTTPException as exc:
            raise ValueError(str(exc.detail)) from exc
        if payload.get("history") is None:
            raise ValueError("No completed NBA games in the recent window.")
        frame, info = store.load(payload["history"]["id"])
        return payload["history"]["id"], frame, info

    @app.post("/api/nba/history")
    @app.post("/api/datasets/nba", status_code=201, deprecated=True)
    def import_nba(body: NBARequest):
        if body.include_previous and (len(body.player_ids) != 1 or int(body.season[:4]) <= 1946):
            raise HTTPException(422, "Pooled recent histories require one player and a season after 1946; use the league player selector.")
        query = body.model_dump(exclude={"refresh", "include_previous"})
        query["player_ids"] = sorted(query["player_ids"])
        if body.include_previous:
            query["include_previous"] = True
        try:
            names = None
            if any(player_id not in static_player_ids() for player_id in body.player_ids):
                names = {p["id"]: p["name"] for p in current_roster()["players"]}
            with nba_lock:
                cached = store.cached_nba(query)
                if cached is not None and not body.refresh and nba_cache_fresh(cached):
                    return {**cached, "cache_hit": True}
                if body.include_previous:
                    frames, warnings = [], []
                    for season in (previous_season(body.season), body.season):
                        try:
                            frames.append(import_game_logs(query["player_ids"], season, body.season_type, player_names=names))
                        except NoGamesError as exc:
                            warnings.append(str(exc))
                    if not frames:
                        raise NoGamesError("No completed NBA games in either requested season.")
                    frame = validate_frame(pd.concat(frames, ignore_index=True))
                else:
                    frame = import_game_logs(query["player_ids"], body.season, body.season_type, player_names=names)
                    warnings = []
                labels = [p["name"] for p in player_list(frame)]
                label = labels[0] if len(labels) == 1 else f"{len(labels)} players"
                info = store.save(frame, f"{label} / {body.season} / {body.season_type}", "nba_api", nba_query=query,
                                  attributes={"history_warnings": warnings})
                return {**info, "cache_hit": False}
        except NBAUnavailableError as exc:
            raise HTTPException(502, f"{exc} Any previously saved NBA history remains unchanged; it has not been refreshed.") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/lines")
    def lines(dataset_id: str, player_id: str, stat: Stat, event_date: date | None = None):
        check_player(dataset_id, player_id)
        return {"lines": store.lines(dataset_id, player_id, stat, event_date.isoformat() if event_date else None)}

    @app.post("/api/lines", status_code=201)
    def add_line(body: LineRequest):
        check_player(body.dataset_id, body.player_id)
        return store.add_line(body.model_dump(mode="json"))

    # ---------- Newsroom ----------
    def feed(call):
        try:
            return call()
        except news.NewsUnavailableError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.get("/api/news/headlines")
    def news_headlines(refresh: bool = False):
        return feed(lambda: newsroom.headlines(refresh))

    @app.get("/api/news/injuries")
    def news_injuries(refresh: bool = False):
        return feed(lambda: newsroom.injuries(refresh))

    @app.get("/api/news/transactions")
    def news_transactions(refresh: bool = False):
        return feed(lambda: newsroom.transactions(refresh))

    @app.get("/api/news/coaches")
    def news_coaches(refresh: bool = False):
        return feed(lambda: newsroom.coaches(refresh))

    @app.get("/api/news/games")
    def news_games(day: date | None = None, refresh: bool = False):
        target = day or newsroom.today()
        if abs((target - newsroom.today()).days) > 366:
            raise HTTPException(422, "Pick a date within a year of today.")
        board = feed(lambda: newsroom.scoreboard(target, refresh))
        report, _ = newsroom.attempt(newsroom.injuries)
        playing = {code for game in board["games"] for code in (game["home"]["team"], game["away"]["team"])}
        injuries = {team["team"]: [entry for entry in team["injuries"] if entry["status_key"] in ("out", "suspended", "doubtful", "questionable", "day_to_day")]
                    for team in (report or {}).get("teams", []) if team["team"] in playing}
        model_error = None
        for game in board["games"]:
            game["model"] = None
            if game["state"] != "pre" or model_error:
                continue
            try:
                game["model"] = team_model.annotate(game)
            except (GameModelUnavailable, NBAUnavailableError) as exc:
                model_error = f"The team model is unavailable: {exc}"
        return {**board, "injuries": injuries, "today": newsroom.today().isoformat(), "model_error": model_error}

    @app.get("/api/news/teams")
    def news_teams(season: Literal["current", "previous"] = "current", refresh: bool = False):
        styles = feed(lambda: newsroom.styles(season, refresh))
        report, _ = newsroom.attempt(newsroom.injuries)
        staffs, _ = newsroom.attempt(newsroom.coaches)
        out = {team["team"]: sum(entry["status_key"] in ("out", "suspended") for entry in team["injuries"])
               for team in (report or {}).get("teams", [])}
        listed = {team["team"]: len(team["injuries"]) for team in (report or {}).get("teams", [])}
        heads = {item["team"]: next((coach["name"] for coach in item["coaches"] if coach["head"]), None)
                 for item in (staffs or {}).get("teams", [])}
        teams = [{**team, "summary": news.style_sentence(team), "injuries_listed": listed.get(team["team"], 0),
                  "injuries_out": out.get(team["team"], 0), "head_coach": heads.get(team["team"])} for team in styles["teams"]]
        median_games = sorted(team["games"] for team in teams)[len(teams) // 2] if teams else 0
        return {**styles, "teams": teams, "which": season, "median_games": median_games}

    @app.get("/api/news/teams/{abbreviation}")
    def news_team(abbreviation: str, refresh: bool = False):
        try:
            return newsroom.team_detail(abbreviation.upper(), refresh)
        except KeyError as exc:
            raise HTTPException(404, "Unknown NBA team abbreviation.") from exc

    @app.get("/api/news/players/{player_id}")
    def news_player(player_id: str, refresh: bool = False):
        return newsroom.player_news(current_player(player_id), refresh)

    dist = ROOT / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    else:
        @app.get("/")
        def missing_frontend():
            return JSONResponse(status_code=503, content={"detail": "Frontend not built. Run npm run build in frontend, then restart the service."})
    return app


app = create_app(background=os.environ.get("COURTSIDE_BACKGROUND", "1") != "0")
