from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
from threading import RLock
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
import pandas as pd
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .data import STATS, Store, demo_frame, nba_cache_fresh, nba_cache_id, player_list, read_csv, validate_frame
from .engine import american_probability, backtest, forecast
from .nba import NBAUnavailableError, NoGamesError, import_game_logs, normalize_name, search_players, static_player_ids
from .league import current_season, fetch_career, fetch_profile, fetch_roster, previous_season, summarize_history

ROOT = Path(__file__).resolve().parent.parent
Stat = Literal["points", "rebounds", "assists", "steals", "blocks"]
Model = Literal["auto", "empirical", "normal", "poisson", "negative_binomial"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ForecastRequest(StrictModel):
    dataset_id: str = Field(default="demo", min_length=1, max_length=80)
    player_id: str = Field(min_length=1, max_length=80)
    stat: Stat = "points"
    line: float = Field(default=26.5, ge=0, le=200)
    model: Model = "auto"
    lookback: int = Field(default=30, ge=10, le=82)
    odds_over: int = -110
    odds_under: int = -110
    expected_minutes: float | None = Field(default=None, ge=1, le=60)
    rest_days: int | None = Field(default=None, ge=0, le=30)
    opponent: str | None = Field(default=None, pattern=r"^[A-Za-z0-9]{2,5}$")
    use_context: bool = True

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


class BacktestRequest(StrictModel):
    dataset_id: str = Field(default="demo", min_length=1, max_length=80)
    player_id: str | None = Field(default=None, max_length=80)
    stat: Stat = "points"
    model: Model = "auto"
    lookback: int = Field(default=30, ge=10, le=82)
    line: float | None = Field(default=None, ge=0, le=200)
    use_context: bool = True

    @field_validator("line")
    @classmethod
    def validate_line(cls, value: float | None) -> float | None:
        if value is not None:
            return ForecastRequest.validate_line(value)
        return value


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


def create_app(data_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Courtside / NBA Prop Forecaster", version="1.2.0")
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

    def current_player(player_id: str):
        if not player_id.isdigit() or len(player_id) > 10:
            raise HTTPException(422, "Use a numeric NBA player ID.")
        roster = current_roster()
        player = next((player for player in roster["players"] if player["id"] == player_id), None)
        if player is None:
            raise HTTPException(404, "Player is not listed on NBA's current roster. Refresh the league directory if a recent signing is missing.")
        return player

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/datasets")
    def datasets():
        return {"datasets": store.datasets(), "stats": [{"id": stat, "label": stat.title()} for stat in STATS]}

    @app.get("/api/players")
    def players(dataset_id: str = "demo"):
        frame, _ = load(dataset_id)
        return {"players": player_list(frame)}

    @app.post("/api/forecast")
    def run_forecast(body: ForecastRequest):
        frame, info = check_player(body.dataset_id, body.player_id)
        try:
            return forecast(frame, info, body.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/backtest")
    def run_backtest(body: BacktestRequest):
        frame, _ = load(body.dataset_id)
        try:
            return backtest(frame, body.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

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

    @app.post("/api/nba/player-history")
    def nba_player_history(body: PlayerHistoryRequest):
        try:
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
                summary, splits = summarize_history(frame)
                return {
                    "player_id": body.player_id, "status": "ready" if len(frame) >= 10 else "limited",
                    "history": info, "summary": summary, "splits": splits,
                    "games": json.loads(frame.to_json(orient="records", force_ascii=False)),
                    "queried_seasons": queried, "warnings": warnings, "cache_hit": cache_hit,
                    "fetched_at": info["fetched_at"],
                }
        except NBAUnavailableError as exc:
            raise HTTPException(502, f"{exc} Previously saved history is unchanged.") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

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

    dist = ROOT / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    else:
        @app.get("/")
        def missing_frontend():
            return JSONResponse(status_code=503, content={"detail": "Frontend not built. Run npm run build in frontend, then restart the service."})
    return app


app = create_app()
