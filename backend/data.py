from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sqlite3
from typing import Iterator
from uuid import uuid4

import numpy as np
import pandas as pd

STATS = ("points", "rebounds", "assists", "steals", "blocks")
REQUIRED = ("date", "player_id", "player_name", "team", "opponent", "minutes", *STATS)
BOX_COUNTS = (
    "field_goals_made", "field_goals_attempted", "three_pointers_made", "three_pointers_attempted",
    "free_throws_made", "free_throws_attempted", "offensive_rebounds", "defensive_rebounds",
    "turnovers", "fouls", "plus_minus",
)
TEAMS = ("BOS", "NYK", "CLE", "DEN", "OKC", "LAL", "GSW", "MIL", "DAL", "SAC", "MIN", "PHX")
NBA_CACHE_TTL = timedelta(hours=6)


def nba_cache_id(query: dict) -> str:
    canonical = {**query, "player_ids": sorted(query["player_ids"])}
    return "nba-" + sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:24]


def nba_cache_fresh(info: dict) -> bool:
    if not info.get("fetched_at"):
        return False
    age = datetime.now(timezone.utc) - datetime.fromisoformat(info["fetched_at"])
    return timedelta(0) <= age < NBA_CACHE_TTL


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError("Missing game-log fields: " + ", ".join(sorted(missing)))
    if not 1 <= len(frame) <= 50000:
        raise ValueError("A dataset must contain 1 to 50,000 player-game rows.")
    optional = [col for col in ("home", "season", "game_id", "win", *BOX_COUNTS) if col in frame]
    frame = frame.loc[:, [*REQUIRED, *optional]].copy()
    if frame.loc[:, list(REQUIRED)].isna().any().any():
        raise ValueError("Required game-log fields cannot be empty.")
    for col in ("player_id", "player_name", "team", "opponent"):
        frame[col] = frame[col].astype(str).str.strip()
        if frame[col].str.len().eq(0).any() or frame[col].str.len().gt(80).any():
            raise ValueError(f"{col} must contain 1 to 80 characters.")
        if frame[col].str.contains(r"[\x00-\x1f]|^[=+@-]", regex=True).any():
            raise ValueError(f"{col} contains an unsafe control character or spreadsheet formula prefix.")
    if not frame["player_id"].str.fullmatch(r"[A-Za-z0-9_-]+").all():
        raise ValueError("Player IDs may contain letters, numbers, underscores and hyphens only.")
    for col in ("team", "opponent"):
        frame[col] = frame[col].str.upper()
        if not frame[col].str.fullmatch(r"[A-Z0-9]{2,5}").all():
            raise ValueError(f"{col} must be a 2 to 5 character team abbreviation.")
    dates = frame["date"].astype(str)
    if not dates.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("Dates must use YYYY-MM-DD format.")
    try:
        parsed = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    except (ValueError, OverflowError) as exc:
        raise ValueError("The game logs contain an invalid calendar date.") from exc
    frame["date"] = parsed.dt.strftime("%Y-%m-%d")
    for col in ("minutes", *STATS):
        try:
            values = pd.to_numeric(frame[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{col} must be numeric.") from exc
        upper = 65 if col == "minutes" else 300
        if not np.isfinite(values).all() or values.lt(0).any() or values.gt(upper).any():
            raise ValueError(f"{col} must contain finite values between 0 and {upper}.")
        if col == "minutes":
            if values.le(0).any():
                raise ValueError("Exclude DNPs: minutes must be greater than zero.")
            frame[col] = values.astype(float)
        else:
            if (values % 1 != 0).any():
                raise ValueError(f"{col} must contain whole-number game totals.")
            frame[col] = values.astype(int)
    for col in BOX_COUNTS:
        if col not in frame:
            continue
        try:
            values = pd.to_numeric(frame[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{col} must contain whole-number counts or missing values.") from exc
        present = values.dropna()
        lower = -300 if col == "plus_minus" else 0
        if not np.isfinite(present).all() or (present % 1 != 0).any() or present.lt(lower).any() or present.gt(300).any():
            raise ValueError(f"{col} contains an invalid box-score value.")
        frame[col] = values.astype("Int64")
    for made, attempted in (
        ("field_goals_made", "field_goals_attempted"), ("three_pointers_made", "three_pointers_attempted"),
        ("free_throws_made", "free_throws_attempted"),
    ):
        if made in frame and attempted in frame and (frame[made] > frame[attempted]).any():
            raise ValueError(f"{made} cannot exceed {attempted}.")
    if "season" in frame and not frame["season"].astype(str).str.fullmatch(r"\d{4}-\d{2}").all():
        raise ValueError("Game-log seasons must use YYYY-YY format.")
    if "game_id" in frame:
        frame["game_id"] = frame["game_id"].astype("string")
        if not frame["game_id"].dropna().str.fullmatch(r"[A-Za-z0-9_-]{1,30}").all():
            raise ValueError("Invalid NBA game identifier.")
    if "win" in frame and not frame["win"].dropna().isin(["W", "L"]).all():
        raise ValueError("Game outcomes must be W, L or missing.")
    if frame.duplicated(["player_id", "date"]).any():
        raise ValueError("Duplicate player/date rows found; keep one completed game per player and date.")
    if frame.groupby("player_id")["player_name"].nunique().gt(1).any():
        raise ValueError("Each player ID must have one consistent player name.")
    if "home" in frame:
        text = frame["home"].astype(str).str.strip().str.lower()
        absent = frame["home"].isna() | text.eq("")
        mapped = text.map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if (mapped.isna() & ~absent).any():
            raise ValueError("Optional home column must be true/false, 1/0, or blank when unknown.")
        frame["home"] = mapped.astype("boolean")
    else:
        frame["home"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame = frame.sort_values(["player_id", "date"], kind="stable").reset_index(drop=True)
    days = pd.to_datetime(frame["date"]).groupby(frame["player_id"]).diff().dt.days
    frame["rest_days"] = (days - 1).fillna(2).clip(lower=0).astype(int)
    return frame


def read_csv(text: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(StringIO(text), dtype={"player_id": str, "game_id": str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError("Cannot parse this CSV. Use the downloadable template.") from exc
    return validate_frame(frame)


@lru_cache(maxsize=1)
def demo_frame() -> pd.DataFrame:
    rng = np.random.default_rng(20261027)
    players = [
        ("203999", "Nikola Jokic", "DEN", 35, [27, 12, 9, 1.4, 0.8]),
        ("1629029", "Luka Doncic", "LAL", 36, [29, 8, 8, 1.5, 0.5]),
        ("203507", "Giannis Antetokounmpo", "MIL", 34, [30, 11, 6, 1.0, 1.2]),
        ("1628369", "Jayson Tatum", "BOS", 36, [27, 8, 5, 1.1, 0.7]),
        ("1630162", "Anthony Edwards", "MIN", 35, [27, 6, 5, 1.3, 0.6]),
        ("201939", "Stephen Curry", "GSW", 33, [25, 5, 6, 1.0, 0.4]),
        ("1628983", "Shai Gilgeous-Alexander", "OKC", 34, [31, 5, 6, 1.8, 0.9]),
        ("1630169", "Tyrese Haliburton", "IND", 34, [20, 4, 10, 1.2, 0.6]),
        ("1627734", "Domantas Sabonis", "SAC", 36, [20, 13, 8, 0.8, 0.5]),
        ("1628973", "Jalen Brunson", "NYK", 35, [28, 4, 7, 0.9, 0.2]),
        ("201142", "Kevin Durant", "HOU", 35, [27, 7, 5, 0.8, 1.1]),
        ("203076", "Anthony Davis", "DAL", 34, [25, 12, 3, 1.1, 2.1]),
    ]
    rows = []
    for player_id, name, team, base_minutes, averages in players:
        date = datetime(2024, 10, 22)
        for game in range(72):
            gap = int(rng.choice([1, 2, 3, 4], p=[0.17, 0.49, 0.29, 0.05]))
            date += timedelta(days=gap)
            opponents = [t for t in TEAMS if t != team]
            opponent = str(rng.choice(opponents))
            minutes = round(float(np.clip(rng.normal(base_minutes, 3.4), 16, 44)), 1)
            rest_factor = 0.92 if gap == 1 else 1.025
            opponent_factor = 0.94 + TEAMS.index(opponent) * 0.012
            form = 1 + 0.045 * np.sin(game / 8)
            mean = np.array(averages) * minutes / base_minutes * rest_factor * opponent_factor * form
            latent = rng.gamma(shape=[30, 22, 18, 10, 9], scale=mean / [30, 22, 18, 10, 9])
            counts = rng.poisson(latent)
            rows.append(dict(
                date=date.strftime("%Y-%m-%d"), player_id=player_id, player_name=name,
                team=team, opponent=opponent, minutes=minutes, home=bool(game % 2),
                **dict(zip(STATS, map(int, counts))),
            ))
    return validate_frame(pd.DataFrame(rows))


def metadata(frame: pd.DataFrame, dataset_id: str, name: str, source: str) -> dict:
    return {
        "id": dataset_id, "name": name, "source": source, "rows": len(frame),
        "players": int(frame["player_id"].nunique()),
        "start_date": str(frame["date"].min()), "end_date": str(frame["date"].max()),
    }


def player_list(frame: pd.DataFrame) -> list[dict]:
    return [
        {
            "id": str(player_id), "name": str(group.iloc[-1]["player_name"]),
            "team": str(group.iloc[-1]["team"]), "games": len(group),
            "averages": {col: round(float(group[col].mean()), 2) for col in (*STATS, "minutes")},
        }
        for player_id, group in frame.groupby("player_id", sort=False)
    ]


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "courtside.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS game_data (id TEXT PRIMARY KEY, metadata TEXT NOT NULL, records TEXT NOT NULL)")
            legacy = db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'datasets'").fetchone()
            if legacy:
                # Preserve the original table as a backup; migrate each old dataset only once.
                rows = db.execute("""SELECT old.* FROM datasets old
                    LEFT JOIN game_data current ON old.id = current.id WHERE current.id IS NULL""").fetchall()
                for row in rows:
                    frame = read_csv(row["csv"])
                    db.execute("INSERT INTO game_data VALUES (?, ?, ?)", (
                        row["id"], row["metadata"], frame.to_json(orient="records", force_ascii=False),
                    ))
            db.execute("""CREATE TABLE IF NOT EXISTS lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                dataset_id TEXT NOT NULL, player_id TEXT NOT NULL, stat TEXT NOT NULL,
                event_date TEXT NOT NULL, book TEXT NOT NULL, line REAL NOT NULL,
                odds_over INTEGER NOT NULL, odds_under INTEGER NOT NULL
            )""")
            db.execute("CREATE TABLE IF NOT EXISTS provider_cache (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def datasets(self) -> list[dict]:
        demo = metadata(demo_frame(), "demo", "Synthetic season / 864 games", "synthetic")
        with self.connect() as db:
            saved = [json.loads(row["metadata"]) for row in db.execute("SELECT metadata FROM game_data ORDER BY rowid DESC")]
        return [demo, *sorted(saved, key=lambda info: info.get("fetched_at", ""), reverse=True)]

    def load(self, dataset_id: str) -> tuple[pd.DataFrame, dict]:
        if dataset_id == "demo":
            frame = demo_frame().copy()
            return frame, metadata(frame, "demo", "Synthetic season / 864 games", "synthetic")
        with self.connect() as db:
            row = db.execute("SELECT metadata, records FROM game_data WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise KeyError("Dataset not found.")
        return validate_frame(pd.DataFrame.from_records(json.loads(row["records"]))), json.loads(row["metadata"])

    def cached_nba(self, query: dict) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT metadata FROM game_data WHERE id = ?", (nba_cache_id(query),)).fetchone()
        return json.loads(row["metadata"]) if row else None

    def resource(self, key: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM provider_cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def save_resource(self, key: str, payload: dict) -> dict:
        result = {**payload, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="microseconds")}
        with self.connect() as db:
            db.execute("INSERT INTO provider_cache VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET payload = excluded.payload",
                       (key, json.dumps(result, allow_nan=False)))
        return result

    def save(self, frame: pd.DataFrame, name: str, source: str, nba_query: dict | None = None, attributes: dict | None = None) -> dict:
        frame = validate_frame(frame)
        dataset_id = nba_cache_id(nba_query) if nba_query is not None else uuid4().hex
        info = metadata(frame, dataset_id, name, source)
        if attributes:
            info.update(attributes)
        if nba_query is not None:
            info.update({
                "nba_query": {**nba_query, "player_ids": sorted(nba_query["player_ids"])},
                "nba_players": [{"id": p["id"], "name": p["name"]} for p in player_list(frame)],
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "provider": "NBA Stats / stats.nba.com",
            })
        with self.connect() as db:
            db.execute("""INSERT INTO game_data VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET metadata = excluded.metadata, records = excluded.records""", (
                dataset_id, json.dumps(info), frame.to_json(orient="records", force_ascii=False),
            ))
        return info

    def lines(self, dataset_id: str, player_id: str, stat: str, event_date: str | None) -> list[dict]:
        query = "SELECT * FROM lines WHERE dataset_id = ? AND player_id = ? AND stat = ?"
        values: list = [dataset_id, player_id, stat]
        if event_date:
            query += " AND event_date = ?"
            values.append(event_date)
        with self.connect() as db:
            return [dict(row) for row in db.execute(query + " ORDER BY id", values)]

    def add_line(self, values: dict) -> dict:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        columns = ("dataset_id", "player_id", "stat", "event_date", "book", "line", "odds_over", "odds_under")
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO lines (timestamp, " + ", ".join(columns) + ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [timestamp, *[values[col] for col in columns]],
            )
            row = db.execute("SELECT * FROM lines WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)
