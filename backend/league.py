from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
import math

from nba_api.stats.endpoints import commonallplayers, commonplayerinfo, playercareerstats
from nba_api.stats.static import teams
import pandas as pd
from requests.exceptions import RequestException

from .data import STATS
from .nba import NBAUnavailableError

AVERAGES = (*STATS, "minutes", "turnovers", "three_pointers_made")
CAREER_COLUMNS = {
    "points": "PTS", "rebounds": "REB", "assists": "AST", "steals": "STL", "blocks": "BLK",
    "minutes": "MIN", "turnovers": "TOV", "three_pointers_made": "FG3M",
}


def current_season(today: date | None = None) -> str:
    today = today or datetime.now(timezone.utc).date()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{(start + 1) % 100:02d}"


def request_tables(request: Callable[[], list[pd.DataFrame]], label: str) -> list[pd.DataFrame]:
    try:
        return request()
    except RequestException as exc:
        raise NBAUnavailableError(f"NBA Stats could not return {label}. The request timed out or was blocked; try again later.") from exc
    except (ValueError, KeyError) as exc:
        raise NBAUnavailableError(f"NBA Stats returned an unexpected {label} response.") from exc


def text(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    result = str(value).strip()
    return result or None


def integer(value) -> int | None:
    if text(value) is None:
        return None
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise NBAUnavailableError("NBA Stats returned an invalid identifier or count.") from exc
    if not math.isfinite(number) or number != int(number):
        raise NBAUnavailableError("NBA Stats returned a non-integer identifier or count.")
    return int(number)


def fetch_roster() -> dict:
    season = current_season()
    tables = request_tables(
        lambda: commonallplayers.CommonAllPlayers(is_only_current_season=1, league_id="00", season=season, timeout=20).get_data_frames(),
        "the current league roster",
    )
    required = {"PERSON_ID", "DISPLAY_FIRST_LAST", "ROSTERSTATUS", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_CITY", "TEAM_NAME"}
    if not tables or not required.issubset(tables[0].columns):
        raise NBAUnavailableError("The current NBA roster schema is incomplete.")
    frame = tables[0]
    if not frame["ROSTERSTATUS"].isin([0, 1]).all():
        raise NBAUnavailableError("NBA's roster response has missing or unrecognized roster-status values.")
    active = frame[frame["ROSTERSTATUS"] == 1]
    if active.empty or active["PERSON_ID"].duplicated().any():
        raise NBAUnavailableError("NBA returned an empty or ambiguous current roster.")
    roster = []
    for row in active.to_dict(orient="records"):
        player_id = integer(row["PERSON_ID"])
        name = text(row["DISPLAY_FIRST_LAST"])
        if player_id is None or name is None:
            raise NBAUnavailableError("An NBA roster entry is missing its identifier or name.")
        roster.append({
            "id": str(player_id), "name": name,
            "team_id": str(integer(row["TEAM_ID"]) or 0),
            "team": text(row["TEAM_ABBREVIATION"]) or "FA",
            "team_name": " ".join(filter(None, [text(row["TEAM_CITY"]), text(row["TEAM_NAME"])])) or "Unassigned",
            "from_year": integer(row.get("FROM_YEAR")), "to_year": integer(row.get("TO_YEAR")),
            "has_played": row.get("GAMES_PLAYED_FLAG") == "Y",
        })
    roster.sort(key=lambda player: player["name"].casefold())
    franchises = []
    for team_id in sorted({p["team_id"] for p in roster} - {"0"}):
        members = [p for p in roster if p["team_id"] == team_id]
        franchises.append({"id": team_id, "abbreviation": members[0]["team"], "name": members[0]["team_name"], "count": len(members)})
    expected_teams = {str(team["id"]) for team in teams.get_teams()}
    missing = expected_teams - {team["id"] for team in franchises}
    if missing:
        raise NBAUnavailableError(f"NBA's roster response is missing {len(missing)} team(s); a partial directory will not be presented as the full league.")
    return {
        "season": season, "players": roster, "teams": sorted(franchises, key=lambda team: team["name"]),
        "total_players": len(roster), "total_teams": len(franchises),
        "provider": "NBA Stats / CommonAllPlayers / current roster status",
    }


def fetch_profile(player_id: str) -> dict:
    tables = request_tables(lambda: commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=20).get_data_frames(), "the player profile")
    if not tables or tables[0].empty or not {"PERSON_ID", "DISPLAY_FIRST_LAST"}.issubset(tables[0].columns):
        raise NBAUnavailableError("NBA did not return a usable player profile.")
    row = tables[0].iloc[0]
    if str(integer(row["PERSON_ID"])) != player_id:
        raise NBAUnavailableError("NBA returned a profile for a different player.")
    birth = text(row.get("BIRTHDATE"))
    try:
        born = datetime.fromisoformat(birth.replace("Z", "+00:00")).date() if birth else None
    except ValueError as exc:
        raise NBAUnavailableError("NBA returned an unrecognized birth-date format.") from exc
    today = datetime.now(timezone.utc).date()
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day)) if born else None
    seasons = set()
    for table in tables[1:]:
        if "SEASON_ID" in table and len(table.columns) == 1:
            for value in table["SEASON_ID"]:
                year = str(value)[-4:]
                if year.isdigit():
                    seasons.add(f"{year}-{(int(year) + 1) % 100:02d}")
    return {
        "id": player_id, "name": str(row["DISPLAY_FIRST_LAST"]),
        "position": text(row.get("POSITION")), "jersey": text(row.get("JERSEY")),
        "height": text(row.get("HEIGHT")), "weight": text(row.get("WEIGHT")),
        "birth_date": born.isoformat() if born else None, "age": age,
        "experience": integer(row.get("SEASON_EXP")), "country": text(row.get("COUNTRY")),
        "school": text(row.get("SCHOOL")), "draft_year": text(row.get("DRAFT_YEAR")),
        "draft_round": text(row.get("DRAFT_ROUND")), "draft_pick": text(row.get("DRAFT_NUMBER")),
        "team": text(row.get("TEAM_ABBREVIATION")),
        "team_name": " ".join(filter(None, [text(row.get("TEAM_CITY")), text(row.get("TEAM_NAME"))])) or None,
        "roster_status": text(row.get("ROSTERSTATUS")),
        "available_seasons": sorted(seasons, reverse=True), "provider": "NBA Stats / CommonPlayerInfo",
    }


def summed(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    try:
        value = pd.to_numeric(frame[column], errors="raise").sum(min_count=1)
    except (ValueError, TypeError) as exc:
        raise NBAUnavailableError(f"NBA returned nonnumeric {column} totals.") from exc
    if pd.isna(value):
        return None
    if not math.isfinite(float(value)):
        raise NBAUnavailableError(f"NBA returned a non-finite {column} total.")
    return float(value)


def shooting(frame: pd.DataFrame, career: bool = False) -> dict:
    pairs = {
        "field_goal": ("FGM", "FGA") if career else ("field_goals_made", "field_goals_attempted"),
        "three_point": ("FG3M", "FG3A") if career else ("three_pointers_made", "three_pointers_attempted"),
        "free_throw": ("FTM", "FTA") if career else ("free_throws_made", "free_throws_attempted"),
    }
    result = {}
    for key, (makes, attempts) in pairs.items():
        if makes not in frame or attempts not in frame:
            result[key] = None
            continue
        paired = frame.dropna(subset=[makes, attempts])
        made, attempted = summed(paired, makes), summed(paired, attempts)
        result[key] = made / attempted if made is not None and attempted is not None and attempted > 0 else None
    return result


def summarize(frame: pd.DataFrame) -> dict:
    return {
        "games": len(frame),
        "averages": {col: float(frame[col].mean()) if col in frame and frame[col].notna().any() else None for col in AVERAGES},
        "shooting": shooting(frame),
    }


def summarize_history(frame: pd.DataFrame) -> tuple[dict, list[dict]]:
    frame = frame.sort_values("date")
    splits = []
    for count in (5, 10, 20):
        splits.append({"label": f"Last {count}", **summarize(frame.tail(count))})
    for home in (True, False):
        subset = frame[frame["home"] == home]
        if not subset.empty:
            splits.append({"label": "Home" if home else "Away", **summarize(subset)})
    if "season" in frame:
        for season, subset in frame.groupby("season", sort=True):
            splits.append({"label": f"Season {season}", **summarize(subset)})
    for opponent, subset in frame.groupby("opponent", sort=True):
        splits.append({"label": f"vs {opponent}", **summarize(subset)})
    return summarize(frame), splits


def career_line(frame: pd.DataFrame, season: str) -> dict:
    games = summed(frame, "GP")
    if games is None or games < 0:
        raise NBAUnavailableError("NBA career data has no valid game count.")
    count = int(games)
    starts = summed(frame, "GS")
    return {
        "season": season, "team": str(frame.iloc[0].get("TEAM_ABBREVIATION", "Career")) if len(frame) == 1 else "Multiple teams",
        "games": count, "starts": int(starts) if starts is not None else None,
        "averages": {key: value / count if count and (value := summed(frame, col)) is not None else None for key, col in CAREER_COLUMNS.items()},
        "shooting": shooting(frame, career=True),
    }


def career_seasons(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    if not {"SEASON_ID", "GP", "TEAM_ID"}.issubset(frame.columns):
        raise NBAUnavailableError("NBA career season fields are incomplete.")
    result = []
    for season, group in frame.groupby("SEASON_ID", sort=True):
        total = group[group["TEAM_ID"] == 0]
        if len(total) > 1:
            raise NBAUnavailableError("NBA returned duplicate career season totals.")
        result.append(career_line(total if not total.empty else group, str(season)))
    return list(reversed(result))


def fetch_career(player_id: str) -> dict:
    def request():
        response = playercareerstats.PlayerCareerStats(player_id=player_id, per_mode36="Totals", timeout=20)
        return [
            response.season_totals_regular_season.get_data_frame(), response.career_totals_regular_season.get_data_frame(),
            response.season_totals_post_season.get_data_frame(), response.career_totals_post_season.get_data_frame(),
        ]
    regular, total_regular, playoffs, total_playoffs = request_tables(request, "the player's career history")
    return {
        "player_id": player_id, "regular_season": career_seasons(regular), "playoffs": career_seasons(playoffs),
        "totals": {
            "regular_season": career_line(total_regular, "Career") if not total_regular.empty else None,
            "playoffs": career_line(total_playoffs, "Career") if not total_playoffs.empty else None,
        },
        "provider": "NBA Stats / PlayerCareerStats / totals converted to per-game averages",
    }
