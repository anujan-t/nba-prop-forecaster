from __future__ import annotations

from datetime import datetime
import unicodedata

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
from requests.exceptions import RequestException

from .data import validate_frame


class NBAUnavailableError(RuntimeError):
    pass


class NoGamesError(ValueError):
    pass


BOX_FIELDS = {
    "FGM": "field_goals_made", "FGA": "field_goals_attempted",
    "FG3M": "three_pointers_made", "FG3A": "three_pointers_attempted",
    "FTM": "free_throws_made", "FTA": "free_throws_attempted",
    "OREB": "offensive_rebounds", "DREB": "defensive_rebounds",
    "TOV": "turnovers", "PF": "fouls", "PLUS_MINUS": "plus_minus",
}


def normalize_name(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", value.casefold()) if not unicodedata.combining(char))


def search_players(search: str) -> list[dict]:
    needle = normalize_name(search.strip())
    matches = [p for p in players.get_players() if needle in normalize_name(p["full_name"])]
    matches.sort(key=lambda p: (not p["is_active"], p["full_name"]))
    return [{"id": str(p["id"]), "name": p["full_name"]} for p in matches[:25]]


def static_player_ids() -> set[str]:
    return {str(player["id"]) for player in players.get_players()}


def import_game_logs(player_ids: list[str], season: str, season_type: str, player_names: dict[str, str] | None = None) -> pd.DataFrame:
    directory = {str(p["id"]): p["full_name"] for p in players.get_players()}
    if player_names:
        directory.update(player_names)
    if any(player_id not in directory for player_id in player_ids):
        raise ValueError("An NBA player ID was not found in nba_api's player directory.")
    rows = []
    for player_id in player_ids:
        try:
            response = playergamelog.PlayerGameLog(
                player_id=player_id, season=season, season_type_all_star=season_type, timeout=20,
            )
            tables = response.get_data_frames()
        except RequestException as exc:
            raise NBAUnavailableError(
                "NBA Stats could not be reached (timeout or network restriction). "
                "Try again later. No synthetic data has been substituted."
            ) from exc
        except (ValueError, KeyError) as exc:
            raise NBAUnavailableError("NBA Stats returned an unexpected response. Try again later.") from exc
        if not tables:
            raise NBAUnavailableError("NBA Stats returned no data table.")
        table = tables[0]
        required = {"GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST", "STL", "BLK"}
        if table.empty:
            raise NoGamesError(f"No completed games for {directory[player_id]} in {season} ({season_type}).")
        if not required.issubset(table.columns):
            raise NBAUnavailableError("NBA Stats changed its game-log schema; no dataset was imported.")
        for row in table.itertuples():
            if pd.isna(row.MIN) or float(row.MIN) <= 0:
                continue
            matchup = str(row.MATCHUP).split()
            if len(matchup) != 3:
                raise NBAUnavailableError("An NBA matchup had an unsupported format.")
            try:
                game_date = datetime.strptime(str(row.GAME_DATE), "%b %d, %Y").strftime("%Y-%m-%d")
            except ValueError as exc:
                raise NBAUnavailableError("An NBA game date had an unsupported format.") from exc
            rows.append({
                "date": game_date, "player_id": player_id, "player_name": directory[player_id],
                "team": matchup[0], "opponent": matchup[2], "home": matchup[1] == "vs.",
                "minutes": float(row.MIN), "points": row.PTS, "rebounds": row.REB,
                "assists": row.AST, "steals": row.STL, "blocks": row.BLK,
                "season": season, "game_id": str(getattr(row, "Game_ID", "")) or None,
                "win": getattr(row, "WL", None),
                **{target: getattr(row, source, None) for source, target in BOX_FIELDS.items()},
            })
    if not rows:
        raise NoGamesError("NBA returned no games with positive minutes.")
    return validate_frame(pd.DataFrame(rows))
