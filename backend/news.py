"""NBA newsroom: headlines, the league injury report and roster moves (ESPN's public JSON feeds), plus coaching staffs
and data-driven team playing styles (NBA Stats via nba_api).

ESPN's site API is unofficial and unauthenticated, like the NBA Stats endpoints: it can change, time out or block
requests. Failures are reported; the app falls back to the last saved copy and says how old it is.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import re
from typing import Callable

import httpx
from nba_api.stats.endpoints import commonteamroster, leaguedashteamstats, synergyplaytypes
from nba_api.stats.static import teams as static_teams
import pandas as pd
from requests.exceptions import RequestException

from .nba import NBAUnavailableError, normalize_name

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
TIMEOUT = 15
HEADERS = {"User-Agent": "Mozilla/5.0 (COURTSIDE local NBA research app)", "Accept": "application/json"}
# ESPN team ids to NBA abbreviations.
ESPN_TEAMS = {
    "1": "ATL", "2": "BOS", "3": "NOP", "4": "CHI", "5": "CLE", "6": "DAL", "7": "DEN", "8": "DET", "9": "GSW", "10": "HOU",
    "11": "IND", "12": "LAC", "13": "LAL", "14": "MIA", "15": "MIL", "16": "MIN", "17": "BKN", "18": "NYK", "19": "ORL",
    "20": "PHI", "21": "PHX", "22": "POR", "23": "SAC", "24": "SAS", "25": "OKC", "26": "UTA", "27": "WAS", "28": "TOR",
    "29": "MEM", "30": "CHA",
}
ESPN_ABBREVIATIONS = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
KINDS = ("injury", "transaction", "coaching", "game", "betting", "other")
KIND_PATTERNS = (
    ("injury", re.compile(r"\b(injur\w*|ruled out|rule out|rules out|questionable|doubtful|probable|day-to-day|week-to-week|"
                          r"sidelined|surgery|sprain\w*|strain\w*|fracture\w*|torn|tear|hamstring|ankle|knee|calf|achilles|"
                          r"groin|wrist|shoulder|concussion|illness|re-evaluated|out for|out at least|out \d|to miss|"
                          r"will miss|misses|(?<!\w)out (?:vs\.?|against|tonight|indefinitely)|available vs|upgraded?|"
                          r"downgraded?|return(?:s|ing)? from|season debut|load management)", re.I)),
    ("coaching", re.compile(r"\b(coach(es|ing)?|head coach|assistant coach|front office|general manager|team president|ceo)\b", re.I)),
    ("transaction", re.compile(r"\b(sign(s|ed|ing)?|waive[sd]?|waiving|trade[sd]?|trading|acquire[sd]?|release[sd]?|"
                               r"team option|player option|exercise[sd]? .*option|extension|contract|two-way|claim(s|ed)?|"
                               r"buyout|free agen\w+|agree[sd]? to|re-sign\w*|convert\w*|suspend\w*|fine[sd]?)\b", re.I)),
    ("betting", re.compile(r"\b(fantasy|bets?|betting|odds|picks|props?|sleepers?|adds and)\b", re.I)),
    ("game", re.compile(r"\b(beat|beats|victory|win over|loss to|highlights|rout|edge[sd]?|top[sp]?|rally|buzzer-beater)\b", re.I)),
)
STATUS_KEYS = {"out": "out", "doubtful": "doubtful", "questionable": "questionable", "probable": "probable",
               "day-to-day": "day_to_day", "suspension": "suspended", "suspended": "suspended"}
STATUS_ORDER = {"out": 0, "suspended": 1, "doubtful": 2, "questionable": 3, "day_to_day": 4, "probable": 5, "other": 6}
PLAY_TYPES = {
    "Transition": "Transition", "PRBallHandler": "Pick-and-roll ball handler", "PRRollman": "Pick-and-roll roll man",
    "Isolation": "Isolation", "Spotup": "Spot-up", "Postup": "Post-up", "Handoff": "Handoff", "Cut": "Cut",
    "OffScreen": "Off screen", "OffRebound": "Putbacks", "Misc": "Miscellaneous",
}


class NewsUnavailableError(RuntimeError):
    pass


def _get(path: str, params: dict | None = None) -> dict:
    try:
        response = httpx.get(f"{ESPN}/{path}", params=params, timeout=TIMEOUT, headers=HEADERS, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise NewsUnavailableError("ESPN's NBA feed could not be reached (timeout or network restriction). Try again later.") from exc
    if response.status_code != 200:
        raise NewsUnavailableError(f"ESPN's NBA feed returned HTTP {response.status_code}. Try again later.")
    try:
        data = response.json()
    except ValueError as exc:
        raise NewsUnavailableError("ESPN's NBA feed returned an unreadable response.") from exc
    if not isinstance(data, dict):
        raise NewsUnavailableError("ESPN's NBA feed returned an unexpected response.")
    return data


def _text(value) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def nba_abbreviation(espn: str | None) -> str | None:
    if not espn:
        return None
    return ESPN_ABBREVIATIONS.get(espn.upper(), espn.upper())


def classify(headline: str, description: str | None, article_type: str | None) -> str:
    """injury (incl. availability), coaching (incl. front office), transaction, betting, game or other."""
    if article_type in ("Recap", "Preview"):
        return "game"
    text = f"{headline} {description or ''}"
    for kind, pattern in KIND_PATTERNS:
        if pattern.search(text):
            return kind
    return "other"


def player_directory(roster: list[dict]) -> dict[tuple[str, str], str]:
    """(normalized name, team) and (normalized name, '') keys to NBA player ids; ambiguous names map to ''."""
    directory: dict[tuple[str, str], str] = {}
    for player in roster:
        key = normalize_name(player["name"]).replace(".", "").replace("'", "")
        for team in (player.get("team") or "", ""):
            directory[(key, team)] = "" if (key, team) in directory and directory[(key, team)] != player["id"] else player["id"]
    return directory


def match_player(directory: dict[tuple[str, str], str], name: str | None, team: str | None) -> str | None:
    if not name:
        return None
    key = normalize_name(name).replace(".", "").replace("'", "")
    for candidate in ((key, team or ""), (key, "")):
        player_id = directory.get(candidate)
        if player_id:
            return player_id
    return None


def fetch_headlines(roster: list[dict] | None = None, limit: int = 60) -> dict:
    data = _get("news", {"limit": limit})
    articles = data.get("articles")
    if not isinstance(articles, list):
        raise NewsUnavailableError("ESPN's headline feed changed its format; no headlines were read.")
    directory = player_directory(roster or [])
    items = []
    for article in articles:
        headline = _text(article.get("headline"))
        if not headline:
            continue
        categories = article.get("categories") or []
        team_codes = sorted({ESPN_TEAMS[str(c.get("teamId"))] for c in categories
                             if c.get("type") == "team" and str(c.get("teamId")) in ESPN_TEAMS})
        athletes = []
        for category in categories:
            if category.get("type") == "athlete" and _text(category.get("description")):
                name = _text(category.get("description"))
                athletes.append({"name": name, "player_id": match_player(directory, name, None)})
        images = article.get("images") or []
        image = next((img.get("url") for img in images if isinstance(img, dict) and str(img.get("url", "")).startswith("https://")), None)
        url = ((article.get("links") or {}).get("web") or {}).get("href")
        description = _text(article.get("description"))
        items.append({
            "id": str(article.get("id") or headline), "type": _text(article.get("type")) or "Story",
            "kind": classify(headline, description, article.get("type")), "headline": headline,
            "description": description if description != headline else None, "published": article.get("published"),
            "byline": _text(article.get("byline")), "url": url if isinstance(url, str) and url.startswith("https://") else None,
            "image": image, "teams": team_codes, "athletes": athletes, "premium": bool(article.get("premium")),
            "video": article.get("type") == "Media",
        })
    items.sort(key=lambda item: item.get("published") or "", reverse=True)
    counts = {kind: sum(item["kind"] == kind for item in items) for kind in KINDS}
    return {"articles": items, "counts": counts, "provider": "ESPN public NBA news feed (unofficial JSON)"}


def fetch_injuries(roster: list[dict] | None = None) -> dict:
    data = _get("injuries")
    teams = data.get("injuries")
    if not isinstance(teams, list):
        raise NewsUnavailableError("ESPN's injury feed changed its format; no injury report was read.")
    directory = player_directory(roster or [])
    result, counts = [], {}
    for team in teams:
        team_code = ESPN_TEAMS.get(str(team.get("id")))
        entries = []
        for injury in team.get("injuries") or []:
            athlete = injury.get("athlete") or {}
            name = _text(athlete.get("displayName"))
            if not name:
                continue
            code = nba_abbreviation((athlete.get("team") or {}).get("abbreviation")) or team_code
            status = _text(injury.get("status")) or "Unknown"
            key = STATUS_KEYS.get(status.lower(), "other")
            details = injury.get("details") or {}
            entries.append({
                "player": name, "player_id": match_player(directory, name, code),
                "position": _text((athlete.get("position") or {}).get("abbreviation")), "team": code,
                "status": status, "status_key": key, "date": injury.get("date"),
                "injury": _text(details.get("type")), "detail": _text(details.get("detail")), "side": _text(details.get("side")),
                "return_date": _text(details.get("returnDate")),
                "short_comment": _text(injury.get("shortComment")), "long_comment": _text(injury.get("longComment")),
                "headshot": ((athlete.get("headshot") or {}).get("href")
                             if str((athlete.get("headshot") or {}).get("href", "")).startswith("https://") else None),
            })
            counts[key] = counts.get(key, 0) + 1
        entries.sort(key=lambda entry: (STATUS_ORDER.get(entry["status_key"], 9), entry["player"]))
        if team_code:
            result.append({"team": team_code, "team_name": _text(team.get("displayName")) or team_code, "injuries": entries})
    result.sort(key=lambda team: team["team_name"])
    return {"teams": result, "counts": counts, "total": sum(counts.values()),
            "provider": "ESPN public NBA injury feed (unofficial JSON); statuses are reported, not official league filings"}


def fetch_transactions(limit: int = 60) -> dict:
    data = _get("transactions", {"limit": limit})
    rows = data.get("transactions")
    if not isinstance(rows, list):
        raise NewsUnavailableError("ESPN's transactions feed changed its format; no roster moves were read.")
    items = []
    for row in rows:
        description = _text(row.get("description"))
        if not description:
            continue
        team = row.get("team") or {}
        items.append({"date": row.get("date"), "description": description,
                      "team": ESPN_TEAMS.get(str(team.get("id"))) or nba_abbreviation(team.get("abbreviation")),
                      "team_name": _text(team.get("displayName"))})
    items.sort(key=lambda item: item.get("date") or "", reverse=True)
    return {"transactions": items, "provider": "ESPN public NBA transactions feed (unofficial JSON)"}


# ---------- NBA Stats: coaches and playing styles ----------

def nba_teams() -> list[dict]:
    return sorted(({"id": str(team["id"]), "abbreviation": team["abbreviation"], "name": team["full_name"],
                    "city": team["city"], "nickname": team["nickname"]} for team in static_teams.get_teams()),
                  key=lambda team: team["name"])


def _request(call: Callable[[], list[pd.DataFrame]], label: str) -> list[pd.DataFrame]:
    try:
        return call()
    except RequestException as exc:
        raise NBAUnavailableError(f"NBA Stats could not return {label}. The request timed out or was blocked; try again later.") from exc
    except (ValueError, KeyError) as exc:
        raise NBAUnavailableError(f"NBA Stats returned an unexpected {label} response.") from exc


def fetch_coaches(team_id: str, season: str) -> dict:
    tables = _request(lambda: commonteamroster.CommonTeamRoster(team_id=team_id, season=season, timeout=20).get_data_frames(),
                      "the coaching staff")
    if len(tables) < 2 or "COACH_NAME" not in tables[1].columns:
        raise NBAUnavailableError("NBA Stats returned no coaching staff for this team.")
    staff = []
    for row in tables[1].to_dict(orient="records"):
        name = _text(row.get("COACH_NAME"))
        if name:
            staff.append({"name": name, "role": _text(row.get("COACH_TYPE")) or "Coach",
                          "head": str(row.get("COACH_TYPE", "")).lower() == "head coach"})
    staff.sort(key=lambda coach: (not coach["head"], coach["role"] != "Assistant Coach", coach["name"]))
    return {"team_id": team_id, "season": season, "coaches": staff, "provider": "NBA Stats / CommonTeamRoster coaching staff"}


def _rank(frame: pd.DataFrame, column: str, ascending: bool = False) -> pd.Series:
    return frame[column].rank(ascending=ascending, method="min").astype(int)


def _ordinal(number: int) -> str:
    suffix = "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


METRICS = (
    # key, source column, label, higher_is_rank_1, percent
    ("pace", "PACE", "Pace (possessions per 48)", True, False),
    ("offense", "OFF_RATING", "Offensive rating", True, False),
    ("defense", "DEF_RATING", "Defensive rating", False, False),
    ("net", "NET_RATING", "Net rating", True, False),
    ("three_rate", "PCT_FGA_3PT", "Share of shots from three", True, True),
    ("paint_share", "PCT_PTS_PAINT", "Share of points in the paint", True, True),
    ("fast_break_share", "PCT_PTS_FB", "Share of points on the fast break", True, True),
    ("assisted", "PCT_AST_FGM", "Made shots assisted", True, True),
    ("free_throw_rate", "FTA_RATE", "Free-throw rate", True, False),
    ("turnover_rate", "TM_TOV_PCT", "Turnover rate", False, True),
    ("offensive_rebounding", "OREB_PCT", "Offensive rebound rate", True, True),
    ("opponent_efg", "OPP_EFG_PCT", "Opponent effective FG%", False, True),
    ("forced_turnovers", "OPP_TOV_PCT", "Opponent turnover rate", True, True),
    ("opponent_points", "OPP_PTS", "Opponent points per game", False, False),
    ("opponent_threes", "OPP_FG3M", "Opponent threes made per game", False, False),
    ("opponent_rebounds", "OPP_REB", "Opponent rebounds per game", False, False),
    ("opponent_assists", "OPP_AST", "Opponent assists per game", False, False),
)


def _style_tags(metrics: dict, play_types: list[dict], teams: int) -> list[str]:
    tags = []
    rank = {key: value["rank"] for key, value in metrics.items()}
    top, bottom = 5, teams - 4

    def check(key: str, high: str, low: str | None = None) -> None:
        if key in rank and rank[key] <= top:
            tags.append(high)
        elif low and key in rank and rank[key] >= bottom:
            tags.append(low)

    check("pace", "Plays fast", "Plays slow")
    check("three_rate", "Three-point heavy", "Few threes")
    check("paint_share", "Paint-first attack")
    check("fast_break_share", "Runs in transition")
    check("assisted", "Ball movement", "Creates off the dribble")
    check("offensive_rebounding", "Crashes the offensive glass")
    check("free_throw_rate", "Lives at the free-throw line")
    check("offense", "Elite offense", "Struggling offense")
    check("defense", "Elite defense", "Leaky defense")
    check("forced_turnovers", "Forces turnovers")
    by_type = {item["play_type"]: item for item in play_types}
    for key, tag in (("Isolation", "Isolation-heavy"), ("PRBallHandler", "Pick-and-roll heavy"), ("Postup", "Post-up heavy"),
                     ("Spotup", "Spot-up shooting"), ("Cut", "Cutting offense"), ("Handoff", "Handoff actions")):
        if key in by_type and by_type[key]["frequency_rank"] <= 3:
            tags.append(tag)
    return list(dict.fromkeys(tags))


def fetch_team_styles(season: str) -> dict:
    """League-wide team profiles for one season: record, ratings, pace, shot profile, four factors, what each team
    allows, and offensive play-type frequencies, each with its league rank."""
    measures = ("Base", "Advanced", "Four Factors", "Misc", "Scoring", "Opponent")

    def measure(kind: str) -> pd.DataFrame:
        return _request(lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense=kind, per_mode_detailed="PerGame", timeout=25,
        ).get_data_frames(), f"team {kind.lower()} statistics")[0]

    def play_type(kind: str) -> pd.DataFrame:
        return _request(lambda: synergyplaytypes.SynergyPlayTypes(
            league_id="00", per_mode_simple="PerGame", player_or_team_abbreviation="T", season_type_all_star="Regular Season",
            season=season, type_grouping_nullable="offensive", play_type_nullable=kind, timeout=25,
        ).get_data_frames(), "play-type statistics")[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = dict(zip(measures, pool.map(measure, measures)))
        play_frames = dict(zip(PLAY_TYPES, pool.map(lambda kind: _safe_play_type(play_type, kind), PLAY_TYPES)))
    merged = None
    for kind, frame in frames.items():
        if frame.empty or "TEAM_ID" not in frame:
            raise NBAUnavailableError(f"NBA Stats returned no {kind.lower()} team statistics for {season}.")
        keep = ["TEAM_ID", *[column for column in frame.columns if column not in ("TEAM_ID",) and not column.endswith("_RANK")
                             and (merged is None or column not in merged.columns)]]
        merged = frame[keep] if merged is None else merged.merge(frame[keep], on="TEAM_ID", how="inner")
    if merged is None or merged.empty:
        raise NBAUnavailableError(f"NBA Stats returned no team statistics for {season}.")
    count = len(merged)
    by_id = {team["id"]: team for team in nba_teams()}
    plays_by_team: dict[str, list[dict]] = {}
    for kind, frame in play_frames.items():
        if frame is None or frame.empty or "POSS_PCT" not in frame:
            continue
        frame = frame.assign(_rank=_rank(frame, "POSS_PCT"))
        for row in frame.to_dict(orient="records"):
            plays_by_team.setdefault(str(int(row["TEAM_ID"])), []).append({
                "play_type": kind, "label": PLAY_TYPES[kind], "frequency": float(row["POSS_PCT"]),
                "frequency_rank": int(row["_rank"]), "points_per_possession": float(row["PPP"]),
                "percentile": float(row["PERCENTILE"]) if pd.notna(row.get("PERCENTILE")) else None,
            })
    ranks = {key: _rank(merged, column, ascending=not higher) for key, column, _, higher, _ in METRICS if column in merged}
    result = []
    for index, row in merged.reset_index(drop=True).iterrows():
        team_id = str(int(row["TEAM_ID"]))
        info = by_id.get(team_id, {"abbreviation": str(row.get("TEAM_ABBREVIATION", "")), "name": str(row.get("TEAM_NAME", ""))})
        metrics = {}
        for key, column, label, _, percent in METRICS:
            if column in merged and pd.notna(row[column]):
                metrics[key] = {"label": label, "value": float(row[column]), "rank": int(ranks[key].iloc[index]), "percent": percent}
        plays = sorted(plays_by_team.get(team_id, []), key=lambda item: -item["frequency"])
        result.append({
            "team_id": team_id, "team": info["abbreviation"], "name": info["name"],
            "games": int(row["GP"]), "wins": int(row["W"]), "losses": int(row["L"]),
            "metrics": metrics, "play_types": plays, "tags": _style_tags(metrics, plays, count),
        })
    result.sort(key=lambda team: team["name"])
    return {"season": season, "teams": result, "team_count": count,
            "provider": "NBA Stats / LeagueDashTeamStats and Synergy play types (offense)"}


def _safe_play_type(fetch: Callable[[str], pd.DataFrame], kind: str) -> pd.DataFrame | None:
    """Play types are a bonus: one failed type leaves the rest of the profile intact."""
    try:
        return fetch(kind)
    except NBAUnavailableError:
        return None


def style_sentence(team: dict) -> str:
    """A plain-English summary of a team's style from its ranks."""
    metrics = team.get("metrics", {})
    parts = []
    if "pace" in metrics:
        parts.append(f"{_ordinal(metrics['pace']['rank'])} in pace")
    if "offense" in metrics and "defense" in metrics:
        parts.append(f"{_ordinal(metrics['offense']['rank'])} on offense and {_ordinal(metrics['defense']['rank'])} on defense")
    if "three_rate" in metrics:
        parts.append(f"{_ordinal(metrics['three_rate']['rank'])} in three-point rate")
    plays = team.get("play_types") or []
    if plays:
        top = plays[0]
        parts.append(f"most-used action: {top['label'].lower()} ({top['frequency']:.0%} of possessions)")
    return "; ".join(parts) + "." if parts else ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
