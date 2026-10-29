"""Every player-prop market COURTSIDE models, as integer per-game series built from box-score columns.

Box-score counts (points, threes, turnovers, ...), sums of them (PRA, stocks, ...) and milestones
(double-double, triple-double) all become one number per game, so every market shares the same forecast,
backtest, parlay, grading and calibration code.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

MILESTONE_CATEGORIES = ("points", "rebounds", "assists", "steals", "blocks")


@dataclass(frozen=True)
class StatDef:
    id: str
    label: str
    short: str
    unit: str
    group: str  # core | combo | shooting | other | milestone
    columns: tuple[str, ...]  # box-score columns summed per game (milestones: the categories counted)
    family: str = "large"  # large: normal under Auto; small: Poisson / negative binomial under Auto
    threshold: int = 0  # milestones: categories that must reach 10
    simulated: bool = True  # the joint game simulator produces this market

    @property
    def milestone(self) -> bool:
        return self.group == "milestone"


CATALOG: tuple[StatDef, ...] = (
    StatDef("points", "Points", "PTS", "pts", "core", ("points",)),
    StatDef("rebounds", "Rebounds", "REB", "reb", "core", ("rebounds",)),
    StatDef("assists", "Assists", "AST", "ast", "core", ("assists",)),
    StatDef("threes", "3-pointers made", "3PM", "3pm", "core", ("three_pointers_made",), "small"),
    StatDef("steals", "Steals", "STL", "stl", "core", ("steals",), "small"),
    StatDef("blocks", "Blocks", "BLK", "blk", "core", ("blocks",), "small"),
    StatDef("turnovers", "Turnovers", "TOV", "tov", "core", ("turnovers",), "small"),
    StatDef("pts_reb_ast", "Pts + Reb + Ast", "PRA", "pra", "combo", ("points", "rebounds", "assists")),
    StatDef("pts_reb", "Pts + Reb", "PR", "pr", "combo", ("points", "rebounds")),
    StatDef("pts_ast", "Pts + Ast", "PA", "pa", "combo", ("points", "assists")),
    StatDef("reb_ast", "Reb + Ast", "RA", "ra", "combo", ("rebounds", "assists")),
    StatDef("stocks", "Steals + Blocks", "STK", "stk", "combo", ("steals", "blocks"), "small"),
    StatDef("field_goals_made", "Field goals made", "FGM", "fgm", "shooting", ("field_goals_made",), "small"),
    StatDef("field_goals_attempted", "Field goal attempts", "FGA", "fga", "shooting", ("field_goals_attempted",)),
    StatDef("threes_attempted", "3-point attempts", "3PA", "3pa", "shooting", ("three_pointers_attempted",), "small"),
    StatDef("free_throws_made", "Free throws made", "FTM", "ftm", "shooting", ("free_throws_made",), "small"),
    StatDef("free_throws_attempted", "Free throw attempts", "FTA", "fta", "shooting", ("free_throws_attempted",), "small"),
    StatDef("offensive_rebounds", "Offensive rebounds", "OREB", "oreb", "other", ("offensive_rebounds",), "small"),
    StatDef("defensive_rebounds", "Defensive rebounds", "DREB", "dreb", "other", ("defensive_rebounds",), "small"),
    StatDef("fouls", "Personal fouls", "PF", "pf", "other", ("fouls",), "small"),
    StatDef("double_double", "Double-double", "DD", "dd", "milestone", MILESTONE_CATEGORIES, threshold=2),
    StatDef("triple_double", "Triple-double", "TD", "td", "milestone", MILESTONE_CATEGORIES, threshold=3),
)
STAT_DEFS: dict[str, StatDef] = {stat.id: stat for stat in CATALOG}
PROP_STATS: tuple[str, ...] = tuple(stat.id for stat in CATALOG)
GROUPS = {"core": "Core", "combo": "Combos", "shooting": "Shooting", "other": "Rebounding & fouls", "milestone": "Milestones"}
MILESTONE_LINE = 0.5


def definition(stat: str) -> StatDef:
    try:
        return STAT_DEFS[stat]
    except KeyError as exc:
        raise ValueError(f"Unknown statistic {stat!r}.") from exc


def label(stat: str) -> str:
    return definition(stat).label


def catalog() -> list[dict]:
    """The markets as the API lists them."""
    return [{"id": s.id, "label": s.label, "short": s.short, "unit": s.unit, "group": s.group, "group_label": GROUPS[s.group],
             "milestone": s.milestone, "simulated": s.simulated, "components": list(s.columns)} for s in CATALOG]


def stat_values(frame: pd.DataFrame, stat: str) -> pd.Series:
    """Per-game float values of a market; NaN where a needed box-score column is missing or blank."""
    spec = definition(stat)
    if any(column not in frame.columns for column in spec.columns):
        return pd.Series(np.nan, index=frame.index, dtype=float)
    parts = [pd.to_numeric(frame[column], errors="coerce").astype(float) for column in spec.columns]
    if spec.milestone:
        known = pd.concat(parts, axis=1).notna().all(axis=1)
        reached = sum((part >= 10).astype(float) for part in parts)
        return (reached >= spec.threshold).astype(float).where(known)
    total = parts[0].copy()
    for part in parts[1:]:
        total = total + part
    return total


def with_stat(frame: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Rows where the market is known, with the market as an integer column named after it."""
    values = stat_values(frame, stat)
    known = values.notna()
    result = frame.loc[known].copy()
    result[stat] = values[known].astype(int)
    return result


def with_stats(frame: pd.DataFrame, stats: Iterable[str]) -> pd.DataFrame:
    """A copy with a float column per market (NaN where unknown); base columns keep their original values."""
    result = frame.copy()
    for stat in stats:
        if stat in result.columns and not definition(stat).milestone and definition(stat).columns == (stat,):
            continue
        result[stat] = stat_values(frame, stat)
    return result


def available(frame: pd.DataFrame, stat: str) -> int:
    """How many rows have every box-score column this market needs."""
    return int(stat_values(frame, stat).notna().sum())


def game_value(game: dict, stat: str) -> float | None:
    """A market's value in one box-score row (dict), or None when a needed column is missing."""
    values = stat_values(pd.DataFrame([game]), stat)
    value = values.iloc[0]
    return None if pd.isna(value) else float(value)


def validate_line(stat: str, line: float) -> None:
    if definition(stat).milestone and line != MILESTONE_LINE:
        raise ValueError(f"{label(stat)} is a yes/no market: use the 0.5 line (over = yes, under = no).")
