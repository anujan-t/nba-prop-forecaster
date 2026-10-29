"""Monte Carlo game simulator: draws a player's whole box score many times from one joint model.

Each simulated game draws
  1. minutes from the player's recent minutes (recency-weighted and kernel-smoothed; a minutes scenario rescales them),
  2. a per-minute rate for every usage component (two- and three-point attempts, free-throw attempts, offensive and
     defensive rebounds, assists, steals, blocks, turnovers, fouls) that includes the uncertainty in estimating it from a
     few dozen games, plus a small shared role-drift factor,
  3. counts with the dispersion real NBA box scores show at fixed minutes (slightly tighter than Poisson for attempts,
     turnovers and fouls; looser for free-throw attempts, which arrive in pairs),
  4. makes from attempts at a shooting percentage drawn from its posterior, so points = 2 x made twos + 3 x made
     threes + made free throws.
Every market (points, threes, PRA, stocks, double-doubles, ...) is read from the same simulated games, so markets are
internally consistent and same-player parlay legs are priced jointly.

League-wide priors were measured on 8,417 real 2024-25 and 2025-26 player-games (70 players, every position and role).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from hashlib import sha256
import math
import threading

import numpy as np
import pandas as pd

from .stats import STAT_DEFS, definition

USAGE = ("fg2a", "fg3a", "fta", "oreb", "dreb", "assists", "steals", "blocks", "turnovers", "fouls")
SHOTS = (("fg2m", "fg2a"), ("fg3m", "fg3a"), ("ftm", "fta"))
BASIC = ("points", "rebounds", "assists", "steals", "blocks")
FULL_COLUMNS = ("field_goals_made", "field_goals_attempted", "three_pointers_made", "three_pointers_attempted",
                "free_throws_made", "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "turnovers", "fouls")

DEFAULT_SIMULATIONS = 250_000
MIN_SIMULATIONS = 1_000
MAX_SIMULATIONS = 1_000_000
SIMULATION_CHOICES = (100_000, 250_000, 500_000, 1_000_000)
EVALUATION_SIMULATIONS = 8_000  # walk-forward scoring: simulation noise far below the forecast error being measured
SEED = 20261028
# Recency half-lives in games (a game this many games older than the latest counts half as much), tuned in walk-forward
# tests: usage and minutes follow role changes quickly; rare events (steals, blocks) and shooting need longer memory.
HALF_LIFE = 12.0
MINUTES_HALF_LIFE = 10.0
SHOOTING_HALF_LIFE = 30.0
COMPONENT_HALF_LIFE = {
    "fg2a": 10.0, "fg3a": 10.0, "fta": 12.0, "assists": 10.0, "oreb": 15.0, "dreb": 12.0, "turnovers": 15.0,
    "fouls": 20.0, "steals": 30.0, "blocks": 30.0, "points": 10.0, "rebounds": 12.0,
}
DISPERSION_PRIOR_GAMES = 30.0
CONTEXT_PRIOR_GAMES = 12
ROLE_DRIFT = 0.0  # optional shared production drift; walk-forward tests found no gain beyond the rate uncertainty
MAX_MINUTES = 60.0
FOUL_LIMIT = 6
CACHE_DRAWS = 4_000_000  # simulated games kept in memory across recent forecasts

# Variance-to-mean ratio of each count at fixed minutes, pooled over 216 thirty-game windows (finite-sample corrected).
DISPERSION_PRIOR = {
    "fg2a": 0.94, "fg3a": 0.84, "fta": 1.96, "oreb": 1.11, "dreb": 1.02, "assists": 0.99, "steals": 1.01,
    "blocks": 1.07, "turnovers": 0.95, "fouls": 0.84, "points": 2.19, "rebounds": 1.07,
}
# League shooting percentage and the attempts of prior weight behind it (spread of true talent between players).
SHOOTING_PRIOR = {"fg2m": (0.54, 100.0), "fg3m": (0.36, 150.0), "ftm": (0.78, 25.0)}


def full_mode(history: pd.DataFrame) -> bool:
    """True when every game has made/attempted shooting, rebound splits, turnovers and fouls."""
    if not all(column in history.columns for column in FULL_COLUMNS):
        return False
    return not history.loc[:, list(FULL_COLUMNS)].isna().any().any()


def usage_counts(history: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    source = {"fg3a": "three_pointers_attempted", "fta": "free_throws_attempted", "oreb": "offensive_rebounds",
              "dreb": "defensive_rebounds"}
    columns = []
    for name in names:
        if name == "fg2a":
            values = history["field_goals_attempted"].astype(float) - history["three_pointers_attempted"].astype(float)
        else:
            values = history[source.get(name, name)].astype(float)
        columns.append(np.maximum(0.0, values.to_numpy(dtype=float)))
    return np.column_stack(columns)


def shot_counts(history: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    made_3 = history["three_pointers_made"].astype(float).to_numpy()
    att_3 = history["three_pointers_attempted"].astype(float).to_numpy()
    return {
        "fg2m": (history["field_goals_made"].astype(float).to_numpy() - made_3,
                 history["field_goals_attempted"].astype(float).to_numpy() - att_3),
        "fg3m": (made_3, att_3),
        "ftm": (history["free_throws_made"].astype(float).to_numpy(), history["free_throws_attempted"].astype(float).to_numpy()),
    }


def recency_weights(n: int, half_life: float = HALF_LIFE) -> np.ndarray:
    """Oldest first; the newest game has weight 1 and a game half_life games older has weight 1/2."""
    if half_life <= 0 or not math.isfinite(half_life):
        return np.ones(n)
    return 0.5 ** ((n - 1 - np.arange(n)) / half_life)


def effective_games(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.sum(weights ** 2))


def dispersion_estimate(counts: np.ndarray, minutes: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted per-minute rate of one count and its variance-to-mean ratio around rate x minutes (corrected for
    estimating the rate)."""
    rate = float(weights @ counts) / max(float(weights @ minutes), 1e-9)
    expected = minutes * rate
    denominator = float(weights @ expected)
    if denominator <= 0:
        return rate, 1.0
    n_eff = effective_games(weights)
    ratio = float(weights @ ((counts - expected) ** 2)) / denominator
    return rate, ratio * n_eff / max(n_eff - 1, 1.0)


def component_half_life(name: str, half_life: float | None) -> float:
    return COMPONENT_HALF_LIFE.get(name, HALF_LIFE) if half_life is None else half_life


@dataclass
class GameModel:
    names: tuple[str, ...]  # usage components drawn as counts
    rates: np.ndarray  # per minute, after context factors
    dispersion: np.ndarray  # variance-to-mean ratio at fixed minutes
    uncertainty: np.ndarray  # relative variance of each estimated rate
    shooting: dict[str, tuple[float, float]]  # made component -> Beta posterior (made, missed) pseudo-counts
    minutes: np.ndarray  # historical minutes, oldest first
    minute_weights: np.ndarray
    bandwidth: float
    minutes_scale: float
    rest_factors: np.ndarray
    opponent_factors: np.ndarray
    games: int
    drift: float = ROLE_DRIFT
    notes: list[str] = field(default_factory=list)

    @property
    def full(self) -> bool:
        return bool(self.shooting)

    def fingerprint(self) -> str:
        parts = [",".join(self.names), self.rates.round(12).tobytes().hex(), self.dispersion.round(12).tobytes().hex(),
                 self.uncertainty.round(12).tobytes().hex(), self.minutes.round(6).tobytes().hex(),
                 self.minute_weights.round(12).tobytes().hex(), f"{self.bandwidth:.10f}", f"{self.minutes_scale:.10f}",
                 f"{self.drift:.6f}", repr(sorted((k, round(a, 9), round(b, 9)) for k, (a, b) in self.shooting.items()))]
        return sha256("|".join(parts).encode()).hexdigest()[:24]

    def percentage(self, made: str) -> float:
        a, b = self.shooting[made]
        return a / (a + b)

    def expected(self, column: str, rates: np.ndarray | None = None) -> float | None:
        """Mean per-minute value of a box-score column under the model (or under alternative usage rates)."""
        rate = dict(zip(self.names, self.rates if rates is None else rates))
        if not self.full:
            return rate.get(column)
        made = {"fg2m": rate["fg2a"] * self.percentage("fg2m"), "fg3m": rate["fg3a"] * self.percentage("fg3m"),
                "ftm": rate["fta"] * self.percentage("ftm")}
        values = {
            "points": 2 * made["fg2m"] + 3 * made["fg3m"] + made["ftm"], "rebounds": rate["oreb"] + rate["dreb"],
            "field_goals_made": made["fg2m"] + made["fg3m"], "field_goals_attempted": rate["fg2a"] + rate["fg3a"],
            "three_pointers_made": made["fg3m"], "three_pointers_attempted": rate["fg3a"], "free_throws_made": made["ftm"],
            "free_throws_attempted": rate["fta"], "offensive_rebounds": rate["oreb"], "defensive_rebounds": rate["dreb"],
        }
        return values.get(column, rate.get(column))

    def factor(self, stat: str, which: str) -> float:
        """The context factor a market inherits from its components: 'rest', 'opponent' or 'combined'."""
        spec = definition(stat)
        if spec.milestone:
            return 1.0
        base = self.rates / (self.rest_factors * self.opponent_factors)
        multiplier = {"rest": self.rest_factors, "opponent": self.opponent_factors,
                      "combined": self.rest_factors * self.opponent_factors}[which]
        before = [self.expected(column, base) for column in spec.columns]
        after = [self.expected(column, base * multiplier) for column in spec.columns]
        if any(value is None for value in (*before, *after)) or sum(before) <= 0:
            return 1.0
        return float(sum(after) / sum(before))


def _context_factor(counts: np.ndarray, minutes: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """Per-component per-minute rate ratio in a subset of games, shrunk toward 1 and capped at ±20%."""
    n = int(subset.sum())
    if n == 0:
        return np.ones(counts.shape[1])
    base = counts.sum(axis=0) / max(minutes.sum(), 1e-9)
    rate = counts[subset].sum(axis=0) / max(minutes[subset].sum(), 1e-9)
    ratio = np.divide(rate, base, out=np.ones_like(base), where=base > 0)
    return np.clip(1 + n / (n + CONTEXT_PRIOR_GAMES) * (ratio - 1), 0.8, 1.2)


def fit_game_model(history: pd.DataFrame, expected_minutes: float | None = None, rest_days: int | None = None,
                   opponent: str | None = None, use_context: bool = True, half_life: float | None = None,
                   drift: float | None = None) -> GameModel:
    """Rates, dispersion, rate uncertainty and shooting from the trailing games (oldest first)."""
    if len(history) < 2:
        raise ValueError("At least two previous games are required.")
    drift = ROLE_DRIFT if drift is None else drift
    full = full_mode(history)
    names = USAGE if full else BASIC
    counts = usage_counts(history, names)
    minutes = history["minutes"].to_numpy(dtype=float)
    n = len(minutes)
    rates = np.zeros(len(names))
    dispersion = np.ones(len(names))
    uncertainty = np.zeros(len(names))
    for k, name in enumerate(names):
        weights_k = recency_weights(n, component_half_life(name, half_life))
        n_eff_k = effective_games(weights_k)
        rates[k], observed = dispersion_estimate(counts[:, k], minutes, weights_k)
        prior = DISPERSION_PRIOR.get(name, 1.0)
        dispersion[k] = float(np.clip((n_eff_k * observed + DISPERSION_PRIOR_GAMES * prior) / (n_eff_k + DISPERSION_PRIOR_GAMES), 0.5, 6.0))
        # Sampling variance of the rate estimate relative to its square (counts with that dispersion).
        total = float(weights_k @ counts[:, k]) * n_eff_k / weights_k.sum()
        uncertainty[k] = min(1.0, dispersion[k] / total) if total > 0 and rates[k] > 0 else 0.0
    minute_weights = recency_weights(n, MINUTES_HALF_LIFE if half_life is None else half_life)
    weighted_minutes = float(np.sum(minute_weights * minutes) / minute_weights.sum())
    notes = []
    if half_life is None:
        notes.append(f"Recent games count more: minutes and usage use a {MINUTES_HALF_LIFE:g}-game half-life, steals and "
                     f"blocks {COMPONENT_HALF_LIFE['steals']:g} games, shooting percentages {SHOOTING_HALF_LIFE:g} games.")
    elif math.isfinite(half_life):
        notes.append(f"Recent games count more: a game {half_life:g} games before the latest has half its weight.")

    scale = 1.0
    rest_factors = np.ones(len(names))
    opponent_factors = np.ones(len(names))
    if use_context:
        if expected_minutes is not None:
            scale = expected_minutes / weighted_minutes
            if not 0.1 <= scale <= 3:
                raise ValueError("Minutes scenarios must stay between 10% and 300% of the player's trailing minutes; this scenario is outside the model's scope.")
            notes.append(f"Minutes scenario: {expected_minutes:.1f}, rescaling recent minutes (recency-weighted average {weighted_minutes:.1f}).")
        else:
            notes.append(f"Minutes are drawn from the recent minutes distribution (recency-weighted average {weighted_minutes:.1f}), never a game's actual future minutes.")
        if rest_days is not None and "rest_days" in history:
            subset = history["rest_days"].clip(upper=3).to_numpy() == min(rest_days, 3)
            rest_factors = _context_factor(counts, minutes, subset)
        if opponent and "opponent" in history:
            subset = history["opponent"].astype(str).str.upper().to_numpy() == opponent.upper()
            opponent_factors = _context_factor(counts, minutes, subset)
    else:
        notes.append("Context adjustments disabled; minutes follow the recent minutes distribution without a scenario.")

    shooting = {}
    if full:
        shot_weights = recency_weights(n, SHOOTING_HALF_LIFE if half_life is None else half_life)
        shot_scale = effective_games(shot_weights) / shot_weights.sum()
        for made_name, (made, attempts) in shot_counts(history).items():
            league, prior_attempts = SHOOTING_PRIOR[made_name]
            weighted_made = float(shot_weights @ made) * shot_scale
            weighted_attempts = float(shot_weights @ attempts) * shot_scale
            shooting[made_name] = (weighted_made + prior_attempts * league,
                                   max(0.0, weighted_attempts - weighted_made) + prior_attempts * (1 - league))

    spread = float(np.sqrt(np.sum(minute_weights * (minutes - weighted_minutes) ** 2) / minute_weights.sum()))
    bandwidth = max(0.75, 0.9 * spread * effective_games(minute_weights) ** -0.2)
    return GameModel(names, rates * rest_factors * opponent_factors, dispersion, uncertainty, shooting, minutes,
                     minute_weights / minute_weights.sum(), bandwidth, scale, rest_factors, opponent_factors, n, drift, notes)


def _counts(rng: np.random.Generator, lam: np.ndarray, dispersion: float) -> np.ndarray:
    """Counts with mean lam and variance dispersion x lam (binomial when tighter than Poisson, gamma-Poisson when looser)."""
    if abs(dispersion - 1) < 0.02:
        return rng.poisson(lam)
    if dispersion < 1:
        trials = np.ceil(lam / (1 - dispersion)).astype(np.int64)
        chance = np.divide(lam, trials, out=np.zeros_like(lam), where=trials > 0)
        return rng.binomial(trials, np.clip(chance, 0.0, 1.0))
    shape = np.maximum(lam / (dispersion - 1), 1e-9)
    return rng.poisson(rng.gamma(shape, 1.0) * (dispersion - 1))


@dataclass
class Simulation:
    columns: dict[str, np.ndarray]  # simulated components and made shots (int16)
    minutes: np.ndarray
    model: GameModel

    @property
    def size(self) -> int:
        return int(len(self.minutes))

    def base(self, column: str) -> np.ndarray | None:
        """A box-score column rebuilt from the simulated components, or None when it is not simulated."""
        get = lambda name: self.columns[name].astype(np.int32)  # noqa: E731
        if not self.model.full:
            return get(column) if column in self.columns else None
        derived = {
            "points": lambda: 2 * get("fg2m") + 3 * get("fg3m") + get("ftm"),
            "rebounds": lambda: get("oreb") + get("dreb"),
            "field_goals_made": lambda: get("fg2m") + get("fg3m"),
            "field_goals_attempted": lambda: get("fg2a") + get("fg3a"),
            "three_pointers_made": lambda: get("fg3m"), "three_pointers_attempted": lambda: get("fg3a"),
            "free_throws_made": lambda: get("ftm"), "free_throws_attempted": lambda: get("fta"),
            "offensive_rebounds": lambda: get("oreb"), "defensive_rebounds": lambda: get("dreb"),
        }
        if column in derived:
            return derived[column]()
        return get(column) if column in self.columns else None

    def supports(self, stat: str) -> bool:
        spec = STAT_DEFS.get(stat)
        return spec is not None and all(self.base(column) is not None for column in spec.columns)

    def market(self, stat: str) -> np.ndarray:
        spec = definition(stat)
        parts = [self.base(column) for column in spec.columns]
        if any(part is None for part in parts):
            raise ValueError(f"The simulator needs shooting, rebound, turnover and foul splits for {spec.label.lower()}.")
        if spec.milestone:
            return (sum((part >= 10).astype(np.int32) for part in parts) >= spec.threshold).astype(np.int32)
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        return total


def simulate(model: GameModel, simulations: int = DEFAULT_SIMULATIONS, seed: int = SEED) -> Simulation:
    """Joint draws of every component for `simulations` games from one seeded generator (repeatable)."""
    simulations = int(simulations)
    if not MIN_SIMULATIONS <= simulations <= MAX_SIMULATIONS:
        raise ValueError(f"Simulations must be between {MIN_SIMULATIONS:,} and {MAX_SIMULATIONS:,}.")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(model.minutes), size=simulations, p=model.minute_weights)
    minutes = np.abs(model.minutes[picks] + model.bandwidth * rng.standard_normal(simulations))
    minutes = np.minimum(minutes * model.minutes_scale, MAX_MINUTES)
    drift = np.exp(model.drift * rng.standard_normal(simulations) - model.drift ** 2 / 2) if model.drift > 0 else 1.0
    columns: dict[str, np.ndarray] = {}
    for index, name in enumerate(model.names):
        rate = model.rates[index]
        if rate <= 0:
            columns[name] = np.zeros(simulations, dtype=np.int64)
            continue
        variance = model.uncertainty[index]
        estimate = rng.gamma(1 / variance, variance, simulations) if variance > 1e-9 else 1.0
        lam = rate * minutes * drift * estimate
        columns[name] = _counts(rng, lam, float(model.dispersion[index]))
    if "fouls" in columns:
        columns["fouls"] = np.minimum(columns["fouls"], FOUL_LIMIT)
    for made_name, attempts_name in SHOTS if model.full else ():
        made_prior, missed_prior = model.shooting[made_name]
        chance = rng.beta(made_prior, missed_prior, simulations)
        columns[made_name] = rng.binomial(columns[attempts_name], chance)
    return Simulation({name: values.astype(np.int16) for name, values in columns.items()}, minutes.astype(np.float32), model)


class SimulationCache:
    """Recent simulations keyed by the fitted model, simulation count and seed (a fitted model is deterministic)."""

    def __init__(self, budget: int = CACHE_DRAWS):
        self.budget = budget
        self._items: OrderedDict[str, Simulation] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, model: GameModel, simulations: int, seed: int = SEED) -> Simulation:
        key = f"{model.fingerprint()}:{int(simulations)}:{seed}"
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        result = simulate(model, simulations, seed)
        with self._lock:
            self._items[key] = result
            while len(self._items) > 1 and sum(item.size for item in self._items.values()) > self.budget:
                self._items.popitem(last=False)
        return result

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


CACHE = SimulationCache()


def standard_error(probability: float, simulations: int) -> float:
    """Binomial simulation error of a probability estimated from `simulations` independent games."""
    return math.sqrt(max(probability * (1 - probability), 0.0) / max(simulations, 1))


def probabilities(values: np.ndarray, line: float) -> tuple[float, float, float]:
    over = float(np.count_nonzero(values > line)) / len(values)
    under = float(np.count_nonzero(values < line)) / len(values)
    return over, under, max(0.0, 1.0 - over - under)
