"""Parlay pricing: per-leg forecasts joined by a Gaussian copula fitted to shared games."""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm, spearmanr

from .engine import american_probability, forecast, profit_per_unit

MAX_LEGS = 10
CORRELATION_WINDOW = 30  # most recent games per player used for correlations and shared-game history
MIN_SHARED_GAMES = 10
SHRINKAGE_GAMES = 10  # estimates are scaled by n / (n + 10) toward zero
MAX_CORRELATION = 0.95
SEED = 20261028


class LegNotFound(LookupError):
    pass


def fair_odds(probability: float) -> dict | None:
    if not 0 < probability < 1:
        return None
    american = -100 * probability / (1 - probability) if probability > 0.5 else 100 * (1 - probability) / probability
    return {"decimal": 1 / probability, "american": int(round(american))}


def latent_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float]:
    """Spearman rank correlation and the shrunk Gaussian-copula correlation it implies."""
    n = len(x)
    if n < MIN_SHARED_GAMES or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, 0.0
    rank = float(spearmanr(x, y).statistic)
    if not math.isfinite(rank):
        return None, 0.0
    # Pearson correlation of a Gaussian copula with this rank correlation, shrunk toward independence.
    latent = 2 * math.sin(math.pi * rank / 6) * n / (n + SHRINKAGE_GAMES)
    return rank, float(np.clip(latent, -MAX_CORRELATION, MAX_CORRELATION))


def valid_correlation(matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    """Nearest usable correlation matrix by eigenvalue clipping; reports whether it changed."""
    values, vectors = np.linalg.eigh(matrix)
    if values.min() >= 1e-8:
        return matrix, False
    repaired = vectors @ np.diag(np.clip(values, 1e-6, None)) @ vectors.T
    scale = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    return repaired, True


def orthant(probabilities: list[float], correlation: np.ndarray) -> float:
    """P(W_i < Φ⁻¹(p_i) for all i) for standard normals W with the given correlation."""
    p = np.asarray(probabilities, dtype=float)
    if (p <= 0).any():
        return 0.0
    keep = p < 1
    if not keep.any():
        return 1.0
    p, correlation = p[keep], correlation[np.ix_(keep, keep)]
    if len(p) == 1 or np.allclose(correlation, np.eye(len(p)), atol=1e-12):
        return float(np.prod(p))
    value = float(multivariate_normal.cdf(
        norm.ppf(p), mean=np.zeros(len(p)), cov=correlation, allow_singular=True,
        abseps=1e-7, releps=1e-6, rng=np.random.default_rng(SEED),
    ))
    # Respect the Fréchet bounds that every joint probability must satisfy.
    return float(np.clip(value, max(0.0, float(p.sum()) - (len(p) - 1)), float(p.min())))


def leg_label(number: int, name: str, stat: str) -> str:
    return f"Leg {number} ({name} {stat})"


def calibrated(priced: list[dict], oriented: np.ndarray, calibrator) -> dict | None:
    """Recalibrates each leg's chance of settling over (when the learning loop has an active map) and re-joins them."""
    if calibrator is None:
        return None
    adjusted = []
    for leg in priced:
        loss = max(0.0, 1 - leg["probability"] - leg["push"])
        settled = leg["probability"] + loss
        if settled <= 1e-12:
            return None
        over = leg["probability"] / settled if leg["side"] == "over" else loss / settled
        mapped = calibrator.calibrate_over(over, leg["stat"])
        if mapped is None:
            return None
        adjusted.append(float((mapped if leg["side"] == "over" else 1 - mapped) * settled))
    status = calibrator.calibration_status()
    return {"probability": orthant(adjusted, oriented), "legs": adjusted, "samples": int(status.get("samples") or 0),
            "note": f"Each leg recalibrated from {int(status.get('samples') or 0):,} graded predictions, then joined with the same correlations."}


def won(values: pd.Series, line: float, side: str) -> pd.Series:
    return values > line if side == "over" else values < line


def price_parlay(legs: list[dict], book_odds: int | None, load: Callable[[str], tuple[pd.DataFrame, dict]],
                 calibrator=None) -> dict:
    if not 1 <= len(legs) <= MAX_LEGS:
        raise ValueError(f"A parlay needs 1 to {MAX_LEGS} legs.")
    if book_odds is not None:
        american_probability(book_odds)
    datasets: dict[str, tuple[pd.DataFrame, dict]] = {}
    seen: dict[tuple[str, str], int] = {}
    priced, warnings = [], []
    for number, leg in enumerate(legs, start=1):
        if leg["dataset_id"] not in datasets:
            try:
                datasets[leg["dataset_id"]] = load(leg["dataset_id"])
            except KeyError as exc:
                raise LegNotFound(f"Leg {number}: its saved history was not found. Re-run the forecast for this player, then add the leg again.") from exc
        frame, info = datasets[leg["dataset_id"]]
        rows = frame[frame["player_id"] == leg["player_id"]]
        if rows.empty:
            raise LegNotFound(f"Leg {number}: the player is not in the selected history. Re-run the forecast, then add the leg again.")
        name = str(rows.iloc[-1]["player_name"])
        key = (leg["player_id"], leg["stat"])
        if key in seen:
            raise ValueError(f"{leg_label(number, name, leg['stat'])}: leg {seen[key]} already uses this player and stat. "
                             "A second line on the same stat is redundant or contradictory; edit or remove one of them.")
        seen[key] = number
        request = {k: v for k, v in leg.items() if k != "side"} | {"odds_over": -110, "odds_under": -110}
        try:
            result = forecast(frame, info, request)
        except ValueError as exc:
            raise ValueError(f"{leg_label(number, name, leg['stat'])}: {exc}") from exc
        chances = result["probabilities"]
        sample = pd.Series([game["value"] for game in result["history"]], dtype=float)
        probability = chances[leg["side"]]
        # Only whole-number lines can land exactly on the line; drop floating-point residue for half points.
        push = chances["push"] if float(leg["line"]).is_integer() else 0.0
        priced.append({
            "index": number - 1, "player": result["player"], "stat": leg["stat"], "line": leg["line"], "side": leg["side"],
            "model": leg["model"], "resolved_model": result["resolved_model"], "sample_size": result["sample_size"],
            "mean": result["summary"]["mean"], "probability": probability, "push": push,
            "hit_rate": {"hits": int(won(sample, leg["line"], leg["side"]).sum()), "games": len(sample)},
            "fair_odds": fair_odds(probability),
        })
        warnings.extend(warning for warning in result["warnings"] if warning not in warnings)

    # Each player's recent games across every history used for their legs (one row per date).
    games: dict[str, pd.DataFrame] = {}
    for leg in legs:
        frame, _ = datasets[leg["dataset_id"]]
        rows = frame.loc[frame["player_id"] == leg["player_id"], ["date", "team", "points", "rebounds", "assists", "steals", "blocks"]]
        games[leg["player_id"]] = pd.concat([games.get(leg["player_id"]), rows]).drop_duplicates("date", keep="last").sort_values("date")
    window: dict[str, int] = {}
    for leg in legs:
        window[leg["player_id"]] = max(window.get(leg["player_id"], CORRELATION_WINDOW), leg["lookback"])
    recent = {player: rows.tail(window[player]) for player, rows in games.items()}
    team = {player: str(rows.iloc[-1]["team"]) for player, rows in games.items()}

    def shared(players: list[str]) -> pd.DataFrame:
        """Recent games of one player, or recent games several players played together for their current team."""
        unique = list(dict.fromkeys(players))
        merged = recent[unique[0]][["date", "team"]]
        if len(unique) == 1:
            return merged
        for player in unique[1:]:
            merged = merged.merge(recent[player][["date", "team"]], on=["date", "team"])
        return merged[merged["team"] == team[unique[0]]]

    def values(player: str, stat: str, dates: pd.DataFrame) -> np.ndarray:
        return dates.merge(recent[player][["date", stat]], on="date")[stat].to_numpy(dtype=float)

    size = len(legs)
    correlation = np.eye(size)
    signs = np.array([-1.0 if leg["side"] == "over" else 1.0 for leg in legs])
    pairs, linked = [], list(range(size))

    def root(i: int) -> int:
        while linked[i] != i:
            linked[i] = linked[linked[i]]
            i = linked[i]
        return i

    for a, b in combinations(range(size), 2):
        first, second = legs[a], legs[b]
        pa, pb = first["player_id"], second["player_id"]
        same = pa == pb
        if not same and team[pa] != team[pb]:
            pairs.append({"a": a, "b": b, "basis": "independent", "games": 0, "rank_correlation": None, "correlation": 0.0,
                          "note": f"Different teams ({team[pa]} and {team[pb]}): treated as independent."})
            continue
        linked[root(a)] = root(b)
        dates = shared([pa, pb])
        x, y = values(pa, first["stat"], dates), values(pb, second["stat"], dates)
        rank, latent = latent_correlation(x, y)
        correlation[a, b] = correlation[b, a] = latent
        subject = f"Same player, {first['stat']} and {second['stat']}" if same else f"Teammates on {team[pa]}"
        if rank is None:
            reason = (f"only {len(dates)} shared games" if len(dates) < MIN_SHARED_GAMES
                      else "one of the stats did not vary over the shared games")
            note = f"{subject}: {reason}, so no linkage is assumed."
        else:
            effect = signs[a] * signs[b] * latent
            strength = "tend to" if abs(effect) >= 0.3 else "lean toward"
            tendency = (f"these legs {strength} {'win together' if effect > 0 else 'work against each other'}"
                        if abs(effect) >= 0.1 else "little linkage between these legs")
            note = f"{subject}: rank correlation {rank:+.2f} over {len(dates)} shared games; {tendency}."
        pairs.append({"a": a, "b": b, "basis": "same_player" if same else "teammates", "games": int(len(dates)),
                      "rank_correlation": rank, "correlation": latent, "note": note})

    correlation, repaired = valid_correlation(correlation)
    oriented = correlation * np.outer(signs, signs)
    p = [leg["probability"] for leg in priced]
    probability = orthant(p, oriented)
    independent = float(np.prod(p))
    pushes = [leg["push"] for leg in priced]
    no_loss = orthant([min(1.0, hit + push) for hit, push in zip(p, pushes)], oriented) if max(pushes) > 0 else None

    history = []
    groups: dict[int, list[int]] = {}
    for index in range(size):
        groups.setdefault(root(index), []).append(index)
    for members in groups.values():
        if len(members) < 2:
            continue
        players = [legs[i]["player_id"] for i in members]
        dates = shared(players)
        hits = pd.Series(True, index=range(len(dates)))
        for i in members:
            leg = legs[i]
            hits &= won(pd.Series(values(leg["player_id"], leg["stat"], dates)), leg["line"], leg["side"])
        history.append({
            "legs": members, "games": int(len(dates)), "hits": int(hits.sum()) if len(dates) else 0,
            "model_probability": orthant([p[i] for i in members], oriented[np.ix_(members, members)]),
        })

    book = None
    if book_odds is not None:
        implied = american_probability(book_odds)
        book = {"odds": book_odds, "implied": implied, "edge": probability - implied,
                "expected_roi": probability * profit_per_unit(book_odds) - (1 - probability)}
    calibration = calibrated(priced, oriented, calibrator)

    if any(pair["basis"] == "independent" for pair in pairs):
        warnings.append("Legs for players on different teams are treated as independent. If those players meet in the same game, "
                        "shared pace and score can make them move together.")
    if any(pair["basis"] != "independent" for pair in pairs):
        warnings.append("Linked-leg correlations come from each player's recent shared games (the last 30, or a longer lookback) "
                        "and are estimates, not guarantees.")
    if repaired:
        warnings.append("The estimated correlations were adjusted slightly so they form a valid joint model.")
    if max(pushes) > 0:
        warnings.append("Whole-number lines can push. The parlay probability counts a push as not winning, and the expected return "
                        "treats it as a loss; sportsbooks usually shorten the parlay instead.")
    return {
        "legs": priced, "pairs": pairs, "probability": probability, "independent_probability": independent,
        "correlation_lift": probability / independent if independent > 0 else None,
        "no_loss_probability": no_loss, "fair_odds": fair_odds(probability), "book": book, "history": history,
        "calibration": calibration,
        "method": ("Each leg uses the same model as the Forecast lab. Legs for the same player, or for current teammates, are linked "
                   "with a Gaussian copula: rank correlations from shared recent games are converted to the latent scale "
                   "(2·sin(πρ/6)) and shrunk toward zero by n/(n+10). Legs for players on different teams are independent. "
                   "The chance that every leg wins is the multivariate normal orthant probability (Genz algorithm)."),
        "warnings": warnings, "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
