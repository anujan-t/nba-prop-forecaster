from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

from .data import STATS

MODELS = ("empirical", "normal", "poisson", "negative_binomial", "auto")
MIN_TRAIN = 15


def american_probability(odds: int) -> float:
    if isinstance(odds, bool) or int(odds) != odds or not 100 <= abs(odds) <= 10000:
        raise ValueError("American odds must be whole numbers with absolute value between 100 and 10,000.")
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def profit_per_unit(odds: int) -> float:
    american_probability(odds)
    return odds / 100 if odds > 0 else 100 / -odds


def market_comparison(over: float, under: float, push: float, odds_over: int, odds_under: int) -> dict:
    implied_over = american_probability(odds_over)
    implied_under = american_probability(odds_under)
    total = implied_over + implied_under
    settled = over + under
    conditional = over / settled if settled > 1e-12 else None
    return {
        "over_implied": implied_over, "under_implied": implied_under,
        "no_vig_over": implied_over / total, "no_vig_under": implied_under / total,
        "overround": total - 1, "model_over_conditional": conditional,
        "edge_over": conditional - implied_over / total if conditional is not None else None,
        "edge_under": 1 - conditional - implied_under / total if conditional is not None else None,
        "expected_roi_over": over * profit_per_unit(odds_over) - under,
        "expected_roi_under": under * profit_per_unit(odds_under) - over,
    }


@dataclass
class CountDistribution:
    model: str
    values: np.ndarray
    mean: float
    std: float
    frozen: object | None = None
    normal_floor: float = 0.0

    def cdf(self, k: float) -> float:
        k = math.floor(k)
        if k < 0:
            return 0.0
        if self.model == "empirical":
            return float(np.mean(self.values <= k))
        if self.model == "normal":
            return float(np.clip(
                (self.frozen.cdf(k + 0.5) - self.normal_floor) / (1 - self.normal_floor), 0, 1
            ))
        return float(self.frozen.cdf(k))

    def probabilities(self, line: float) -> tuple[float, float, float]:
        if not math.isfinite(line) or line < 0 or line > 200 or line * 2 != int(line * 2):
            raise ValueError("The prop line must be a non-negative integer or half-integer, at most 200.")
        over = max(0.0, 1 - self.cdf(math.floor(line)))
        under = self.cdf(math.ceil(line) - 1)
        push = max(0.0, 1 - over - under)
        return over, under, push

    def quantile(self, probability: float) -> int:
        if self.model == "empirical":
            return int(np.quantile(self.values, probability, method="inverted_cdf"))
        if self.model == "normal":
            value = self.frozen.ppf(self.normal_floor + probability * (1 - self.normal_floor))
            return max(0, int(math.floor(float(value) + 0.5)))
        return max(0, int(self.frozen.ppf(probability)))

    def histogram(self) -> list[dict]:
        maximum = max(5, self.quantile(0.9995))
        previous = 0.0
        histogram = []
        for value in range(maximum + 1):
            current = self.cdf(value)
            histogram.append({"value": value, "probability": max(0.0, current - previous)})
            previous = current
        return histogram


def context_adjustment(
    history: pd.DataFrame, stat: str, expected_minutes: float | None,
    rest_days: int | None, opponent: str | None, enabled: bool,
) -> tuple[float, dict, list[str]]:
    historical_minutes = float(history["minutes"].mean())
    projected = expected_minutes if expected_minutes is not None else historical_minutes
    details = {"projected_minutes": projected, "rest_factor": 1.0, "opponent_factor": 1.0}
    notes = []
    if not enabled:
        details["projected_minutes"] = historical_minutes
        return 1.0, details, ["Context adjustments disabled; using the trailing game totals."]
    base_rate = float(history[stat].sum() / history["minutes"].sum())
    factor = projected / historical_minutes
    if not 0.1 <= factor <= 3:
        raise ValueError("Minutes scenarios must stay between 10% and 300% of the player's trailing minutes; this scenario is outside the model's scope.")
    if expected_minutes is not None:
        notes.append(f"Minutes scenario: {projected:.1f} versus a trailing average of {historical_minutes:.1f}.")
    else:
        notes.append(f"Projected minutes use the trailing average ({historical_minutes:.1f}); no actual future minutes.")
    if base_rate > 0:
        subsets = []
        if rest_days is not None:
            rest_bucket = min(rest_days, 3)
            subsets.append(("rest_factor", history[history["rest_days"].clip(upper=3) == rest_bucket], "rest"))
        if opponent:
            subsets.append(("opponent_factor", history[history["opponent"] == opponent.upper()], "opponent"))
        for key, subset, label in subsets:
            n = len(subset)
            if n:
                ratio = float(subset[stat].sum() / subset["minutes"].sum()) / base_rate
                shrinkage = n / (n + 12)
                adjustment = float(np.clip(1 + shrinkage * (ratio - 1), 0.8, 1.2))
                details[key] = adjustment
                factor *= adjustment
                notes.append(f"{label.title()} rate adjustment {adjustment:.3f}x from {n} prior games, shrunk with 12-game prior weight.")
            else:
                notes.append(f"No prior games for this {label}; no {label} adjustment applied.")
    else:
        notes.append("All selected stat totals are zero; context cannot estimate a non-zero rate.")
    return factor, details, notes


def fit_distribution(history: pd.DataFrame, stat: str, model: str, factor: float = 1.0) -> CountDistribution:
    if stat not in STATS or model not in MODELS:
        raise ValueError("Unknown stat or model.")
    if len(history) < 2:
        raise ValueError("At least two previous games are required.")
    values = history[stat].to_numpy(dtype=float)
    observed_mean = float(values.mean())
    observed_var = float(values.var(ddof=1))
    mean = observed_mean * factor
    # A count-rate scenario scales the Poisson noise linearly and excess variance quadratically.
    variance = max(mean, mean + max(0.0, observed_var - observed_mean) * factor * factor)
    if model == "auto":
        model = ("negative_binomial" if observed_var > 1.5 * observed_mean else "poisson") if stat in ("steals", "blocks") else "normal"
    if model == "empirical":
        adjusted = np.maximum(0, np.floor(values * factor + 0.5))
        return CountDistribution(model, adjusted, float(adjusted.mean()), float(adjusted.std(ddof=1)))
    if model == "normal":
        std = max(0.5, math.sqrt(observed_var) * factor)
        frozen = norm(loc=mean, scale=std)
        floor = float(frozen.cdf(-0.5))
        support = np.arange(max(5, int(math.ceil(float(frozen.ppf(1 - 1e-10))))) + 1)
        masses = (frozen.cdf(support + 0.5) - frozen.cdf(support - 0.5)) / (1 - floor)
        discrete_mean = float(np.sum(support * masses))
        discrete_std = math.sqrt(float(np.sum((support - discrete_mean) ** 2 * masses)))
        return CountDistribution(model, values, discrete_mean, discrete_std, frozen, floor)
    if model == "negative_binomial" and variance > mean + 1e-8 and mean > 0:
        size = mean * mean / (variance - mean)
        frozen = nbinom(size, size / (size + mean))
        return CountDistribution(model, values, mean, math.sqrt(variance), frozen)
    return CountDistribution("poisson", values, mean, math.sqrt(mean), poisson(mean))


def forecast(frame: pd.DataFrame, dataset: dict, request: dict) -> dict:
    stat = request["stat"]
    player_games = frame[frame["player_id"] == request["player_id"]].sort_values("date")
    if player_games.empty:
        raise ValueError("Player not found in the selected dataset.")
    if len(player_games) < 10:
        raise ValueError("At least 10 completed games are required for a forecast.")
    history = player_games.tail(request["lookback"])
    factor, details, notes = context_adjustment(
        history, stat, request.get("expected_minutes"), request.get("rest_days"),
        request.get("opponent"), request["use_context"],
    )
    fitted = fit_distribution(history, stat, request["model"], factor)
    over, under, push = fitted.probabilities(request["line"])
    comparisons = []
    for model in MODELS[:-1]:
        candidate = fit_distribution(history, stat, model, factor)
        p_over, p_under, p_push = candidate.probabilities(request["line"])
        comparisons.append({"model": model, "over": p_over, "under": p_under, "push": p_push, "mean": candidate.mean})
    warnings = []
    if dataset["source"] == "synthetic":
        warnings.append("Synthetic demonstration data, not actual NBA results. Named players are illustrative. This is not evidence of real predictive accuracy.")
    elif dataset["source"] == "csv":
        warnings.append("User-supplied CSV data; provenance has not been independently verified.")
    if len(history) < 30:
        warnings.append("Small sample: fewer than 30 games. Calibration and tail estimates are unstable.")
    variance = float(history[stat].var(ddof=1))
    if fitted.model == "poisson" and variance > 1.5 * max(float(history[stat].mean()), 0.01):
        warnings.append("Observed variance exceeds the Poisson mean assumption. Compare the negative-binomial and empirical models.")
    if fitted.model == "normal" and stat in ("steals", "blocks"):
        warnings.append("Normal approximation is weak for low counts. Compare a discrete count model.")
    if fitted.model == "empirical":
        warnings.append("Empirical probabilities cannot assign mass beyond observed counts; scenario adjustment rounds scaled counts.")
    if request["model"] == "negative_binomial" and fitted.model == "poisson":
        warnings.append("No excess variance was estimated; the negative-binomial model uses its Poisson limit.")
    if request["model"] == "auto":
        notes.append("Auto is a transparent rule, not tuned model selection: normal for points/rebounds/assists; Poisson or overdispersed negative-binomial for steals/blocks.")
    warnings.append("No injury, lineup, usage-change or live odds feed. Context adjustments are exploratory, not causal effects.")
    last = player_games.iloc[-1]
    market = market_comparison(over, under, push, request["odds_over"], request["odds_under"])
    if market["overround"] < 0:
        warnings.append("Entered two-sided odds imply a negative overround; verify they refer to the same book and event.")
    if market["model_over_conditional"] is None:
        warnings.append("This model predicts only a push at this line; a conditional over probability or edge is undefined.")
    return {
        "dataset": dataset, "player": {"id": request["player_id"], "name": str(last["player_name"]), "team": str(last["team"])},
        "stat": stat, "line": request["line"], "model": request["model"], "resolved_model": fitted.model,
        "sample_size": len(history),
        "summary": {
            "mean": fitted.mean, "median": fitted.quantile(0.5), "std": fitted.std,
            "observed_mean": float(history[stat].mean()), "context_factor": factor, **details,
        },
        "probabilities": {"over": over, "under": under, "push": push}, "market": market,
        "distribution": fitted.histogram(),
        "interval": {"low": fitted.quantile(0.1), "high": fitted.quantile(0.9), "coverage": 0.8},
        "history": [
            {"date": row.date, "value": int(getattr(row, stat)), "minutes": float(row.minutes),
             "opponent": row.opponent, "rest_days": int(row.rest_days), "home": None if pd.isna(row.home) else bool(row.home)}
            for row in history.itertuples()
        ],
        "comparisons": comparisons, "warnings": warnings, "context_notes": notes,
    }


def score_predictions(predictions: list[dict]) -> tuple[float, float]:
    probabilities = np.array([p["probability"] for p in predictions])
    outcomes = np.array([p["outcome"] for p in predictions])
    brier = float(np.mean((probabilities - outcomes) ** 2))
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    log_loss = float(-np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log1p(-clipped)))
    return brier, log_loss


def backtest(frame: pd.DataFrame, request: dict) -> dict:
    if request.get("player_id"):
        frame = frame[frame["player_id"] == request["player_id"]]
        if frame.empty:
            raise ValueError("Player not found in the selected dataset.")
    records = {model: [] for model in MODELS}
    if int((frame.groupby("player_id").size() - MIN_TRAIN).clip(lower=0).sum()) > 5000:
        raise ValueError("Select a player or a shorter dataset: one backtest is limited to 5,000 evaluation games.")
    pushes = 0
    stat = request["stat"]
    for player_id, player_games in frame.groupby("player_id"):
        player_games = player_games.sort_values("date").reset_index(drop=True)
        for index in range(MIN_TRAIN, len(player_games)):
            history = player_games.iloc[max(0, index - request["lookback"]):index]
            target = player_games.iloc[index]
            line = request.get("line")
            if line is None:
                line = float(math.floor(float(history[stat].median())) + 0.5)
            actual = int(target[stat])
            if actual == line:
                pushes += 1
                continue
            # Rest and opponent are schedule features. Minutes are estimated strictly from older games.
            factor, _, _ = context_adjustment(history, stat, None, int(target["rest_days"]), str(target["opponent"]), request["use_context"])
            candidates = {}
            for model in MODELS[:-1]:
                fitted = fit_distribution(history, stat, model, factor)
                over, under, _ = fitted.probabilities(line)
                # A tiny symmetric smoothing prior keeps all-push empirical windows scoreable.
                conditional = (over + 1e-6) / (over + under + 2e-6)
                candidates[model] = conditional
                records[model].append({
                    "date": str(target["date"]), "player_id": str(player_id), "line": line,
                    "actual": actual, "probability": conditional, "outcome": int(actual > line),
                })
            auto = fit_distribution(history, stat, "auto", factor).model
            records["auto"].append({
                "date": str(target["date"]), "player_id": str(player_id), "line": line,
                "actual": actual, "probability": candidates[auto], "outcome": int(actual > line),
            })
    predictions = sorted(records[request["model"]], key=lambda p: (p["date"], p["player_id"]))
    if not predictions:
        raise ValueError("No settled evaluation games. Need at least 16 games per player and a line with non-push outcomes.")
    brier, log_loss = score_predictions(predictions)
    calibration = []
    for bucket_index in range(10):
        lower, upper = bucket_index / 10, (bucket_index + 1) / 10
        bucket = [p for p in predictions if lower <= p["probability"] and (p["probability"] < upper or upper == 1.0)]
        if bucket:
            calibration.append({
                "lower": float(lower), "upper": upper, "count": len(bucket),
                "predicted": float(np.mean([p["probability"] for p in bucket])),
                "observed": float(np.mean([p["outcome"] for p in bucket])),
            })
    return {
        "evaluations": len(predictions), "pushes_excluded": pushes, "brier_score": brier,
        "log_loss": log_loss, "baseline_brier": 0.25, "skill_score": 1 - brier / 0.25,
        "calibration": calibration,
        "model_scores": [
            {"model": model, "brier_score": score_predictions(rows)[0],
             "log_loss": score_predictions(rows)[1], "evaluations": len(rows)}
            for model, rows in records.items()
        ],
        "predictions": predictions,
        "description": (
            "Chronological walk-forward evaluation after 15 training games; only older games fit each forecast. "
            "Minutes are trailing estimates, never the held-out game's actual minutes. "
            + ("Adaptive half-point lines come from the prior rolling median, not historical sportsbook lines. "
               if request.get("line") is None else "The chosen line is held fixed across history, not historical sportsbook lines. ")
            + "Pushes are excluded and probabilities conditioned on a settled outcome; a 1e-6 symmetric prior handles degenerate predictions. "
            "Baseline is a constant 50% forecast (Brier 0.25); log loss clips probabilities at 1e-6. "
            "Comparing these models is exploratory, not an independent holdout after model selection. No profitability claim."
        ),
    }
