"""Learning loop: conversation memory, a bet ledger graded against real box scores, and a probability
calibration that switches on only when it beats the raw model on the most recent graded predictions.

Everything is stored in the app's SQLite database in tables prefixed ``chat_``, ``ledger_`` and ``learning_``.
"""
from __future__ import annotations

from collections.abc import Callable
import copy
from datetime import date, datetime, timedelta, timezone
import json
import logging
import math
import re
import threading
from typing import Any
import unicodedata
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from . import nlu, simulation
from .data import BOX_COUNTS, STATS
from .engine import (MIN_TRAIN, CountDistribution, american_probability, context_adjustment, fit_distribution,
                     profit_per_unit, resolve_model, simulation_supported)
from .nlu import name_key
from .stats import PROP_STATS, STAT_DEFS, game_value, label as stat_label, validate_line as validate_market_line, with_stats

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
MAX_TURNS = 5000
MAX_LEGS = 10
MAX_STAKE = 1_000_000
MAX_NOTE = 200
MAX_FEEDBACK_NOTE = 1000
ALIAS_SOURCES = ("taught", "clarified")
# Words the assistant reads as stats (including ones it can't forecast); they can't double as nicknames.
STAT_TERMS = frozenset(name_key(word) for word in (*STATS, *PROP_STATS, *nlu.STAT_LOOKUP, *getattr(nlu, "UNSUPPORTED", {})))
TURN_JSON_FIELDS = ("intents", "entities", "actions", "engine", "slip_before", "slip_after", "tool_calls")
SIDES = ("over", "under")
WINDOW = 60  # trailing games behind each walk-forward forecast (the default lookback)
LINE_STEPS = (-2, -1, 0, 1, 2)
CLIP = 1e-4
RIDGE = 10.0
STAT_RIDGE = 25.0  # each market's own correction is shrunk toward the shared one
HOLDOUT_SHARE = 4  # the newest quarter of graded predictions is held out
MIN_HOLDOUT = 200
MIN_SAMPLES = MIN_HOLDOUT * HOLDOUT_SHARE
STAT_MIN_HOLDOUT = 100  # a market needs this many held-out predictions (and as many to train on) for its own test
MIN_GAIN = 0.001
CALIBRATION_VERSION = 2  # 2: per-market corrections learned on the game simulator's forecasts
BOX_SCORE_HOUR = 6  # Eastern hour after which a missing box score from the previous day means "did not play"

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS chat_turns (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, conversation_id TEXT NOT NULL,
        created_at TEXT NOT NULL, message TEXT NOT NULL, reply TEXT NOT NULL, intents TEXT NOT NULL,
        entities TEXT NOT NULL, actions TEXT NOT NULL, engine TEXT NOT NULL, slip_before TEXT, slip_after TEXT,
        tool_calls TEXT NOT NULL, rating INTEGER, note TEXT, rated_at TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS chat_turns_conversation ON chat_turns (conversation_id, seq)",
    """CREATE TABLE IF NOT EXISTS chat_aliases (
        phrase TEXT PRIMARY KEY, player_id TEXT NOT NULL, player TEXT NOT NULL, source TEXT NOT NULL,
        uses INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS ledger_bets (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
        event_date TEXT NOT NULL, book_odds INTEGER, stake REAL, probability REAL NOT NULL,
        independent_probability REAL NOT NULL, status TEXT NOT NULL, profit REAL, note TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS ledger_legs (
        bet_id TEXT NOT NULL, position INTEGER NOT NULL, player_id TEXT NOT NULL, player TEXT NOT NULL,
        team TEXT NOT NULL, stat TEXT NOT NULL, line REAL NOT NULL, side TEXT NOT NULL, probability REAL NOT NULL,
        push REAL NOT NULL, status TEXT NOT NULL, actual INTEGER, settled_at TEXT, request TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'player',
        PRIMARY KEY (bet_id, position)
    )""",
    "CREATE INDEX IF NOT EXISTS ledger_legs_status ON ledger_legs (status, player_id)",
    """CREATE TABLE IF NOT EXISTS learning_calibration (
        id INTEGER PRIMARY KEY CHECK (id = 1), trained_at TEXT NOT NULL, active INTEGER NOT NULL,
        intercept REAL, slope REAL, status TEXT NOT NULL
    )""",
)


# ---------- Text helpers ----------

def _trimmable(char: str) -> bool:
    return char.isspace() or unicodedata.category(char)[0] in "PS"


def normalize_phrase(text: str) -> str:
    """Casefold, strip accents, collapse whitespace and trim punctuation or symbols at both ends."""
    decomposed = unicodedata.normalize("NFKD", str(text).casefold())
    collapsed = " ".join("".join(char for char in decomposed if not unicodedata.combining(char)).split())
    start, end = 0, len(collapsed)
    while start < end and _trimmable(collapsed[start]):
        start += 1
    while end > start and _trimmable(collapsed[end - 1]):
        end -= 1
    return collapsed[start:end]


def tokens(text: str) -> set[str]:
    """Normalized word tokens (letters and digits) used for similarity between messages."""
    return set(re.findall(r"[^\W_]+", normalize_phrase(text)))


def similarity(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _now_text(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def _load(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def _plain(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def _plural(count: int, word: str) -> str:
    return f"{count:,} {word}{'' if count == 1 else 's'}"


def _legs_stay(count: int) -> str:
    return "1 leg stays" if count == 1 else f"{count} legs stay"


def _reason(exc: Exception) -> str:
    """A one-line, user-facing reason from a provider error; never a traceback."""
    detail = getattr(exc, "detail", None)
    text = detail if isinstance(detail, str) else ("" if isinstance(exc, KeyError) else str(exc))
    text = " ".join(text.split()).rstrip(".")
    return text if 0 < len(text) <= 200 else ""


# ---------- Validation helpers ----------

def _probability(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a probability between 0 and 1.") from exc
    if not math.isfinite(number) or not -1e-9 <= number <= 1 + 1e-9:
        raise ValueError(f"{label} must be a probability between 0 and 1.")
    return min(1.0, max(0.0, number))


def _line(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: the line must be a number.") from exc
    if not math.isfinite(number) or not 0 <= number <= 200 or number * 2 != int(number * 2):
        raise ValueError(f"{label}: use a whole-number or half-point line between 0 and 200.")
    return number


def _stake(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("Stake must be a number.")
    stake = float(value)
    if not math.isfinite(stake) or not 0 < stake <= MAX_STAKE:
        raise ValueError("Stake must be greater than 0 and at most 1,000,000.")
    return stake


def _note(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Notes must be text.")
    text = value.strip()
    if len(text) > limit:
        raise ValueError(f"Notes are limited to {limit} characters.")
    return text or None


def _event_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            pass
    raise ValueError("The event date must be a calendar date (YYYY-MM-DD).")


# ---------- Bet grading ----------

def grade_leg(actual: int, line: float, side: str) -> str:
    """Over wins above the line, under wins below it; landing exactly on a whole-number line is a push."""
    if actual == line:
        return "push"
    return "won" if (actual > line) == (side == "over") else "lost"


def grade_game_leg(margin: int, market: str, line: float | None) -> str:
    """A moneyline wins when the team wins; a spread wins when the team's margin plus its line is positive, pushes at zero."""
    if market == "moneyline":
        return "won" if margin > 0 else "lost"
    total = margin + float(line or 0.0)
    return "push" if total == 0 else "won" if total > 0 else "lost"


def bet_outcome(statuses: list[str], stake: float | None, book_odds: int | None) -> tuple[str, float | None]:
    """Bet status and profit (stake units) from its leg statuses."""
    if "lost" in statuses:
        return "lost", (-stake if stake is not None else None)
    if "pending" in statuses:
        return "pending", None
    if "won" not in statuses:
        # Every leg void or pushed: books refund a parlay with no live legs.
        return "void", (0.0 if stake is not None else None)
    if all(status == "won" for status in statuses):
        if stake is None or book_odds is None:
            return "won", None
        return "won", round(stake * profit_per_unit(book_odds), 6)
    return "reduced", None


# ---------- Calibration ----------

def _logit(p: Any) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=float), CLIP, 1 - CLIP))


def recalibrate(p: Any, intercept: float, slope: float) -> Any:
    """q = σ(a + b·logit(p)) with p clipped to [1e-4, 1 − 1e-4]; scalars in, scalars out."""
    q = expit(intercept + slope * _logit(p))
    return float(q) if np.ndim(q) == 0 else q


def fit_recalibration(p: Any, outcomes: Any, penalty: float = RIDGE, iterations: int = 50,
                      tolerance: float = 1e-10, prior: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    """Penalized maximum likelihood for logit(q) = a + b·logit(p) with ridge penalty λ·((a − a₀)² + (b − b₀)²).

    Newton's method from the prior map (a₀, b₀), by default the identity (0, 1); the penalty keeps the fit near the
    prior when graded predictions are scarce and makes the objective strictly convex.
    """
    x = _logit(p)
    y = np.asarray(outcomes, dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    prior = np.array([float(prior[0]), float(prior[1])])
    theta = prior.copy()

    def objective(candidate: np.ndarray) -> float:
        z = design @ candidate
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + penalty * np.sum((candidate - prior) ** 2))

    current = objective(theta)
    for _ in range(iterations):
        q = expit(design @ theta)
        gradient = design.T @ (q - y) + 2 * penalty * (theta - prior)
        hessian = (design * (q * (1 - q))[:, None]).T @ design + 2 * penalty * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        if np.max(np.abs(step)) < tolerance:
            theta = theta - step
            break
        scale = 1.0
        while True:
            candidate = theta - scale * step
            value = objective(candidate)
            # Step halving guards against overshooting; the slack absorbs rounding near the optimum.
            if value <= current + 1e-12 * (1 + abs(current)) or scale < 1e-6:
                break
            scale /= 2
        theta, current = candidate, value
    return float(theta[0]), float(theta[1])


def _scores(p: np.ndarray, y: np.ndarray, intercept: float, slope: float) -> tuple[float, float]:
    """Log loss and Brier score of σ(a + b·logit(p)); (0, 1) scores the raw model."""
    z = intercept + slope * _logit(p)
    log_loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
    brier = float(np.mean((expit(z) - y) ** 2))
    return log_loss, brier


def calibration_curve(p: Any, outcomes: Any, intercept: float | None, slope: float | None) -> list[dict]:
    """Ten equal-width bins of the raw probability; empty bins are omitted."""
    raw = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(outcomes, dtype=float)
    if not len(raw):
        return []
    calibrated = raw if intercept is None or slope is None else np.asarray(recalibrate(raw, intercept, slope))
    bins = np.minimum((raw * 10).astype(int), 9)
    curve = []
    for index in range(10):
        mask = bins == index
        count = int(mask.sum())
        if count:
            curve.append({
                "predicted": round(float(raw[mask].mean()), 6), "observed": round(float(y[mask].mean()), 6),
                "calibrated": round(float(calibrated[mask].mean()), 6), "count": count,
            })
    return curve


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def untrained_status() -> dict:
    return {
        "active": False, "trained_at": None, "samples": 0, "ledger_samples": 0, "players": 0,
        "intercept": None, "slope": None, "holdout": None, "curve": [], "stats": [],
        "message": ("Not trained yet. The calibration learns from graded predictions: walk-forward forecasts on saved NBA "
                    "game logs plus your settled tracked legs, separately for every market. A correction switches on only "
                    "if it beats the raw model on the most recent games."),
    }


def _direction(intercept: float, slope: float) -> str:
    """One sentence on what the fitted correction found, judged at a 50% raw probability."""
    shift = recalibrate(0.5, intercept, slope) - 0.5
    parts = []
    if shift > 0.01:
        parts.append("overs have hit more often than forecast")
    elif shift < -0.01:
        parts.append("overs have hit less often than forecast")
    if slope < 0.95:
        parts.append("strong leans were overconfident, so they are pulled toward 50%")
    elif slope > 1.05:
        parts.append("strong leans were too timid, so they are pushed away from 50%")
    if not parts:
        return ""
    sentence = " and ".join(parts)
    return f" {sentence[0].upper()}{sentence[1:]}."


def _holdout_scores(raw: np.ndarray, y: np.ndarray, intercept: float, slope: float) -> dict:
    log_loss, brier = _scores(raw, y, intercept, slope)
    return {"log_loss": round(log_loss, 6), "brier": round(brier, 6)}


def _stat_calibration(stat: str, group: pd.DataFrame, training: np.ndarray, pooled_train: tuple[float, float] | None,
                      pooled: tuple[bool, float | None, float | None]) -> tuple[dict, tuple[bool, float | None, float | None]]:
    """One market's own correction, tested on its share of the newest predictions against the raw model and the
    shared correction. ``training`` marks rows outside the global holdout (rows are in date order)."""
    name = stat_label(stat)
    raw = group["p"].to_numpy(dtype=float)
    y = group["outcome"].to_numpy(dtype=float)
    count = len(group)
    ledger = int((group["source"] == "ledger").sum())
    pooled_active, pooled_intercept, pooled_slope = pooled
    held = int((~training).sum())
    entry = {"stat": stat, "label": name, "samples": count, "ledger_samples": ledger, "choice": "none", "active": False,
             "intercept": None, "slope": None, "holdout": None, "curve": [], "message": ""}

    def shared(reason: str) -> tuple[dict, tuple[bool, float | None, float | None]]:
        if pooled_active:
            entry.update(choice="shared", active=True, intercept=_rounded(pooled_intercept), slope=_rounded(pooled_slope),
                         curve=calibration_curve(raw, y, pooled_intercept, pooled_slope),
                         message=f"{reason} Uses the shared correction, which beat the raw model across every market.")
            return entry, (True, pooled_intercept, pooled_slope)
        entry.update(curve=calibration_curve(raw, y, None, None), message=f"{reason} Keeps the raw model.")
        return entry, (False, None, None)

    if pooled_train is None or held < STAT_MIN_HOLDOUT or count - held < STAT_MIN_HOLDOUT:
        return shared(f"{_plural(count, 'graded prediction')}: too few for a separate {name.lower()} test "
                      f"(needs {STAT_MIN_HOLDOUT:,} recent and {STAT_MIN_HOLDOUT:,} older).")
    own = fit_recalibration(raw[training], y[training], penalty=STAT_RIDGE, prior=pooled_train)
    test_raw, test_y = raw[~training], y[~training]
    scores = {"raw": _holdout_scores(test_raw, test_y, 0.0, 1.0), "shared": _holdout_scores(test_raw, test_y, *pooled_train),
              "own": _holdout_scores(test_raw, test_y, *own)}
    entry["holdout"] = {"samples": held, **{f"{key}_{metric}": value[metric] for key, value in scores.items()
                                            for metric in ("log_loss", "brier")}}
    base = scores["raw"]
    passing = [key for key in ("own", "shared") if scores[key]["log_loss"] < base["log_loss"] - MIN_GAIN
               and scores[key]["brier"] <= base["brier"]]
    choice = min(passing, key=lambda key: scores[key]["log_loss"]) if passing else "none"
    if choice == "own":
        prior = (pooled_intercept, pooled_slope) if pooled_intercept is not None and pooled_slope is not None else pooled_train
        intercept, slope = fit_recalibration(raw, y, penalty=STAT_RIDGE, prior=prior)
        entry.update(choice="own", active=True, intercept=_rounded(intercept), slope=_rounded(slope),
                     curve=calibration_curve(raw, y, intercept, slope),
                     message=(f"Own {name.lower()} correction from {_plural(count, 'graded prediction')}: on the newest {held:,} "
                              f"it lowered log loss from {base['log_loss']:.3f} to {scores['own']['log_loss']:.3f}"
                              + (f" (shared correction: {scores['shared']['log_loss']:.3f})." if pooled_active else ".")
                              + _direction(intercept, slope)))
        return entry, (True, intercept, slope)
    if choice == "shared":
        entry.update(choice="shared", active=True, intercept=_rounded(pooled_intercept), slope=_rounded(pooled_slope),
                     curve=calibration_curve(raw, y, pooled_intercept, pooled_slope),
                     message=(f"On the newest {held:,} {name.lower()} predictions the shared correction did best (log loss "
                              f"{base['log_loss']:.3f} raw, {scores['shared']['log_loss']:.3f} shared, {scores['own']['log_loss']:.3f} own), "
                              "so it uses the shared one."))
        return entry, (True, pooled_intercept, pooled_slope)
    entry.update(curve=calibration_curve(raw, y, None, None),
                 message=(f"Raw {name.lower()} forecasts were already as well calibrated on the newest {held:,} (log loss "
                          f"{base['log_loss']:.3f} raw, {scores['own']['log_loss']:.3f} own, {scores['shared']['log_loss']:.3f} shared), "
                          "so no correction is applied."))
    return entry, (False, None, None)


def fit_calibration(samples: pd.DataFrame, trained_at: str | None = None) -> tuple[dict, tuple]:
    """CalibrationStatus plus the parameters to apply, from graded over/under predictions.

    ``samples`` has columns date (YYYY-MM-DD), player_id, stat, p (raw over probability), outcome (1 when the over won)
    and source ('walk_forward' or 'ledger'). The newest quarter is a holdout: the shared correction is switched on only
    if, fitted on the older three quarters, it lowers holdout log loss by more than 0.001 without raising the Brier
    score. Each market then gets its own correction (shrunk toward the shared one) and keeps whichever of raw, shared
    or its own did best on its part of the holdout. Every active correction is refitted on all its samples.

    Returns (status, (active, intercept, slope, per_stat)) where per_stat maps a market to (active, intercept, slope).
    """
    count = len(samples)
    ledger = int((samples["source"] == "ledger").sum()) if count else 0
    players = int(samples["player_id"].nunique()) if count else 0
    raw = np.zeros(0)
    y = np.zeros(0)
    ordered = samples
    if count:
        order = np.argsort(samples["date"].astype(str).to_numpy(), kind="stable")
        ordered = samples.iloc[order].reset_index(drop=True)
        raw = ordered["p"].to_numpy(dtype=float)
        y = ordered["outcome"].to_numpy(dtype=float)
    held = count // HOLDOUT_SHARE
    holdout = None
    active = False
    intercept: float | None = None
    slope: float | None = None
    pooled_train: tuple[float, float] | None = None
    if held < MIN_HOLDOUT:
        if count:
            intercept, slope = fit_recalibration(raw, y)
        message = (f"Needs at least {MIN_SAMPLES:,} graded predictions (has {count:,}); keeps the raw model for now. "
                   "Saved NBA game logs and settled bets add more.")
    else:
        cut = count - held
        intercept, slope = fit_recalibration(raw[:cut], y[:cut])
        pooled_train = (intercept, slope)
        raw_log_loss, raw_brier = _scores(raw[cut:], y[cut:], 0.0, 1.0)
        calibrated_log_loss, calibrated_brier = _scores(raw[cut:], y[cut:], intercept, slope)
        active = calibrated_log_loss < raw_log_loss - MIN_GAIN and calibrated_brier <= raw_brier
        holdout = {
            "samples": held, "raw_log_loss": round(raw_log_loss, 6), "calibrated_log_loss": round(calibrated_log_loss, 6),
            "raw_brier": round(raw_brier, 6), "calibrated_brier": round(calibrated_brier, 6),
        }
        source = f"{_plural(count, 'graded prediction')}" + (f" ({ledger:,} from your tracked legs)" if ledger else "")
        if active:
            intercept, slope = fit_recalibration(raw, y)
            message = (f"Learned from {source} across {_plural(players, 'player')}. On the most recent {held:,}, the correction "
                       f"lowered log loss from {raw_log_loss:.3f} to {calibrated_log_loss:.3f}, so it is switched on."
                       + _direction(intercept, slope))
        elif calibrated_log_loss < raw_log_loss - MIN_GAIN:
            message = (f"On the most recent {held:,} graded predictions the correction lowered log loss ({raw_log_loss:.3f} to "
                       f"{calibrated_log_loss:.3f}) but not the Brier score ({raw_brier:.4f} vs {calibrated_brier:.4f}), "
                       "so no correction is applied.")
        else:
            message = (f"The raw model is already as well calibrated on the most recent {held:,} graded predictions "
                       f"(log loss {raw_log_loss:.3f} raw vs {calibrated_log_loss:.3f} corrected), so no correction is applied.")
    per_stat: dict[str, tuple[bool, float | None, float | None]] = {}
    entries = []
    if count and "stat" in ordered:
        training = np.arange(count) < count - held if held >= MIN_HOLDOUT else np.ones(count, dtype=bool)
        order_of = {stat: index for index, stat in enumerate(PROP_STATS)}
        for stat in sorted(ordered["stat"].unique(), key=lambda value: order_of.get(value, len(order_of))):
            if stat not in STAT_DEFS:
                continue
            mask = (ordered["stat"] == stat).to_numpy()
            entry, params = _stat_calibration(stat, ordered[mask], training[mask], pooled_train, (bool(active), intercept, slope))
            entries.append(entry)
            per_stat[stat] = params
    if entries:
        own = sum(entry["choice"] == "own" for entry in entries)
        shared = sum(entry["choice"] == "shared" for entry in entries)
        message += (f" Calibrated per market: {own} of {len(entries)} use their own correction, {shared} the shared one and "
                    f"{len(entries) - own - shared} the raw model.")
    status = {
        "active": bool(active), "trained_at": trained_at, "samples": count, "ledger_samples": ledger, "players": players,
        "intercept": _rounded(intercept), "slope": _rounded(slope), "holdout": holdout,
        "curve": calibration_curve(raw, y, intercept, slope), "message": message, "stats": entries,
        "version": CALIBRATION_VERSION,
    }
    return status, (bool(active), intercept, slope, per_stat)


def nba_game_logs(store) -> pd.DataFrame:
    """Every stored NBA (nba_api) game, one row per player and date, with rest days recomputed on the merged timeline."""
    frames = []
    for info in store.datasets():
        if info.get("source") != "nba_api":
            continue
        try:
            frame, _ = store.load(info["id"])
        except (KeyError, ValueError):
            continue
        frames.append(frame.loc[:, ["date", "player_id", "opponent", "minutes", *STATS,
                                    *[column for column in BOX_COUNTS if column in frame.columns]]])
    if not frames:
        return pd.DataFrame(columns=["date", "player_id", "opponent", "minutes", *STATS, "rest_days"])
    combined = pd.concat(frames, ignore_index=True)
    combined["player_id"] = combined["player_id"].astype(str)
    combined["date"] = combined["date"].astype(str)
    # Datasets arrive most recently fetched first, so a refreshed copy of a game wins.
    combined = combined.drop_duplicates(["player_id", "date"], keep="first")
    combined = combined.sort_values(["player_id", "date"], kind="stable").reset_index(drop=True)
    days = pd.to_datetime(combined["date"]).groupby(combined["player_id"]).diff().dt.days
    combined["rest_days"] = (days - 1).fillna(2).clip(lower=0).astype(int)
    return combined


def _lines(values: np.ndarray, milestone: bool) -> list[float]:
    """The trailing median's half point and ±1, ±2 steps of half a standard deviation (at least one); 0.5 for yes/no."""
    if milestone:
        return [0.5]
    base = math.floor(float(np.median(values))) + 0.5
    step = max(1, int(round(float(np.std(values, ddof=1)) / 2)))
    return [line for line in (base + offset * step for offset in LINE_STEPS) if 0.5 <= line <= 200]


def walk_forward_samples(frame: pd.DataFrame, simulations: int = simulation.EVALUATION_SIMULATIONS) -> pd.DataFrame:
    """Walk-forward over probabilities for every player, market and target game after the first 15, from Auto.

    Each forecast uses the player's previous 30 games, the target game's rest days and opponent (schedule fields)
    and never the target game's minutes or box score. One simulation per target game prices every simulated market.
    Lines sit at the trailing median's half point and ±1, ±2 steps of half a standard deviation (at least one), so
    probabilities spread across the range; yes/no markets use 0.5.
    """
    records: dict[str, list] = {key: [] for key in ("date", "player_id", "stat", "line", "p", "outcome")}
    if len(frame):
        for player_id, games in frame.groupby("player_id", sort=True):
            games = games.sort_values("date", kind="stable").reset_index(drop=True)
            markets = [stat for stat in PROP_STATS if all(column in games.columns for column in STAT_DEFS[stat].columns)]
            games = with_stats(games, markets)
            dates = games["date"].astype(str).tolist()
            rest = games["rest_days"].astype(int).tolist()
            opponents = games["opponent"].astype(str).tolist()
            for index in range(MIN_TRAIN, len(games)):
                history_all = games.iloc[max(0, index - WINDOW):index]
                simulated = None
                for stat in markets:
                    actual = games.at[index, stat]
                    if pd.isna(actual):
                        continue
                    history = history_all[history_all[stat].notna()]
                    if len(history) < MIN_TRAIN:
                        continue
                    spec = STAT_DEFS[stat]
                    try:
                        model = resolve_model(history, stat, "auto")
                        if model == "simulation":
                            if simulated is None or len(history) != len(history_all):
                                game_model = simulation.fit_game_model(history, None, rest[index], opponents[index], True)
                                draws = simulation.simulate(game_model, simulations, simulation.SEED + index)
                                if len(history) == len(history_all):
                                    simulated = draws
                            else:
                                draws = simulated
                            fitted = CountDistribution.from_draws(draws.market(stat))
                        else:
                            factor = 1.0 if spec.milestone else context_adjustment(history, stat, None, rest[index], opponents[index], True)[0]
                            fitted = fit_distribution(history, stat, model, factor)
                    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                        continue
                    for line in _lines(history[stat].to_numpy(dtype=float), spec.milestone):
                        over, under, _ = fitted.probabilities(line)
                        total = over + under
                        if total < 1e-9:
                            continue
                        p = over / total
                        if not math.isfinite(p):
                            continue
                        records["date"].append(dates[index])
                        records["player_id"].append(str(player_id))
                        records["stat"].append(stat)
                        records["line"].append(line)
                        records["p"].append(min(1.0, max(0.0, p)))
                        records["outcome"].append(int(float(actual) > line))
    samples = pd.DataFrame(records)
    samples["source"] = "walk_forward"
    return samples


# ---------- Row shapes ----------

def _alias(row) -> dict:
    return {"phrase": row["phrase"], "player_id": row["player_id"], "player": row["player"],
            "source": row["source"], "uses": int(row["uses"])}


def _turn(row) -> dict:
    """The turn exactly as recorded (JSON round-tripped), plus its id, rating and creation time."""
    turn = {"conversation_id": row["conversation_id"], "message": row["message"], "reply": row["reply"]}
    turn.update({field: _load(row[field]) for field in TURN_JSON_FIELDS})
    return {**turn, "turn_id": row["id"], "rating": row["rating"], "created_at": row["created_at"]}


def _leg(row) -> dict:
    kind = row["kind"] if "kind" in row.keys() else "player"
    leg = {
        "kind": kind, "player_id": row["player_id"], "player": row["player"], "team": row["team"], "stat": row["stat"],
        "line": float(row["line"]), "side": row["side"], "probability": float(row["probability"]),
        "status": row["status"], "actual": None if row["actual"] is None else int(row["actual"]),
        "settled_at": row["settled_at"],
    }
    if kind == "game":
        request = _load(row["request"]) or {}
        leg.update({"market": row["stat"], "line": None if row["stat"] == "moneyline" else float(row["line"]),
                    "game_id": str(request.get("game_id") or ""), "date": request.get("date"), "opponent": request.get("opponent"),
                    "home": bool(request.get("home"))})
    return leg


def _bet(row, legs: list) -> dict:
    return {
        "id": row["id"], "created_at": row["created_at"], "event_date": row["event_date"],
        "legs": [_leg(leg) for leg in legs], "book_odds": row["book_odds"], "stake": row["stake"],
        "probability": float(row["probability"]), "independent_probability": float(row["independent_probability"]),
        "status": row["status"], "profit": row["profit"], "note": row["note"],
    }


class Learning:
    """Memory, bet ledger and calibration for one Store. Safe for a single-user app with background threads."""

    max_turns = MAX_TURNS
    box_score_hour = BOX_SCORE_HOUR

    def __init__(self, store, history: Callable[[str, bool], tuple[str, pd.DataFrame, dict]] | None = None,
                 clock: Callable[[], datetime] | None = None, today: Callable[[], date] | None = None,
                 scores: Callable[[date], dict] | None = None):
        self.store = store
        self._history = history
        self._scores = scores
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._today = today or (lambda: self._clock().astimezone(EASTERN).date())
        self._lock = threading.RLock()
        self._train_lock = threading.Lock()
        self._settle_lock = threading.Lock()
        self._training = False
        self._worker: threading.Thread | None = None
        self._calibration = untrained_status()
        self._params: tuple[bool, float | None, float | None] = (False, None, None)
        self._stat_params: dict[str, tuple[bool, float | None, float | None]] = {}
        with self._lock, store.connect() as db:
            for statement in SCHEMA:
                db.execute(statement)
            if "kind" not in {row["name"] for row in db.execute("PRAGMA table_info(ledger_legs)")}:
                db.execute("ALTER TABLE ledger_legs ADD COLUMN kind TEXT NOT NULL DEFAULT 'player'")
            self._rekey_aliases(db)
            row = db.execute("SELECT active, intercept, slope, status FROM learning_calibration WHERE id = 1").fetchone()
        if row is not None:
            try:
                saved = json.loads(row["status"])
                if saved.get("version") == CALIBRATION_VERSION:
                    self._calibration = saved
                    self._params = (bool(row["active"]), row["intercept"], row["slope"])
                    self._stat_params = self._saved_stat_params(saved)
                else:
                    # Learned on an older model's forecasts: never apply it to the current model; a retrain rebuilds it.
                    self._calibration = {**untrained_status(), "message": (
                        "Rebuilding the calibration for the game simulator, separately for every market. It retrains in the "
                        "background; until then forecasts are shown uncorrected.")}
            except (TypeError, ValueError, KeyError):
                logger.warning("Ignoring an unreadable saved calibration; it will be rebuilt on the next retrain.")

    @staticmethod
    def _saved_stat_params(status: dict) -> dict[str, tuple[bool, float | None, float | None]]:
        """Per-market parameters exactly as fitted, from the saved status (rounded copies are for display only)."""
        params = {}
        for entry in status.get("stats") or []:
            exact = entry.get("params") or [entry.get("active"), entry.get("intercept"), entry.get("slope")]
            params[entry["stat"]] = (bool(exact[0]), exact[1], exact[2])
        return params

    def _now(self) -> str:
        return _now_text(self._clock())

    # ---------- Memory: aliases ----------

    @staticmethod
    def _alias_key(phrase: str) -> str:
        """The assistant's comparable form of a nickname (nlu.name_key), checked for use as a nickname."""
        if not isinstance(phrase, str):
            raise ValueError("A nickname must be text.")
        key = name_key(phrase)
        if not 2 <= len(key) <= 40:
            raise ValueError("A nickname must be 2 to 40 characters long.")
        if key.replace(" ", "").isdigit():
            raise ValueError(f"“{key}” is just a number, and numbers are read as lines or odds, so it can't be a nickname. "
                             "Pick a word instead.")
        if key in STAT_TERMS:
            raise ValueError(f"“{key}” is a stat word, so it can't be a nickname for a player. Pick a different word.")
        return key

    @staticmethod
    def _rekey_aliases(db) -> None:
        """Move nicknames saved under an older normalization to the assistant's name key; the newest meaning wins."""
        for row in db.execute("SELECT * FROM chat_aliases ORDER BY updated_at").fetchall():
            key = name_key(row["phrase"])
            if key == row["phrase"]:
                continue
            db.execute("DELETE FROM chat_aliases WHERE phrase = ?", (row["phrase"],))
            kept = db.execute("SELECT * FROM chat_aliases WHERE phrase = ?", (key,)).fetchone()
            if not 2 <= len(key) <= 40 or (kept is not None and kept["updated_at"] > row["updated_at"]):
                continue
            uses = int(row["uses"]) + (int(kept["uses"]) if kept is not None and kept["player_id"] == row["player_id"] else 0)
            db.execute("""INSERT OR REPLACE INTO chat_aliases (phrase, player_id, player, source, uses, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""", (key, row["player_id"], row["player"], row["source"], uses,
                                                  row["created_at"], row["updated_at"]))

    def learn_alias(self, phrase: str, player_id: str, player_name: str, source: str = "taught") -> dict:
        """Save a nickname under the assistant's name key. Re-learning it overwrites the old meaning; the use
        count carries over only when it still means the same player."""
        key = self._alias_key(phrase)
        player_id = "" if player_id is None else str(player_id).strip()
        if not player_id:
            raise ValueError("A nickname needs a player to point to, but no player ID was given.")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", player_id):
            raise ValueError("Player IDs may contain letters, numbers, underscores and hyphens only.")
        player = " ".join(str(player_name).split())
        if not 1 <= len(player) <= 80:
            raise ValueError("The player name must be 1 to 80 characters.")
        if source not in ALIAS_SOURCES:
            raise ValueError("Alias source must be 'taught' or 'clarified'.")
        now = self._now()
        with self._lock, self.store.connect() as db:
            db.execute("""INSERT INTO chat_aliases (phrase, player_id, player, source, uses, created_at, updated_at)
                VALUES (?, ?, ?, ?, 0, ?, ?) ON CONFLICT(phrase) DO UPDATE SET
                    uses = CASE WHEN chat_aliases.player_id = excluded.player_id THEN chat_aliases.uses ELSE 0 END,
                    player_id = excluded.player_id, player = excluded.player, source = excluded.source,
                    updated_at = excluded.updated_at""", (key, player_id, player, source, now, now))
            row = db.execute("SELECT * FROM chat_aliases WHERE phrase = ?", (key,)).fetchone()
        return _alias(row)

    def aliases(self) -> dict[str, dict]:
        """Saved nicknames keyed by nlu.name_key(phrase)."""
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM chat_aliases ORDER BY phrase").fetchall()
        return {row["phrase"]: _alias(row) for row in rows}

    def alias(self, phrase: str) -> dict | None:
        if not isinstance(phrase, str):
            return None
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM chat_aliases WHERE phrase = ?", (name_key(phrase),)).fetchone()
        return _alias(row) if row else None

    def use_alias(self, phrase: str) -> dict | None:
        """Count one use of a saved nickname; returns it, or None if it isn't saved."""
        if not isinstance(phrase, str):
            return None
        key = name_key(phrase)
        with self._lock, self.store.connect() as db:
            db.execute("UPDATE chat_aliases SET uses = uses + 1 WHERE phrase = ?", (key,))
            row = db.execute("SELECT * FROM chat_aliases WHERE phrase = ?", (key,)).fetchone()
        return _alias(row) if row else None

    def forget_alias(self, phrase: str) -> bool:
        if not isinstance(phrase, str):
            return False
        with self._lock, self.store.connect() as db:
            return db.execute("DELETE FROM chat_aliases WHERE phrase = ?", (name_key(phrase),)).rowcount > 0

    # ---------- Memory: turns and feedback ----------

    def record_turn(self, turn: dict) -> str:
        """Save a turn; every field other than the three texts is stored as JSON exactly as passed (None kept)."""
        conversation_id, message, reply = turn.get("conversation_id"), turn.get("message"), turn.get("reply")
        if not isinstance(conversation_id, str) or not 1 <= len(conversation_id) <= 100:
            raise ValueError("A turn needs a conversation ID of 1 to 100 characters.")
        if not isinstance(message, str) or not isinstance(reply, str):
            raise ValueError("A turn needs the message and reply text.")
        turn_id = uuid4().hex[:16]
        values = (turn_id, conversation_id, self._now(), message, reply,
                  *(_dump(turn.get(field)) for field in TURN_JSON_FIELDS))
        with self._lock, self.store.connect() as db:
            db.execute("""INSERT INTO chat_turns (id, conversation_id, created_at, message, reply, intents, entities,
                actions, engine, slip_before, slip_after, tool_calls) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", values)
            db.execute("""DELETE FROM chat_turns WHERE seq <= (
                SELECT seq FROM chat_turns ORDER BY seq DESC LIMIT 1 OFFSET ?)""", (max(1, int(self.max_turns)),))
        return turn_id

    def turn(self, turn_id: str) -> dict | None:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM chat_turns WHERE id = ?", (turn_id,)).fetchone()
        return _turn(row) if row else None

    def last_turns(self, conversation_id: str, limit: int = 10) -> list[dict]:
        if limit <= 0:
            return []
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM chat_turns WHERE conversation_id = ? ORDER BY seq DESC LIMIT ?",
                              (conversation_id, int(limit))).fetchall()
        return [_turn(row) for row in rows]

    def feedback(self, turn_id: str, rating: int, note: str | None = None) -> dict:
        if isinstance(rating, bool) or rating not in (1, -1):
            raise ValueError("Rating must be 1 (helpful) or -1 (not helpful).")
        note = _note(note, MAX_FEEDBACK_NOTE)
        with self._lock, self.store.connect() as db:
            updated = db.execute("UPDATE chat_turns SET rating = ?, note = ?, rated_at = ? WHERE id = ?",
                                 (int(rating), note, self._now(), turn_id)).rowcount
        if not updated:
            raise KeyError("Turn not found; it may be older than the saved conversation history.")
        return {"turn_id": turn_id, "rating": int(rating)}

    def examples(self, message: str, limit: int = 3) -> list[dict]:
        """Liked turns most similar to ``message`` (token Jaccard > 0), newest first among ties."""
        query = tokens(message)
        if limit <= 0 or not query:
            return []
        with self.store.connect() as db:
            rows = db.execute("SELECT seq, message, reply, intents, tool_calls FROM chat_turns WHERE rating = 1 ORDER BY seq DESC").fetchall()
        scored, seen = [], set()
        for row in rows:
            key = normalize_phrase(row["message"])
            if key in seen:
                continue
            seen.add(key)
            score = similarity(query, tokens(row["message"]))
            if score > 0:
                scored.append((score, row["seq"], row))
        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [{"message": row["message"], "reply": row["reply"], "intents": _load(row["intents"]),
                 "tool_calls": _load(row["tool_calls"])} for _, _, row in scored[:limit]]

    def memory_summary(self) -> dict:
        with self.store.connect() as db:
            aliases = [_alias(row) for row in db.execute("SELECT * FROM chat_aliases ORDER BY phrase")]
            counts = db.execute("SELECT COUNT(*) AS turns, SUM(rating = 1) AS liked, SUM(rating = -1) AS disliked FROM chat_turns").fetchone()
        return {"aliases": aliases, "turns": int(counts["turns"]), "liked": int(counts["liked"] or 0),
                "disliked": int(counts["disliked"] or 0)}

    # ---------- Bet tracker ----------

    def track(self, slip_legs: list[dict], book_odds: int | None, stake: float | None, event_date: date | None,
              priced: dict, note: str | None = None) -> dict:
        """Save a slip with the model's probabilities at tracking time; ``priced`` is price_parlay() for these legs."""
        legs = [_plain(leg) for leg in (slip_legs or [])]
        if not 1 <= len(legs) <= MAX_LEGS:
            raise ValueError(f"Track a slip with 1 to {MAX_LEGS} legs.")
        quotes = (priced or {}).get("legs") or []
        if len(quotes) != len(legs):
            raise ValueError("The priced slip does not match the legs being tracked; price the slip again.")
        if book_odds is not None:
            american_probability(book_odds)
            book_odds = int(book_odds)
        stake = _stake(stake)
        note = _note(note, MAX_NOTE)
        day = (self._today() if event_date is None else _event_date(event_date)).isoformat()
        rows, seen = [], set()
        for number, (leg, quote) in enumerate(zip(legs, quotes), start=1):
            request = _plain(leg.get("request", leg))
            label = leg.get("label") or {}
            if request.get("kind") == "game":
                rows.append(self._game_row(number, request, label, quote, seen))
                continue
            player_id = str(request.get("player_id", "")).strip()
            stat, side = request.get("stat"), request.get("side")
            if not player_id or stat not in PROP_STATS or side not in SIDES:
                raise ValueError(f"Leg {number} needs a player, a supported market (such as points, threes or PRA) and a side (over or under).")
            line = _line(request.get("line"), f"Leg {number}")
            try:
                validate_market_line(stat, line)
            except ValueError as exc:
                raise ValueError(f"Leg {number}: {exc}") from exc
            if (player_id, stat) in seen:
                raise ValueError(f"Leg {number} repeats a player and stat already on this slip.")
            seen.add((player_id, stat))
            player = quote.get("player") or {}
            try:
                matches = (str(player.get("id", player_id)) == player_id and quote.get("stat", stat) == stat
                           and quote.get("side", side) == side and float(quote.get("line", line)) == line)
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(f"Leg {number} does not match its priced forecast; price the slip again.")
            probability = _probability(quote.get("probability"), f"Leg {number} probability")
            push = _probability(quote.get("push") or 0.0, f"Leg {number} push probability")
            name = " ".join(str(label.get("player") or player.get("name") or player_id).split())[:80]
            team = str(label.get("team") or player.get("team") or "").strip()[:10]
            rows.append((number - 1, player_id, name, team, stat, line, side, probability, push, _dump(request), "player"))
        joint = _probability(priced.get("probability"), "Parlay probability")
        independent = _probability(priced.get("independent_probability"), "Independent parlay probability")
        bet_id = uuid4().hex[:12]
        with self._lock, self.store.connect() as db:
            db.execute("""INSERT INTO ledger_bets (id, created_at, event_date, book_odds, stake, probability,
                independent_probability, status, profit, note) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?)""",
                       (bet_id, self._now(), day, book_odds, stake, joint, independent, note))
            db.executemany("""INSERT INTO ledger_legs (bet_id, position, player_id, player, team, stat, line, side,
                probability, push, status, actual, settled_at, request, kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, ?, ?)""", [(bet_id, *row) for row in rows])
        return self.bet(bet_id)

    @staticmethod
    def _game_row(number: int, request: dict, label: dict, quote: dict, seen: set) -> tuple:
        """A moneyline or spread leg as a ledger row: the team in player_id/team, the market in stat, home/away in side."""
        market = request.get("market")
        team = str(request.get("team") or "").strip().upper()
        opponent = str(request.get("opponent") or "").strip().upper()
        game_id = str(request.get("game_id") or "").strip()
        if market not in ("moneyline", "spread") or not re.fullmatch(r"[A-Z]{2,4}", team) \
                or not re.fullmatch(r"[A-Z]{2,4}", opponent) or not re.fullmatch(r"\d{1,20}", game_id):
            raise ValueError(f"Leg {number} needs a game, a team and a market (moneyline or spread).")
        _event_date(request.get("date"))
        line = 0.0
        if market == "spread":
            try:
                line = float(request.get("line"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Leg {number}: the spread must be a number.") from exc
            if not math.isfinite(line) or abs(line) > 60 or line * 2 != int(line * 2):
                raise ValueError(f"Leg {number}: use a whole-number or half-point spread between -60 and +60.")
        if ("game", game_id, market) in seen:
            raise ValueError(f"Leg {number} repeats a game's {market} already on this slip.")
        seen.add(("game", game_id, market))
        try:
            matches = (quote.get("kind") == "game" and quote.get("team") == team and quote.get("market") == market
                       and (market == "moneyline" or float(quote.get("line")) == line))
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(f"Leg {number} does not match its priced forecast; price the slip again.")
        probability = _probability(quote.get("probability"), f"Leg {number} probability")
        push = _probability(quote.get("push") or 0.0, f"Leg {number} push probability")
        name = " ".join(str(label.get("player") or team).split())[:80]
        return (number - 1, team, name, team, market, line, "home" if request.get("home") else "away", probability, push,
                _dump(request), "game")

    def bet(self, bet_id: str) -> dict | None:
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM ledger_bets WHERE id = ?", (bet_id,)).fetchone()
            legs = db.execute("SELECT * FROM ledger_legs WHERE bet_id = ? ORDER BY position", (bet_id,)).fetchall()
        return _bet(row, legs) if row else None

    def bets(self) -> list[dict]:
        """Every tracked bet, newest first."""
        with self.store.connect() as db:
            rows = db.execute("SELECT * FROM ledger_bets ORDER BY seq DESC").fetchall()
            legs = db.execute("SELECT * FROM ledger_legs ORDER BY bet_id, position").fetchall()
        grouped: dict[str, list] = {}
        for leg in legs:
            grouped.setdefault(leg["bet_id"], []).append(leg)
        return [_bet(row, grouped.get(row["id"], [])) for row in rows]

    def delete_bet(self, bet_id: str) -> bool:
        with self._lock, self.store.connect() as db:
            db.execute("DELETE FROM ledger_legs WHERE bet_id = ?", (bet_id,))
            return db.execute("DELETE FROM ledger_bets WHERE id = ?", (bet_id,)).rowcount > 0

    def settle(self) -> dict:
        """Grade pending legs from games before today (US Eastern) against each player's refreshed game log.

        One history call per player. A leg whose player has no game on the event date is void (did not play),
        except that a game from the previous day stays pending until 6 AM Eastern, since late games end after
        midnight and box scores post a little later. Provider failures leave the player's legs pending.
        """
        with self._settle_lock:
            today = self._today()
            yesterday = (today - timedelta(days=1)).isoformat()
            eastern = self._clock().astimezone(EASTERN)
            early = eastern.date() == today and eastern.hour < self.box_score_hour
            with self.store.connect() as db:
                pending = db.execute("""SELECT l.bet_id, l.position, l.player_id, l.player, l.stat, l.line, l.side, b.event_date
                    FROM ledger_legs l JOIN ledger_bets b ON b.id = l.bet_id
                    WHERE l.status = 'pending' AND l.kind = 'player' AND b.event_date < ? ORDER BY b.seq, l.position""",
                                     (today.isoformat(),)).fetchall()
                game_legs = db.execute("""SELECT l.bet_id, l.position, l.player_id, l.player, l.stat, l.line, l.request, b.event_date
                    FROM ledger_legs l JOIN ledger_bets b ON b.id = l.bet_id
                    WHERE l.status = 'pending' AND l.kind = 'game' ORDER BY b.seq, l.position""").fetchall()
            players: dict[str, list] = {}
            for leg in pending:
                players.setdefault(leg["player_id"], []).append(leg)
            warnings: list[str] = []
            updates = []
            self._settle_games(game_legs, today, updates, warnings)
            if players and self._history is None:
                warnings.append(f"Settling needs NBA game logs, but no game-log source is connected; {_legs_stay(len(pending))} pending.")
                players = {}
            for player_id, legs in players.items():
                name = legs[0]["player"]
                try:
                    _, frame, _ = self._history(player_id, True)
                    rows = frame
                    if "player_id" in frame.columns:
                        rows = frame[frame["player_id"].astype(str) == player_id]
                    games = {str(day)[:10]: game for day, game in zip(rows["date"], rows.to_dict("records"))}
                except Exception as exc:  # noqa: BLE001 - provider errors must never abort settling other players
                    reason = _reason(exc)
                    warnings.append(f"Could not load {name}'s recent games; {_legs_stay(len(legs))} pending. Settle again later."
                                    + (f" Details: {reason}." if reason else ""))
                    continue
                settled_at = self._now()
                earliest = min(games) if games else None
                waiting: dict[str, int] = {}
                unseen = 0
                for leg in legs:
                    game = games.get(leg["event_date"])
                    if game is None:
                        if early and leg["event_date"] == yesterday:
                            waiting[leg["event_date"]] = waiting.get(leg["event_date"], 0) + 1
                        elif earliest is not None and leg["event_date"] < earliest:
                            # The returned log does not reach back to this date, so "no game" cannot be concluded.
                            unseen += 1
                        else:
                            updates.append(("void", None, settled_at, leg["bet_id"], leg["position"]))
                        continue
                    try:
                        actual = game_value(game, leg["stat"])
                        if actual is None or not math.isfinite(actual):
                            raise ValueError
                    except (KeyError, TypeError, ValueError):
                        warnings.append(f"Could not read {name}'s {stat_label(leg['stat']).lower()} on {leg['event_date']}; {_legs_stay(1)} pending.")
                        continue
                    updates.append((grade_leg(int(actual), float(leg["line"]), leg["side"]), int(actual), settled_at,
                                    leg["bet_id"], leg["position"]))
                for day, count in waiting.items():
                    warnings.append(f"No box score yet for {name} on {day}. Late games finish after midnight Eastern, "
                                    f"so {_legs_stay(count)} pending until {self.box_score_hour} AM ET.")
                if unseen:
                    warnings.append(f"{name}'s recent game log starts on {earliest}, after the game date of "
                                    f"{_plural(unseen, 'tracked leg')}; {'it stays' if unseen == 1 else 'they stay'} pending. "
                                    "Delete the bet if it cannot be graded.")
            settled, touched = 0, []
            with self._lock, self.store.connect() as db:
                for status, actual, settled_at, bet_id, position in updates:
                    changed = db.execute("""UPDATE ledger_legs SET status = ?, actual = ?, settled_at = ?
                        WHERE bet_id = ? AND position = ? AND status = 'pending'""",
                                         (status, actual, settled_at, bet_id, position)).rowcount
                    if changed:
                        settled += 1
                        if bet_id not in touched:
                            touched.append(bet_id)
                for bet_id in touched:
                    self._refresh_bet(db, bet_id)
            return {"settled_legs": settled, "updated_bets": len(touched), "warnings": warnings, "record": self.track_record()}

    def _settle_games(self, legs: list, today: date, updates: list, warnings: list[str]) -> None:
        """Grade moneyline and spread legs from final scores on the scoreboard, as soon as each game is final."""
        by_day: dict[str, list[tuple]] = {}
        for leg in legs:
            request = _load(leg["request"]) or {}
            day = str(request.get("date") or leg["event_date"])[:10]
            if day <= today.isoformat():
                by_day.setdefault(day, []).append((leg, request))
        if by_day and self._scores is None:
            count = sum(len(items) for items in by_day.values())
            warnings.append(f"Grading moneylines and spreads needs final scores, but no scoreboard is connected; {_legs_stay(count)} pending.")
            return
        for day, items in sorted(by_day.items()):
            try:
                board = self._scores(date.fromisoformat(day))
            except Exception as exc:  # noqa: BLE001 - a provider failure must never abort settling other legs
                reason = _reason(exc)
                warnings.append(f"Could not load the scores for {day}; {_legs_stay(len(items))} pending. Settle again later."
                                + (f" Details: {reason}." if reason else ""))
                continue
            found = {str(game.get("id")): game for game in board.get("games", [])}
            settled_at = self._now()
            for leg, request in items:
                game = found.get(str(request.get("game_id")))
                team = leg["player_id"]
                if game is None:
                    warnings.append(f"The {leg['player']} game on {day} is not on that day's scoreboard; {_legs_stay(1)} pending. "
                                    "Delete the bet if the game moved.")
                    continue
                detail = str(game.get("detail") or "").lower()
                if any(word in detail for word in ("postponed", "canceled", "cancelled")):
                    updates.append(("void", None, settled_at, leg["bet_id"], leg["position"]))
                    continue
                sides = {game["home"]["team"]: game["home"], game["away"]["team"]: game["away"]}
                mine = sides.get(team)
                other = next((side for code, side in sides.items() if code != team), None)
                if mine is None or other is None:
                    warnings.append(f"The {day} scoreboard game {request.get('game_id')} doesn't include {team}; {_legs_stay(1)} pending.")
                    continue
                if game.get("state") != "post" or mine.get("score") is None or other.get("score") is None:
                    continue
                margin = int(mine["score"]) - int(other["score"])
                updates.append((grade_game_leg(margin, leg["stat"], float(leg["line"])), margin, settled_at, leg["bet_id"], leg["position"]))

    @staticmethod
    def _refresh_bet(db, bet_id: str) -> None:
        bet = db.execute("SELECT stake, book_odds FROM ledger_bets WHERE id = ?", (bet_id,)).fetchone()
        if bet is None:
            return
        statuses = [row["status"] for row in db.execute("SELECT status FROM ledger_legs WHERE bet_id = ?", (bet_id,))]
        status, profit = bet_outcome(statuses, bet["stake"], bet["book_odds"])
        db.execute("UPDATE ledger_bets SET status = ?, profit = ? WHERE id = ?", (status, profit, bet_id))

    def track_record(self) -> dict:
        with self.store.connect() as db:
            counts = {row["status"]: int(row["n"]) for row in db.execute("SELECT status, COUNT(*) AS n FROM ledger_bets GROUP BY status")}
            money = db.execute("""SELECT COALESCE(SUM(stake), 0) AS staked, COALESCE(SUM(profit), 0) AS profit FROM ledger_bets
                WHERE status != 'pending' AND stake IS NOT NULL AND profit IS NOT NULL""").fetchone()
            graded = db.execute("SELECT probability, status FROM ledger_legs WHERE status IN ('won', 'lost')").fetchall()
        p = np.array([row["probability"] for row in graded], dtype=float)
        y = np.array([row["status"] == "won" for row in graded], dtype=float)
        staked, profit = float(money["staked"]), float(money["profit"])
        return {
            "bets": sum(counts.values()), "pending": counts.get("pending", 0), "won": counts.get("won", 0),
            "lost": counts.get("lost", 0), "void": counts.get("void", 0), "reduced": counts.get("reduced", 0),
            "staked": staked, "profit": profit, "roi": profit / staked if staked > 0 else None,
            "legs_graded": len(p), "legs_won": int(y.sum()), "expected_legs_won": float(p.sum()),
            "brier": float(np.mean((p - y) ** 2)) if len(p) else None,
        }

    # ---------- Calibration ----------

    def ledger_samples(self) -> pd.DataFrame:
        """Graded (won/lost) tracked legs as over predictions; pushes are excluded, so p is conditional on no push."""
        with self.store.connect() as db:
            rows = db.execute("""SELECT l.player_id, l.stat, l.line, l.side, l.probability, l.push, l.status, b.event_date
                FROM ledger_legs l JOIN ledger_bets b ON b.id = l.bet_id WHERE l.status IN ('won', 'lost') AND l.kind = 'player'
                ORDER BY b.seq, l.position""").fetchall()
        records: dict[str, list] = {key: [] for key in ("date", "player_id", "stat", "line", "p", "outcome")}
        for row in rows:
            backed, push = float(row["probability"]), float(row["push"])
            other = max(0.0, 1.0 - backed - push)
            over, under = (backed, other) if row["side"] == "over" else (other, backed)
            if over + under < 1e-9:
                continue
            records["date"].append(row["event_date"])
            records["player_id"].append(row["player_id"])
            records["stat"].append(row["stat"])
            records["line"].append(float(row["line"]))
            records["p"].append(over / (over + under))
            records["outcome"].append(int((row["status"] == "won") == (row["side"] == "over")))
        samples = pd.DataFrame(records)
        samples["source"] = "ledger"
        return samples

    def retrain(self) -> dict:
        """Rebuild the calibration from stored NBA game logs and settled legs; persists and returns its status."""
        with self._train_lock:
            self._training = True
            try:
                parts = [walk_forward_samples(nba_game_logs(self.store)), self.ledger_samples()]
                samples = pd.concat([part for part in parts if len(part)], ignore_index=True) if any(len(part) for part in parts) else parts[0]
                status, params = fit_calibration(samples, self._now())
                active, intercept, slope, per_stat = params
                for entry in status.get("stats", []):
                    entry["params"] = list(per_stat.get(entry["stat"], (False, None, None)))
                with self._lock, self.store.connect() as db:
                    db.execute("""INSERT INTO learning_calibration (id, trained_at, active, intercept, slope, status)
                        VALUES (1, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET trained_at = excluded.trained_at,
                        active = excluded.active, intercept = excluded.intercept, slope = excluded.slope,
                        status = excluded.status""", (status["trained_at"], int(active), intercept, slope, _dump(status)))
                    self._calibration, self._params, self._stat_params = status, (active, intercept, slope), dict(per_stat)
                return copy.deepcopy(status)
            finally:
                self._training = False

    def retrain_in_background(self) -> threading.Thread | None:
        """Start retrain() on a daemon thread; returns None when a retrain is already running."""
        with self._lock:
            if self._training or (self._worker is not None and self._worker.is_alive()):
                return None
            self._training = True
            self._worker = threading.Thread(target=self._retrain_quietly, name="courtside-learning", daemon=True)
            worker = self._worker
        worker.start()
        return worker

    def _retrain_quietly(self) -> None:
        try:
            self.retrain()
        except Exception:  # noqa: BLE001 - a failed background run keeps the previous calibration
            logger.exception("Calibration retrain failed; the previous calibration is kept.")
        finally:
            self._training = False

    @property
    def training(self) -> bool:
        return self._training

    def calibrate_over(self, p_over: float, stat: str | None = None) -> float | None:
        """Recalibrated chance of the over, strictly inside (0, 1); None while no correction is on for this market or
        when p is not a number. p is clipped into [0, 1] (and to [1e-4, 1 − 1e-4] before the logit).

        Each market uses its own learned correction (or the shared one when that tested better for it); a market with
        no graded predictions yet falls back to the shared correction. Without a market, the shared one is used.
        For an under leg use 1 − calibrate_over(1 − p_under) (half-point lines, where over + under = 1).
        """
        active, intercept, slope = self._stat_params.get(stat, self._params) if stat is not None else self._params
        if not active or intercept is None or slope is None:
            return None
        try:
            p = float(p_over)
        except (TypeError, ValueError):
            return None
        if math.isnan(p):
            return None
        q = recalibrate(min(1.0, max(0.0, p)), intercept, slope)
        return float(min(1 - CLIP, max(CLIP, q)))

    calibrate = calibrate_over

    def calibration_for(self, stat: str) -> dict | None:
        """The calibration entry for one market (samples, choice, held-out scores), or None before training."""
        return next((copy.deepcopy(entry) for entry in self._calibration.get("stats") or [] if entry.get("stat") == stat), None)

    def calibration_status(self) -> dict:
        return copy.deepcopy(self._calibration)

    def status(self) -> dict:
        return {"calibration": self.calibration_status(), "memory": self.memory_summary(),
                "record": self.track_record(), "training": self.training}
