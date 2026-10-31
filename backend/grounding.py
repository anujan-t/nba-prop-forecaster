"""Grounding check for language-model answers: every number in a reply must come from the data the model was given.

A reply number is supported when some number in the evidence (tool results, the user's question, the prompt's facts:
today's date and what is being discussed) rounds to it at the precision shown: 26.83 supports "26.8" and "27", 0.5432
supports "54%" and "54.3%" (a percentage may also be a 0-1 probability times 100). Signs are ignored, so hyphens in
records and seasons can't trip it. List lengths and numbers in field names count too ("the top 5 defenses", an
"interval_80" is an 80% range), so pass tool results as objects rather than JSON text. Dates and clock times are matched
as dates and times ("Oct 28" against 2026-10-28, "7:30 PM" against "7:30 PM ET"), so the day of the month can't vouch
for a statistic. Anything else is reported as unsupported.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# Digits that are part of names, phrases or the problem-gambling helpline rather than statistics.
IGNORED = re.compile(
    r"1-800-[A-Z]+(?: \(1-800-\d{3}-\d{4}\))?|1-800-\d{3}-\d{4}|\b24/7\b|\b76ers\b|\b3-?(?:point(?:ers?|s)?|pt|pm|pa)\b|"
    r"\b(?:2|3)PT\b|\band-1\b|\bp(?:<|≤)?0?\.0?5\b",
    re.IGNORECASE)
LIST_MARKER = re.compile(r"(?m)^\s*(?:\d{1,2}[.)]|#\d{1,2})\s+")
NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|(?<![\w.])\d+(?:\.\d+)?(?:(?=st|nd|rd|th)|(?![\w.]?\d))")
PERCENT = re.compile(r"\s*(?:%|percent\b|pct\b|per cent\b)", re.IGNORECASE)
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
MONTH = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
# 2026-10-28 and 2026-10-28T23:30Z; "Oct 28", "October 28, 2026", "28 Oct"; 10/28/2026 and ESPN's "10/28 - 7:30 PM" (a bare
# 3/2 stays two numbers: it may be fractional odds or 3 of 2); "7:30", "7:30 PM", "7 pm".
ISO_DATE = re.compile(r"(?<![\w.])(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?(?![\w.]?\d)")
SLASH_DATE = re.compile(r"(?<![\w./])(\d{1,2})/(\d{1,2})(?:/(\d{4}|\d{2})(?![\w/])|(?![\d/])(?=\s*(?:[-\u2013\u2014,@]|at)?\s*\d{1,2}(?::[0-5]\d)?\s*[ap]\.?m\b))",
                        re.IGNORECASE)
NAMED_DATE = re.compile(rf"\b({MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?![.,]\d)(?:,?\s+(\d{{4}})\b)?|"
                        rf"(?<![\d.,])\b(\d{{1,2}})(?:(?:st|nd|rd|th)\s+(?:of\s+)?|\s+of\s+|\s+(?!may\b(?!,?\s+\d{{4}}\b)))({MONTH})\b\.?"
                        rf"(?:,?\s+(\d{{4}})\b)?", re.IGNORECASE)  # "26.5 may be high" and "the top 3 may sit" are not dates
CLOCK = re.compile(r"(?<![\w.:])(\d{1,2}):([0-5]\d)(?![\d:])(?:\s*([ap])\.?m\b)?|(?<![\w.:])(\d{1,2})\s*([ap])\.?m\b|"
                   r"(?<![\w.:])(\d{1,2})\s*(?-i:E[SD]?T|[CMP][SD]T)\b", re.IGNORECASE)  # "tips at 10 ET"
# A bare 10/28 is a date only after a date word ("on 10/28", "Fri 10/28", "(10/28") and never before a shooting word:
# "went 1/18 from the field" is a shooting line even when Jan 18 is a date the tools returned.
DATED_SLASH = re.compile(r"(\b(?:on|since|until|till|by|before|after|through|thru)\s+|\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\.?,?\s+|\(\s*)"
                         r"(\d{1,2})/(\d{1,2})(?![\w/])(?!\s*(?:fg|ft|3p|3pt|from\b|shooting|on\s+(?:threes|3s|field|free)|for\b))",
                         re.IGNORECASE)


@dataclass(frozen=True)
class Number:
    text: str
    value: float
    decimals: int
    percent: bool


@dataclass
class Evidence:
    """What a reply may quote: numbers (absolute values), dates (year, or 0 when not given; month; day) and clock times
    (hour on a 12-hour clock, minute)."""
    numbers: set[float] = field(default_factory=set)
    dates: set[tuple[int, int, int]] = field(default_factory=set)
    times: set[tuple[int, int]] = field(default_factory=set)


def _iso(match: re.Match) -> tuple[int, int, int, tuple[int, int] | None] | None:
    year, month, day = int(match[1]), int(match[2]), int(match[3])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    clock = (int(match[4]) % 12, int(match[5])) if match[4] and int(match[4]) <= 23 and int(match[5]) <= 59 else None
    return year, month, day, clock


def _named(match: re.Match) -> tuple[int | None, int, int] | None:
    if match[1]:
        month, day, year = MONTHS.index(match[1][:3].lower()) + 1, int(match[2]), match[3]
    else:
        day, month, year = int(match[4]), MONTHS.index(match[5][:3].lower()) + 1, match[6]
    return (int(year) if year else None, month, day) if 1 <= day <= 31 else None


def _slash(match: re.Match) -> tuple[int | None, int, int] | None:
    month, day, year = int(match[1]), int(match[2]), match[3]
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return (int(year) + (2000 if len(year) == 2 else 0) if year else None), month, day


def _clock(match: re.Match) -> tuple[int, int] | None:
    hour = int(next(group for group in (match[1], match[4], match[6]) if group is not None))
    minute = int(match[2]) if match[2] is not None else 0
    half = match[3] or match[5]
    if (half and not 1 <= hour <= 12) or (not half and hour > 23):
        return None
    return hour % 12, minute


def _without(pattern: re.Pattern, parse, text: str) -> str:
    return pattern.sub(lambda match: " " if parse(match) else match[0], text)


def _undated(text: str) -> str:
    return _without(NAMED_DATE, _named, _without(SLASH_DATE, _slash, _without(ISO_DATE, _iso, text)))


def dates_in(text: str) -> list[tuple[str, int | None, int, int]]:
    """Calendar dates in `text` as (label, year or None, month, day)."""
    text = IGNORED.sub(" ", text.replace("\u2212", "-"))
    found = [(match[0], *_iso(match)[:3]) for match in ISO_DATE.finditer(text) if _iso(match)]
    text = _without(ISO_DATE, _iso, text)
    found += [(match[0], *_slash(match)) for match in SLASH_DATE.finditer(text) if _slash(match)]
    text = _without(SLASH_DATE, _slash, text)
    found += [(match[0].strip(), *_named(match)) for match in NAMED_DATE.finditer(text) if _named(match)]
    return found


def clocks_in(text: str) -> list[tuple[str, int, int]]:
    """Clock times in `text`, including those in ISO timestamps, as (label, hour on a 12-hour clock, minute). "3PM" is
    threes made, not a time."""
    text = IGNORED.sub(" ", text.replace("\u2212", "-"))
    found = [(match[0], *_iso(match)[3]) for match in ISO_DATE.finditer(text) if _iso(match) and _iso(match)[3]]
    found += [(match[0].strip(), *_clock(match)) for match in CLOCK.finditer(_undated(text)) if _clock(match)]
    return found


def numbers_in(text: str) -> list[Number]:
    """The numbers a reader would see as figures in `text`: not list markers, team names, the helpline, dates or times."""
    cleaned = LIST_MARKER.sub(" ", IGNORED.sub(" ", text.replace("\u2212", "-")))
    cleaned = _without(CLOCK, _clock, _undated(cleaned))
    found = []
    for match in NUMBER.finditer(cleaned):
        raw = match.group(0)
        digits = raw.replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        found.append(Number(raw, float(digits), decimals, bool(PERCENT.match(cleaned, match.end()))))
    return found


def evidence_values(*sources: Any) -> Evidence:
    """Every number, date and clock time in the given tool results, texts and values, plus the length of every list."""
    evidence = Evidence()

    def read(text: str) -> None:
        for _, year, month, day in dates_in(text):
            evidence.dates.add((year or 0, month, day))
            if year:
                evidence.numbers.add(float(year))
        evidence.times.update((hour, minute) for _, hour, minute in clocks_in(text))
        evidence.numbers.update(number.value for number in numbers_in(text))

    def walk(item: Any) -> None:
        if isinstance(item, bool) or item is None:
            return
        if isinstance(item, (int, float)):
            if math.isfinite(item):
                evidence.numbers.add(abs(float(item)))
            return
        if isinstance(item, str):
            read(item)
            return
        if isinstance(item, dict):
            for key, value in item.items():
                if isinstance(key, str):  # field names carry meaning too ("interval_80" is an 80% range)
                    evidence.numbers.update(number.value for number in numbers_in(key.replace("_", " ")))
                walk(value)
            return
        if isinstance(item, (list, tuple, set)):
            evidence.numbers.add(float(len(item)))
            for value in item:
                walk(value)
            return
        walk(json.loads(json.dumps(item, default=str)))

    for source in sources:
        walk(source)
    return evidence


def supported(number: Number, evidence: Iterable[float]) -> bool:
    tolerance = 0.5 * 10 ** -number.decimals + 1e-9
    for value in evidence:
        if abs(number.value - value) <= tolerance:
            return True
        if number.percent and value <= 1 and abs(number.value - value * 100) <= tolerance:
            return True
    return False


def unsupported_numbers(text: str, evidence: Evidence) -> list[str]:
    """The dates, clock times and figures in `text` that the evidence doesn't support, without repeats."""
    missing: list[str] = []

    def report(label: str) -> None:
        if label not in missing:
            missing.append(label)

    def known_day(match: re.Match) -> bool:
        return any(m == int(match[2]) and d == int(match[3]) for _, m, d in evidence.dates)

    text = DATED_SLASH.sub(lambda match: match[1] + " " if known_day(match) else match[0], text)  # "on 10/28" is a given date
    for label, year, month, day in dates_in(text):
        if not any(m == month and d == day and (year is None or y in (0, year)) for y, m, d in evidence.dates):
            report(label)
    for label, hour, minute in clocks_in(text):
        if (hour, minute) not in evidence.times:
            report(label)
    for number in numbers_in(text):
        if not supported(number, evidence.numbers):
            report(number.text + ("%" if number.percent else ""))
    return missing
