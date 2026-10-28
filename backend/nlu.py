"""Deterministic language understanding for COURTSIDE Coach: players, stats, sides, lines, odds and intents."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import difflib
import re
import unicodedata

STATS = ("points", "rebounds", "assists", "steals", "blocks")
STAT_WORDS = {
    "points": ("points", "point", "pts", "pt", "pnts", "scoring", "buckets"),
    "rebounds": ("rebounds", "rebound", "rebs", "reb", "boards", "board"),
    "assists": ("assists", "assist", "asts", "ast", "dimes", "dime"),
    "steals": ("steals", "steal", "stl", "stls"),
    "blocks": ("blocks", "block", "blk", "blks", "swats", "rejections"),
}
STAT_LOOKUP = {word: stat for stat, words in STAT_WORDS.items() for word in words}
UNSUPPORTED = {
    "threes": "three-pointers", "three": "three-pointers", "3s": "three-pointers", "3pm": "three-pointers", "3pt": "three-pointers",
    "3pts": "three-pointers", "triples": "three-pointers", "pra": "points + rebounds + assists", "turnovers": "turnovers",
    "tos": "turnovers", "fantasy": "fantasy points", "minutes": "minutes", "mins": "minutes", "doubles": "double-doubles",
    "double-double": "double-doubles", "triple-double": "triple-doubles",
}
# Common nicknames; each is used only when its player is on the current roster.
NICKNAMES = {
    "joker": "Nikola Jokić", "the joker": "Nikola Jokić", "sga": "Shai Gilgeous-Alexander", "shai": "Shai Gilgeous-Alexander",
    "wemby": "Victor Wembanyama", "wemb": "Victor Wembanyama", "kd": "Kevin Durant", "durantula": "Kevin Durant",
    "lebron": "LeBron James", "bron": "LeBron James", "king james": "LeBron James", "steph": "Stephen Curry", "chef curry": "Stephen Curry",
    "ant": "Anthony Edwards", "ant man": "Anthony Edwards", "antman": "Anthony Edwards", "ad": "Anthony Davis", "the brow": "Anthony Davis",
    "greek freak": "Giannis Antetokounmpo", "luka": "Luka Dončić", "luka magic": "Luka Dončić", "jt": "Jayson Tatum", "jb": "Jaylen Brown",
    "cp3": "Chris Paul", "pg13": "Paul George", "kat": "Karl-Anthony Towns", "dame": "Damian Lillard", "spida": "Donovan Mitchell",
    "ja": "Ja Morant", "chet": "Chet Holmgren", "jjj": "Jaren Jackson Jr.", "the process": "Joel Embiid", "dlo": "D'Angelo Russell",
    "bam": "Bam Adebayo", "jimmy buckets": "Jimmy Butler III", "tyhal": "Tyrese Haliburton", "hali": "Tyrese Haliburton",
    "brunson": "Jalen Brunson", "cade": "Cade Cunningham", "paolo": "Paolo Banchero", "zion": "Zion Williamson", "trae": "Trae Young",
    "book": "", "scoot": "Scoot Henderson", "flagg": "Cooper Flagg",
}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# Words that are never read as a one-word player name (full names still match, e.g. "Danuel House").
STOPWORDS = {
    "over", "under", "points", "odds", "book", "house", "will", "smart", "love", "white", "black", "brown", "green", "young", "rose",
    "holiday", "price", "king", "line", "lines", "leg", "legs", "add", "remove", "drop", "flip", "change", "make", "parlay", "slip",
    "bet", "bets", "more", "less", "than", "least", "most", "same", "value", "edge", "fair", "good", "the", "and", "his", "him", "for",
    "with", "what", "whats", "chance", "chances", "probability", "track", "record", "grade", "undo", "clear", "help", "tonight",
    "today", "tomorrow", "yesterday", "this", "that", "game", "games", "last", "first", "second", "third", "next", "average",
    "averaging", "stats", "show", "tell", "give", "put", "take", "move", "set", "switch", "swap", "also", "plus", "then", "pays",
    "paying", "offered", "offering", "money", "even", "dollars", "bucks", "units", "stake", "compare", "against", "vig", "juice",
    "worth", "should", "would", "could", "doing", "playing", "form", "about", "explain", "why", "how", "often", "times", "season",
    "learned", "learning", "smarter", "means", "mean", "call", "remember", "jordan", "max", "bridges", "mobley", "james",
    "williams", "johnson", "jones", "smith", "miller", "mitchell", "murray", "porter", "jackson", "harris", "thompson", "walker",
    "allen", "anderson", "davis", "brooks", "hill", "martin", "robinson", "wallace", "watson", "wright", "collins", "carter", "bell",
    "lewis", "cook", "washington", "russell", "reed", "ball", "bey", "nance", "payton", "trent", "fox", "george", "paul",
}
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
ORDINALS = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5, "5th": 5,
            "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9, "tenth": 10, "10th": 10}


def fold(text: str) -> str:
    """Case-folded, accent-free text with typographic minus signs and quotes normalized."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    for source, target in (("−", "-"), ("–", "-"), ("—", "-"), ("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip()


def name_key(text: str) -> str:
    """Comparable form of a player name or phrase: no accents, apostrophes, periods or generational suffixes."""
    text = re.sub(r"['.,]", "", fold(text)).replace("-", " ")
    tokens = re.sub(r"[^a-z0-9 ]", " ", text).split()
    while len(tokens) > 1 and tokens[-1] in SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


@dataclass
class Mention:
    start: int
    end: int
    phrase: str
    candidates: list[dict]
    kind: str  # alias | full | last | first | fuzzy

    @property
    def certain(self) -> bool:
        return len(self.candidates) == 1 and self.kind != "fuzzy"


class PlayerIndex:
    """Finds current-roster players in free text by full name, surname, first name, nickname or learned alias."""

    def __init__(self, players: list[dict], aliases: dict[str, dict] | None = None):
        self.players = {str(player["id"]): player for player in players}
        self.keys: dict[str, dict[str, set[str]]] = {kind: {} for kind in ("alias", "full", "last", "first")}
        for player in players:
            tokens = name_key(player["name"]).split()
            if not tokens:
                continue
            self._add("full", " ".join(tokens), player["id"])
            if len(tokens) > 1:
                self._add("last", " ".join(tokens[1:]), player["id"])
                self._add("last", tokens[-1], player["id"])
                self._add("first", tokens[0], player["id"])
        for phrase, full in NICKNAMES.items():
            matches = self.keys["full"].get(name_key(full), set()) if full else set()
            if len(matches) == 1:
                self._add("alias", name_key(phrase), next(iter(matches)))
        self.learned = {}
        for phrase, alias in (aliases or {}).items():
            if str(alias["player_id"]) in self.players:
                key = name_key(phrase)
                self.keys["alias"][key] = {str(alias["player_id"])}
                self.learned[key] = str(alias["player_id"])

    def _add(self, kind: str, key: str, player_id) -> None:
        self.keys[kind].setdefault(key, set()).add(str(player_id))

    def by_name(self, text: str) -> list[dict]:
        """Exact lookup of a whole phrase (alias, full name, surname or first name), most specific first."""
        key = name_key(text)
        for kind in ("alias", "full", "last", "first"):
            ids = self.keys[kind].get(key)
            if ids:
                return sorted((self.players[i] for i in ids), key=lambda p: p["name"])
        return []

    def find(self, text: str) -> list[Mention]:
        tokens = name_key(text).split()
        used = [False] * len(tokens)
        mentions: list[Mention] = []
        for size in (4, 3, 2, 1):
            for start in range(0, len(tokens) - size + 1):
                if any(used[start:start + size]):
                    continue
                phrase = " ".join(tokens[start:start + size])
                for kind in ("alias", "full", "last", "first"):
                    if size == 1 and kind != "alias" and (phrase in STOPWORDS and kind != "full" or phrase in STAT_LOOKUP
                                                          or phrase in UNSUPPORTED or len(phrase) < 3 or phrase.isdigit()):
                        continue
                    if size == 1 and kind == "first" and phrase in STOPWORDS:
                        continue
                    ids = self.keys[kind].get(phrase)
                    if ids:
                        candidates = sorted((self.players[i] for i in ids), key=lambda p: p["name"])
                        mentions.append(Mention(start, start + size, phrase, candidates, kind))
                        used[start:start + size] = [True] * size
                        break
        # Ambiguous common surnames (e.g. "murray") are still worth offering as choices.
        for start, token in enumerate(tokens):
            if not used[start] and token in STOPWORDS and token in self.keys["last"] and len(self.keys["last"][token]) >= 1 \
                    and token not in {"james", "jordan", "max", "paul", "george"} and len(token) >= 4 and self._near_leg_words(tokens, start):
                ids = self.keys["last"][token]
                mentions.append(Mention(start, start + 1, token, sorted((self.players[i] for i in ids), key=lambda p: p["name"]), "last"))
                used[start] = True
        if not mentions:
            mentions = self._fuzzy(tokens)
        return sorted(mentions, key=lambda mention: mention.start)

    @staticmethod
    def _near_leg_words(tokens: list[str], index: int) -> bool:
        """A surname that is also a stopword counts only next to prop words (e.g. "murray over 20.5")."""
        window = tokens[max(0, index - 2):index + 4]
        return any(token in STAT_LOOKUP or token in {"over", "under", "o", "u"} or re.fullmatch(r"\d+(\.\d)?", token) for token in window)

    def _fuzzy(self, tokens: list[str]) -> list[Mention]:
        names = list(self.keys["full"]) + [key for key in self.keys["last"] if " " not in key]
        for size in (3, 2, 1):
            for start in range(0, len(tokens) - size + 1):
                phrase = " ".join(tokens[start:start + size])
                if len(phrase) < 5 or phrase in STOPWORDS or phrase in STAT_LOOKUP or any(t.isdigit() for t in tokens[start:start + size]):
                    continue
                match = difflib.get_close_matches(phrase, names, n=1, cutoff=0.84)
                if match:
                    ids = self.keys["full"].get(match[0]) or self.keys["last"].get(match[0])
                    return [Mention(start, start + size, phrase, sorted((self.players[i] for i in ids), key=lambda p: p["name"]), "fuzzy")]
        return []


@dataclass
class LegSpec:
    stat: str | None = None
    side: str | None = None
    line: float | None = None
    unsupported: str | None = None
    problem: str | None = None


ODDS_CONTEXT = re.compile(r"\b(odds|price|priced|pays?|paying|offer(?:ed|ing|s)?|book|books|sportsbook|house|vig|juice|line is|"
                          r"at|ev|value|worth|implied|break.?even|payout|fair)\b")


def find_odds(text: str) -> tuple[int | None, str, str | None]:
    """American odds named in the text (signed, even money, decimal or fractional), the text without them, and any problem."""
    t = fold(text)
    patterns = [
        (r"(?<![\w.$])([+-]\s?\d{3,5})(?![\w.%])", lambda m: int(m.group(1).replace(" ", ""))),
        (r"\b(even money|evens|even odds)\b", lambda m: 100),
        (r"\b(\d{1,3}\.\d{1,2})\s*(?:x|decimal|dec)\b", lambda m: decimal_to_american(float(m.group(1)))),
    ]
    if ODDS_CONTEXT.search(t):
        patterns.append((r"(?<![\d/])(\d{1,3})\s*/\s*(\d{1,3})(?![\d/])", lambda m: decimal_to_american(1 + int(m.group(1)) / max(1, int(m.group(2))))))
        patterns.append((r"\b(?:odds|price|priced at|pays|paying|offering|offered at|at)\s+(\d{3,5})\b(?!\s*(?:pts|points|reb|ast))",
                         lambda m: int(m.group(1))))
    for pattern, convert in patterns:
        match = re.search(pattern, t)
        if match:
            value = convert(match)
            rest = (t[:match.start()] + " " + t[match.end():]).strip()
            if value is None or not 100 <= abs(value) <= 10000:
                return None, rest, "Odds must be American odds from −10000 to −100 or +100 to +10000 (decimal 1.01–101 or fractions like 9/2 also work)."
            return value, rest, None
    return None, t, None


def decimal_to_american(value: float) -> int | None:
    if not value > 1:
        return None
    return int(round((value - 1) * 100)) if value >= 2 else int(round(-100 / (value - 1)))


def find_stake(text: str) -> float | None:
    t = fold(text)
    for pattern in (r"\$\s?(\d+(?:\.\d{1,2})?)", r"\b(\d+(?:\.\d{1,2})?)\s*(?:dollars|bucks|units)\b", r"\bstake\s+(?:of\s+)?(\d+(?:\.\d{1,2})?)\b"):
        match = re.search(pattern, t)
        if match:
            return float(match.group(1))
    return None


def find_date(text: str, today: date) -> date | None:
    t = fold(text)
    if re.search(r"\b(tonight|today)\b", t):
        return today
    if re.search(r"\btomorrow\b", t):
        return today + timedelta(days=1)
    if re.search(r"\byesterday\b", t):
        return today - timedelta(days=1)
    match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    match = re.search(r"\b(?:on|for)\s+(\d{1,2})/(\d{1,2})\b", t)
    if match:
        try:
            return date(today.year, int(match.group(1)), int(match.group(2)))
        except ValueError:
            return None
    return None


def find_leg_numbers(text: str) -> list[int]:
    t = fold(text)
    numbers = [int(m.group(1)) for m in re.finditer(r"\blegs?\s*#?\s*(\d{1,2})\b", t)]
    numbers += [WORD_NUMBERS[m.group(1)] for m in re.finditer(r"\blegs?\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b", t)]
    numbers += [ORDINALS[m.group(1)] for m in re.finditer(r"\b(" + "|".join(ORDINALS) + r")\s+leg\b", t)]
    numbers += [-1 for _ in re.finditer(r"\blast\s+leg\b", t)]
    return list(dict.fromkeys(numbers))


def _prepare(text: str) -> str:
    t = fold(text)
    for word, number in ORDINALS.items():
        t = re.sub(rf"\b{word}\b", f"ord{number}", t)
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)  # 26.5pts -> 26.5 pts
    t = re.sub(r"\b([ou])(\d)", r"\1 \2", t)  # o26.5 -> o 26.5
    return t


def parse_leg(text: str) -> LegSpec:
    """Stat, side and line from a phrase like "Jokić o26.5 pts", "10+ assists" or "under 11.5 boards"."""
    _, t, _ = find_odds(text)
    t = _prepare(t)
    t = re.sub(r"\blegs?\s*#?\s*\d{1,2}\b|\$\s?\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:dollars|bucks|units)\b|\b\d{4}-\d{2}-\d{2}\b"
               r"|\b\d{1,2}/\d{1,2}\b|\b(?:last|past|previous)\s+\d{1,2}\s+games?\b|\bin\s+\d{1,2}\s+(?:of|out of)\s+\d{1,2}\b", " ", t)
    spec = LegSpec()
    tokens = re.findall(r"[a-z0-9+#]+|\d+\.\d+", t)
    for token in tokens:
        if token in STAT_LOOKUP and spec.stat is None:
            spec.stat = STAT_LOOKUP[token]
        elif token in UNSUPPORTED and spec.unsupported is None:
            spec.unsupported = UNSUPPORTED[token]
    number = r"(\d{1,3}(?:\.\d+)?)"
    rules = [
        (rf"\b(?:o|over)\s+{number}", "over", 0.0, False),
        (rf"\b(?:u|under)\s+{number}", "under", 0.0, False),
        (rf"\b(?:more than|greater than|above)\s+{number}", "over", 0.5, True),
        (rf"\b(?:at least|minimum of|min)\s+{number}", "over", -0.5, True),
        (rf"(?<![\d.]){number}\s*\+(?!\d)", "over", -0.5, True),
        (rf"\b(?:less than|fewer than|below)\s+{number}", "under", -0.5, True),
        (rf"\b(?:at most|no more than|maximum of|max)\s+{number}", "under", 0.5, True),
    ]
    for pattern, side, shift, whole_only in rules:
        match = re.search(pattern, t)
        if match:
            value = float(match.group(1))
            spec.side, spec.line = side, value + (shift if value.is_integer() or not whole_only else 0.0)
            break
    if spec.side is None:
        if re.search(r"\b(?:the\s+)?overs?\b", t):
            spec.side = "over"
        elif re.search(r"\b(?:the\s+)?unders?\b", t):
            spec.side = "under"
    if spec.line is None:
        match = re.search(r"(?<![\d.+-])(\d{1,3}(?:\.\d+)?)(?![\d.%/])", t)
        if match and (spec.stat or spec.side):
            spec.line = float(match.group(1))
    if spec.line is not None and (spec.line < 0 or spec.line > 200 or spec.line * 2 != int(spec.line * 2)):
        spec.problem = f"{spec.line:g} is not a valid line: use a whole or half point from 0 to 200."
        spec.line = None
    return spec


INTENTS = [
    ("teach", r"^(?:please\s+)?(?:remember|learn|note)\b.*\b(?:means|is)\b|^\s*when i say\b|\bmeans\b"),
    ("undo", r"^\s*(?:undo|go back|revert|take (?:that|it) back|nevermind|never mind)\b"),
    ("clear", r"\b(?:clear|reset|empty|wipe)\b.*\b(?:slip|parlay|everything|all|legs)\b|^\s*(?:clear|start over|reset)\s*(?:it)?\s*$|\bremove (?:all|everything)\b|\bstart over\b"),
    ("help", r"^\s*(?:help|commands|what can you do|how do (?:i|you) use|what do you do)\b"),
    ("greet", r"^\s*(?:hi|hey|hello|yo|sup|good (?:morning|afternoon|evening)|what'?s up)\b[\s!.?]*$"),
    ("track", r"\b(?:track|log|save|record)\s+(?:this|that|it|my|the|a)\b|\b(?:i (?:placed|took|locked in|put money on)|placed (?:it|this|the|my|a)|wagered)\b"),
    ("forget", r"^\s*forget\s+\S"),
    ("settle", r"\b(?:grade|settle)\b|\b(?:track record|my record|results|how did (?:my|i|it|the)|did (?:my|i|it) (?:win|hit|cash)|how am i doing|my bets|bet history|performance)\b"),
    ("learning", r"\b(?:smarter|learn(?:ing|ed|s)?|calibrat\w*|getting better|improv\w*|train\w*)\b"),
    ("remove", r"\b(?:remove|delete|drop|scratch|take (?:off|out)|get rid of|cut)\b"),
    ("update", r"\b(?:flip|switch|swap|change|move|bump|raise|lower)\b|\bmake (?:it|that|leg|the)\b|\bset (?:leg|the line|it|that)\b"),
    ("house", r"\b(?:house|vig|juice|edge|ev|\+ev|-ev|expected value|value|worth it|worth|good (?:price|bet|odds)|bad (?:price|bet)|fair (?:odds|price)|compare|break.?even|implied|overpriced|underpriced|beat the book|good)\b"),
    ("explain", r"\b(?:why|explain|correlat\w*|linked|together|independent)\b"),
    ("summary", r"\b(?:averag\w*|stats for|stat line|how(?:'s| is| has)\s+\w+(?:\s+\w+)?\s+(?:doing|playing|been)|recent form|game log|last \d+ games)\b"),
    ("hit_rate", r"\b(?:how often|hit rate|how many (?:times|games)|cleared|has (?:he|she) hit)\b"),
    ("forecast", r"\?|\b(?:chance|chances|probability|likely|odds of|odds that|will|does|can|projection|project|forecast|predict\w*|expect\w*|what about)\b"),
    ("parlay", r"\b(?:my (?:parlay|slip|bet|ticket|odds|chances?)|our odds|the (?:parlay|slip)|parlay|slip|price (?:it|this|my)|all (?:\d+|the|my) legs)\b"),
    ("add", r"\b(?:add|put|throw|include|tack|stack|plus|also|parlay it|give me|lock in|i like)\b"),
]


def intents(text: str) -> list[str]:
    t = fold(text).replace("track record", "trackrecord")
    found = [name for name, pattern in INTENTS if re.search(pattern, t)]
    if "trackrecord" in t and "settle" not in found:
        found.append("settle")
    return found


def split_segments(text: str) -> list[str]:
    """Splits "add Jokić o26.5 pts and Murray 21+ points" into one segment per leg."""
    parts = re.split(r"\s*(?:,|;|&|\band\b|\balso\b|\bplus\b|\bthen\b|\n)\s*", text)
    segments: list[str] = []
    for part in parts:
        if not part.strip():
            continue
        spec = parse_leg(part)
        if segments and not (spec.stat or spec.side or spec.line is not None):
            segments[-1] = f"{segments[-1]} {part}"
        else:
            segments.append(part)
    return segments or [text]


@dataclass
class Teach:
    phrase: str
    name: str


def parse_teach(text: str) -> Teach | None:
    t = fold(text).strip(" .!?")
    for pattern in (r"^(?:please\s+)?(?:remember|learn|note)\s+(?:that\s+)?['\"]?(?P<alias>.+?)['\"]?\s+(?:means|is)\s+(?P<name>.+)$",
                    r"^when i say\s+['\"]?(?P<alias>.+?)['\"]?,?\s+i mean\s+(?P<name>.+)$",
                    r"^['\"]?(?P<alias>[a-z0-9 .'-]{2,40}?)['\"]?\s+means\s+(?P<name>.+)$"):
        match = re.match(pattern, t)
        if match:
            return Teach(match.group("alias").strip(" '\""), match.group("name").strip(" '\""))
    return None


@dataclass
class Parsed:
    text: str
    intents: list[str]
    odds: int | None
    odds_problem: str | None
    stake: float | None
    event_date: date | None
    legs: list[int]
    segments: list[str] = field(default_factory=list)


def parse(text: str, today: date) -> Parsed:
    odds, _, problem = find_odds(text)
    return Parsed(text, intents(text), odds, problem, find_stake(text), find_date(text, today), find_leg_numbers(text), split_segments(text))
