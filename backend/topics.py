"""Understanding league questions for COURTSIDE Coach: NBA team mentions, weekdays, ranking metrics and the topic of
a question (injuries, news, roster moves, coaches, playing style, rankings, standings, schedule, scores, game lines,
rosters, player bios, odds math, betting advice, site help or off-topic)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import re

from .nlu import fold

TEAM_NAMES: dict[str, tuple[str, ...]] = {
    "ATL": ("atlanta hawks", "hawks", "atlanta"),
    "BOS": ("boston celtics", "celtics", "celts", "boston"),
    "BKN": ("brooklyn nets", "nets", "brooklyn"),
    "CHA": ("charlotte hornets", "hornets", "charlotte"),
    "CHI": ("chicago bulls", "bulls", "chicago"),
    "CLE": ("cleveland cavaliers", "cavaliers", "cavs", "cleveland"),
    "DAL": ("dallas mavericks", "mavericks", "mavs", "dallas"),
    "DEN": ("denver nuggets", "nuggets", "nugs", "denver"),
    "DET": ("detroit pistons", "pistons", "detroit"),
    "GSW": ("golden state warriors", "warriors", "golden state", "dubs"),
    "HOU": ("houston rockets", "rockets", "houston"),
    "IND": ("indiana pacers", "pacers", "indiana"),
    "LAC": ("los angeles clippers", "la clippers", "clippers", "clips"),
    "LAL": ("los angeles lakers", "la lakers", "lakers", "lake show"),
    "MEM": ("memphis grizzlies", "grizzlies", "grizz", "memphis"),
    "MIA": ("miami heat", "heat", "miami"),
    "MIL": ("milwaukee bucks", "bucks", "milwaukee"),
    "MIN": ("minnesota timberwolves", "timberwolves", "t wolves", "twolves", "wolves", "minnesota"),
    "NOP": ("new orleans pelicans", "pelicans", "pels", "new orleans", "nola"),
    "NYK": ("new york knicks", "knicks", "new york"),
    "OKC": ("oklahoma city thunder", "thunder", "oklahoma city", "oklahoma"),
    "ORL": ("orlando magic", "magic", "orlando"),
    "PHI": ("philadelphia 76ers", "philadelphia sixers", "76ers", "sixers", "philadelphia", "philly"),
    "PHX": ("phoenix suns", "suns", "phoenix"),
    "POR": ("portland trail blazers", "trail blazers", "blazers", "portland", "rip city"),
    "SAC": ("sacramento kings", "kings", "sacramento"),
    "SAS": ("san antonio spurs", "spurs", "san antonio"),
    "TOR": ("toronto raptors", "raptors", "raps", "toronto"),
    "UTA": ("utah jazz", "jazz", "utah"),
    "WAS": ("washington wizards", "wizards", "wiz", "washington"),
}
# Abbreviations count in lower case only when they can't be ordinary words; MIN, WAS, IND and the like need capitals.
SAFE_CODES = {"atl", "bos", "bkn", "cle", "dal", "gsw", "hou", "lac", "lal", "mil", "nop", "nyk", "okc", "orl", "phi", "phx",
              "por", "sas", "tor", "uta", "sac", "mem", "cha", "det", "chi", "wsh"}
CODE_ALIASES = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "WSH": "WAS", "PHO": "PHX", "BRK": "BKN", "UTAH": "UTA", "NOLA": "NOP"}
TEAM_DISPLAY = {code: names[0].title().replace("76Ers", "76ers") for code, names in TEAM_NAMES.items()}
NICKNAME = {code: names[1].title() if code != "PHI" else "76ers" for code, names in TEAM_NAMES.items()}
NICKNAME.update({"POR": "Trail Blazers", "LAC": "Clippers", "LAL": "Lakers", "GSW": "Warriors", "MIN": "Timberwolves",
                 "NOP": "Pelicans", "NYK": "Knicks", "OKC": "Thunder", "PHX": "Suns", "SAS": "Spurs", "SAC": "Kings"})


@dataclass(frozen=True)
class TeamMention:
    code: str
    phrase: str
    start: int
    end: int


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9+$ ]", " ", fold(text).replace("-", " ").replace("'s", "").replace("’s", ""))


_PHRASES = sorted(((phrase, code) for code, names in TEAM_NAMES.items() for phrase in names), key=lambda item: -len(item[0]))


def find_teams(text: str) -> list[TeamMention]:
    """NBA teams named in the text by full name, nickname, city or abbreviation, in order of appearance."""
    normalized = f" {_normalize(text)} "
    used = [False] * len(normalized)
    found: list[TeamMention] = []
    for phrase, code in _PHRASES:
        for match in re.finditer(rf"(?<=\s){re.escape(phrase)}(?=\s)", normalized):
            start, end = match.span()
            if any(used[start:end]):
                continue
            if phrase == "bucks" and re.search(r"(?:\d|\$)\s*$", normalized[:start]):
                continue  # "10 bucks" is money
            if phrase == "heat" and re.search(r"\bheat check\b", normalized):
                continue
            used[start:end] = [True] * (end - start)
            found.append(TeamMention(code, phrase, start - 1, end - 1))
    letters = [char for char in text if char.isalpha()]
    shouting = len(letters) >= 8 and sum(char.isupper() for char in letters) / len(letters) > 0.6
    for match in re.finditer(r"\b[A-Za-z]{2,4}\b", text):
        token = match.group(0)
        if shouting and token.lower() not in SAFE_CODES:
            continue
        code = CODE_ALIASES.get(token.upper(), token.upper())
        if code not in TEAM_NAMES and token.upper() not in CODE_ALIASES:
            continue
        if not (token.isupper() or token.lower() in SAFE_CODES and len(token) >= 3):
            continue
        if any(mention.code == code for mention in found):
            continue
        found.append(TeamMention(code, token.lower(), match.start(), match.end()))
    found.sort(key=lambda mention: mention.start)
    unique: list[TeamMention] = []
    for mention in found:
        if all(other.code != mention.code for other in unique):
            unique.append(mention)
    return unique


WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def find_day(text: str, today: date) -> date | None:
    """today/tonight, tomorrow, yesterday/last night, a weekday (the next one, or the last one after "last"), or a date."""
    t = fold(text)
    if re.search(r"\b(?:last night|yesterday)\b", t):
        return today - timedelta(days=1)
    if re.search(r"\btomorrow\b", t):
        return today + timedelta(days=1)
    if re.search(r"\b(?:tonight|today|this evening)\b", t):
        return today
    for index, name in enumerate(WEEKDAYS):
        match = re.search(rf"\b(last\s+)?{name}\b", t)
        if match:
            delta = (index - today.weekday()) % 7
            if match.group(1):
                return today - timedelta(days=(today.weekday() - index) % 7 or 7)
            return today + timedelta(days=delta)
    match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    match = re.search(r"\b(?:on|for)?\s*(\d{1,2})/(\d{1,2})\b(?!/)", t)
    if match:
        try:
            return date(today.year, int(match.group(1)), int(match.group(2)))
        except ValueError:
            return None
    return None


# (metric, pattern, direction of "more" in the stat: True when a bigger value is the league rank 1)
METRIC_WORDS: tuple[tuple[str, str], ...] = (
    ("opponent_threes", r"\b(?:allow|give|gives|giving|allows|allowed|surrender|concede)\w*\b.*\b(?:threes|3s|3 pointers|three pointers|3pt|triples)\b|\bthrees allowed\b|\bopponent threes\b"),
    ("opponent_rebounds", r"\b(?:allow|give|gives|allows|allowed|concede)\w*\b.*\b(?:rebounds|boards)\b"),
    ("opponent_assists", r"\b(?:allow|give|gives|allows|allowed|concede)\w*\b.*\bassists\b"),
    ("opponent_points", r"\b(?:allow|give|gives|allows|allowed|concede)\w*\b.*\bpoints\b|\bpoints allowed\b"),
    ("opponent_efg", r"\b(?:opponent|allowed)\b.*\b(?:efg|effective|shooting|fg)\b|\bshot defense\b|\bdefend the (?:three|perimeter)\b"),
    ("forced_turnovers", r"\bforc\w*\b.*\bturnovers?\b|\bsteals\b|\bdisrupt\w*\b"),
    ("turnover_rate", r"\bturnovers?\b|\bcareless\b|\bball security\b"),
    ("offensive_rebounding", r"\boffensive rebound\w*\b|\boffensive glass\b|\bsecond chance\w*\b|\bcrash\w*\b"),
    ("three_rate", r"\bthrees?\b|\b3s\b|\b3 point\w*\b|\bthree point\w*\b|\b3pt\b|\bfrom deep\b|\bfrom three\b"),
    ("paint_share", r"\bpaint\b|\binside\b|\bat the rim\b"),
    ("fast_break_share", r"\bfast break\w*\b|\btransition\b|\bin the open court\b"),
    ("assisted", r"\bassist\w*\b|\bball movement\b|\bpass\w*\b|\bunselfish\b"),
    ("free_throw_rate", r"\bfree throws?\b|\bfoul line\b|\bfree throw line\b|\bget to the line\b|\bft rate\b"),
    ("pace", r"\bpace\b|\bfast(?:est)?\b|\bslow(?:est)?\b|\btempo\b|\bpossessions\b|\bup.?tempo\b"),
    ("net", r"\bnet rating\b|\bnet\b|\boverall\b|\bbest team\b|\bworst team\b"),
    ("defense", r"\bdefen[cs]\w*\b|\bdefending\b|\bstop\w*\b"),
    ("offense", r"\boffen[cs]\w*\b|\bscor\w*\b|\befficien\w*\b"),
)
MORE = re.compile(r"\b(?:most|more|highest|high|fastest|top|leads?|leading|lots|many|biggest|heaviest|max)\b")
LESS = re.compile(r"\b(?:least|fewest|less|lowest|low|slowest|bottom|minimum|smallest)\b")
BEST = re.compile(r"\b(?:best|elite|good|great|strongest|top)\b")
WORST = re.compile(r"\b(?:worst|bad|terrible|weakest|leaky|poor|bottom)\b")


def find_metric(text: str) -> str | None:
    t = fold(text)
    for metric, pattern in METRIC_WORDS:
        if re.search(pattern, t):
            return metric
    return None


def rank_direction(text: str, metric: str, higher_is_first: bool) -> bool:
    """True to list the league's rank 1 downwards, False to list from the bottom."""
    t = fold(text)
    if metric == "pace" and re.search(r"\bslow", t):
        return False
    if WORST.search(t) and not BEST.search(t):
        return False
    if BEST.search(t) and not WORST.search(t):
        return True
    if LESS.search(t):
        return not higher_is_first
    if MORE.search(t):
        return higher_is_first
    return True


OTHER_SPORTS = re.compile(
    r"\b(?:nfl|super bowl|touchdowns?|quarterback|mlb|baseball|home runs?|world series|nhl|hockey|stanley cup|soccer|premier league|"
    r"champions league|world cup|la liga|mls|ufc|mma|boxing|tennis|wimbledon|golf|pga|masters|nascar|formula 1|f1|cricket|"
    r"college football|ncaaf|march madness|ncaa|wnba|euroleague|horse racing|kentucky derby|esports)\b")
SPORTS_WORDS = re.compile(
    r"\b(?:nba|basketball|bet\w*|odds|parlay|prop\w*|line|lines|spread|moneyline|total|over|under|points?|rebounds?|assists?|"
    r"threes?|steals?|blocks?|turnovers?|pra|team|teams|player|players|game|games|season|playoffs?|injur\w*|coach\w*|roster|trade\w*|"
    r"sportsbook|book|vig|juice|units?|bankroll|slip|leg|legs|wager\w*|stake|ev|edge|model|forecast\w*|simulat\w*|calibrat\w*|"
    r"courtside|news|standings?|schedule|tonight|draft|rookie|mvp|all.?star|dunk|hoops?|ball|court|scor\w*|record)\b")


# (topic, strong pattern: enough on its own, weak pattern: counts only when a team or player is named)
TOPICS: tuple[tuple[str, str, str], ...] = (
    ("injury",
     r"\binjur\w*|\bhurt\b|\bruled out\b|\bquestionable\b|\bdoubtful\b|\bday to day\b|\bday-to-day\b|\bgtd\b|\bgame.time decision\b|"
     r"\bsidelined\b|\bwho(?:'s| is| are)? (?:out|injured|hurt|sitting|resting|missing)\b|\binjury report\b|\bload management\b|\bdnp\b",
     r"\bhealthy\b|\bhealth\b|\bout (?:tonight|today|tomorrow|for|with|vs|against|again|indefinitely|long)\b|\bprobable\b|\bresting\b|"
     r"\bsit(?:s|ting)? (?:out|tonight)\b|\bavailab\w+\b|\bsuit(?:ing)? up\b|\bcoming back\b|\bback from\b|\breturn(?:s|ing)? (?:from|to|tonight|date)\b|"
     r"\bwhen (?:will|does|is) .{1,30} (?:return|be back|back)\b|\bstatus\b|\bis .{1,30} (?:playing|out|active|in tonight|going)\b|"
     r"\bwill .{1,30} (?:play|suit up|be out|be back|return|miss)\b|\bmiss(?:es|ing)? (?:the |tonight|tomorrow|time|games?)\b|"
     r"\bplaying (?:tonight|today|tomorrow)\b|\bis .{1,30} out\b|\bout\??$"),
    ("moves",
     r"\btrades?\b|\btraded\b|\btrading\b|\btransactions?\b|\broster moves?\b|\bfree agen\w+\b|\bwaive[ds]?\b|\bwaiving\b|\bbuyout\b|"
     r"\btwo.way\b|\bg league\b|\bsign(?:ed|ing|ings)\b|\bacquir\w+\b|\bany moves\b|\bmade (?:a|any) moves?\b",
     r"\bsigns?\b|\breleased?\b|\bextension\b|\bcontract\b|\bclaim(?:ed|s)?\b|\bcut\b|\bcalled up\b|\bmoves?\b"),
    ("coaching",
     r"\bhead coach\w*\b|\bcoaching staff\b|\bassistant coach\w*\b|\bwho coaches\b|\bcoaches\b|\bcoached\b|"
     r"\bwho(?:'s| is) (?:the |their )?(?:head )?coach\b|\bwho(?:'s| is) coaching\b",
     r"\bcoach\b|\bfired\b|\bhired\b|\bstaff\b|\bcoaching\b"),
    ("lines",
     r"\bspreads?\b|\bmoney ?lines?\b|\bpoint spread\b|\bgame (?:lines?|odds)\b|\bvegas\b|\bwho(?:'s| is) favou?red\b|"
     r"\bfavou?rites? (?:tonight|today|tomorrow)\b|\bunderdogs? (?:tonight|today|tomorrow)\b|\bbetting lines?\b",
     r"\bml\b|\bover ?/ ?under\b|\bover under\b|\bo/u\b|\btotals?\b|\bfavou?rites?\b|\bfavou?red\b|\bunderdogs?\b|\bdogs?\b|\blines?\b|"
     r"\bodds\b|\bwho wins\b|\bwho(?:'s| is) (?:going to|gonna) win\b|\bcover\b|\bpredict\w*\b|\bwin tonight\b"),
    ("scores",
     r"\bscores?\b|\bwho won\b|\blast night\b|\bresults? (?:of|from) (?:last|yesterday|the)\b|\bbox scores?\b|\bfinal\b",
     r"\bwon\b|\blost\b|\bbeat\b|\bwin\b|\blose\b|\bhow did .{1,30} (?:do|play|look)\b|\blast game\b|\byesterday\b|\bblew out\b|\bresult\b"),
    ("schedule",
     r"\bschedule\b|\bslate\b|\bwho(?:'s| is| are) playing\b|\bwhat games\b|\bgames? (?:tonight|today|tomorrow|on \w+|this week)\b|"
     r"\btonight'?s games\b|\btomorrow'?s games\b|\bany games\b|\bwhere can i watch\b|\bon tv\b|\bwho plays\b|\bwho(?:'s| is) on (?:tonight|tomorrow)\b",
     r"\bnext game\b|\bplay(?:s|ing)? next\b|\btip ?off\b|\bstart time\b|\bwhat time\b|\bwhen (?:do|does|is|are)\b|"
     r"\bwho (?:do|does|are|is) .{1,30} play(?:ing)?\b|\bplaying who\b|\bmatchups?\b|\bchannel\b|\bhome or away\b|\btonight\b|"
     r"\btomorrow\b|\bvs\.?\b|\bversus\b|\bplay(?:ing)? (?:against|the)\b|\bgame\b"),
    ("standings",
     r"\bstandings?\b|\bseed(?:s|ed|ing)?\b|\bplayoff (?:picture|race|spot)\b|\bconference\b|\bbest record\b|\bworst record\b|"
     r"\bundefeated\b|\bwinless\b|\bfirst place\b|\blast place\b",
     r"\brecords?\b|\bwins and losses\b|\bhow many (?:wins|games have)\b|\bw-l\b"),
    ("roster",
     r"\broster\b|\bdepth chart\b|\bstarting (?:five|lineup|5)\b|\bwho plays for\b|\bplayers on\b",
     r"\blineups?\b|\bstarters\b|\bwho(?:'s| is| are) on the\b|\bwho do (?:the|they) have\b"),
    ("player_info",
     r"(?!x)x",
     r"\bhow old\b|\bage\b|\bborn\b|\bbirthday\b|\bheight\b|\bhow tall\b|\bweigh\w*\b|\bposition\b|\bcollege\b|\bschool\b|"
     r"\bwhere did .{1,30} go to\b|\bdraft(?:ed)?\b|\bexperience\b|\byears in the league\b|\bjersey\b|\bwhat number\b|"
     r"\bwhat team (?:does|is)\b|\bwho does .{1,30} play for\b|\bplays? for\b|\bwhere does .{1,30} play\b|\bcountry\b|\bnationality\b|"
     r"\bwhere is .{1,30} from\b|\bbio\b|\bwho is\b|\bwho'?s\b|\btell me about\b"),
    ("odds_math",
     r"\bconvert\w*\b|\bin decimal\b|\bdecimal odds\b|\bfractional\b|\bimplied (?:probability|odds|chance)\b|\bbreak ?even\b|\bpayout\b|"
     r"\bpays? out\b|\bhow much (?:would|will|do|can|could|does) (?:i|it|that|a|you)\b.{0,40}\b(?:win|make|get|profit|pay|return)\b|"
     r"\bwhat does [+-]?\d{3,5} mean\b|\bwhat(?:'s| is) [+-]\d{3,5}\b|[+-]\d{3,5} (?:mean|means|in decimal|as a (?:probability|percentage))\b|"
     r"\bto win \$?\d+\b|\bin american\b|\bamerican odds\b|\bto american\b|\bas american\b|\bto decimal\b|\bas decimal\b",
     r"(?!x)x"),
    ("advice",
     r"\bshould i (?:bet|take|play|tail|fade|hammer|parlay|lock)\b|\bbest bets?\b|\block of the\b|\bsure thing\b|\bguarantee\w*\b|"
     r"\bwho should i\b|\bwhat should i (?:bet|take|play)\b|\bbet of the day\b|\bfree money\b|\bcan'?t lose\b|\bgood bets?\b|"
     r"\bwho do you like\b|\bwhat do you like\b|\bany value\b|\bwhat(?:'s| is) the play\b|\bgive me (?:a |some |your )?picks?\b|\byour picks?\b|"
     r"\bpicks? (?:tonight|today|for)\b|\bwhat to bet\b|\bwho to bet\b|\bwho(?:'s| is) a lock\b",
     r"\blocks?\b|\brecommend\w*\b|\bpicks?\b"),
    ("news",
     r"\bnews\b|\bheadlines?\b|\bwhat(?:'s| is) (?:new|happening|going on)\b|\bgoing on with\b|\brumou?rs?\b|\bany updates?\b|"
     r"\bupdates? on\b|\bupdate me\b|\bcatch me up\b|\bbrief(?:ing)?\b|\bwhat happened (?:to|with)\b",
     r"\blatest\b|\bstor(?:y|ies)\b|\barticles?\b|\breports?\b|\breported\b|\bbuzz\b|\bwhat happened\b|\bupdate\b"),
    ("style",
     r"\bplay(?:ing)? style\b|\bplay ?types?\b|\bscouting report\b",
     r"\bstyle\b|\bhow do(?:es)? (?:the )?.{1,25} play\b|\bidentity\b|\bscheme\b|\btendenc\w+\b|\bpace\b|\btempo\b|\boffen[cs]e\b|"
     r"\bdefen[cs]e\b|\bdefensive\b|\boffensive\b|\bpick and roll\b|\bisolation\b|\btransition\b|\bnet rating\b|\bstrengths?\b|"
     r"\bweakness\w*\b|\bgive(?:s)? up\b|\ballow(?:s|ed)?\b|\bgood at\b|\bbad at\b|\bprofile\b|\bscout\b|\bbreak ?down\b|"
     r"\bhow (?:good|bad) (?:is|are)\b|\bthrees?\b|\bshoot\w*\b|\brebound\w*\b|\bfast\b|\bslow\b|\bgood\b|\bbad\b"),
    ("rank",
     r"(?!x)x",
     r"\brank\w*\b|\bbest\b|\bworst\b|\bmost\b|\bleast\b|\bfewest\b|\bfastest\b|\bslowest\b|\bhighest\b|\blowest\b|\btop\b|\bbottom\b|"
     r"\bleads? the league\b|\bleague leaders?\b|\bwhich teams?\b|\bwhat teams?\b|\bwho (?:has|plays|gives|allows|shoots|forces)\b"),
)


def topics(text: str) -> dict[str, str]:
    """Topic -> "strong" or "weak" for every topic the text touches."""
    variants = (" ".join(_normalize(text).replace("$", "").split()), fold(text))
    found: dict[str, str] = {}
    for name, strong, weak in TOPICS:
        if any(re.search(strong, variant) for variant in variants):
            found[name] = "strong"
        elif any(re.search(weak, variant) for variant in variants):
            found[name] = "weak"
    return found


def off_topic(text: str) -> str | None:
    """"other_sport" or "unrelated" when the message clearly isn't about NBA betting or the app."""
    t = fold(text)
    if OTHER_SPORTS.search(t):
        return "other_sport"
    if not SPORTS_WORDS.search(t):
        return "unrelated"
    return None
