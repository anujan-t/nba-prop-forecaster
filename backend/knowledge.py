"""What COURTSIDE Coach knows about the app itself and about betting terms, with a small BM25 search over it.

Each entry is short, accurate to the app, and optionally links to the page that shows it (a hash route).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata


@dataclass(frozen=True)
class Entry:
    id: str
    title: str
    keywords: tuple[str, ...]
    text: str
    route: str | None = None
    route_label: str | None = None
    kind: str = "site"  # site | glossary


# Walk-forward test behind the default model (mirrors frontend/src/modelReport.ts).
WALK_FORWARD = {
    "players": 70, "games": 8417, "predictions": 232288, "period": "Nov 2024 – Apr 2026",
    "overall": {"old_log_loss": 0.5239, "simulator_log_loss": 0.5150, "old_brier": 0.1749, "simulator_brier": 0.1718},
    "joint_pairs": {"pairs": 28464, "simulator": 0.5104, "history_copula": 0.5141, "independent": 0.5323},
    "markets": {
        "points": (0.5551, 0.5452), "rebounds": (0.5652, 0.5593), "assists": (0.5352, 0.5219), "threes": (0.4885, 0.4773),
        "steals": (0.4295, 0.4286), "blocks": (0.3315, 0.3299), "turnovers": (0.4728, 0.4686), "pts_reb_ast": (0.5632, 0.5512),
        "pts_reb": (0.5633, 0.5539), "pts_ast": (0.5647, 0.5521), "reb_ast": (0.5700, 0.5639), "stocks": (0.4833, 0.4800),
        "field_goals_made": (0.5838, 0.5660), "field_goals_attempted": (0.5661, 0.5485), "threes_attempted": (0.5555, 0.5378),
        "free_throws_made": (0.5930, 0.5847), "free_throws_attempted": (0.5996, 0.5931), "offensive_rebounds": (0.4522, 0.4508),
        "defensive_rebounds": (0.5602, 0.5536), "fouls": (0.4754, 0.4738), "double_double": (0.3184, 0.2937),
        "triple_double": (0.0480, 0.0377),
    },
}

ENTRIES: tuple[Entry, ...] = (
    # ------------------------------------------------------------------ the app
    Entry("overview", "What COURTSIDE is",
          ("courtside", "app", "website", "site", "about", "purpose", "what is this", "what does this do", "features", "overview"),
          "COURTSIDE is your NBA player-prop research app. It forecasts the chance a player goes over or under a line in 22 "
          "markets, prices parlays with correlated legs, tracks your bets and grades them from box scores, and keeps a newsroom "
          "of headlines, injuries, roster moves, team styles and the day's games with betting lines. It runs on your computer; "
          "it is research, not betting advice.", "#/players", "Open Players"),
    Entry("navigation", "Where everything is",
          ("tab", "tabs", "page", "pages", "menu", "navigate", "navigation", "where", "find", "section", "sections", "layout", "header"),
          "- **Players**: search any current player, then Props (forecast a line), Game log, Career and Lines (your line notebook).\n"
          "- **Parlay**: build and price a slip.\n- **My Bets**: tracked bets, grading, record and ROI.\n"
          "- **News**: headlines, injury report, teams & coaches, roster moves, and games with betting lines.\n"
          "- **Coach**: this chat.\nThe header menu holds **How it works** (Method, Accuracy) and **Settings** (Appearance, Model, Data, Coach).",
          "#/players", "Open Players"),
    Entry("props", "Forecasting a prop",
          ("prop", "props", "forecast", "forecasting", "projection", "project", "line", "over", "under", "probability", "chance",
           "how do i forecast", "run a forecast", "player page", "edge", "expected value"),
          "Open **Players**, pick anyone on a current roster, then in **Props** choose a market and line. Auto (the game simulator) is "
          "the default. You can set the lookback window, the book's over and under prices, expected minutes, rest days and opponent. "
          "You get the over, under and push chances, the projected average, an 80% range, fair odds, and the edge and expected value "
          "against the prices you entered, with the recent games behind it. Or just ask me: **Jokić over 26.5 points?**",
          "#/players", "Open Players"),
    Entry("markets", "Markets covered",
          ("markets", "market", "stats", "stat", "supported", "combos", "combo", "pra", "pr", "pa", "ra", "stocks", "threes", "3pm",
           "double double", "triple double", "fantasy", "minutes", "which props", "what props", "blocks", "steals", "turnovers",
           "free throws", "field goals", "rebounds", "assists", "points"),
          "22 markets: points, rebounds, assists, threes made, steals, blocks and turnovers; combos **PRA** (points + rebounds + "
          "assists), **PR** (points + rebounds), **PA** (points + assists), **RA** (rebounds + assists) and **stocks** (steals + "
          "blocks); shooting: field goals made and attempted, threes attempted, free throws made and attempted; offensive and "
          "defensive rebounds; fouls; and the yes/no markets double-double and triple-double. Fantasy points and minutes aren't "
          "modelled.", "#/players", "Open Players"),
    Entry("simulator", "How the game simulator works",
          ("simulator", "simulation", "simulations", "monte carlo", "auto", "model", "how does it work", "how it works", "algorithm",
           "method", "methodology", "engine", "half life", "minutes", "usage", "rates", "math behind"),
          "Auto plays the player's next game **250,000 times** by default. Each simulated game draws minutes from his recent minutes, "
          "then a per-minute rate for every shot attempt, rebound, assist, steal, block, turnover and foul, including the uncertainty "
          "in estimating those rates. Counts are as spread out as real box scores, and makes come from attempts at a shooting "
          "percentage drawn from its posterior. Recent games count more: usage and minutes have a 10-game half-life, steals and "
          "blocks 30. Every market comes from the same simulated games, so combos and same-player parlays stay consistent.",
          "#/how-it-works/method", "Open How it works"),
    Entry("simulation_count", "How many simulations run",
          ("how many simulations", "simulations", "sims", "runs", "250000", "250k", "million", "1m", "more simulations",
           "simulation count", "precision", "noise", "monte carlo error"),
          "250,000 simulated games per forecast by default; choose 100K, 250K, 500K or 1M in **Settings › Model**. Results are "
          "seeded and cached, so the same inputs give the same numbers. At 250K the simulation noise is about ±0.1 percentage "
          "points (±0.05 at 1M). More simulations remove noise but not model error, which is why the walk-forward test matters "
          "more than the count.", "#/settings/model", "Open Settings › Model"),
    Entry("models", "Model choices",
          ("models", "model", "empirical", "normal", "poisson", "negative binomial", "distribution", "auto", "which model",
           "difference between models", "compare models"),
          "**Auto** is the game simulator (it beat the previous default on all 22 markets in the walk-forward test). The "
          "alternatives are **Empirical** (the observed distribution of recent games), **Normal** (a mean and spread), **Poisson** "
          "(a count model whose variance equals its mean) and **Negative binomial** (allows extra variance). Use them to compare; "
          "calibration only applies to Auto.", "#/how-it-works/method", "Open How it works"),
    Entry("accuracy", "How accurate the model is",
          ("accuracy", "accurate", "how good", "how reliable", "reliable", "trust", "track record of the model", "walk forward",
           "test", "tested", "validation", "log loss", "brier", "error", "performance", "results", "proof", "precise", "pinpoint"),
          "In a walk-forward test on 8,417 real games for 70 players (Nov 2024 – Apr 2026; 232,288 predictions, each made only from "
          "games before it), the simulator beat the previous model on **all 22 markets**: log loss 0.5239 → 0.5150 and Brier score "
          "0.1749 → 0.1718 (lower is better). For two legs on the same player, the joint simulation scored 0.5104 against 0.5323 "
          "when the legs were treated as independent. Every market's numbers and the calibration curves are in How it works › "
          "Accuracy.", "#/how-it-works/accuracy", "Open Accuracy"),
    Entry("calibration", "Calibration and learning",
          ("calibration", "calibrate", "calibrated", "learning", "learn", "smarter", "improve", "retrain", "training", "correction",
           "adjust", "feedback", "per market", "per stat"),
          "After games finish, COURTSIDE grades its past predictions and learns a correction **for each market**: its own "
          "correction, a shared one across markets, or none, whichever scored best on recent games it didn't train on. It applies "
          "to Auto forecasts only and retrains in the background (every 6 hours, or when new games or graded bets arrive). Your "
          "graded bets feed it too. Ask **are you getting smarter?** for the current status.",
          "#/how-it-works/accuracy", "Open Accuracy"),
    Entry("backtest", "Backtesting a player",
          ("backtest", "backtesting", "replay", "historical test", "test a player", "validate", "hit rate history"),
          "How it works › Accuracy has a backtest: pick a player and market and it replays past games, forecasting each one only "
          "from the games before it, then scores the predictions and draws a calibration chart. From Props, **Test the evidence** "
          "opens it for the player you're looking at.", "#/how-it-works/accuracy", "Open Accuracy"),
    Entry("context", "Minutes, rest and opponent",
          ("context", "expected minutes", "minutes", "rest", "rest days", "back to back", "b2b", "opponent", "matchup", "adjust",
           "scenario", "blowout", "role"),
          "In Props › context you can set **expected minutes** (scales the projection), **rest days** (0 means a back-to-back) and "
          "the **opponent**; rest and opponent apply bounded adjustments learned from history. Leave them blank and the model uses "
          "recent history. When I forecast a prop in chat I use the same defaults and show the player's next game and injury "
          "status alongside.",
          "#/players", "Open Players"),
    Entry("injuries_model", "How injuries are handled",
          ("injury", "injuries", "injured", "out", "teammate out", "without", "absence", "usage bump", "affect forecast",
           "adjust for injury", "questionable", "does the model know"),
          "Injuries are **not** applied automatically. If a player is on the injury report, his forecast and any parlay leg show a "
          "warning, and the forecast assumes he plays. When a teammate is out and you expect a bigger role, raise **expected "
          "minutes** in Props to model it. The full report is in News › Injury report, or ask me **who's out tonight?**",
          "#/news/injuries", "Open Injury report"),
    Entry("parlay", "The parlay builder",
          ("parlay", "parlays", "slip", "ticket", "legs", "leg", "build", "builder", "same game parlay", "sgp", "price my parlay",
           "add a leg", "combine"),
          "Add legs from Props or tell me (**add Jokić over 26.5 points**). The Parlay tab prices the chance that every leg hits: "
          "legs on the same player come from the same simulated games, so their link is exact, and teammates and opponents use "
          "correlations measured from games they shared. Enter the book's parlay odds to see the break-even chance, the model's "
          "chance, fair odds and expected value. Up to 10 legs.", "#/parlay", "Open Parlay"),
    Entry("correlation", "Why legs are linked",
          ("correlation", "correlated", "linked", "link", "independent", "together", "copula", "dependence", "why are my legs"),
          "Legs aren't independent. Two legs on the same player are priced from the same simulated games (points and assists rise "
          "together in a big-minutes night). Teammates and opponents use correlations measured from their shared games, which "
          "raise or lower the joint chance compared with simply multiplying. Ask **why are my legs linked?** to see each pair.",
          "#/parlay", "Open Parlay"),
    Entry("bets", "Tracking and grading bets",
          ("my bets", "track", "tracking", "tracked", "grade", "grading", "settle", "record", "roi", "profit", "history",
           "ledger", "bet log", "results", "won", "lost"),
          "Track a slip from the Parlay tab or say **track this for $10 at +450**. After the games, COURTSIDE grades every leg from "
          "the box scores and keeps your record, ROI and Brier score in **My Bets**; graded legs also feed calibration. If a leg "
          "pushes or is void while the rest win, the bet is marked reduced and its profit left blank, because books recalculate "
          "those payouts differently.", "#/bets", "Open My Bets"),
    Entry("news_tab", "The News tab",
          ("news", "news tab", "headlines", "newsroom", "feed", "injury report", "teams", "coaches", "roster moves", "games",
           "scoreboard", "lines", "where is the news"),
          "News has **Headlines** (filter by injuries, roster moves, coaching or betting), the league **Injury report**, **Teams "
          "& coaches** (playing style, ranks, play types, what each team allows, and coaching staffs), **Roster moves**, and "
          "**Games** (the day's schedule with spreads, totals and moneylines). Feeds refresh every few minutes and fall back to the "
          "last saved copy when a provider is down.", "#/news", "Open News"),
    Entry("team_styles", "Team styles explained",
          ("style", "styles", "playing style", "pace", "offensive rating", "defensive rating", "net rating", "play types",
           "synergy", "four factors", "what they allow", "tags", "team profile"),
          "Team profiles come from NBA Stats: pace (possessions per 48 minutes), offensive and defensive rating (points per 100 "
          "possessions), shot profile, four factors, what opponents score against them, and Synergy play types (how often they "
          "run pick-and-roll, isolation, spot-ups and so on, and how well). Ranks are 1 = best; tags like **Plays fast** mark top-5 "
          "or bottom-5 ranks.", "#/news/teams", "Open Teams & coaches"),
    Entry("data", "Where the data comes from",
          ("data", "source", "sources", "where does the data", "nba stats", "nba_api", "espn", "provider", "api", "updated",
           "refresh", "live", "live odds", "odds feed", "stale", "cache", "csv", "import"),
          "Player game logs, rosters, profiles, team stats and play types come from **NBA Stats** (through nba_api); headlines, "
          "injuries, roster moves and the scoreboard with game lines come from **ESPN's** public feeds. Both are unofficial, so the "
          "app caches results and says when a copy is stale. There is **no live player-prop odds feed**: enter the book's price "
          "yourself. Settings › Data manages saved histories and CSV imports.", "#/settings/data", "Open Settings › Data"),
    Entry("lines", "The line notebook",
          ("line notebook", "notebook", "snapshots", "snapshot", "line movement", "save a line", "record lines", "lines tab",
           "track lines", "steam"),
          "Players › Lines is a notebook for prices you see: save the event date, book, line and over/under odds, and it charts how "
          "the line moved. It is not an odds feed, and saving a snapshot doesn't change your forecast.", "#/players/lines",
          "Open Lines"),
    Entry("settings", "Settings",
          ("settings", "theme", "dark mode", "light mode", "night", "day", "appearance", "colors", "preferences", "configure",
           "options"),
          "Settings has **Appearance** (Day game or Night game theme), **Model** (simulations per forecast, 100K to 1M), **Data** "
          "(saved NBA histories and CSV imports) and **Coach** (language-model status and setup).", "#/settings/appearance",
          "Open Settings"),
    Entry("coach", "What I can do",
          ("coach", "chatbot", "bot", "assistant", "you", "what can you do", "capabilities", "help", "commands", "ask"),
          "I can build and edit your slip by chat, price it against the book, forecast any prop, show a player's averages, "
          "answer news, injury, schedule, betting-line, team, coaching and standings questions, explain how the app and its model "
          "work, convert odds, and track and grade your bets. I run on your computer; a language model can be connected for "
          "free-form conversation (Settings › Coach).", "#/coach", "Open Coach"),
    Entry("llm", "Connecting a language model",
          ("language model", "llm", "ai", "gpt", "openai", "chatgpt", "claude", "anthropic", "ollama", "openrouter", "api key",
           "connect", "smarter chat", "local model"),
          "I work without one. To add one, put **COURTSIDE_LLM_MODEL** (plus an API key and base URL if needed) in a .env file in "
          "the app folder and restart: OpenAI, Anthropic, OpenRouter, Azure OpenAI or a local Ollama or LM Studio server all work. "
          "Every number still comes from the app's own tools.", "#/settings/coach", "Open Settings › Coach"),
    Entry("privacy", "Privacy",
          ("privacy", "private", "data stored", "stored", "where is my data", "local", "secure", "security", "share"),
          "Your bets, slips, chat history and learned nicknames stay in the app's local runtime folder. Only feed requests leave "
          "your computer (NBA Stats and ESPN), plus the language model if you connect one."),
    Entry("limits", "What the model doesn't know",
          ("limits", "limitations", "weakness", "doesn't know", "blind spots", "lineups", "blowout", "motivation", "coaching decisions",
           "guarantee", "why was it wrong", "missed"),
          "It sees each player's own history and the context you give it. It doesn't see injuries or lineup changes unless you "
          "adjust minutes, and it has no read on coaching decisions, motivation or live odds. Blowouts and foul trouble are "
          "already part of the history it simulates. Treat a forecast as one input, not a certainty."),
    Entry("responsible", "Responsible gambling",
          ("responsible", "gambling problem", "addiction", "budget", "limit", "losing", "chasing", "chase", "help line", "helpline",
           "lost money", "stop betting", "sure thing", "lock", "guaranteed", "guarantee"),
          "No forecast makes a bet a sure thing. Set a budget, never chase losses, and size bets you can afford to lose. If betting "
          "stops feeling like research, the National Problem Gambling Helpline is **1-800-GAMBLER** (1-800-426-2537), free and "
          "confidential, 24/7."),
    # ------------------------------------------------------------------ betting terms
    Entry("american_odds", "American odds",
          ("american odds", "odds", "plus", "minus", "+150", "-110", "what does +", "what does -", "moneyline odds", "favorite",
           "underdog", "payout", "pays"),
          "**+150** means a $100 bet wins $150 profit; **−150** means you risk $150 to win $100. The break-even chance is "
          "100 ÷ (odds + 100) for plus odds and odds ÷ (odds + 100) for minus odds, so +150 needs 40% and −150 needs 60%. Ask me "
          "**what does +250 mean?** for any number.", kind="glossary"),
    Entry("implied_probability", "Implied probability and break-even",
          ("implied probability", "implied", "break even", "breakeven", "break-even", "probability of odds", "chance the book gives"),
          "Implied probability is the win rate a price needs to break even: +200 implies 33.3%, −110 implies 52.4%. If the model's "
          "chance is above the implied probability, the price beats the model's fair price; below it, the book has the edge.",
          kind="glossary"),
    Entry("decimal_odds", "Decimal and fractional odds",
          ("decimal odds", "decimal", "fractional", "european odds", "convert odds", "conversion", "9/2", "2.50"),
          "Decimal odds include the stake: 2.50 returns $2.50 per $1 (a $1.50 profit). +150 = 2.50, −150 ≈ 1.67, −110 ≈ 1.91. "
          "Fractional 3/2 = +150. I can convert any price: **convert −135 to decimal**.", kind="glossary"),
    Entry("vig", "Vig (juice)",
          ("vig", "vigorish", "juice", "hold", "margin", "overround", "house edge", "how do books make money", "sportsbook margin"),
          "The vig is the book's margin. At −110 on both sides each side implies 52.4%, adding to 104.8%; the extra 4.8% is the "
          "margin. That's why you must win 52.4% of −110 bets just to break even. Parlays multiply the margin of every leg.",
          kind="glossary"),
    Entry("expected_value", "Expected value (EV)",
          ("ev", "+ev", "expected value", "positive ev", "negative ev", "value bet", "value", "profitable", "worth it", "edge"),
          "Expected value is the average profit per bet if the probability is right: chance × profit − (1 − chance) × stake. "
          "Positive EV means the price pays more than the model's fair price. COURTSIDE shows EV per $1 against the price you "
          "enter; it's only as good as the model's probability.", kind="glossary"),
    Entry("fair_odds", "Fair odds",
          ("fair odds", "fair price", "no vig", "no-vig", "true odds", "devig", "fair line"),
          "Fair odds are the price with no margin for a given chance: 50% is +100, 60% is −150, 25% is +300. COURTSIDE shows the "
          "model's fair odds next to each forecast and parlay so you can compare them with the book's price.", kind="glossary"),
    Entry("push", "Pushes",
          ("push", "pushes", "tie", "refund", "whole number line", "half point", "hook"),
          "A push happens when the result lands exactly on a whole-number line (say 25 points on a 25 line): the stake is refunded. "
          "Half-point lines such as 25.5 can't push. In a parlay, a pushed leg usually drops out and the payout is recalculated.",
          kind="glossary"),
    Entry("parlay_term", "What a parlay is",
          ("what is a parlay", "parlay meaning", "multi", "accumulator", "acca", "same game parlay", "sgp", "all legs"),
          "A parlay combines legs that must all win. Payouts multiply, and so do the book's margins. Legs from the same game are "
          "often correlated, so the true chance can differ a lot from multiplying the legs; books price same-game parlays with "
          "their own adjustments.", kind="glossary"),
    Entry("player_prop", "Player props",
          ("player prop", "prop bet", "what is a prop", "over under prop", "props meaning"),
          "A player prop is a bet on one player's stat line, usually over or under a number such as 24.5 points. COURTSIDE "
          "estimates both sides' chances from the player's game history.", kind="glossary"),
    Entry("spread", "Point spread",
          ("spread", "point spread", "ats", "against the spread", "cover", "covering", "handicap", "line for the game"),
          "The favourite must win by more than the spread: −5.5 means by 6 or more. The underdog at +5.5 covers by losing by 5 or "
          "fewer or by winning. Ask me **what's the spread for the Lakers game?** for a current line.", kind="glossary"),
    Entry("moneyline", "Moneyline",
          ("moneyline", "money line", "ml", "straight up", "outright", "who wins"),
          "A moneyline bet is on who wins the game outright. The favourite has minus odds (−200: risk $200 to win $100), the "
          "underdog plus odds (+170: risk $100 to win $170).", kind="glossary"),
    Entry("total", "Game totals",
          ("total", "totals", "over under", "o/u", "game total", "points total"),
          "A game total is an over/under on both teams' combined points, e.g. 228.5. Ask **what's the total for Celtics–Knicks?** "
          "for a current number.", kind="glossary"),
    Entry("bankroll", "Bankroll and units",
          ("bankroll", "unit", "units", "stake size", "how much to bet", "kelly", "staking", "bet sizing", "money management"),
          "A unit is a fixed slice of your bankroll, often 1–2%, so a bad run can't wipe you out. Kelly sizing scales the stake by "
          "edge but swings hard and assumes the probability is exactly right; most bettors who use it bet a small fraction of it. "
          "Only bet money you can afford to lose.", kind="glossary"),
    Entry("clv", "Closing line value",
          ("closing line", "clv", "line movement", "steam", "sharp", "beat the close", "opening line"),
          "Closing line value means getting a better number than the final price before tip-off. Consistently beating the close is "
          "one of the best signs your bets have an edge. Players › Lines can record how a prop's line moves.", kind="glossary"),
    Entry("scoring_rules", "Log loss and Brier score",
          ("log loss", "brier", "brier score", "scoring rule", "calibration curve", "what is log loss", "cross entropy"),
          "Both score probability forecasts, and lower is better. The Brier score is the average squared gap between the forecast "
          "and what happened (0 or 1); log loss punishes confident misses much harder. A 50/50 guess scores 0.25 Brier and 0.693 "
          "log loss.", kind="glossary"),
    Entry("double_double", "Double-doubles and triple-doubles",
          ("double double", "triple double", "double-double", "triple-double", "dd", "td"),
          "A double-double is 10 or more in two of points, rebounds, assists, steals and blocks; a triple-double is 10 or more in "
          "three. COURTSIDE prices both as yes/no markets from the same simulated games.", kind="glossary"),
    Entry("stocks_term", "Combos and stocks",
          ("stocks", "pra", "pts reb ast", "points rebounds assists", "pr", "pa", "ra", "combo", "combos"),
          "**PRA** is points + rebounds + assists; **PR**, **PA** and **RA** are the two-stat combos (points + rebounds, points + "
          "assists, rebounds + assists), and **stocks** are steals + blocks. Because COURTSIDE simulates every stat from the same "
          "games, a combo's chance is consistent with its parts.", kind="glossary"),
    Entry("pace_term", "Pace and ratings",
          ("pace", "possessions", "tempo", "offensive rating", "defensive rating", "net rating", "efficiency", "per 100"),
          "Pace is possessions per 48 minutes: a faster game gives every player more chances. Offensive and defensive rating are "
          "points scored and allowed per 100 possessions; net rating is the difference. A fast opponent with a weak defense is a "
          "friendlier spot for overs.", kind="glossary"),
    Entry("usage", "Usage and minutes",
          ("usage", "usage rate", "role", "minutes", "playing time", "load management", "rotation"),
          "Usage is the share of team possessions a player finishes while he's on the floor. The simulator works per minute, so "
          "minutes drive volume: if a player's minutes jump because a teammate is out, raise expected minutes in Props.",
          kind="glossary"),
    Entry("back_to_back", "Back-to-backs and rest",
          ("back to back", "back-to-back", "b2b", "rest", "rest days", "fatigue", "schedule spot"),
          "A back-to-back is games on consecutive days (rest = 0). Players often play fewer minutes or sit. Set rest days in "
          "Props › context to account for it.", kind="glossary"),
)

STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "and", "or", "it", "its", "this", "that",
    "do", "does", "did", "i", "you", "me", "my", "your", "we", "us", "can", "could", "would", "should", "will", "how", "what",
    "whats", "why", "when", "where", "which", "who", "with", "about", "tell", "explain", "please", "there", "any", "some", "get",
    "have", "has", "had", "at", "by", "from", "as", "if", "so", "just", "like", "work", "works", "mean", "means", "use", "using",
}
SYNONYMS = {
    "sims": "simulation", "sim": "simulation", "simulate": "simulation", "simulated": "simulation", "simulations": "simulation",
    "simulator": "simulation", "montecarlo": "simulation",
    "accurate": "accuracy", "precise": "accuracy", "reliable": "accuracy", "pinpoint": "accuracy", "correct": "accuracy",
    "calibrate": "calibration", "calibrated": "calibration", "injured": "injury", "injuries": "injury", "hurt": "injury",
    "theme": "appearance", "dark": "appearance", "juice": "vig", "vigorish": "vig", "ev": "expected", "bets": "bet",
    "betting": "bet", "parlays": "parlay", "sgp": "parlay", "props": "prop", "forecasts": "forecast", "chatbot": "coach",
    "bot": "coach", "ai": "llm", "gpt": "llm", "chatgpt": "llm", "claude": "llm", "openai": "llm", "ollama": "llm",
    "breakeven": "break", "o/u": "total", "ml": "moneyline", "kelly": "bankroll", "units": "unit", "b2b": "back",
}


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in text if not unicodedata.combining(char)).replace("−", "-")


def stem(token: str) -> str:
    for suffix, replacement, minimum in (("ies", "y", 5), ("ing", "", 6), ("ed", "", 5), ("es", "", 5), ("s", "", 4)):
        if token.endswith(suffix) and len(token) >= minimum and not token.endswith("ss"):
            return token[: -len(suffix)] + replacement
    return token


def tokens(text: str, expand: bool = False) -> list[str]:
    words = re.findall(r"[a-z0-9+/]+", fold(text))
    out = []
    for word in words:
        if word in STOP:
            continue
        if expand and word in SYNONYMS:
            out.append(stem(SYNONYMS[word]))
        out.append(stem(word))
    return out


class Knowledge:
    """BM25 over title (×3), keywords (×3) and text; phrases among the keywords add a bonus when they appear verbatim."""

    def __init__(self, entries: tuple[Entry, ...] = ENTRIES, k1: float = 1.4, b: float = 0.6):
        self.entries = entries
        self.k1, self.b = k1, b
        self.docs: list[Counter] = []
        for entry in entries:
            bag = Counter(tokens(entry.text))
            for _ in range(3):
                bag.update(tokens(entry.title))
                for keyword in entry.keywords:
                    bag.update(tokens(keyword))
            self.docs.append(bag)
        self.avg = sum(sum(doc.values()) for doc in self.docs) / len(self.docs)
        frequency = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in frequency.items()}

    def search(self, query: str, limit: int = 3, kind: str | None = None) -> list[tuple[Entry, float]]:
        terms = tokens(query, expand=True)
        folded = f" {' '.join(re.findall(r'[a-z0-9+/-]+', fold(query)))} "
        scored = []
        for entry, doc in zip(self.entries, self.docs):
            if kind and entry.kind != kind:
                continue
            length = sum(doc.values())
            score = 0.0
            for term in set(terms):
                tf = doc.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
            for keyword in entry.keywords:
                if " " in keyword and f" {keyword} " in folded:
                    score += 2.5
            if score > 0:
                scored.append((entry, score))
        scored.sort(key=lambda item: -item[1])
        return scored[:limit]

    def best(self, query: str, threshold: float = 3.5, kind: str | None = None) -> Entry | None:
        found = self.search(query, 1, kind)
        return found[0][0] if found and found[0][1] >= threshold else None

    def get(self, entry_id: str) -> Entry:
        return next(entry for entry in self.entries if entry.id == entry_id)


KNOWLEDGE = Knowledge()
