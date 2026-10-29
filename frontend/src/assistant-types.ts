import type { CoachEntry, Game, NewsArticle, Transaction } from './news-types';
import type { FairOdds, ParlayLegRequest, ParlayResult, Side } from './parlay-types';
import type { InjuryEntry, Model, Stat } from './types';

/**
 * API contract for the COURTSIDE assistant, the bet tracker and the learning loop.
 * Probabilities are 0–1 fractions; odds are American unless named decimal.
 */

// ---------- Chat ----------

/** A slip leg as the client stores it: the request plus display labels. */
export interface ChatSlipLeg {
  request: ParlayLegRequest;
  label: { player: string; team: string };
}

export interface ChatSlip {
  legs: ChatSlipLeg[]; // 0–10, unique player + stat
  book_odds: number | null; // American odds entered for the whole parlay
}

/** What the user is looking at in the Forecast lab, so "add the over" or "this player" can resolve. */
export interface ChatFocus {
  dataset_id: string;
  player_id: string;
  player: string;
  team: string;
  stat: Stat;
  line: number;
  model: Model;
  lookback: number;
  expected_minutes: number | null;
  rest_days: number | null;
  opponent: string | null;
  use_context: boolean;
}

export interface ChatRequest {
  message: string; // 1–1000 characters
  conversation_id: string; // client-generated, 8–64 chars [A-Za-z0-9_-]
  slip: ChatSlip;
  focus: ChatFocus | null;
}

export type ChatActionType = 'add_leg' | 'replace_leg' | 'remove_leg' | 'update_leg' | 'set_book_odds' | 'clear_slip' | 'undo' | 'track_bet';

export interface ChatAction {
  type: ChatActionType;
  leg: number | null; // 1-based leg number affected, when relevant
  description: string; // e.g. "Added leg 3: Jamal Murray Over 20.5 points"
}

/** Price comparison between an offered price and the model. */
export interface HouseComparison {
  basis: 'entered' | 'standard'; // 'standard' = assumed −110 on every leg, priced as independent (no book odds entered)
  odds: number; // American odds compared
  decimal: number; // decimal odds (stake included)
  implied: number; // break-even probability of those odds
  model: number; // model probability of winning
  fair_odds: FairOdds | null; // the model's no-margin price
  edge: number; // model − implied; positive means the price is better than the model's fair price
  expected_roi: number; // expected profit per 1 unit staked if the model is right
  house_edge: number; // −expected_roi: the book's expected take per unit staked if the model is right
  vig: number | null; // two-sided overround when both sides of a single prop were priced, else null
  verdict: 'value' | 'fair' | 'house'; // |edge| < 0.01 → 'fair'
  note: string; // one plain-English sentence; never advice to bet
}

export interface ForecastCard {
  type: 'forecast';
  player: { id: string; name: string; team: string };
  stat: Stat;
  line: number;
  model: Model;
  resolved_model: Model;
  sample_size: number;
  mean: number;
  interval: { low: number; high: number; coverage: number };
  over: number;
  under: number;
  push: number;
  hit_rate: { over: number; under: number; games: number }; // games in the sample above / below the line
  fair: { over: FairOdds | null; under: FairOdds | null };
  calibrated: { over: number; under: number } | null; // learned calibration, when active
  house: HouseComparison | null; // when the message named a price for one side
  side: Side | null; // the side the user asked about, if any
}

export interface ParlayCard {
  type: 'parlay';
  result: ParlayResult;
  house: HouseComparison; // basis 'entered' when the slip has book odds, otherwise 'standard'
}

export interface PlayerCard {
  type: 'player';
  player: { id: string; name: string; team: string };
  games: number; // games in the window
  averages: Record<'points' | 'rebounds' | 'assists' | 'steals' | 'blocks' | 'minutes', number> & { threes?: number }; // over the last `games` games
  last_games: { date: string; opponent: string; minutes: number; points: number; rebounds: number; assists: number; steals: number; blocks: number }[]; // most recent first, up to 5
}

export interface TrackRecordCard { type: 'track_record'; record: TrackRecord }
export interface LearningCard { type: 'learning'; status: LearningStatus }
export interface BetCard { type: 'bet'; bet: TrackedBet }

// League and news cards. `route` is an in-app hash such as "#/news/teams/lal" or "#/how-it-works/accuracy".

/** A headline as Coach sends it: the news feed's article without images or athletes. */
export type ChatArticle = Pick<NewsArticle, 'id' | 'headline' | 'description' | 'published' | 'url' | 'kind' | 'teams' | 'byline'>;

export interface NewsCard {
  type: 'news';
  title: string;
  articles: ChatArticle[];
  fetched_at: string | null;
  stale: boolean; // a saved copy, shown because the feed failed
  route: string;
}

/** An injury-report entry as Coach sends it (no long comment or headshot). */
export type ChatInjury = Pick<InjuryEntry, 'player' | 'player_id' | 'position' | 'status' | 'status_key' | 'injury' | 'detail' | 'side' | 'return_date' | 'short_comment' | 'date'> & { team: string };

export interface InjuriesCard {
  type: 'injuries';
  title: string;
  entries: ChatInjury[]; // up to 30
  total: number; // everyone who matched, even past the entries sent
  fetched_at: string | null;
  stale: boolean;
  route: string;
}

export interface GamesCard {
  type: 'games';
  title: string;
  date: string; // YYYY-MM-DD (US Eastern)
  games: Game[];
  route: string;
}

export interface TeamCard {
  type: 'team';
  team: string; // abbreviation, e.g. "LAL"
  name: string;
  season: string | null;
  record: { wins: number; losses: number; seed: number | null; conference: string | null; games_back: number | null } | null;
  head_coach: string | null;
  summary: string | null; // one plain-English sentence on how they play
  tags: string[];
  metrics: { key: string; label: string; value: number; rank: number; percent: boolean }[]; // percent: value is a 0–1 fraction
  play_types: { label: string; frequency: number; points_per_possession: number; frequency_rank: number }[]; // frequency 0–1
  injuries: { player: string; status: string; status_key: InjuryEntry['status_key']; injury: string | null }[];
  next_game: Game | null;
  team_count: number | null; // ranks are out of this many teams
  route: string;
}

export interface MatchupCard { type: 'matchup'; a: TeamCard; b: TeamCard; game: Game | null }

export interface RankingCard {
  type: 'ranking';
  title: string;
  metric: string;
  label: string;
  percent: boolean;
  season: string;
  rows: { team: string; name: string; label: string; value: number; rank: number; percent: boolean }[];
  team_count: number;
  route: string;
}

export interface StandingsCard {
  type: 'standings';
  season: string;
  teams: { team: string; name: string; conference: 'East' | 'West' | null; wins: number; losses: number; pct: number; seed?: number; games_back?: number }[];
  route: string;
}

export interface CoachesCard { type: 'coaches'; team: string; name: string; season: string | null; coaches: CoachEntry[]; route: string }

export interface MovesCard {
  type: 'moves';
  title: string;
  transactions: Transaction[];
  route: string;
  fetched_at: string | null;
  stale: boolean;
}

/** A plain "Open …" button to a page in the app. */
export interface LinkCard { type: 'link'; label: string; route: string }

export type LeagueCard = NewsCard | InjuriesCard | GamesCard | TeamCard | MatchupCard | RankingCard | StandingsCard | CoachesCard | MovesCard | LinkCard;

export type ChatCard = ForecastCard | ParlayCard | PlayerCard | TrackRecordCard | LearningCard | BetCard | LeagueCard;

export interface ChatClarification {
  prompt: string; // e.g. "Which Williams?"
  options: { label: string; message: string }[]; // sending `message` resolves the question
}

export interface ChatResponse {
  turn_id: string; // for feedback
  reply: string; // plain text with **bold**, line breaks, "- " bullet lines and whole-line _asides_ only
  slip: ChatSlip | null; // the full updated slip when it changed; null when unchanged
  actions: ChatAction[];
  cards: ChatCard[];
  suggestions: string[]; // up to 4 short follow-up messages
  clarification: ChatClarification | null;
  learned: string[]; // e.g. 'Learned that "joker" means Nikola Jokić.'
  engine: { mode: 'built_in' | 'llm'; model: string | null };
}

export interface FeedbackRequest { turn_id: string; rating: 1 | -1; note: string | null }

export interface AssistantStatus {
  mode: 'built_in' | 'llm';
  model: string | null; // configured LLM model name
  endpoint: string | null; // host of the OpenAI-compatible endpoint, never a key
  ready: boolean; // false when an LLM is configured but unreachable (chat falls back to built-in)
  detail: string; // plain-English status, including how to connect an LLM
}

// ---------- Bet tracker ----------

export type LegStatus = 'pending' | 'won' | 'lost' | 'push' | 'void';
export type BetStatus = 'pending' | 'won' | 'lost' | 'void' | 'reduced';

export interface TrackedLeg {
  player_id: string;
  player: string;
  team: string;
  stat: Stat;
  line: number;
  side: Side;
  probability: number; // model probability when tracked
  status: LegStatus;
  actual: number | null; // the stat in the graded game
  settled_at: string | null;
}

export interface TrackedBet {
  id: string;
  created_at: string;
  event_date: string; // YYYY-MM-DD (US Eastern game date)
  legs: TrackedLeg[];
  book_odds: number | null;
  stake: number | null;
  probability: number; // model joint probability when tracked
  independent_probability: number;
  status: BetStatus; // 'reduced' = some legs void/push, the rest won: books recalculate the payout, so profit is unknown
  profit: number | null; // in stake units; null when unknown or no stake
  note: string | null;
}

export interface TrackBetRequest {
  slip: ChatSlip; // at least one leg
  stake: number | null; // > 0, ≤ 1,000,000
  event_date: string | null; // YYYY-MM-DD; defaults to today (US Eastern)
  note: string | null; // ≤ 200 characters
}

export interface TrackRecord {
  bets: number;
  pending: number;
  won: number;
  lost: number;
  void: number;
  reduced: number;
  staked: number; // total stake on bets with a known profit
  profit: number; // total profit on those bets
  roi: number | null; // profit / staked
  legs_graded: number; // won + lost legs
  legs_won: number;
  expected_legs_won: number; // sum of model probabilities for graded legs
  brier: number | null; // Brier score of graded legs (lower is better)
}

export interface SettleResult {
  settled_legs: number;
  updated_bets: number;
  warnings: string[];
  record: TrackRecord;
}

// ---------- Learning ----------

export interface CalibrationStatus {
  active: boolean; // true only when the learned correction beat the raw model on held-out recent games
  trained_at: string | null;
  samples: number; // graded predictions used
  ledger_samples: number; // of which came from your tracked legs
  players: number;
  intercept: number | null; // logistic recalibration: logit(q) = intercept + slope · logit(p)
  slope: number | null;
  holdout: null | { samples: number; raw_log_loss: number; calibrated_log_loss: number; raw_brier: number; calibrated_brier: number };
  curve: { predicted: number; observed: number; calibrated: number; count: number }[]; // up to 10 bins
  message: string; // plain-English summary of what was learned or why it is off
  stats?: StatCalibration[]; // one entry per market with graded predictions (absent before the first per-market training)
}

/** One market's calibration: its own correction, the shared one, or the raw model, whichever did best on recent games. */
export interface StatCalibration {
  stat: Stat;
  label: string;
  samples: number;
  ledger_samples: number;
  choice: 'own' | 'shared' | 'none';
  active: boolean;
  intercept: number | null;
  slope: number | null;
  holdout: null | {
    samples: number;
    raw_log_loss: number; raw_brier: number;
    shared_log_loss: number; shared_brier: number;
    own_log_loss: number; own_brier: number;
  };
  curve: { predicted: number; observed: number; calibrated: number; count: number }[];
  message: string;
}

export interface LearningStatus {
  calibration: CalibrationStatus;
  memory: {
    aliases: { phrase: string; player_id: string; player: string; source: 'taught' | 'clarified'; uses: number }[];
    turns: number;
    liked: number;
    disliked: number;
  };
  record: TrackRecord;
  training: boolean; // a retrain is running in the background
}
