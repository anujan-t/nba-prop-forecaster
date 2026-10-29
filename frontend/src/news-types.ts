import type { InjuryEntry } from './types';

/** API contract for /api/news/*. Feeds are cached by the server; `stale` marks a saved copy shown because the provider failed. */

export type NewsKind = 'injury' | 'transaction' | 'coaching' | 'game' | 'betting' | 'other';

export interface Feed {
  fetched_at: string;
  stale: boolean;
  error: string | null;
  provider: string;
}

export interface NewsArticle {
  id: string;
  type: string;
  kind: NewsKind;
  headline: string;
  description: string | null;
  published: string | null;
  byline: string | null;
  url: string | null;
  image: string | null;
  teams: string[];
  athletes: { name: string; player_id: string | null }[];
  premium: boolean;
  video: boolean;
}

export interface Headlines extends Feed { articles: NewsArticle[]; counts: Record<NewsKind, number> }

export interface InjuryTeam { team: string; team_name: string; injuries: InjuryEntry[] }
export interface InjuryReport extends Feed {
  teams: InjuryTeam[];
  counts: Partial<Record<InjuryEntry['status_key'], number>>;
  total: number;
}

export interface Transaction { date: string | null; description: string; team: string | null; team_name: string | null }
export interface Transactions extends Feed { transactions: Transaction[] }

export interface TeamMetric { label: string; value: number; rank: number; percent: boolean }
export interface PlayType {
  play_type: string;
  label: string;
  frequency: number; // share of possessions
  frequency_rank: number; // 1 = uses it most
  points_per_possession: number;
  percentile: number | null; // efficiency percentile, 0–1
}
export interface TeamProfile {
  team_id: string;
  team: string;
  name: string;
  games: number;
  wins: number;
  losses: number;
  metrics: Record<string, TeamMetric>;
  play_types: PlayType[];
  tags: string[];
  summary?: string;
  injuries_listed?: number;
  injuries_out?: number;
  head_coach?: string | null;
}
export interface TeamStyles extends Feed {
  season: string;
  teams: TeamProfile[];
  team_count: number;
  which: 'current' | 'previous';
  median_games: number;
}

export interface CoachEntry { name: string; role: string; head: boolean }
export type SeasonProfile = TeamProfile & { season: string; team_count: number; fetched_at?: string; stale?: boolean };
export interface TeamDetail {
  team: { id: string; abbreviation: string; name: string; city: string; nickname: string };
  profiles: Partial<Record<'current' | 'previous', SeasonProfile>>;
  conference?: 'East' | 'West' | null;
  coaches: CoachEntry[];
  coaches_season: string | null;
  injuries: InjuryEntry[];
  headlines: NewsArticle[];
  transactions: Transaction[];
  next_game?: Game | null; // the next game that isn't final, within the lookahead window
  errors: string[];
}

// ---------- Games (ESPN scoreboard) ----------

export interface GameSide {
  team: string; // abbreviation, e.g. "LAL"
  name: string;
  record: string | null; // e.g. "12-3"
  score: number | null;
  winner: boolean | null;
}

/** The sportsbook line ESPN lists for a scheduled game, shown for reference only. American odds. */
export interface GameOdds {
  provider: string | null; // e.g. "DraftKings"
  details: string | null; // e.g. "LAL -1.5"
  home_spread: number | null;
  away_spread: number | null;
  total: number | null;
  favorite: string | null;
  home_moneyline: number | null;
  away_moneyline: number | null;
  home_spread_odds: number | null;
  away_spread_odds: number | null;
  over_odds: number | null;
  under_odds: number | null;
}

export interface Game {
  id: string;
  start: string | null; // ISO UTC
  tipoff: string | null; // e.g. "8:00 PM ET"
  state: 'pre' | 'in' | 'post';
  detail: string | null; // "10/29 - 8:00 PM EDT", "Q3 5:12", "Final", "Final/OT"
  home: GameSide;
  away: GameSide;
  odds: GameOdds | null; // only before tipoff
  broadcast: string | null; // e.g. "NBA TV, ESPN"
  note: string | null; // e.g. "NBA Cup - Group Play"
  date?: string | null; // YYYY-MM-DD (US Eastern), on games looked up for a team
}

export interface GamesFeed extends Feed {
  date: string; // YYYY-MM-DD shown
  today: string; // today in US Eastern time
  games: Game[];
  /** Per team playing that day: players listed out, suspended, doubtful, questionable or day-to-day. */
  injuries: Record<string, InjuryEntry[]>;
}

export interface PlayerNews {
  player_id: string;
  team: string | null;
  injury: InjuryEntry | null;
  teammates: InjuryEntry[];
  headlines: NewsArticle[];
  report_fetched_at: string | null;
  stale: boolean;
  error: string | null;
}
