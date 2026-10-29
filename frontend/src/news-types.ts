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
  coaches: CoachEntry[];
  coaches_season: string | null;
  injuries: InjuryEntry[];
  headlines: NewsArticle[];
  transactions: Transaction[];
  errors: string[];
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
