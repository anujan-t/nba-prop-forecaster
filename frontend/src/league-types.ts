import type { Dataset, Stat } from './types';

export interface RosterPlayer {
  id: string;
  name: string;
  team: string;
  team_id: string;
  team_name: string;
  from_year: number | null;
  to_year: number | null;
  has_played: boolean;
}
export interface LeagueRoster {
  season: string;
  players: RosterPlayer[];
  teams: { id: string; abbreviation: string; name: string; count: number }[];
  total_players: number;
  total_teams: number;
  fetched_at: string;
  provider: string;
}
export interface PlayerProfile {
  id: string;
  name: string;
  position: string | null;
  jersey: string | null;
  height: string | null;
  weight: string | null;
  birth_date: string | null;
  age: number | null;
  experience: number | null;
  country: string | null;
  school: string | null;
  draft_year: string | null;
  draft_round: string | null;
  draft_pick: string | null;
  team: string | null;
  team_name: string | null;
  roster_status: string | null;
  available_seasons: string[];
  fetched_at: string;
  provider: string;
}
export type OverviewStat = Stat | 'minutes' | 'turnovers' | 'three_pointers_made';
export interface ShootingSummary {
  field_goal: number | null;
  three_point: number | null;
  free_throw: number | null;
}
export interface PlayerSummary {
  games: number;
  averages: Record<OverviewStat, number | null>;
  shooting: ShootingSummary;
}
export interface PlayerSplit extends PlayerSummary { label: string }
export interface PlayerGame {
  date: string;
  player_id: string;
  player_name: string;
  team: string;
  opponent: string;
  minutes: number;
  points: number;
  rebounds: number;
  assists: number;
  steals: number;
  blocks: number;
  rest_days: number;
  home: boolean | null;
  season?: string;
  game_id?: string;
  win?: string | null;
  field_goals_made?: number | null;
  field_goals_attempted?: number | null;
  three_pointers_made?: number | null;
  three_pointers_attempted?: number | null;
  free_throws_made?: number | null;
  free_throws_attempted?: number | null;
  offensive_rebounds?: number | null;
  defensive_rebounds?: number | null;
  turnovers?: number | null;
  fouls?: number | null;
  plus_minus?: number | null;
}
export interface PlayerHistory {
  player_id: string;
  status: 'ready' | 'limited' | 'empty';
  history: Dataset | null;
  summary: PlayerSummary | null;
  splits: PlayerSplit[];
  games: PlayerGame[];
  queried_seasons: string[];
  warnings: string[];
  fetched_at: string;
  cache_hit: boolean;
}
export interface CareerLine extends PlayerSummary {
  season: string;
  team: string;
  starts: number | null;
}
export interface PlayerCareer {
  player_id: string;
  regular_season: CareerLine[];
  playoffs: CareerLine[];
  totals: { regular_season: CareerLine | null; playoffs: CareerLine | null };
  fetched_at: string;
  provider: string;
}
export interface LeagueSelectorProps {
  roster: LeagueRoster | null;
  loading: boolean;
  error: string;
  selectedId: string;
  onSelect: (id: string) => void;
  onRefresh: () => void;
  historyWindow: string;
  onHistoryWindowChange: (window: string) => void;
  seasonType: 'Regular Season' | 'Playoffs';
  onSeasonTypeChange: (type: 'Regular Season' | 'Playoffs') => void;
  busy: boolean;
  onReload: () => void;
  onDetails: () => void;
}
export interface PlayerIntelligenceProps {
  player: RosterPlayer | null;
  profile: PlayerProfile | null;
  profileLoading: boolean;
  profileError: string;
  career: PlayerCareer | null;
  careerLoading: boolean;
  careerError: string;
  history: PlayerHistory | null;
  historyLoading: boolean;
  historyError: string;
}
