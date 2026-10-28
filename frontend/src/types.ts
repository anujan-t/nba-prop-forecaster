export type Stat = 'points' | 'rebounds' | 'assists' | 'steals' | 'blocks';
export type Model = 'auto' | 'empirical' | 'normal' | 'poisson' | 'negative_binomial';
export type View = 'forecast' | 'player' | 'validation' | 'notebook' | 'data' | 'methodology';
export interface NBAHistoryQuery {
  player_ids: string[];
  season: string;
  season_type: 'Regular Season' | 'Playoffs';
  include_previous?: boolean;
}

export interface Dataset {
  id: string;
  name: string;
  source: 'synthetic' | 'csv' | 'nba_api';
  rows: number;
  players: number;
  start_date: string;
  end_date: string;
  nba_query?: NBAHistoryQuery;
  nba_players?: NbaPlayer[];
  fetched_at?: string;
  provider?: string;
  cache_hit?: boolean;
}
export interface Player {
  id: string;
  name: string;
  team: string;
  games: number;
  averages: Record<Stat | 'minutes', number>;
}
export interface ForecastRequest {
  dataset_id: string;
  player_id: string;
  stat: Stat;
  line: number;
  model: Model;
  lookback: number;
  odds_over: number;
  odds_under: number;
  expected_minutes: number | null;
  rest_days: number | null;
  opponent: string | null;
  use_context: boolean;
}
export interface ForecastForm {
  player_id: string;
  stat: Stat;
  line: string;
  model: Model;
  lookback: string;
  odds_over: string;
  odds_under: string;
  expected_minutes: string;
  rest_days: string;
  opponent: string;
  use_context: boolean;
}
export interface Forecast {
  dataset: Dataset;
  player: Pick<Player, 'id' | 'name' | 'team'>;
  stat: Stat;
  line: number;
  model: Model;
  resolved_model: Model;
  sample_size: number;
  summary: {
    mean: number;
    median: number;
    std: number;
    observed_mean: number;
    projected_minutes: number;
    context_factor: number;
    rest_factor: number;
    opponent_factor: number;
  };
  probabilities: { over: number; under: number; push: number };
  market: {
    over_implied: number;
    under_implied: number;
    no_vig_over: number;
    no_vig_under: number;
    overround: number;
    model_over_conditional: number | null;
    edge_over: number | null;
    edge_under: number | null;
    expected_roi_over: number;
    expected_roi_under: number;
  };
  distribution: { value: number; probability: number }[];
  interval: { low: number; high: number; coverage: number };
  history: { date: string; value: number; minutes: number; opponent: string; rest_days: number; home: boolean | null }[];
  comparisons: { model: Model; over: number; under: number; push: number; mean: number }[];
  warnings: string[];
  context_notes: string[];
}
export interface AppliedForecast { data: Forecast; request: ForecastRequest }
export interface BacktestRequest {
  dataset_id: string;
  player_id: string | null;
  stat: Stat;
  model: Model;
  lookback: number;
  line: number | null;
  use_context: boolean;
}
export interface Backtest {
  evaluations: number;
  pushes_excluded: number;
  brier_score: number;
  log_loss: number;
  baseline_brier: number;
  skill_score: number;
  calibration: { lower: number; upper: number; predicted: number; observed: number; count: number }[];
  model_scores: { model: Model; brier_score: number; log_loss: number; evaluations: number }[];
  predictions: { date: string; player_id: string; line: number; actual: number; probability: number; outcome: number }[];
  description: string;
}
export interface AppliedBacktest { data: Backtest; request: BacktestRequest; playerName: string; dataset: Dataset }
export interface LineSnapshot {
  id: string;
  timestamp: string;
  dataset_id: string;
  player_id: string;
  stat: Stat;
  event_date: string;
  book: string;
  line: number;
  odds_over: number;
  odds_under: number;
}
export interface NbaPlayer { id: string; name: string }
