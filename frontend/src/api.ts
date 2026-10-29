import type { Dataset, ForecastForm, ForecastRequest, Model, Player, Stat } from './types';

export const statNames: Record<Stat, string> = {
  points: 'Points', rebounds: 'Rebounds', assists: 'Assists', threes: '3-pointers made', steals: 'Steals', blocks: 'Blocks',
  turnovers: 'Turnovers', pts_reb_ast: 'Pts + Reb + Ast', pts_reb: 'Pts + Reb', pts_ast: 'Pts + Ast', reb_ast: 'Reb + Ast',
  stocks: 'Steals + Blocks', field_goals_made: 'Field goals made', field_goals_attempted: 'Field goal attempts',
  threes_attempted: '3-point attempts', free_throws_made: 'Free throws made', free_throws_attempted: 'Free throw attempts',
  offensive_rebounds: 'Offensive rebounds', defensive_rebounds: 'Defensive rebounds', fouls: 'Personal fouls',
  double_double: 'Double-double', triple_double: 'Triple-double',
};
/** Sportsbook shorthand for each market. */
export const statShort: Record<Stat, string> = {
  points: 'PTS', rebounds: 'REB', assists: 'AST', threes: '3PM', steals: 'STL', blocks: 'BLK', turnovers: 'TOV',
  pts_reb_ast: 'PRA', pts_reb: 'PR', pts_ast: 'PA', reb_ast: 'RA', stocks: 'STL+BLK', field_goals_made: 'FGM',
  field_goals_attempted: 'FGA', threes_attempted: '3PA', free_throws_made: 'FTM', free_throws_attempted: 'FTA',
  offensive_rebounds: 'OREB', defensive_rebounds: 'DREB', fouls: 'PF', double_double: 'DD', triple_double: 'TD',
};
export const statUnits: Record<Stat, string> = {
  points: 'pts', rebounds: 'reb', assists: 'ast', threes: '3pm', steals: 'stl', blocks: 'blk', turnovers: 'tov',
  pts_reb_ast: 'pra', pts_reb: 'pr', pts_ast: 'pa', reb_ast: 'ra', stocks: 'stk', field_goals_made: 'fgm',
  field_goals_attempted: 'fga', threes_attempted: '3pa', free_throws_made: 'ftm', free_throws_attempted: 'fta',
  offensive_rebounds: 'oreb', defensive_rebounds: 'dreb', fouls: 'pf', double_double: '', triple_double: '',
};
export const statGroups: { label: string; stats: Stat[] }[] = [
  { label: 'Core', stats: ['points', 'rebounds', 'assists', 'threes', 'steals', 'blocks', 'turnovers'] },
  { label: 'Combos', stats: ['pts_reb_ast', 'pts_reb', 'pts_ast', 'reb_ast', 'stocks'] },
  { label: 'Shooting', stats: ['field_goals_made', 'field_goals_attempted', 'threes_attempted', 'free_throws_made', 'free_throws_attempted'] },
  { label: 'Rebounding & fouls', stats: ['offensive_rebounds', 'defensive_rebounds', 'fouls'] },
  { label: 'Milestones', stats: ['double_double', 'triple_double'] },
];
/** Box-score columns behind each market, used to suggest a starting line from a player's averages. */
const statParts: Partial<Record<Stat, string[]>> = {
  points: ['points'], rebounds: ['rebounds'], assists: ['assists'], steals: ['steals'], blocks: ['blocks'],
  threes: ['three_pointers_made'], turnovers: ['turnovers'],
  pts_reb_ast: ['points', 'rebounds', 'assists'], pts_reb: ['points', 'rebounds'], pts_ast: ['points', 'assists'],
  reb_ast: ['rebounds', 'assists'], stocks: ['steals', 'blocks'], field_goals_made: ['field_goals_made'],
  field_goals_attempted: ['field_goals_attempted'], threes_attempted: ['three_pointers_attempted'],
  free_throws_made: ['free_throws_made'], free_throws_attempted: ['free_throws_attempted'],
  offensive_rebounds: ['offensive_rebounds'], defensive_rebounds: ['defensive_rebounds'], fouls: ['fouls'],
};
export const isMilestone = (stat: Stat) => stat === 'double_double' || stat === 'triple_double';
/** A starting half-point line near the player's average, or null when the average is not known. */
export function suggestedLine(stat: Stat, player: Player | undefined): string | null {
  if (isMilestone(stat)) return '0.5';
  const parts = statParts[stat];
  if (!player || !parts || parts.some(part => typeof player.averages[part] !== 'number')) return null;
  const total = parts.reduce((sum, part) => sum + (player.averages[part] ?? 0), 0);
  return Number.isFinite(total) ? String(Math.max(0.5, Math.floor(total) + 0.5)) : null;
}
export const modelNames: Record<Model, string> = {
  auto: 'Auto select', simulation: 'Game simulator', empirical: 'Empirical', normal: 'Normal', poisson: 'Poisson', negative_binomial: 'Negative binomial',
};
export const DEFAULT_LOOKBACK = 60;
export const defaultForm: ForecastForm = {
  player_id: '203999', stat: 'points', line: '26.5', model: 'auto', lookback: String(DEFAULT_LOOKBACK),
  odds_over: '-110', odds_under: '-110', expected_minutes: '', rest_days: '', opponent: '', use_context: true,
};
export function preferredDataset(datasets: Dataset[]): string {
  return datasets.find(dataset => dataset.source === 'nba_api' && dataset.nba_query)?.id
    ?? datasets.find(dataset => dataset.source === 'nba_api')?.id
    ?? '';
}
export function latestCompletedSeason(): string {
  const now = new Date();
  const start = now.getFullYear() - (now.getMonth() >= 6 ? 1 : 2);
  return `${start}-${String((start + 1) % 100).padStart(2, '0')}`;
}
export function historyIsStale(dataset: Dataset): boolean {
  return !dataset.fetched_at || Date.now() - Date.parse(dataset.fetched_at) >= 6 * 60 * 60 * 1000;
}
export function historySyncLabel(dataset: Dataset): string {
  if (dataset.source === 'synthetic') return 'Development fixture, not NBA results';
  if (!dataset.fetched_at) return 'Previously saved NBA logs; sync time unavailable';
  return `Fetched ${new Date(dataset.fetched_at).toLocaleString()}${historyIsStale(dataset) ? ' · refresh available' : ''}`;
}
function detailMessage(detail: unknown): string {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((item: unknown) => {
      if (typeof item === 'object' && item !== null && 'msg' in item) {
        const location = 'loc' in item && Array.isArray(item.loc) ? item.loc.filter((part: unknown) => part !== 'body').join(' → ') : '';
        return `${location ? `${location}: ` : ''}${String(item.msg)}`;
      }
      return typeof item === 'string' ? item : 'The server could not validate this value.';
    }).join('\n');
  }
  return 'The request could not be completed. Please try again.';
}
export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...options,
      headers: options.body instanceof FormData ? options.headers : { 'Content-Type': 'application/json', ...options.headers },
    });
  } catch (error) {
    if (isAbort(error)) throw error;
    throw new Error('Unable to reach the research engine. Check that the Python server is running, then try again.');
  }
  if (!response.ok) {
    let body: unknown;
    try { body = await response.json(); } catch { throw new Error(`The server returned HTTP ${response.status}. Please try again.`); }
    throw new Error(typeof body === 'object' && body !== null && 'detail' in body
      ? detailMessage(body.detail) : `The server returned HTTP ${response.status}. Please try again.`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
export function isAbort(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}
export function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : 'Something went wrong. Please try again.';
}
export function requestFromForm(datasetId: string, form: ForecastForm, simulations: number): ForecastRequest {
  const optional = (value: string): number | null => value.trim() === '' ? null : Number(value);
  return {
    dataset_id: datasetId, player_id: form.player_id, stat: form.stat, line: isMilestone(form.stat) ? 0.5 : Number(form.line),
    model: form.model, lookback: Number(form.lookback), odds_over: Number(form.odds_over),
    odds_under: Number(form.odds_under), expected_minutes: optional(form.expected_minutes),
    rest_days: optional(form.rest_days), opponent: form.opponent.trim() || null, use_context: form.use_context,
    simulations,
  };
}
export function formFromRequest(request: ForecastRequest): ForecastForm {
  return {
    player_id: request.player_id, stat: request.stat, line: String(request.line), model: request.model,
    lookback: String(request.lookback), odds_over: String(request.odds_over), odds_under: String(request.odds_under),
    expected_minutes: request.expected_minutes === null ? '' : String(request.expected_minutes),
    rest_days: request.rest_days === null ? '' : String(request.rest_days),
    opponent: request.opponent ?? '', use_context: request.use_context,
  };
}
export function validateOdds(over: number, under: number): string | null {
  return [over, under].every(value => Number.isInteger(value) && Math.abs(value) >= 100 && Math.abs(value) <= 10000)
    ? null : 'American odds must be whole numbers from −10000 to −100 or +100 to +10000.';
}
export function pct(value: number | null, digits = 1): string {
  if (value === null) return 'N/A';
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—';
}
export function edgePoints(value: number | null, includeUnit = true): string {
  return value === null ? 'N/A' : `${signed(value * 100)}${includeUnit ? 'pp' : ''}`;
}
export function decimal(value: number, digits = 1): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '—';
}
export function signed(value: number, digits = 1): string {
  return Number.isFinite(value) ? `${value > 0 ? '+' : ''}${value.toFixed(digits)}` : '—';
}
export function odds(value: number): string { return value > 0 ? `+${value}` : String(value); }
export function venueLabel(home: boolean | null): string {
  return home === true ? 'Home' : home === false ? 'Away' : 'Venue unknown';
}
export function dateLabel(value: string, full = false): string {
  const date = new Date(value.length === 10 ? `${value}T12:00:00` : value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', ...(full ? { year: 'numeric' } : {}) });
}
export function sourceLabel(source: string): string {
  return source === 'synthetic' ? 'Synthetic data' : source === 'nba_api' ? 'NBA historical logs' : 'User-supplied CSV';
}
export function tomorrow(): string {
  const date = new Date();
  date.setDate(date.getDate() + 1);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}
/** Today's date in US Eastern time, the NBA's game-date convention, as YYYY-MM-DD. */
export function todayEastern(now = new Date()): string {
  const parts = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).formatToParts(now);
  const part = (type: Intl.DateTimeFormatPartTypes) => parts.find(item => item.type === type)?.value ?? '';
  return `${part('year')}-${part('month')}-${part('day')}`;
}
export function downloadJson(value: unknown, filename: string): void {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
/** "2 hr ago", "yesterday", "Oct 26" — for news and report timestamps. */
export function timeAgo(value: string | null | undefined, now = Date.now()): string {
  if (!value) return '';
  const time = Date.parse(value);
  if (Number.isNaN(time)) return '';
  const minutes = Math.round((now - time) / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  const days = Math.round(hours / 24);
  if (days === 1) return 'yesterday';
  if (days < 7) return `${days} days ago`;
  return new Date(time).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}
/** Whole simulations as "250,000" or "1M". */
export function simulationsLabel(value: number, compact = false): string {
  if (compact) return value >= 1_000_000 ? `${value / 1_000_000}M` : `${Math.round(value / 1000)}K`;
  return value.toLocaleString('en-US');
}
