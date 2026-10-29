import { isMilestone, modelNames, odds, statNames } from './api';
import type { ChatSlip } from './assistant-types';
import type { ParlayLegRequest, ParlayPair, ParlayResult, Side } from './parlay-types';
import type { AppliedForecast, Model, Stat } from './types';

export const MAX_LEGS = 10;
export const SLIP_STORAGE_KEY = 'courtside.parlay-slip.v1';

export interface LegLabel { player: string; team: string }
export interface SlipLeg { request: ParlayLegRequest; label: LegLabel }
export interface SlipState { legs: SlipLeg[]; bookOdds: string }
export interface ParlayError { general: string[]; legs: Record<number, string[]> }

export const sideNames: Record<Side, string> = { over: 'Over', under: 'Under' };

export function legKey(leg: Pick<ParlayLegRequest, 'player_id' | 'stat'>): string {
  return `${leg.player_id}:${leg.stat}`;
}

export function legNumber(index: number): string {
  return String(index + 1).padStart(2, '0');
}

/** A leg is the applied Forecast lab request without its odds, plus the side being backed. */
export function legFromForecast(applied: AppliedForecast, side: Side): ParlayLegRequest {
  const request = applied.request;
  return {
    dataset_id: request.dataset_id, player_id: request.player_id, stat: request.stat, line: request.line, side,
    model: request.model, lookback: request.lookback, expected_minutes: request.expected_minutes,
    rest_days: request.rest_days, opponent: request.opponent, use_context: request.use_context,
  };
}

export function sameLeg(a: ParlayLegRequest, b: ParlayLegRequest): boolean {
  return a.dataset_id === b.dataset_id && a.player_id === b.player_id && a.stat === b.stat && a.line === b.line
    && a.side === b.side && a.model === b.model && a.lookback === b.lookback && a.expected_minutes === b.expected_minutes
    && a.rest_days === b.rest_days && a.opponent === b.opponent && a.use_context === b.use_context;
}

/** Whole or half point, clamped to 0–200; null when the input is not a number. */
export function normalizeLine(value: number): number | null {
  if (!Number.isFinite(value)) return null;
  return Math.min(200, Math.max(0, Math.round(value * 2) / 2));
}

export function statLabel(stat: Stat): string {
  const name = statNames[stat];
  return name.includes('+') ? name : name.toLocaleLowerCase();
}

export function legDescription(leg: SlipLeg): string {
  if (isMilestone(leg.request.stat)) return `${leg.label.player} ${leg.request.side === 'over' ? 'to get a' : 'no'} ${statLabel(leg.request.stat)}`;
  return `${leg.label.player} ${sideNames[leg.request.side]} ${leg.request.line} ${statLabel(leg.request.stat)}`;
}

export function parseBookOdds(text: string): { value: number | null; error: string } {
  const cleaned = text.trim().replace(/[−–]/g, '-').replace(/^\+/, '').replace(/,/g, '');
  if (!cleaned) return { value: null, error: '' };
  if (!/^-?\d+$/.test(cleaned)) return { value: null, error: 'Enter whole-number American odds, such as +450 or -120.' };
  const value = Number(cleaned);
  if (Math.abs(value) < 100 || Math.abs(value) > 10000) return { value: null, error: 'American odds must be from −10000 to −100 or +100 to +10000.' };
  return { value, error: '' };
}

/** Odds as typed in the book-odds field ("+450", "-120"), or "" when there is no price. */
export function bookOddsText(value: number | null): string {
  return value === null ? '' : odds(value);
}

/** Whole-number American odds the parlay endpoint accepts, or null. */
export function validBookOdds(value: unknown): number | null {
  return typeof value === 'number' && Number.isInteger(value) && Math.abs(value) >= 100 && Math.abs(value) <= 10000 ? value : null;
}

/** The slip as the assistant and the bet tracker receive it. */
export function chatSlip(legs: SlipLeg[], bookOdds: string): ChatSlip {
  return {
    legs: legs.map(leg => ({ request: leg.request, label: leg.label })),
    book_odds: parseBookOdds(bookOdds).value,
  };
}

export function americanLabel(value: number): string {
  const size = Math.abs(value);
  const digits = size >= 10000 ? size.toLocaleString('en-US') : String(size);
  return value > 0 ? `+${digits}` : `−${digits}`;
}

export function decimalOdds(value: number): string {
  if (!Number.isFinite(value)) return '—';
  return value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/** Parlay chances can be tiny; keep two significant figures below 1% instead of rounding them to 0.0%. */
export function chance(value: number | null): string {
  if (value === null) return 'N/A';
  if (!Number.isFinite(value)) return '—';
  const percent = value * 100;
  if (percent <= 0) return '0.0%';
  if (percent < 0.01) return '<0.01%';
  if (percent < 0.995) return `${Number(percent.toPrecision(2))}%`;
  return `${percent.toFixed(1)}%`;
}

export function chanceDigits(value: number): string {
  return chance(value).replace('%', '');
}

/** Splits an API error into slip-wide lines and lines that name a specific leg. */
export function parseParlayError(message: string, legCount: number): ParlayError {
  const parsed: ParlayError = { general: [], legs: {} };
  for (const raw of message.split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    let index: number | null = null;
    let text = line;
    const named = /^Leg (\d+)\b/.exec(line);
    const located = /^legs → (\d+)(?:(?: → |: )(.*))?$/.exec(line);
    if (named) index = Number(named[1]) - 1;
    else if (located) {
      index = Number(located[1]);
      text = (located[2] ?? line).replace(/Value error, /g, '');
    }
    if (index !== null && index >= 0 && index < legCount) (parsed.legs[index] ??= []).push(text);
    else parsed.general.push(line);
  }
  return parsed;
}

/** Correlation as it affects the parlay: positive means the two legs tend to win together. */
export function pairEffect(pair: ParlayPair, legs: ParlayLegRequest[]): number {
  const a = legs[pair.a];
  const b = legs[pair.b];
  if (!a || !b) return pair.correlation;
  return (a.side === b.side ? 1 : -1) * pair.correlation;
}

export function effectStrength(effect: number): 'none' | 'weak' | 'moderate' | 'strong' {
  const size = Math.abs(effect);
  return size < 0.1 ? 'none' : size < 0.3 ? 'weak' : size < 0.5 ? 'moderate' : 'strong';
}

export const basisNames: Record<ParlayPair['basis'], string> = {
  same_player: 'Same player', teammates: 'Teammates', independent: 'Independent',
};

export function liftSentence(result: ParlayResult): string {
  const independent = chance(result.independent_probability);
  const linked = chance(result.probability);
  if (result.legs.length === 1) return `With one leg, the parlay is simply that leg: ${linked}.`;
  if (result.correlation_lift === null) return `The product of the leg probabilities is 0%, so linking the legs cannot be measured.`;
  if (result.pairs.every(pair => pair.basis === 'independent')) return `These legs are treated as independent, so the chance they all hit is the product of the legs: ${linked}.`;
  const lift = result.correlation_lift;
  if (lift >= 1.005) return `Linked legs move together, raising the chance from ${independent} to ${linked} (×${lift.toFixed(2)}).`;
  if (lift <= 0.995) return `Linked legs pull against each other, lowering the chance from ${independent} to ${linked} (×${lift.toFixed(2)}).`;
  return `The links between these legs barely move the needle: ${independent} if independent, ${linked} linked.`;
}

const stats = new Set(Object.keys(statNames));
const models = new Set(Object.keys(modelNames));
const emptySlip: SlipState = { legs: [], bookOdds: '' };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function nullableNumber(value: unknown, integer = false): number | null | undefined {
  if (value === null) return null;
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && (!integer || Number.isInteger(value)) ? value : undefined;
}

export function sanitizeLeg(value: unknown): SlipLeg | null {
  if (!isRecord(value) || !isRecord(value.request) || !isRecord(value.label)) return null;
  const request = value.request;
  const text = (item: unknown) => typeof item === 'string' && item.length > 0 && item.length <= 120 ? item : null;
  const datasetId = text(request.dataset_id);
  const playerId = text(request.player_id);
  const stat = typeof request.stat === 'string' && stats.has(request.stat) ? request.stat as Stat : null;
  const model = typeof request.model === 'string' && models.has(request.model) ? request.model as Model : null;
  const side = request.side === 'over' || request.side === 'under' ? request.side : null;
  const line = typeof request.line === 'number' ? normalizeLine(request.line) : null;
  const lookback = typeof request.lookback === 'number' && Number.isInteger(request.lookback) ? request.lookback : null;
  const minutes = nullableNumber(request.expected_minutes);
  const rest = nullableNumber(request.rest_days, true);
  const opponent = request.opponent === null ? null : typeof request.opponent === 'string' && request.opponent.length <= 5 ? request.opponent : undefined;
  const context = typeof request.use_context === 'boolean' ? request.use_context : null;
  if (!datasetId || !playerId || !stat || !model || !side || line === null || lookback === null
    || minutes === undefined || rest === undefined || opponent === undefined || context === null) return null;
  const player = typeof value.label.player === 'string' && value.label.player.trim() ? value.label.player.slice(0, 120) : `Player ${playerId}`;
  const team = typeof value.label.team === 'string' ? value.label.team.slice(0, 60) : '';
  return {
    request: {
      dataset_id: datasetId, player_id: playerId, stat, line, side, model, lookback,
      expected_minutes: minutes, rest_days: rest, opponent, use_context: context,
    },
    label: { player, team },
  };
}

/** Sanitizes each leg, keeps the first leg for each player + stat and caps the slip at MAX_LEGS. */
export function sanitizeLegs(items: unknown[]): { legs: SlipLeg[]; dropped: number } {
  const legs: SlipLeg[] = [];
  const seen = new Set<string>();
  for (const item of items) {
    const leg = sanitizeLeg(item);
    if (!leg || seen.has(legKey(leg.request)) || legs.length >= MAX_LEGS) continue;
    seen.add(legKey(leg.request));
    legs.push(leg);
  }
  return { legs, dropped: items.length - legs.length };
}

export function parseSlip(raw: string | null): SlipState {
  if (!raw) return emptySlip;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed) || parsed.version !== 1 || !Array.isArray(parsed.legs)) return emptySlip;
    const { legs } = sanitizeLegs(parsed.legs);
    return { legs, bookOdds: typeof parsed.bookOdds === 'string' ? parsed.bookOdds.slice(0, 12) : '' };
  } catch {
    return emptySlip;
  }
}

export function loadSlip(): SlipState {
  try {
    return parseSlip(window.localStorage.getItem(SLIP_STORAGE_KEY));
  } catch {
    return emptySlip;
  }
}

export function saveSlip(state: SlipState): void {
  try {
    window.localStorage.setItem(SLIP_STORAGE_KEY, JSON.stringify({ version: 1, legs: state.legs, bookOdds: state.bookOdds }));
  } catch {
    // Storage can be unavailable (private browsing or quota); the slip still works for this session.
  }
}
