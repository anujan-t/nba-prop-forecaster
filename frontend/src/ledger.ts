import { api } from './api';
import type { BetStatus, LearningStatus, LegStatus, SettleResult, TrackBetRequest, TrackRecord, TrackedBet, TrackedLeg } from './assistant-types';
import { sideNames, statLabel } from './parlay';

export interface Ledger { bets: TrackedBet[]; record: TrackRecord }

export const betStatusNames: Record<BetStatus, string> = { pending: 'Pending', won: 'Won', lost: 'Lost', void: 'Void', reduced: 'Reduced' };
export const legStatusNames: Record<LegStatus, string> = { pending: 'Pending', won: 'Won', lost: 'Lost', push: 'Push', void: 'Void' };

export const fetchLedger = (signal?: AbortSignal) => api<Ledger>('/api/ledger', { signal });
export const fetchLearning = (signal?: AbortSignal) => api<LearningStatus>('/api/learning', { signal });
export const trackBet = (body: TrackBetRequest) => api<TrackedBet>('/api/ledger', { method: 'POST', body: JSON.stringify(body) });
export const settleBets = () => api<SettleResult>('/api/ledger/settle', { method: 'POST' });
export const deleteBet = (id: string) => api<void>(`/api/ledger/${encodeURIComponent(id)}`, { method: 'DELETE' });
export const retrainLearning = () => api<LearningStatus>('/api/learning/retrain', { method: 'POST' });
export const forgetAlias = (phrase: string) => api<void>(`/api/learning/aliases/${encodeURIComponent(phrase)}`, { method: 'DELETE' });

/** Stake units with two decimals, no sign. */
export function units(value: number): string {
  return Math.abs(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/** Stake units with a typographic sign: +4.20, −10.00. */
export function signedUnits(value: number): string {
  const rounded = Math.round(value * 100) / 100;
  return `${rounded > 0 ? '+' : rounded < 0 ? '−' : ''}${units(rounded)}`;
}

export function roiLabel(roi: number | null): string {
  if (roi === null || !Number.isFinite(roi)) return 'N/A';
  const percent = Math.round(roi * 1000) / 10;
  return `${percent > 0 ? '+' : percent < 0 ? '−' : ''}${Math.abs(percent).toFixed(1)}%`;
}

export function legLine(leg: Pick<TrackedLeg, 'player' | 'side' | 'line' | 'stat'>): string {
  return `${leg.player} ${sideNames[leg.side]} ${leg.line} ${statLabel(leg.stat)}`;
}

export function profitLabel(bet: TrackedBet): string {
  if (bet.status === 'pending') return 'Pending';
  if (bet.status === 'reduced') return 'Book recalculates';
  if (bet.profit === null) return bet.stake === null ? 'No stake' : 'Unknown';
  return signedUnits(bet.profit);
}

/** "3–2", plus whatever else the record holds, in words. */
export function recordDetail(record: TrackRecord): string {
  const parts = [`${record.void} void`, `${record.reduced} reduced`, `${record.pending} pending`];
  return parts.join(' · ');
}

export function legsDetail(record: TrackRecord): string {
  return record.legs_graded
    ? `${record.legs_won} of ${record.legs_graded} graded legs won · model expected ${record.expected_legs_won.toFixed(1)}`
    : 'No graded legs yet';
}
