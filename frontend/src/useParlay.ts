import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, isAbort, messageOf } from './api';
import { MAX_LEGS, SLIP_STORAGE_KEY, bookOddsText, legKey, loadSlip, parseBookOdds, parseParlayError, parseSlip, sameLeg, sanitizeLegs, saveSlip, validBookOdds } from './parlay';
import type { LegLabel, ParlayError, SlipLeg } from './parlay';
import type { ParlayLeg, ParlayLegRequest, ParlayRequest, ParlayResult, Side } from './parlay-types';

export interface ParlayComputation { data: ParlayResult; request: ParlayRequest; key: string }
export type UpsertOutcome = 'added' | 'replaced' | 'unchanged' | 'full';
/** What happened when a whole slip from elsewhere (the assistant) replaced the current one. */
export interface ReplaceOutcome { legs: SlipLeg[]; dropped: number; oddsDropped: boolean }

export interface ParlayState {
  legs: SlipLeg[];
  full: boolean;
  bookOdds: string;
  bookOddsError: string;
  setBookOdds: (value: string) => void;
  result: ParlayComputation | null;
  stale: boolean;
  computing: boolean;
  error: ParlayError | null;
  retry: () => void;
  upsertLeg: (request: ParlayLegRequest, label: LegLabel) => UpsertOutcome;
  replaceSlip: (slip: { legs: unknown[]; book_odds: unknown }) => ReplaceOutcome;
  setSide: (index: number, side: Side) => void;
  setLine: (index: number, line: number) => void;
  removeLeg: (index: number) => void;
  clear: () => void;
}

/** The computed leg for a slip row, and whether it still matches the row's current settings. */
export function computedLeg(result: ParlayComputation | null, leg: SlipLeg, index: number): { leg: ParlayLeg; fresh: boolean } | null {
  const requested = result?.request.legs[index];
  const computed = result?.data.legs[index];
  if (!requested || !computed || legKey(requested) !== legKey(leg.request)) return null;
  return { leg: computed, fresh: sameLeg(requested, leg.request) };
}

export function useParlay(): ParlayState {
  const [initial] = useState(loadSlip);
  const [legs, setLegs] = useState<SlipLeg[]>(initial.legs);
  const [bookOdds, setBookOdds] = useState(initial.bookOdds);
  const [result, setResult] = useState<ParlayComputation | null>(null);
  const [error, setError] = useState<ParlayError | null>(null);
  const [computing, setComputing] = useState(false);
  const [attempt, setAttempt] = useState(0);

  const odds = parseBookOdds(bookOdds);
  const request = useMemo<ParlayRequest | null>(
    () => legs.length ? { legs: legs.map(leg => leg.request), book_odds: odds.value } : null,
    [legs, odds.value],
  );
  const key = request ? JSON.stringify(request) : '';

  useEffect(() => { saveSlip({ legs, bookOdds }); }, [legs, bookOdds]);

  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key !== SLIP_STORAGE_KEY) return;
      const next = parseSlip(event.newValue);
      setLegs(next.legs);
      setBookOdds(next.bookOdds);
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  useEffect(() => {
    setError(null);
    if (!request) {
      setResult(null);
      setComputing(false);
      return;
    }
    setComputing(true);
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      api<ParlayResult>('/api/parlay', { method: 'POST', body: key, signal: controller.signal })
        .then(data => {
          if (controller.signal.aborted) return;
          setResult({ data, request, key });
          setError(null);
        })
        .catch((reason: unknown) => {
          if (controller.signal.aborted || isAbort(reason)) return;
          setError(parseParlayError(messageOf(reason), request.legs.length));
        })
        .finally(() => { if (!controller.signal.aborted) setComputing(false); });
    }, 400);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
    // `request` is fully described by `key`; `attempt` re-runs the same request on retry.
  }, [key, attempt]);

  const upsertLeg = useCallback((next: ParlayLegRequest, label: LegLabel): UpsertOutcome => {
    const existing = legs.find(leg => legKey(leg.request) === legKey(next));
    if (existing && sameLeg(existing.request, next)) return 'unchanged';
    if (!existing && legs.length >= MAX_LEGS) return 'full';
    setLegs(current => {
      if (current.some(leg => legKey(leg.request) === legKey(next))) {
        return current.map(leg => legKey(leg.request) === legKey(next) ? { request: next, label } : leg);
      }
      return current.length >= MAX_LEGS ? current : [...current, { request: next, label }];
    });
    return existing ? 'replaced' : 'added';
  }, [legs]);

  const patchLeg = useCallback((index: number, patch: Partial<Pick<ParlayLegRequest, 'side' | 'line'>>) => {
    setLegs(current => current.map((leg, position) => position === index ? { ...leg, request: { ...leg.request, ...patch } } : leg));
  }, []);

  /** Replaces the whole slip with the same rules as a stored slip; the debounced computation then runs as usual. */
  const replaceSlip = useCallback((slip: { legs: unknown[]; book_odds: unknown }): ReplaceOutcome => {
    const { legs: next, dropped } = sanitizeLegs(Array.isArray(slip.legs) ? slip.legs : []);
    const price = validBookOdds(slip.book_odds);
    setLegs(next);
    setBookOdds(current => {
      const parsed = parseBookOdds(current);
      return !parsed.error && parsed.value === price ? current : bookOddsText(price);
    });
    return { legs: next, dropped, oddsDropped: slip.book_odds != null && price === null };
  }, []);

  return {
    legs,
    full: legs.length >= MAX_LEGS,
    bookOdds,
    bookOddsError: odds.error,
    setBookOdds,
    result,
    stale: result !== null && result.key !== key,
    computing,
    error,
    retry: useCallback(() => setAttempt(value => value + 1), []),
    upsertLeg,
    replaceSlip,
    setSide: useCallback((index: number, side: Side) => patchLeg(index, { side }), [patchLeg]),
    setLine: useCallback((index: number, line: number) => patchLeg(index, { line }), [patchLeg]),
    removeLeg: useCallback((index: number) => setLegs(current => current.filter((_, position) => position !== index)), []),
    clear: useCallback(() => setLegs([]), []),
  };
}
