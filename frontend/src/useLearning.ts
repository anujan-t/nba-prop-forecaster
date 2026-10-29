import { useCallback, useEffect, useRef, useState } from 'react';
import { isAbort, messageOf } from './api';
import type { LearningStatus } from './assistant-types';
import { fetchLearning, forgetAlias, retrainLearning } from './ledger';

export interface LearningState {
  status: LearningStatus | null;
  loading: boolean;
  error: string;
  retraining: boolean;
  retrainError: string;
  /** Seconds since the current retrain started. */
  elapsed: number;
  forgetting: string | null;
  aliasError: string;
  load: (quiet?: boolean) => void;
  /** Resolves to the new status, or null when the retrain failed. */
  retrain: () => Promise<LearningStatus | null>;
  /** Resolves to true once the nickname is forgotten. */
  forget: (phrase: string) => Promise<boolean>;
}

/**
 * One learning-loop status for every screen that shows it (My Bets, How it works › Accuracy, Settings › Coach).
 * It refreshes while one of them is on screen, and whenever the ledger changes.
 */
export function useLearning(active: boolean, version: number): LearningState {
  const [status, setStatus] = useState<LearningStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [retraining, setRetraining] = useState(false);
  const [retrainError, setRetrainError] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const [forgetting, setForgetting] = useState<string | null>(null);
  const [aliasError, setAliasError] = useState('');
  const controller = useRef<AbortController | null>(null);
  const retrainingRef = useRef(false);

  const load = useCallback((quiet = false) => {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    if (!quiet) { setLoading(true); setError(''); }
    fetchLearning(current.signal)
      .then(data => { if (!current.signal.aborted) { setStatus(data); setError(''); } })
      .catch((reason: unknown) => { if (!isAbort(reason) && !current.signal.aborted && !quiet) setError(messageOf(reason)); })
      .finally(() => { if (controller.current === current) setLoading(false); });
  }, []);

  useEffect(() => {
    if (!active || retrainingRef.current) return;
    load();
  }, [active, version, load]);

  useEffect(() => () => controller.current?.abort(), []);

  // A background retrain (started by grading or by Coach) finishes on its own; check back while it is on screen.
  useEffect(() => {
    if (!active || retraining || !status?.training) return;
    const timer = window.setTimeout(() => load(true), 4000);
    return () => window.clearTimeout(timer);
  }, [active, retraining, status, load]);

  useEffect(() => {
    if (!retraining) return;
    const started = Date.now();
    setElapsed(0);
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [retraining]);

  const retrain = useCallback(async (): Promise<LearningStatus | null> => {
    if (retrainingRef.current) return null;
    controller.current?.abort();
    retrainingRef.current = true;
    setRetraining(true);
    setRetrainError('');
    try {
      const next = await retrainLearning();
      setStatus(next);
      return next;
    } catch (reason) {
      setRetrainError(messageOf(reason));
      return null;
    } finally {
      retrainingRef.current = false;
      setRetraining(false);
    }
  }, []);

  const forget = useCallback(async (phrase: string): Promise<boolean> => {
    setForgetting(phrase);
    setAliasError('');
    try {
      await forgetAlias(phrase);
      setStatus(current => current ? { ...current, memory: { ...current.memory, aliases: current.memory.aliases.filter(alias => alias.phrase !== phrase) } } : current);
      return true;
    } catch (reason) {
      setAliasError(messageOf(reason));
      return false;
    } finally {
      setForgetting(null);
    }
  }, []);

  return { status, loading, error, retraining, retrainError, elapsed, forgetting, aliasError, load, retrain, forget };
}
