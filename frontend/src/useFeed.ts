import { useCallback, useEffect, useRef, useState } from 'react';
import { api, isAbort, messageOf } from './api';

export interface FeedState<T> {
  data: T | null;
  loading: boolean;
  error: string;
  /** Fetch again; `force` asks the server to bypass its cache. */
  refresh: (force?: boolean) => void;
}

/**
 * One cached server feed. It loads the first time `enabled` becomes true and whenever the path changes, keeps the
 * last good copy on screen while refreshing, and never refetches just because the page re-renders.
 */
export function useFeed<T>(path: string | null, enabled: boolean): FeedState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [request, setRequest] = useState<{ path: string; force: boolean; attempt: number } | null>(null);
  const loaded = useRef<string | null>(null);

  useEffect(() => {
    if (!enabled || !path || loaded.current === path) return;
    loaded.current = path;
    setData(null);
    setRequest(current => ({ path, force: false, attempt: (current?.attempt ?? 0) + 1 }));
  }, [enabled, path]);

  useEffect(() => {
    if (!request) return;
    const controller = new AbortController();
    setLoading(true);
    setError('');
    const separator = request.path.includes('?') ? '&' : '?';
    api<T>(request.force ? `${request.path}${separator}refresh=true` : request.path, { signal: controller.signal })
      .then(result => { if (!controller.signal.aborted) setData(result); })
      .catch((reason: unknown) => { if (!controller.signal.aborted && !isAbort(reason)) setError(messageOf(reason)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [request]);

  const refresh = useCallback((force = false) => {
    if (!path) return;
    loaded.current = path;
    setRequest(current => ({ path, force, attempt: (current?.attempt ?? 0) + 1 }));
  }, [path]);

  return { data, loading, error, refresh };
}
