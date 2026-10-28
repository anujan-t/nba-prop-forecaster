import { useCallback, useEffect, useRef, useState } from 'react';
import { api, isAbort, messageOf } from './api';

export function usePlayerResource<T>(kind: 'profile' | 'career', playerId: string, enabled: boolean) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const controller = useRef<AbortController | null>(null);
  const load = useCallback(async (refresh = false) => {
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setData(null);
    setError('');
    setLoading(true);
    try {
      const result = await api<T>(`/api/nba/players/${encodeURIComponent(playerId)}/${kind}?refresh=${refresh}`, { signal: current.signal });
      if (!current.signal.aborted) setData(result);
    } catch (reason) { if (!isAbort(reason)) setError(messageOf(reason)); }
    finally { if (!current.signal.aborted) setLoading(false); }
  }, [kind, playerId]);
  useEffect(() => {
    if (enabled) void load();
    else { controller.current?.abort(); setData(null); setError(''); setLoading(false); }
    const current = controller.current;
    return () => current?.abort();
  }, [enabled, load]);
  return { data, loading, error, refresh: () => { if (enabled) void load(true); } };
}
