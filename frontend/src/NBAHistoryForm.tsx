import { useEffect, useId, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ArrowRight, CheckCircle2, Globe2, Info, Plus, RefreshCw, Search, X } from 'lucide-react';
import { api, historySyncLabel, isAbort, latestCompletedSeason, messageOf } from './api';
import { Busy, ErrorNotice, Field } from './components';
import type { Dataset, NBAHistoryQuery, NbaPlayer } from './types';

interface Props {
  active?: Dataset;
  onLoaded: (dataset: Dataset) => void;
  compact?: boolean;
}

export default function NBAHistoryForm({ active, onLoaded, compact = false }: Props) {
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState<NbaPlayer[]>([]);
  const [matches, setMatches] = useState<NbaPlayer[]>([]);
  const [searchBusy, setSearchBusy] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [searched, setSearched] = useState(false);
  const [season, setSeason] = useState(latestCompletedSeason);
  const [seasonType, setSeasonType] = useState<NBAHistoryQuery['season_type']>('Regular Season');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const requestController = useRef<AbortController | null>(null);
  const helpId = useId();

  useEffect(() => () => requestController.current?.abort(), []);
  useEffect(() => {
    setMatches([]);
    setSearchError('');
    setSearched(false);
    if (query.trim().length < 2 || selected.length >= 5) { setSearchBusy(false); return; }
    const controller = new AbortController();
    setSearchBusy(true);
    const timer = window.setTimeout(async () => {
      try {
        const data = await api<{ players: NbaPlayer[] }>(`/api/nba/players?search=${encodeURIComponent(query.trim())}`, { signal: controller.signal });
        if (!controller.signal.aborted) { setMatches(data.players); setSearched(true); }
      } catch (reason) { if (!isAbort(reason)) setSearchError(messageOf(reason)); }
      finally { if (!controller.signal.aborted) setSearchBusy(false); }
    }, 300);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [query, selected.length]);

  async function load(history: NBAHistoryQuery, refresh: boolean) {
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    setBusy(true);
    setError('');
    setSuccess('');
    try {
      const dataset = await api<Dataset>('/api/nba/history', {
        method: 'POST', signal: controller.signal, body: JSON.stringify({ ...history, refresh }),
      });
      if (!controller.signal.aborted) {
        onLoaded(dataset);
        setSuccess(`${dataset.cache_hit ? 'Loaded cached NBA history' : 'Fetched directly from NBA Stats'}: ${dataset.rows} completed games. ${historySyncLabel(dataset)}.`);
      }
    } catch (reason) { if (!isAbort(reason)) setError(messageOf(reason)); }
    finally { if (!controller.signal.aborted) setBusy(false); }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selected.length) { setError('Choose at least one player from the NBA directory.'); return; }
    void load({ player_ids: selected.map(player => player.id), season, season_type: seasonType }, false);
  }

  return <div className={compact ? 'nba-history-form compact-history' : 'nba-history-form'}>
    <div className="nba-history-heading"><div><span className="eyebrow">DIRECT FROM NBA.COM</span><h2>Choose the player. Pull the history.</h2></div><span className="pill"><Globe2 size={13} />No API key needed</span></div>
    <form onSubmit={submit}>
      <div className="nba-history-fields">
        <div className="nba-player-search">
          <Field label={`Find NBA players (${selected.length}/5)`}><div className="search-input"><Search size={16} /><input type="search" placeholder="Search any NBA player..." value={query} onChange={event => setQuery(event.target.value)} disabled={busy || selected.length >= 5} autoComplete="off" aria-describedby={helpId} /></div></Field>
          {searchBusy && <div className="search-status"><Busy text="Searching the NBA directory..." /></div>}
          {searchError && <ErrorNotice message={searchError} />}
          {matches.filter(player => !selected.some(item => item.id === player.id)).length > 0 && <ul className="player-search-results" aria-label="Matching NBA players">{matches.filter(player => !selected.some(item => item.id === player.id)).map(player => <li key={player.id}><button type="button" disabled={busy} onClick={() => { setSelected(current => [...current, player]); setQuery(''); }}><span>{player.name}</span><Plus size={15} /></button></li>)}</ul>}
          {searched && !searchBusy && matches.length === 0 && <p className="field-hint">No matches. Try another spelling or a surname.</p>}
        </div>
        <Field label="NBA season"><input required pattern="[0-9]{4}-[0-9]{2}" title="Season format: 2025-26" placeholder="2025-26" value={season} onChange={event => setSeason(event.target.value)} disabled={busy} /></Field>
        <Field label="Game type"><select value={seasonType} onChange={event => setSeasonType(event.target.value === 'Playoffs' ? 'Playoffs' : 'Regular Season')} disabled={busy}><option>Regular Season</option><option>Playoffs</option></select></Field>
        <button className="primary-button" type="submit" disabled={busy || !selected.length}>{busy ? <Busy text="Loading NBA history" /> : <>Load NBA history <ArrowRight size={15} /></>}</button>
      </div>
      {selected.length > 0 && <div className="selected-players">{selected.map(player => <span className="player-token" key={player.id}>{player.name}<button type="button" disabled={busy} aria-label={`Remove ${player.name}`} onClick={() => setSelected(current => current.filter(item => item.id !== player.id))}><X size={13} /></button></span>)}</div>}
      <p id={helpId} className="field-hint">Search by name, select up to five players, then load a season. Completed games only; at least 10 are needed to forecast. New seasons may not have enough history yet.</p>
    </form>
    <div className="nba-sync-row"><div><strong>{active?.source === 'nba_api' ? 'Active NBA history' : 'History source'}</strong><span>{active ? historySyncLabel(active) : 'Select a player and season to connect directly.'}</span></div>
      {active?.nba_query && <button className="secondary-button small-button" type="button" disabled={busy} onClick={() => { if (active.nba_query) void load(active.nba_query, true); }}><RefreshCw size={14} />Refresh from NBA</button>}
    </div>
    {busy && <p className="search-status" role="status">Requesting completed game logs from NBA Stats. Allow up to 20 seconds per player.</p>}
    {error && <ErrorNotice message={error} />}
    {success && <p className="saved-message" role="status"><CheckCircle2 size={15} />{success}</p>}
    {!compact && <div className="quiet-note"><Info size={15} /><span>NBA Stats returns structured data through the community-maintained <code>nba_api</code> client. Responses are cached locally for six hours; Load refetches expired history and Refresh always contacts NBA. No files, scraping or synthetic substitutes are required.</span></div>}
  </div>;
}
