import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ArrowUpRight, BookOpen, ChartNoAxesCombined, ChevronRight, Database, FlaskConical, Moon, PanelTop, Sun, UserRound } from 'lucide-react';
import { api, defaultForm, formFromRequest, historySyncLabel, isAbort, messageOf, requestFromForm, sourceLabel, validateOdds } from './api';
import { Basketball, CourtMotif, ErrorNotice } from './components';
import DataWorkspace from './DataWorkspace';
import ForecastPanel from './ForecastPanel';
import LeagueSelector from './LeagueSelector';
import PlayerIntelligence from './PlayerIntelligence';
import LineNotebook from './LineNotebook';
import Methodology from './Methodology';
import ValidationPanel from './ValidationPanel';
import { usePlayerResource } from './usePlayerResource';
import type { AppliedForecast, Dataset, Forecast, ForecastForm, ForecastRequest, Player, View } from './types';
import type { LeagueRoster, PlayerCareer, PlayerHistory, PlayerProfile } from './league-types';
import './league.css';

const views: { id: View; label: string; short: string; icon: typeof ChartNoAxesCombined; eyebrow: string; title: string; description: string }[] = [
  { id: 'forecast', label: 'Forecast lab', short: 'Prop Lab', icon: ChartNoAxesCombined, eyebrow: '01 / EVERY CURRENT NBA PLAYER', title: 'Every line has a story.', description: 'The whole league. Real game history. Your assumptions, kept in view.' },
  { id: 'player', label: 'Player intel', short: 'Player intel', icon: UserRound, eyebrow: '02 / THE PLAYER BEHIND THE NUMBERS', title: 'Know the whole picture.', description: 'Current profiles, shooting, recent form, game logs and career history. No sample-player ceiling.' },
  { id: 'validation', label: 'Model validation', short: 'Validation', icon: FlaskConical, eyebrow: '03 / PUT IT TO THE TEST', title: 'Evidence over intuition.', description: 'Out-of-sample evaluation. Honest baselines. No victory laps on training data.' },
  { id: 'notebook', label: 'Market notebook', short: 'Notebook', icon: BookOpen, eyebrow: '04 / FOLLOW THE QUESTION', title: 'Watch the line, not the noise.', description: 'A deliberate record of manual market snapshots, one game and one book at a time.' },
  { id: 'data', label: 'NBA history', short: 'NBA history', icon: Database, eyebrow: '05 / DIRECT FROM NBA.COM', title: 'Straight from the source.', description: 'Saved histories are your research cache, not a limit on which players you can access.' },
  { id: 'methodology', label: 'Methodology', short: 'Methodology', icon: PanelTop, eyebrow: '06 / OPEN THE BLACK BOX', title: 'Nothing up our sleeve.', description: 'The math, the assumptions, and the things this model cannot tell you.' },
];

export default function App() {
  const [view, setView] = useState<View>('forecast');
  const [theme, setTheme] = useState(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');
  const [mode, setMode] = useState<'league' | 'saved'>('league');
  const [roster, setRoster] = useState<LeagueRoster | null>(null);
  const [rosterLoading, setRosterLoading] = useState(true);
  const [rosterError, setRosterError] = useState('');
  const [selectedId, setSelectedId] = useState('203999');
  const [historyWindow, setHistoryWindow] = useState('recent');
  const [seasonType, setSeasonType] = useState<'Regular Season' | 'Playoffs'>('Regular Season');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState('');
  const [savedId, setSavedId] = useState('');
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetError, setDatasetError] = useState('');
  const [players, setPlayers] = useState<Player[]>([]);
  const [history, setHistory] = useState<PlayerHistory | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState('');
  const [form, setForm] = useState<ForecastForm>(defaultForm);
  const [applied, setApplied] = useState<AppliedForecast | null>(null);
  const [forecastBusy, setForecastBusy] = useState(false);
  const [forecastError, setForecastError] = useState('');
  const [engine, setEngine] = useState<'checking' | 'ready' | 'offline'>('checking');
  const rosterController = useRef<AbortController | null>(null);
  const datasetController = useRef<AbortController | null>(null);
  const historyController = useRef<AbortController | null>(null);
  const forecastController = useRef<AbortController | null>(null);
  const formRef = useRef(form);
  formRef.current = form;
  const headingRef = useRef<HTMLHeadingElement | null>(null);
  const selectedPlayer = roster?.players.find(player => player.id === selectedId) ?? null;
  const profile = usePlayerResource<PlayerProfile>('profile', selectedId, mode === 'league' && !!selectedPlayer);
  const career = usePlayerResource<PlayerCareer>('career', selectedId, mode === 'league' && !!selectedPlayer);
  const activeDataset = datasets.find(dataset => dataset.id === datasetId);
  const page = views.find(item => item.id === view) ?? views[0];
  const rosterSeason = roster?.season;
  const selectedAvailable = !!selectedPlayer;

  const mergeDataset = useCallback((dataset: Dataset) => {
    setDatasets(current => [...current.filter(item => item.id !== dataset.id), dataset]);
  }, []);
  const loadDatasets = useCallback(async () => {
    datasetController.current?.abort();
    const controller = new AbortController();
    datasetController.current = controller;
    setDatasetsLoading(true);
    setDatasetError('');
    try {
      const data = await api<{ datasets: Dataset[] }>('/api/datasets', { signal: controller.signal });
      if (!controller.signal.aborted) setDatasets(data.datasets.filter(dataset => dataset.source !== 'csv'));
    } catch (reason) { if (!isAbort(reason)) setDatasetError(messageOf(reason)); }
    finally { if (!controller.signal.aborted) setDatasetsLoading(false); }
  }, []);
  const loadRoster = useCallback(async (refresh = false) => {
    rosterController.current?.abort();
    const controller = new AbortController();
    rosterController.current = controller;
    setRosterLoading(true);
    setRosterError('');
    try {
      const data = await api<LeagueRoster>(`/api/nba/roster?refresh=${refresh}`, { signal: controller.signal });
      if (!data.players.length) throw new Error('NBA returned no current roster players.');
      if (!controller.signal.aborted) {
        setRoster(data);
        setSelectedId(current => data.players.some(player => player.id === current) ? current : data.players[0].id);
      }
    } catch (reason) { if (!isAbort(reason)) setRosterError(messageOf(reason)); }
    finally { if (!controller.signal.aborted) setRosterLoading(false); }
  }, []);
  const executeForecast = useCallback(async (request: ForecastRequest) => {
    forecastController.current?.abort();
    const controller = new AbortController();
    forecastController.current = controller;
    setForecastBusy(true);
    setForecastError('');
    try {
      const data = await api<Forecast>('/api/forecast', { method: 'POST', body: JSON.stringify(request), signal: controller.signal });
      if (!controller.signal.aborted) { setApplied({ data, request }); mergeDataset(data.dataset); }
    } catch (reason) { if (!isAbort(reason)) setForecastError(messageOf(reason)); }
    finally { if (!controller.signal.aborted) setForecastBusy(false); }
  }, [mergeDataset]);

  const loadHistory = useCallback(async (playerId: string, window: string, season_type: 'Regular Season' | 'Playoffs', refresh = false) => {
    historyController.current?.abort();
    forecastController.current?.abort();
    const controller = new AbortController();
    historyController.current = controller;
    setHistoryLoading(true); setHistoryError(''); setHistory(null);
    setPlayers([]); setApplied(null); setForecastBusy(false); setForecastError(''); setDatasetId('');
    try {
      const result = await api<PlayerHistory>('/api/nba/player-history', {
        method: 'POST', body: JSON.stringify({ player_id: playerId, window, season_type, refresh }), signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      setHistory(result);
      if (result.history) {
        mergeDataset(result.history);
        setDatasetId(result.history.id);
        const list = await api<{ players: Player[] }>(`/api/players?dataset_id=${encodeURIComponent(result.history.id)}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setPlayers(list.players);
        const next = { ...formRef.current, player_id: playerId };
        setForm(next);
        if (result.status === 'ready') void executeForecast(requestFromForm(result.history.id, next));
      }
    } catch (reason) { if (!isAbort(reason)) setHistoryError(messageOf(reason)); }
    finally { if (!controller.signal.aborted) setHistoryLoading(false); }
  }, [executeForecast, mergeDataset]);

  const loadSaved = useCallback(async (id: string) => {
    historyController.current?.abort(); forecastController.current?.abort();
    const controller = new AbortController();
    historyController.current = controller;
    setHistory(null); setHistoryLoading(true); setHistoryError('');
    setApplied(null); setPlayers([]); setForecastBusy(false); setForecastError(''); setDatasetId(id);
    try {
      const data = await api<{ players: Player[] }>(`/api/players?dataset_id=${encodeURIComponent(id)}`, { signal: controller.signal });
      if (controller.signal.aborted) return;
      setPlayers(data.players);
      const player = data.players.find(item => item.id === formRef.current.player_id) ?? data.players[0];
      if (!player) { setHistoryError('No players exist in this saved history.'); return; }
      const next = { ...formRef.current, player_id: player.id };
      setForm(next);
      if (player.games >= 10) void executeForecast(requestFromForm(id, next));
    } catch (reason) { if (!isAbort(reason)) setHistoryError(messageOf(reason)); }
    finally { if (!controller.signal.aborted) setHistoryLoading(false); }
  }, [executeForecast]);

  useEffect(() => {
    void loadDatasets(); void loadRoster();
    return () => { datasetController.current?.abort(); rosterController.current?.abort(); forecastController.current?.abort(); };
  }, [loadDatasets, loadRoster]);
  useEffect(() => {
    if (mode === 'league' && rosterSeason && selectedAvailable) void loadHistory(selectedId, historyWindow, seasonType);
    else if (mode === 'saved' && savedId) void loadSaved(savedId);
    const controller = historyController.current;
    return () => controller?.abort();
  }, [mode, rosterSeason, selectedAvailable, selectedId, historyWindow, seasonType, savedId, loadHistory, loadSaved]);
  useEffect(() => {
    const controller = new AbortController();
    async function check() {
      try {
        const data = await api<{ status: string }>('/api/health', { signal: controller.signal });
        if (!controller.signal.aborted) setEngine(data.status === 'ok' ? 'ready' : 'offline');
      } catch (reason) { if (!isAbort(reason)) setEngine('offline'); }
    }
    void check();
    const interval = window.setInterval(() => { void check(); }, 30000);
    return () => { controller.abort(); window.clearInterval(interval); };
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', getComputedStyle(document.documentElement).getPropertyValue('--cp-bg').trim());
  }, [theme]);

  function navigate(next: View) {
    if (next === 'player' && mode === 'saved') {
      if (roster?.players.some(player => player.id === form.player_id)) setSelectedId(form.player_id);
      setMode('league');
    }
    setView(next); window.scrollTo({ top: 0, behavior: 'instant' });
    window.setTimeout(() => headingRef.current?.focus({ preventScroll: true }), 0);
  }
  function browseLeague() { setMode('league'); navigate('forecast'); }
  function selectPlayer(id: string) { setApplied(null); setMode('league'); setSelectedId(id); setForm(current => ({ ...current, player_id: id })); }
  function selectSaved(id: string) { setMode('saved'); setSavedId(id); navigate('forecast'); }
  function refreshPlayer() {
    if (selectedPlayer) { void loadHistory(selectedId, historyWindow, seasonType, true); profile.refresh(); career.refresh(); }
  }
  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    const url = new URL(window.location.href); url.searchParams.set('scoutTheme', next); window.history.replaceState({}, '', url);
  }
  function run(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!datasetId || !players.some(player => player.id === form.player_id && player.games >= 10)) { setForecastError('At least 10 completed games are required. Player details remain available without a forecast.'); return; }
    const request = requestFromForm(datasetId, form);
    const error = validateOdds(request.odds_over, request.odds_under);
    if (error) { setForecastError(error); return; }
    void executeForecast(request);
  }
  const visibleApplied = mode === 'league' && applied?.request.player_id !== selectedId ? null : applied;
  const dirty = visibleApplied !== null && JSON.stringify(requestFromForm(datasetId, form)) !== JSON.stringify(visibleApplied.request);
  const historyNotice = mode === 'league' && history
    ? `${history.status === 'empty' ? 'No completed games in this window. ' : history.status === 'limited' ? 'Not enough games for a forecast yet. ' : ''}${history.warnings.join(' ')}`
    : '';

  return <div className="app-shell">
    <a href="#main-content" className="skip-link">Skip to main content</a>
    <aside className="sidebar" aria-label="Research navigation">
      <a className="brand" href="#main-content" onClick={() => navigate('forecast')} aria-label="Courtside home"><div className="brand-mark"><Basketball /></div><div><strong>COURTSIDE</strong><small>THE PROP RESEARCH LAB</small></div></a>
      <div className="sidebar-divider" /><span className="nav-eyebrow">THE RESEARCH LAB</span>
      <nav>{views.map(item => { const Icon = item.icon; return <button key={item.id} className={`nav-item ${view === item.id ? 'active' : ''}`} aria-current={view === item.id ? 'page' : undefined} onClick={() => navigate(item.id)}><Icon size={18} strokeWidth={1.7} /><span>{item.label}</span>{view === item.id && <span className="nav-active-dot" />}</button>; })}</nav>
      <div className="sidebar-bottom"><div className="research-card"><span className="eyebrow">THE RESEARCH MINDSET</span><p>Less noise.<br /><strong>Better questions.</strong></p><Basketball /><span className="research-rule" /></div><div className="engine-stack"><span className="tiny-dot" /><span>Python under the hood</span></div><small className="stack-names">Pandas · NumPy · SciPy</small><div className="sidebar-disclaimer">An independent portfolio project.<br />Educational use. Not betting advice.</div></div>
    </aside>
    <div className="workspace">
      <header className="topbar"><div className="breadcrumb"><span>Research</span><ChevronRight size={12} /><strong>{page.short}</strong></div><div className="topbar-right">
        <span className="pill">{roster ? `${roster.season} · ${roster.total_players} current players` : 'Connecting to the NBA roster'}</span>
        <div className={`engine-status ${engine}`}><span />{engine === 'ready' ? 'Engine ready' : engine === 'checking' ? 'Connecting' : 'Engine offline'}</div><div className="topbar-divider" />
        <button className="icon-button theme-button" onClick={toggleTheme} title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}>{theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}</button>
      </div></header>
      <main id="main-content" className="main-content">
        <section className="page-hero"><div className="hero-copy"><div className="hero-kicker"><span className="eyebrow">{page.eyebrow}</span><span className="hero-kicker-line" /></div><h1 ref={headingRef} tabIndex={-1}>{page.title}</h1><p>{page.description}</p></div><div className="hero-art"><CourtMotif /><span>PROBABILITIES, NOT CERTAINTIES.</span></div></section>
        <div className="provenance-strip"><span className="source-badge"><span className="tiny-dot" />{mode === 'league' ? 'Current NBA roster' : 'Saved research study'}</span><span>{mode === 'league' ? roster ? `${roster.total_players} players · ${roster.total_teams} teams` : 'Waiting for NBA directory' : activeDataset ? `${sourceLabel(activeDataset.source)} · ${activeDataset.rows} game rows` : 'Loading saved study'}</span><button className="text-button" onClick={() => navigate('player')}>Player details <ArrowUpRight size={13} /></button></div>
        {datasetError && <ErrorNotice message={datasetError} onRetry={() => { void loadDatasets(); }} />}
        {(view === 'forecast' || view === 'player') && mode === 'league' && <LeagueSelector roster={roster} loading={rosterLoading} error={rosterError} selectedId={selectedId} onSelect={selectPlayer} onRefresh={() => { void loadRoster(true); }} historyWindow={historyWindow} onHistoryWindowChange={setHistoryWindow} seasonType={seasonType} onSeasonTypeChange={setSeasonType} busy={historyLoading} onReload={refreshPlayer} onDetails={() => navigate('player')} />}
        {mode === 'saved' && <div className="notice research-notice"><Database size={18} /><div><strong>Saved study mode, not a league roster limit.</strong><p>{activeDataset ? historySyncLabel(activeDataset) : 'Reading saved game rows.'}</p><button className="text-button" onClick={browseLeague}>Browse every current NBA player <ArrowUpRight size={13} /></button></div></div>}
        <section hidden={view !== 'forecast'} aria-label="Forecast lab">
          {historyError && <ErrorNotice message={historyError} onRetry={mode === 'league' ? () => { void loadHistory(selectedId, historyWindow, seasonType); } : () => { void loadSaved(savedId); }} />}
          <ForecastPanel form={form} onChange={setForm} players={players} playersLoading={historyLoading || (mode === 'league' && rosterLoading)} applied={visibleApplied} busy={forecastBusy} dirty={dirty} error={forecastError} onRun={run} onRevert={() => { if (visibleApplied) setForm(formFromRequest(visibleApplied.request)); }} onValidation={() => navigate('validation')} onData={() => navigate('player')} showPlayerSelect={mode === 'saved'} historyNotice={historyNotice} currentTeam={mode === 'league' ? profile.data?.team ?? selectedPlayer?.team : undefined} />
        </section>
        <section hidden={view !== 'player'} aria-label="Player intel"><PlayerIntelligence player={selectedPlayer} profile={profile.data} profileLoading={profile.loading} profileError={profile.error} career={career.data} careerLoading={career.loading} careerError={career.error} history={history} historyLoading={historyLoading} historyError={historyError} /></section>
        <section hidden={view !== 'validation'} aria-label="Model validation"><ValidationPanel applied={visibleApplied} datasetId={datasetId} onForecast={() => navigate('forecast')} /></section>
        <section hidden={view !== 'notebook'} aria-label="Market notebook"><LineNotebook applied={visibleApplied} onForecast={() => navigate('forecast')} /></section>
        <section hidden={view !== 'data'} aria-label="NBA history"><DataWorkspace datasets={datasets} datasetId={datasetId} onSelect={selectSaved} onBrowseLeague={browseLeague} onRefresh={() => { void loadDatasets(); }} refreshing={datasetsLoading} /></section>
        <section hidden={view !== 'methodology'} aria-label="Methodology"><Methodology onValidation={() => navigate('validation')} onData={() => navigate('data')} /></section>
        <footer className="page-footer"><div><Basketball /><strong>COURTSIDE</strong><span>Built for better questions.</span></div><span>Research only <span className="slash">/</span> No guaranteed outcomes</span></footer>
      </main>
    </div>
  </div>;
}
