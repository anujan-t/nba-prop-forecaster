import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ArrowUpRight, BookOpen, ChartNoAxesCombined, Database, FlaskConical, Layers, Moon, PanelTop, Sun, UserRound } from 'lucide-react';
import { api, defaultForm, formFromRequest, historySyncLabel, isAbort, messageOf, requestFromForm, sourceLabel, validateOdds } from './api';
import { Ball, CourtArt, ErrorNotice, Jersey } from './components';
import CourtNav from './CourtNav';
import type { NavItem } from './CourtNav';
import DataWorkspace from './DataWorkspace';
import ForecastPanel from './ForecastPanel';
import LeagueSelector from './LeagueSelector';
import PlayerIntelligence from './PlayerIntelligence';
import LineNotebook from './LineNotebook';
import Methodology from './Methodology';
import { launchBall } from './motion';
import { MAX_LEGS, legFromForecast, sideNames, statLabel } from './parlay';
import type { Side } from './parlay-types';
import { SlipButton, SlipDrawer } from './ParlaySlip';
import ParlayView from './ParlayView';
import { useParlay } from './useParlay';
import ValidationPanel from './ValidationPanel';
import { usePlayerResource } from './usePlayerResource';
import type { AppliedForecast, Dataset, Forecast, ForecastForm, ForecastRequest, Player, View } from './types';
import type { LeagueRoster, PlayerCareer, PlayerHistory, PlayerProfile } from './league-types';

type Theme = 'light' | 'dark';
const themeStorageKey = 'courtside.theme';

const views: (NavItem & { number: string; eyebrow: string; title: string; description: string })[] = [
  { id: 'forecast', label: 'Forecast lab', icon: ChartNoAxesCombined, number: '01', eyebrow: 'EVERY CURRENT NBA PLAYER', title: 'Every line has a story.', description: 'The whole league. Real game history. Your assumptions, kept in view.' },
  { id: 'parlay', label: 'Parlay', icon: Layers, number: '02', eyebrow: 'STACK THE LEGS', title: 'Every leg has to land.', description: 'Combine props into one parlay and see the chance that all of them hit, with same-player and teammate links built in.' },
  { id: 'player', label: 'Player intel', icon: UserRound, number: '03', eyebrow: 'THE PLAYER BEHIND THE NUMBERS', title: 'Know the whole picture.', description: 'Current profiles, shooting, recent form, game logs and career history. No sample-player ceiling.' },
  { id: 'validation', label: 'Model validation', icon: FlaskConical, number: '04', eyebrow: 'PUT IT TO THE TEST', title: 'Evidence over intuition.', description: 'Out-of-sample evaluation. Honest baselines. No victory laps on training data.' },
  { id: 'notebook', label: 'Market notebook', icon: BookOpen, number: '05', eyebrow: 'FOLLOW THE QUESTION', title: 'Watch the line, not the noise.', description: 'A deliberate record of manual market snapshots, one game and one book at a time.' },
  { id: 'data', label: 'NBA history', icon: Database, number: '06', eyebrow: 'DIRECT FROM NBA.COM', title: 'Straight from the source.', description: 'Saved histories are your research cache, not a limit on which players you can access.' },
  { id: 'methodology', label: 'Methodology', icon: PanelTop, number: '07', eyebrow: 'OPEN THE BLACK BOX', title: 'Nothing up our sleeve.', description: 'The math, the assumptions, and the things this model cannot tell you.' },
];

export default function App() {
  const [view, setView] = useState<View>('forecast');
  const [theme, setTheme] = useState<Theme>(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');
  const parlay = useParlay();
  const [slipOpen, setSlipOpen] = useState(false);
  const [bounce, setBounce] = useState(0);
  const [announcement, setAnnouncement] = useState('');
  const focusAfterClose = useRef<'fab' | 'none' | null>(null);
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
    const color = getComputedStyle(document.documentElement).getPropertyValue('--theme-color').trim();
    if (color) document.querySelector('meta[name="theme-color"]')?.setAttribute('content', color);
  }, [theme]);
  useEffect(() => {
    if (slipOpen) return;
    const target = focusAfterClose.current;
    focusAfterClose.current = null;
    if (target === 'fab') document.getElementById('parlay-slip-button')?.focus();
  }, [slipOpen]);

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
  function chooseTheme(next: Theme) {
    setTheme(next);
    try { window.localStorage.setItem(themeStorageKey, next); } catch { /* storage unavailable: the choice lasts for this page */ }
    const url = new URL(window.location.href); url.searchParams.set('scoutTheme', next); window.history.replaceState(window.history.state, '', url);
  }
  function announce(text: string) {
    setAnnouncement(current => current === text ? `${text}\u00a0` : text);
  }
  const openSlip = useCallback(() => setSlipOpen(true), []);
  const closeSlip = useCallback(() => {
    focusAfterClose.current ??= 'fab';
    setSlipOpen(false);
  }, []);
  function viewParlayFromSlip() {
    focusAfterClose.current = 'none';
    setSlipOpen(false);
    navigate('parlay');
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
  const currentTeam = mode === 'league' ? profile.data?.team ?? selectedPlayer?.team : undefined;

  function addLeg(side: Side, origin: HTMLElement) {
    if (!visibleApplied) return;
    const request = legFromForecast(visibleApplied, side);
    const label = { player: visibleApplied.data.player.name, team: currentTeam ?? visibleApplied.data.player.team };
    const description = `${label.player} ${sideNames[side]} ${request.line} ${statLabel(request.stat)}`;
    const outcome = parlay.upsertLeg(request, label);
    if (outcome === 'full') { announce(`The slip is full. Remove a leg before adding another; ${MAX_LEGS} legs is the maximum.`); return; }
    if (outcome === 'unchanged') { announce(`${description} is already in the slip.`); return; }
    const count = outcome === 'added' ? parlay.legs.length + 1 : parlay.legs.length;
    announce(`${outcome === 'added' ? 'Added' : 'Replaced the leg with'} ${description}. The slip has ${count} ${count === 1 ? 'leg' : 'legs'}.`);
    launchBall(origin, document.getElementById('parlay-slip-button'), () => setBounce(value => value + 1));
  }

  return <>
    <div className="app-shell" inert={slipOpen}>
      <a href="#main-content" className="skip-link">Skip to main content</a>
      <header className="arena-header">
        <div className="arena-header-inner">
          <a className="brand" href="#main-content" onClick={() => navigate('forecast')} aria-label="Courtside, go to the Forecast lab">
            <Ball className="brand-ball" />
            <span className="brand-text"><strong>COURTSIDE</strong><small>NBA prop research lab</small></span>
          </a>
          <div className="header-meta">
            <span className="season-chip">{roster ? `${roster.season} · ${roster.total_players} players` : 'Connecting to the NBA roster'}</span>
            <span className={`engine-status ${engine}`}><span aria-hidden="true" />{engine === 'ready' ? 'Engine ready' : engine === 'checking' ? 'Connecting' : 'Engine offline'}</span>
            <div className="theme-switch" role="group" aria-label="Colour theme">
              <button type="button" aria-pressed={theme === 'light'} onClick={() => chooseTheme('light')} title="Day game: light hardwood theme"><Sun size={15} aria-hidden="true" /><span>Day game</span></button>
              <button type="button" aria-pressed={theme === 'dark'} onClick={() => chooseTheme('dark')} title="Night game: dark arena theme"><Moon size={15} aria-hidden="true" /><span>Night game</span></button>
            </div>
          </div>
        </div>
      </header>
      <CourtNav items={views} view={view} onNavigate={navigate} legCount={parlay.legs.length} />
      <main id="main-content" className="main-content">
        <section className="page-hero" key={view}>
          <div className="hero-copy">
            <div className="hero-kicker"><Jersey label={page.number} className="hero-jersey" /><span className="eyebrow">{page.eyebrow}</span></div>
            <h1 ref={headingRef} tabIndex={-1}>{page.title}</h1>
            <p>{page.description}</p>
          </div>
          <div className="hero-art"><CourtArt /><span className="hero-tag">Probabilities, not certainties.</span></div>
        </section>
        <div className="provenance-strip"><span className="source-badge"><span className="tiny-dot" />{mode === 'league' ? 'Current NBA roster' : 'Saved research study'}</span><span>{mode === 'league' ? roster ? `${roster.total_players} players · ${roster.total_teams} teams` : 'Waiting for NBA directory' : activeDataset ? `${sourceLabel(activeDataset.source)} · ${activeDataset.rows} game rows` : 'Loading saved study'}</span><button className="text-button" onClick={() => navigate('player')}>Player details <ArrowUpRight size={13} /></button></div>
        {datasetError && <ErrorNotice message={datasetError} onRetry={() => { void loadDatasets(); }} />}
        {(view === 'forecast' || view === 'player') && mode === 'league' && <LeagueSelector roster={roster} loading={rosterLoading} error={rosterError} selectedId={selectedId} onSelect={selectPlayer} onRefresh={() => { void loadRoster(true); }} historyWindow={historyWindow} onHistoryWindowChange={setHistoryWindow} seasonType={seasonType} onSeasonTypeChange={setSeasonType} busy={historyLoading} onReload={refreshPlayer} onDetails={() => navigate('player')} />}
        {mode === 'saved' && view !== 'parlay' && <div className="notice research-notice"><Database size={18} /><div><strong>Saved study mode, not a league roster limit.</strong><p>{activeDataset ? historySyncLabel(activeDataset) : 'Reading saved game rows.'}</p><button className="text-button" onClick={browseLeague}>Browse every current NBA player <ArrowUpRight size={13} /></button></div></div>}
        <section className="view-section" hidden={view !== 'forecast'} aria-label="Forecast lab">
          {historyError && <ErrorNotice message={historyError} onRetry={mode === 'league' ? () => { void loadHistory(selectedId, historyWindow, seasonType); } : () => { void loadSaved(savedId); }} />}
          <ForecastPanel form={form} onChange={setForm} players={players} playersLoading={historyLoading || (mode === 'league' && rosterLoading)} applied={visibleApplied} busy={forecastBusy} dirty={dirty} error={forecastError} onRun={run} onRevert={() => { if (visibleApplied) setForm(formFromRequest(visibleApplied.request)); }} onValidation={() => navigate('validation')} onData={() => navigate('player')} showPlayerSelect={mode === 'saved'} historyNotice={historyNotice} currentTeam={currentTeam}
            slipLegs={parlay.legs} slipFull={parlay.full} onAddLeg={addLeg} onOpenSlip={openSlip} />
        </section>
        <section className="view-section" hidden={view !== 'parlay'} aria-label="Parlay"><ParlayView parlay={parlay} onForecast={() => navigate('forecast')} /></section>
        <section className="view-section" hidden={view !== 'player'} aria-label="Player intel"><PlayerIntelligence player={selectedPlayer} profile={profile.data} profileLoading={profile.loading} profileError={profile.error} career={career.data} careerLoading={career.loading} careerError={career.error} history={history} historyLoading={historyLoading} historyError={historyError} /></section>
        <section className="view-section" hidden={view !== 'validation'} aria-label="Model validation"><ValidationPanel applied={visibleApplied} datasetId={datasetId} onForecast={() => navigate('forecast')} /></section>
        <section className="view-section" hidden={view !== 'notebook'} aria-label="Market notebook"><LineNotebook applied={visibleApplied} onForecast={() => navigate('forecast')} /></section>
        <section className="view-section" hidden={view !== 'data'} aria-label="NBA history"><DataWorkspace datasets={datasets} datasetId={datasetId} onSelect={selectSaved} onBrowseLeague={browseLeague} onRefresh={() => { void loadDatasets(); }} refreshing={datasetsLoading} /></section>
        <section className="view-section" hidden={view !== 'methodology'} aria-label="Methodology"><Methodology onValidation={() => navigate('validation')} onData={() => navigate('data')} onParlay={() => navigate('parlay')} /></section>
        <footer className="page-footer"><div className="footer-brand"><Ball /><strong>COURTSIDE</strong><span>Built for better questions.</span></div><span>Probabilities, not certainties <span className="slash">/</span> Research only, not betting advice</span></footer>
      </main>
      <SlipButton count={parlay.legs.length} open={slipOpen} bounce={bounce} onOpen={openSlip} />
    </div>
    <SlipDrawer parlay={parlay} open={slipOpen} onClose={closeSlip} onViewParlay={viewParlayFromSlip} />
    <div className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</div>
  </>;
}
