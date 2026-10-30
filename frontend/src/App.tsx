import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';
import { ArrowLeft, ArrowUpRight, BookOpen, BrainCircuit, ChartNoAxesCombined, ClipboardList, CornerDownLeft, Cpu, Database, Ellipsis, FlaskConical, IdCard, Layers, ListOrdered, MessageCircle, Moon, Newspaper, NotebookPen, Palette, Sun, UsersRound, X } from 'lucide-react';
import { api, dateLabel, defaultForm, formFromRequest, historySyncLabel, isAbort, messageOf, requestFromForm, sourceLabel, validateOdds } from './api';
import type { ChatFocus, TrackedBet } from './assistant-types';
import { slipHash } from './chat';
import { ChatLauncher, ChatPage, ChatPanel } from './ChatPanel';
import type { ConversationProps } from './ChatPanel';
import { Ball, CourtArt, EmptyState, ErrorNotice, Jersey } from './components';
import CourtNav from './CourtNav';
import type { NavItem } from './CourtNav';
import DataWorkspace from './DataWorkspace';
import ForecastPanel from './ForecastPanel';
import LearningPanel from './LearningPanel';
import LeagueSelector from './LeagueSelector';
import { units } from './ledger';
import LineNotebook from './LineNotebook';
import Methodology from './Methodology';
import { launchBall, prefersReducedMotion } from './motion';
import MyBets from './MyBets';
import NewsPage, { newsTabs } from './NewsPage';
import type { GameSlipActions } from './NewsPage';
import { MAX_LEGS, bookOddsText, chatSlip, legDescription, legFromForecast, sideNames, statLabel, validBookOdds } from './parlay';
import type { LegLabel } from './parlay';
import type { GameLegRequest, Side } from './parlay-types';
import { SlipButton, SlipDrawer } from './ParlaySlip';
import ParlayView from './ParlayView';
import PlayerIntelligence from './PlayerIntelligence';
import PlayerNewsStrip from './PlayerNewsStrip';
import { GoContext, RouteLink } from './RouteLink';
import type { Go } from './RouteLink';
import { parseHash, routeHash, sameRoute, useHashRoute } from './routes';
import type { HowSection, NewsSection, Page, PlayerSection, Route, SettingsSection } from './routes';
import { ModelReportCard, ModelSettings } from './ModelSettings';
import { useSimulations } from './simulations';
import { AppearanceSettings, CoachSettings } from './SettingsPage';
import type { Theme } from './SettingsPage';
import SubNav from './SubNav';
import { Toast } from './TrackBet';
import type { ToastData } from './TrackBet';
import { useChat } from './useChat';
import type { SlipApplication } from './useChat';
import { useLearning } from './useLearning';
import { useParlay } from './useParlay';
import ValidationPanel from './ValidationPanel';
import { usePlayerResource } from './usePlayerResource';
import type { AppliedForecast, Dataset, Forecast, ForecastForm, ForecastRequest, Player } from './types';
import type { LeagueRoster, PlayerCareer, PlayerHistory, PlayerProfile } from './league-types';

type Drawer = 'slip' | 'chat' | null;
const themeStorageKey = 'courtside.theme';
const defaultPlayerId = '203999';
/** Phones and small tablets: the tabs move to a bottom bar and Coach opens as a page instead of a drawer. */
const compactQuery = '(max-width: 768px)';

interface PageCopy { number: string; eyebrow: string; title: string; description: string; documentTitle: string }
const pageCopy: Record<Page, PageCopy> = {
  players: { number: '01', eyebrow: 'EVERY CURRENT NBA PLAYER', title: 'Every line has a story.', description: 'The whole league and every market: points, threes, PRA, stocks, double-doubles and more. Real game history, simulated a quarter-million times.', documentTitle: 'Players' },
  parlay: { number: '02', eyebrow: 'STACK THE LEGS', title: 'Every leg has to land.', description: 'Combine props, moneylines and spreads into one parlay and see the chance that all of them hit. Same-player legs are priced together from the same simulated games; same-game legs from the team model.', documentTitle: 'Parlay' },
  bets: { number: '03', eyebrow: 'KEEP SCORE', title: 'The scoreboard settles it.', description: 'Track the parlays you research, grade them against final box scores, and see whether the model is earning its keep.', documentTitle: 'My Bets' },
  news: { number: '04', eyebrow: 'THE NEWSROOM', title: 'Know before the line moves.', description: 'Headlines, tonight’s games with the model’s moneyline and spread chances, injuries, roster moves, coaching staffs and how every team plays, in one place.', documentTitle: 'News' },
  coach: { number: '05', eyebrow: 'THE HUDDLE', title: 'Talk it through.', description: 'Ask who’s out, what’s on tonight, how a team plays or what a price means, check your slip against the house, or build the parlay by command. The same conversation as the Ask Coach drawer.', documentTitle: 'Coach' },
  how: { number: '06', eyebrow: 'HOW IT WORKS / OPEN THE BLACK BOX', title: 'Nothing up our sleeve.', description: 'The math, the assumptions, the evidence, and the things this model cannot tell you.', documentTitle: 'How it works' },
  settings: { number: '07', eyebrow: 'SETTINGS / THE LOCKER ROOM', title: 'Set up your bench.', description: 'How COURTSIDE looks, how hard the simulator works, the NBA history saved on this computer, and how Coach answers.', documentTitle: 'Settings' },
};

interface SectionTab<T extends string> { id: T; label: string; icon: ReactNode }
const playerTabs: SectionTab<PlayerSection>[] = [
  { id: 'props', label: 'Props', icon: <ChartNoAxesCombined size={16} aria-hidden="true" /> },
  { id: 'log', label: 'Game log', icon: <ListOrdered size={16} aria-hidden="true" /> },
  { id: 'career', label: 'Profile & career', icon: <IdCard size={16} aria-hidden="true" /> },
  { id: 'lines', label: 'Lines', icon: <NotebookPen size={16} aria-hidden="true" /> },
];
const howTabs: SectionTab<HowSection>[] = [
  { id: 'method', label: 'Method', icon: <BookOpen size={16} aria-hidden="true" /> },
  { id: 'accuracy', label: 'Accuracy', icon: <FlaskConical size={16} aria-hidden="true" /> },
];
const settingsTabs: SectionTab<SettingsSection>[] = [
  { id: 'appearance', label: 'Appearance', icon: <Palette size={16} aria-hidden="true" /> },
  { id: 'model', label: 'Model', icon: <Cpu size={16} aria-hidden="true" /> },
  { id: 'data', label: 'Data', icon: <Database size={16} aria-hidden="true" /> },
  { id: 'coach', label: 'Coach', icon: <BrainCircuit size={16} aria-hidden="true" /> },
];
const sectionLabels: Record<string, string> = Object.fromEntries([...playerTabs, ...newsTabs, ...howTabs, ...settingsTabs].map(tab => [tab.id, tab.label]));

interface MenuLink { route: Route; label: string; detail: string; icon: ReactNode }
const menuGroups: { id: string; title: string; links: MenuLink[] }[] = [
  { id: 'how', title: 'How it works', links: [
    { route: { page: 'how', section: 'method' }, label: 'Method', detail: 'The simulator, the math and their limits', icon: <BookOpen size={18} aria-hidden="true" /> },
    { route: { page: 'how', section: 'accuracy' }, label: 'Accuracy', detail: 'Backtests and per-market calibration', icon: <FlaskConical size={18} aria-hidden="true" /> },
  ] },
  { id: 'settings', title: 'Settings', links: [
    { route: { page: 'settings', section: 'appearance' }, label: 'Appearance', detail: 'Day game or Night game', icon: <Palette size={18} aria-hidden="true" /> },
    { route: { page: 'settings', section: 'model' }, label: 'Model', detail: 'Simulations per forecast', icon: <Cpu size={18} aria-hidden="true" /> },
    { route: { page: 'settings', section: 'data' }, label: 'Data', detail: 'Saved NBA history and studies', icon: <Database size={18} aria-hidden="true" /> },
    { route: { page: 'settings', section: 'coach' }, label: 'Coach', detail: 'Engine, LLM set-up and memory', icon: <BrainCircuit size={18} aria-hidden="true" /> },
  ] },
];

const playbook: { title: string; plays: string[] }[] = [
  { title: 'Build the slip', plays: ['Add Nikola Jokić over 45.5 PRA', 'Add Jamal Murray over 2.5 threes', 'Add Jokić double-double', 'Add Celtics -5.5', 'Add Nuggets moneyline', 'Flip leg 2', 'Undo'] },
  { title: 'Check the price', plays: ['Book has my parlay at +450, is that good?', 'Jokić over 26.5 points at -115?', 'Who wins Lakers vs Celtics?', 'Is Knicks +3.5 a good bet?', 'Why are my legs linked?'] },
  { title: 'Scout a player', plays: ['What’s SGA averaging?', 'How often has Jokić gone over 11.5 rebounds in his last 10 games?'] },
  { title: 'Keep score', plays: ['Track this for $10', 'Grade my tracked bets', 'Are you getting smarter?'] },
  { title: 'League & news', plays: ['Who’s injured on the Lakers?', 'Is LeBron James playing tonight?', 'Latest NBA news', 'What’s the spread for the Celtics game?', 'How do the Thunder play?', 'Who coaches the Warriors?', 'Which team plays the fastest?', 'Standings'] },
  { title: 'The app & betting basics', plays: ['How does the simulator work?', 'How accurate is the model?', 'What does +150 mean?', 'What’s the vig?'] },
];

const sectionOf = (route: Route): string => 'section' in route ? route.section : '';

/** Focus an element by id, scrolling it clear of the sticky header. False when it is missing or hidden. */
function focusTarget(id: string): boolean {
  const element = document.getElementById(id);
  if (!element || !element.getClientRects().length) return false;
  element.scrollIntoView({ block: element.matches('input, select, textarea') ? 'center' : 'start' });
  element.focus({ preventScroll: true });
  return true;
}

export default function App() {
  const [route, navigate] = useHashRoute();
  const routeRef = useRef(route);
  routeRef.current = route;
  const page = route.page;
  const lastPlayerSection = useRef<PlayerSection>('props');
  const lastNewsSection = useRef<NewsSection>('headlines');
  const lastHowSection = useRef<HowSection>('method');
  const lastSettingsSection = useRef<SettingsSection>('appearance');
  if (route.page === 'players') lastPlayerSection.current = route.section;
  if (route.page === 'news') lastNewsSection.current = route.section;
  if (route.page === 'how') lastHowSection.current = route.section;
  if (route.page === 'settings') lastSettingsSection.current = route.section;
  const playerSection = route.page === 'players' ? route.section : lastPlayerSection.current;
  const newsSection = route.page === 'news' ? route.section : lastNewsSection.current;
  const newsTeam = route.page === 'news' ? route.team : null;
  const howSection = route.page === 'how' ? route.section : lastHowSection.current;
  const settingsSection = route.page === 'settings' ? route.section : lastSettingsSection.current;
  const pendingFocus = useRef<string | null>(null);
  const previousRoute = useRef(route);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuButton = useRef<HTMLButtonElement>(null);
  const [prefill, setPrefill] = useState(0);
  const [theme, setTheme] = useState<Theme>(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');
  const [simulations, setSimulations] = useSimulations();
  const simulationsRef = useRef(simulations);
  simulationsRef.current = simulations;
  const parlay = useParlay(simulations);
  const [drawer, setDrawer] = useState<Drawer>(null);
  const slipOpen = drawer === 'slip';
  const chatOpen = drawer === 'chat';
  const drawerRef = useRef<Drawer>(drawer);
  drawerRef.current = drawer;
  const previousDrawer = useRef<Drawer>(null);
  const [bounce, setBounce] = useState(0);
  const [chatBounce, setChatBounce] = useState(0);
  const [chatUnread, setChatUnread] = useState(false);
  const [ledgerVersion, setLedgerVersion] = useState(0);
  const [toast, setToast] = useState<ToastData | null>(null);
  const [announcement, setAnnouncement] = useState('');
  const focusAfterClose = useRef<'fab' | 'chat' | 'none' | null>(null);
  const [mode, setMode] = useState<'league' | 'saved'>('league');
  const [roster, setRoster] = useState<LeagueRoster | null>(null);
  const [rosterLoading, setRosterLoading] = useState(true);
  const [rosterError, setRosterError] = useState('');
  const [selectedId, setSelectedId] = useState(() => {
    const initial = parseHash(window.location.hash);
    return initial?.page === 'players' && initial.playerId ? initial.playerId : defaultPlayerId;
  });
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
  const copy = pageCopy[page];
  const rosterSeason = roster?.season;
  const selectedAvailable = !!selectedPlayer;
  const learning = useLearning(page === 'bets' || (page === 'how' && howSection === 'accuracy') || (page === 'settings' && settingsSection === 'coach'), ledgerVersion);

  const go = useCallback<Go>((next, options) => {
    setMenuOpen(false);
    if (sameRoute(next, routeRef.current)) {
      const id = options?.focus;
      if (!id || !focusTarget(id)) window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
      return;
    }
    pendingFocus.current = options?.focus ?? null;
    navigate(next, { replace: options?.replace });
  }, [navigate]);
  const playersRoute = (section: PlayerSection = lastPlayerSection.current): Route => ({ page: 'players', playerId: mode === 'league' ? selectedId : null, section });
  const goPlayers = (section: PlayerSection, focus?: string) => go(playersRoute(section), { focus });

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
        const known = (id: string) => data.players.some(player => player.id === id);
        setSelectedId(current => known(current) ? current : known(defaultPlayerId) ? defaultPlayerId : data.players[0].id);
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
        if (result.status === 'ready') void executeForecast(requestFromForm(result.history.id, next, simulationsRef.current));
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
      if (player.games >= 10) void executeForecast(requestFromForm(id, next, simulationsRef.current));
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
    const previous = previousDrawer.current;
    previousDrawer.current = drawer;
    if (drawer !== null || previous === null) return;
    const target = focusAfterClose.current;
    focusAfterClose.current = null;
    if (target === 'fab') document.getElementById('parlay-slip-button')?.focus();
    else if (target === 'chat') document.getElementById('chat-launcher')?.focus();
  }, [drawer]);

  /** A new page: scroll to the top and focus its heading (or the element a link asked for). A new section: keep its tabs in view. */
  useLayoutEffect(() => {
    const previous = previousRoute.current;
    previousRoute.current = route;
    if (previous === route) return;
    const target = pendingFocus.current;
    pendingFocus.current = null;
    const pageChanged = previous.page !== route.page;
    const sectionChanged = !pageChanged && sectionOf(previous) !== sectionOf(route);
    const place = () => {
      if (target && focusTarget(target)) return;
      if (pageChanged) {
        window.scrollTo({ top: 0, behavior: 'instant' });
        headingRef.current?.focus({ preventScroll: true });
      } else if (sectionChanged) {
        const nav = document.querySelector<HTMLElement>('.view-section:not([hidden]) .sub-nav');
        const headerBottom = document.querySelector('.arena-header')?.getBoundingClientRect().bottom ?? 0;
        if (nav && nav.getBoundingClientRect().top < headerBottom) nav.scrollIntoView({ block: 'start' });
        const active = document.activeElement as HTMLElement | null;
        if (!active || active === document.body || !active.getClientRects().length) nav?.querySelector<HTMLElement>('[aria-current="page"]')?.focus({ preventScroll: true });
      }
    };
    if (!pageChanged && !sectionChanged) { if (target) place(); return; }
    setMenuOpen(false);
    if (drawerRef.current !== null) {
      // Back or Forward while a drawer is open: close it, then move focus once the page is interactive again.
      focusAfterClose.current = 'none';
      setDrawer(null);
      const frame = requestAnimationFrame(place);
      return () => cancelAnimationFrame(frame);
    }
    place();
  }, [route]);

  useEffect(() => {
    const label = route.page === 'players' || route.page === 'how' || route.page === 'settings' || route.page === 'news'
      ? `${route.page === 'news' && route.team ? `${route.team} · ` : `${sectionLabels[sectionOf(route)] ?? ''} · `}` : '';
    document.title = `${label}${pageCopy[route.page].documentTitle} · COURTSIDE`;
  }, [route]);

  // On Players the URL names the league player: deep links and Back/Forward choose them, and the URL follows a new pick.
  useEffect(() => {
    if (route.page !== 'players') return;
    const { playerId, section } = route;
    if (playerId) {
      if (!roster) return;
      if (!roster.players.some(player => player.id === playerId)) navigate({ page: 'players', playerId: mode === 'league' ? selectedId : null, section }, { replace: true });
      else if (mode !== 'league' || playerId !== selectedId) choosePlayer(playerId);
    } else if (mode === 'league') {
      navigate({ page: 'players', playerId: selectedId, section }, { replace: true });
    }
  }, [route, roster, mode, selectedId, navigate]);

  useEffect(() => {
    if (!menuOpen) return;
    const onPointer = (event: PointerEvent) => { if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false); };
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setMenuOpen(false);
      menuButton.current?.focus();
    };
    document.addEventListener('pointerdown', onPointer);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('pointerdown', onPointer); document.removeEventListener('keydown', onKey); };
  }, [menuOpen]);

  function choosePlayer(id: string) { setApplied(null); setMode('league'); setSelectedId(id); setForm(current => ({ ...current, player_id: id })); }
  function selectPlayer(id: string) {
    choosePlayer(id);
    navigate({ page: 'players', playerId: id, section: playerSection }, { replace: true });
  }
  function browseLeague() { setMode('league'); go({ page: 'players', playerId: selectedId, section: 'props' }, { focus: 'page-title' }); }
  function openInLeague() {
    const id = roster?.players.some(player => player.id === form.player_id) ? form.player_id : selectedId;
    go({ page: 'players', playerId: id, section: playerSection }, { focus: 'page-title' });
  }
  function selectSaved(id: string) { setMode('saved'); setSavedId(id); go({ page: 'players', playerId: null, section: 'props' }); }
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
  const openSlip = useCallback(() => setDrawer('slip'), []);
  const closeSlip = useCallback(() => {
    focusAfterClose.current ??= 'fab';
    setDrawer(current => current === 'slip' ? null : current);
  }, []);
  const openChat = useCallback(() => { setChatUnread(false); setDrawer('chat'); }, []);
  const closeChat = useCallback(() => {
    focusAfterClose.current ??= 'chat';
    setDrawer(current => current === 'chat' ? null : current);
  }, []);
  /** Coach from anywhere: the drawer on wide screens, the Coach tab on phones. */
  const openCoach = useCallback(() => {
    if (routeRef.current.page === 'coach') { document.getElementById('chat-input')?.focus(); return; }
    if (window.matchMedia(compactQuery).matches) { go({ page: 'coach' }); return; }
    openChat();
  }, [go, openChat]);
  function leaveDrawerFor(next: Route, focusId?: string) {
    if (drawerRef.current !== null) { focusAfterClose.current = 'none'; setDrawer(null); }
    go(next, { focus: focusId });
  }
  const onTracked = useCallback((bet: TrackedBet) => {
    const legs = `${bet.legs.length} ${bet.legs.length === 1 ? 'leg' : 'legs'}`;
    const stake = bet.stake !== null ? ` · stake ${units(bet.stake)}` : '';
    setToast({ id: Date.now(), title: 'Parlay tracked', detail: `${legs} · game date ${dateLabel(bet.event_date, true)}${stake}` });
    setLedgerVersion(value => value + 1);
  }, []);
  const closeToast = useCallback(() => setToast(null), []);
  function run(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!datasetId || !players.some(player => player.id === form.player_id && player.games >= 10)) { setForecastError('At least 10 completed games are required. The game log, profile and career remain available without a forecast.'); return; }
    const request = requestFromForm(datasetId, form, simulations);
    const error = validateOdds(request.odds_over, request.odds_under);
    if (error) { setForecastError(error); return; }
    void executeForecast(request);
  }
  const visibleApplied = mode === 'league' && applied?.request.player_id !== selectedId ? null : applied;
  const dirty = visibleApplied !== null && JSON.stringify(requestFromForm(datasetId, form, simulations)) !== JSON.stringify(visibleApplied.request);
  const historyNotice = mode === 'league' && history
    ? `${history.status === 'empty' ? 'No completed games in this window. ' : history.status === 'limited' ? 'Not enough games for a forecast yet. ' : ''}${history.warnings.join(' ')}`
    : '';
  const currentTeam = mode === 'league' ? profile.data?.team ?? selectedPlayer?.team : undefined;
  const newsTeams = useMemo(() => (roster?.teams ?? []).map(team => team.abbreviation).sort(), [roster]);
  const studyPlayer = players.find(player => player.id === form.player_id);
  const studyPlayerInLeague = !!studyPlayer && !!roster?.players.some(player => player.id === studyPlayer.id);

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
  function addGameLeg(request: GameLegRequest, label: LegLabel, origin: HTMLElement) {
    const description = legDescription({ request, label });
    const outcome = parlay.upsertLeg(request, label);
    if (outcome === 'full') { announce(`The slip is full. Remove a leg before adding another; ${MAX_LEGS} legs is the maximum.`); return; }
    if (outcome === 'unchanged') { announce(`${description} is already in the slip.`); return; }
    const count = outcome === 'added' ? parlay.legs.length + 1 : parlay.legs.length;
    announce(`${outcome === 'added' ? 'Added' : `Replaced this game's ${request.market} with`} ${description}. The slip has ${count} ${count === 1 ? 'leg' : 'legs'}.`);
    launchBall(origin, document.getElementById('parlay-slip-button'), () => setBounce(value => value + 1));
  }
  const gameSlip: GameSlipActions = { legs: parlay.legs, full: parlay.full, onAdd: addGameLeg };
  function saveLine() {
    setPrefill(value => value + 1);
    goPlayers('lines', 'line-event-date');
  }

  const focus = useMemo<ChatFocus | null>(() => {
    if (!visibleApplied) return null;
    const { request, data } = visibleApplied;
    return {
      dataset_id: request.dataset_id, player_id: request.player_id, player: data.player.name, team: currentTeam ?? data.player.team,
      stat: request.stat, line: request.line, model: request.model, lookback: request.lookback,
      expected_minutes: request.expected_minutes, rest_days: request.rest_days, opponent: request.opponent, use_context: request.use_context,
    };
  }, [visibleApplied, currentTeam]);
  const currentSlip = useMemo(() => chatSlip(parlay.legs, parlay.bookOdds), [parlay.legs, parlay.bookOdds]);
  const currentSlipHash = useMemo(() => slipHash(currentSlip), [currentSlip]);
  const chat = useChat({
    getSlip: () => currentSlip,
    getFocus: () => focus,
    applySlip: (slip, sent): SlipApplication => {
      if (JSON.stringify(currentSlip) !== JSON.stringify(sent)) {
        return { applied: false, hash: null, notes: ['Your slip changed while Coach was answering, so this change was not applied. Ask again to apply it to the slip as it is now.'] };
      }
      const outcome = parlay.replaceSlip(slip);
      const notes: string[] = [];
      if (outcome.dropped) notes.push(`${outcome.dropped} ${outcome.dropped === 1 ? 'leg' : 'legs'} from Coach could not be used and ${outcome.dropped === 1 ? 'was' : 'were'} left out of the slip.`);
      if (outcome.oddsDropped) notes.push('Coach sent book odds outside −10000 to −100 or +100 to +10000, so the book odds were cleared.');
      return { applied: true, hash: slipHash(chatSlip(outcome.legs, bookOddsText(validBookOdds(slip.book_odds)))), notes };
    },
  });
  const chatReply = chat.reply;
  const chatMessages = chat.conversation.messages;
  useEffect(() => {
    if (!chatReply) return;
    const message = chatMessages.find(item => item.id === chatReply.messageId);
    if (!message || message.role !== 'coach') return;
    if (drawerRef.current !== 'chat' && routeRef.current.page !== 'coach') {
      setChatUnread(true);
      announce('Coach replied. Open Coach to read the answer.');
    }
    const { actions, cards } = message.reply;
    if (actions.some(action => action.type === 'track_bet') || cards.some(card => card.type === 'bet' || card.type === 'track_record' || card.type === 'learning')) {
      setLedgerVersion(value => value + 1);
    }
    // Runs once per reply; the message list is read here, not watched.
  }, [chatReply]);
  useEffect(() => { if (page === 'coach') setChatUnread(false); }, [page]);
  // Narrowing to a phone layout with the drawer open: Coach carries on as a page.
  useEffect(() => {
    const query = window.matchMedia(compactQuery);
    const onChange = () => {
      if (!query.matches || drawerRef.current !== 'chat') return;
      focusAfterClose.current = 'none';
      setDrawer(null);
      go({ page: 'coach' });
    };
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, [go]);
  const slipEvent = chat.slipEvent;
  useEffect(() => {
    if (!slipEvent) return;
    const bump = () => { setBounce(value => value + 1); setChatBounce(value => value + 1); };
    const inChat = drawerRef.current === 'chat' || routeRef.current.page === 'coach';
    const message = document.querySelector<HTMLElement>(`[data-message="${slipEvent.messageId}"]`);
    const launcher = document.getElementById('chat-launcher');
    const origin = inChat ? message?.querySelector<HTMLElement>('.chat-receipts') ?? message ?? null : launcher?.getClientRects().length ? launcher : null;
    const target = document.getElementById(inChat ? 'chat-slip-chip' : 'parlay-slip-button');
    if (!origin || !target) { bump(); return; }
    launchBall(origin, target, bump);
  }, [slipEvent]);
  const toastNode = toast && <Toast key={toast.id} toast={toast} onClose={closeToast} onOpenBets={() => { setToast(null); leaveDrawerFor({ page: 'bets' }); }} />;

  const conversation: ConversationProps = {
    chat, focus, slipCount: parlay.legs.length,
    slipProbability: parlay.result && !parlay.stale ? parlay.result.data.probability : null,
    slipBusy: parlay.computing, slipHash: currentSlipHash, bounce: chatBounce,
    onOpenSlip: openSlip,
    onOpenParlay: () => leaveDrawerFor({ page: 'parlay' }),
    onOpenRecord: () => leaveDrawerFor({ page: 'bets' }),
    onOpenAccuracy: () => leaveDrawerFor({ page: 'how', section: 'accuracy' }, 'learning-title'),
    onOpenRoute: (next, options) => leaveDrawerFor(next, options?.focus),
  };
  const putPlay = (play: string) => {
    chat.setDraft(play);
    requestAnimationFrame(() => document.getElementById('chat-input')?.focus());
  };

  const legCount = parlay.legs.length;
  const tabs: NavItem[] = [
    { id: 'players', label: 'Players', icon: <UsersRound size={19} />, href: routeHash(playersRoute()) },
    { id: 'parlay', label: 'Parlay', icon: <Layers size={19} />, href: routeHash({ page: 'parlay' }), badge: legCount ? { value: legCount, label: `${legCount} ${legCount === 1 ? 'leg' : 'legs'} in the slip` } : null },
    { id: 'bets', label: 'My Bets', icon: <ClipboardList size={19} />, href: routeHash({ page: 'bets' }) },
    { id: 'news', label: 'News', icon: <Newspaper size={19} />, href: routeHash({ page: 'news', section: lastNewsSection.current, team: null }) },
    { id: 'coach', label: 'Coach', icon: <MessageCircle size={19} />, href: routeHash({ page: 'coach' }), dot: chatUnread ? 'new reply from Coach' : null },
  ];
  const activeTab = page === 'players' || page === 'parlay' || page === 'bets' || page === 'news' || page === 'coach' ? page : null;
  const goTab = (id: string) => go(id === 'players' ? playersRoute() : id === 'news' ? { page: 'news', section: lastNewsSection.current, team: null } : { page: id } as Route);
  const menuPage = page === 'how' ? 'How it works' : page === 'settings' ? 'Settings' : null;
  const themeButtons = <>
    <button type="button" aria-pressed={theme === 'light'} onClick={() => chooseTheme('light')} title="Day game: light hardwood theme"><Sun size={15} aria-hidden="true" /><span>Day game</span></button>
    <button type="button" aria-pressed={theme === 'dark'} onClick={() => chooseTheme('dark')} title="Night game: dark arena theme"><Moon size={15} aria-hidden="true" /><span>Night game</span></button>
  </>;
  const engineStatus = <span className={`engine-status ${engine}`}><span aria-hidden="true" />{engine === 'ready' ? 'Engine ready' : engine === 'checking' ? 'Connecting' : 'Engine offline'}</span>;
  const playerName = mode === 'league' ? selectedPlayer?.name : studyPlayer?.name;

  return <GoContext.Provider value={go}>
    <div className="app-shell" inert={drawer !== null}>
      <a href="#main-content" className="skip-link" onClick={event => {
        event.preventDefault();
        const main = document.getElementById('main-content');
        main?.focus({ preventScroll: true });
        main?.scrollIntoView();
      }}>Skip to main content</a>
      <header className={`arena-header${menuOpen ? ' menu-open' : ''}`}>
        <div className="arena-header-inner">
          <RouteLink className="brand" to={playersRoute()} aria-label="Courtside, go to Players">
            <Ball className="brand-ball" />
            <span className="brand-text"><strong>COURTSIDE</strong><small>NBA prop research lab</small></span>
          </RouteLink>
          <CourtNav variant="top" items={tabs} active={activeTab} onNavigate={goTab} label="Main" />
          <div className="header-tools">
            {engineStatus}
            <div className="theme-switch header-theme" role="group" aria-label="Colour theme">{themeButtons}</div>
            <div className="more" ref={menuRef} onBlur={event => {
              const next = event.relatedTarget as Node | null;
              if (menuOpen && next && !event.currentTarget.contains(next)) setMenuOpen(false);
            }}>
              <button ref={menuButton} id="more-button" type="button" className={`more-button${menuPage ? ' is-current' : ''}`} aria-expanded={menuOpen} aria-controls="more-menu" onClick={() => setMenuOpen(open => !open)}>
                {menuOpen ? <X size={18} aria-hidden="true" /> : <Ellipsis size={18} aria-hidden="true" />}<span className="more-label">More</span>
                {menuPage && <span className="sr-only">, current page: {menuPage}</span>}
              </button>
              <div id="more-menu" className="more-menu" hidden={!menuOpen}>
                <nav className="more-nav" aria-label="How it works and Settings">
                  {menuGroups.map(group => <div className="more-group" key={group.id}>
                    <p className="more-title" id={`more-${group.id}`}>{group.title}</p>
                    <ul aria-labelledby={`more-${group.id}`}>
                      {group.links.map(link => {
                        const current = sameRoute(link.route, route);
                        return <li key={link.label}>
                          <RouteLink className={`more-link${current ? ' active' : ''}`} to={link.route} aria-current={current ? 'page' : undefined}>
                            <span className="more-link-icon">{link.icon}</span>
                            <span className="more-link-text"><strong>{link.label}</strong><small>{link.detail}</small></span>
                          </RouteLink>
                        </li>;
                      })}
                    </ul>
                  </div>)}
                </nav>
                <div className="more-footer">
                  <div className="theme-switch menu-theme" role="group" aria-label="Colour theme">{themeButtons}</div>
                  <div className="more-status">
                    {engineStatus}
                    <span className="season-chip">{roster ? `${roster.season} · ${roster.total_players} players` : 'Connecting to the NBA roster'}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>
      <main id="main-content" className="main-content" tabIndex={-1}>
        <section className={`page-hero${page === 'coach' ? ' compact' : ''}`} key={page}>
          <div className="hero-copy">
            <div className="hero-kicker"><Jersey label={copy.number} className="hero-jersey" /><span className="eyebrow">{copy.eyebrow}</span></div>
            <h1 id="page-title" ref={headingRef} tabIndex={-1}>{copy.title}</h1>
            <p>{copy.description}</p>
          </div>
          {page !== 'coach' && <div className="hero-art"><CourtArt /><span className="hero-tag">Probabilities, not certainties.</span></div>}
        </section>

        <section className="view-section" hidden={page !== 'players'} aria-label="Players">
          <div className="provenance-strip"><span className="source-badge"><span className="tiny-dot" />{mode === 'league' ? 'Current NBA roster' : 'Saved research study'}</span><span>{mode === 'league' ? roster ? `${roster.total_players} players · ${roster.total_teams} teams` : 'Waiting for NBA directory' : activeDataset ? `${sourceLabel(activeDataset.source)} · ${activeDataset.rows} game rows` : 'Loading saved study'}</span></div>
          {datasetError && <ErrorNotice message={datasetError} onRetry={() => { void loadDatasets(); }} />}
          {mode === 'league'
            ? <LeagueSelector roster={roster} loading={rosterLoading} error={rosterError} selectedId={selectedId} onSelect={selectPlayer} onRefresh={() => { void loadRoster(true); }} historyWindow={historyWindow} onHistoryWindowChange={setHistoryWindow} seasonType={seasonType} onSeasonTypeChange={setSeasonType} busy={historyLoading} onReload={refreshPlayer} onDetails={playerSection === 'career' ? undefined : () => goPlayers('career')} />
            : <div className="notice research-notice saved-banner">
              <Database size={18} aria-hidden="true" />
              <div><strong>Saved study{activeDataset ? `: ${activeDataset.name}` : ''}</strong><p>{activeDataset ? `${historySyncLabel(activeDataset)}. ` : 'Reading saved game rows. '}Pick a player from this study in Props. A saved study is not a limit on the league.</p></div>
              <button type="button" className="secondary-button small-button" onClick={browseLeague}><ArrowLeft size={15} aria-hidden="true" />Back to the league</button>
            </div>}
          {mode === 'league' && page === 'players' && <PlayerNewsStrip playerId={selectedPlayer ? selectedId : null} playerName={selectedPlayer?.name} team={currentTeam} />}
          <SubNav items={playerTabs.map(tab => ({ ...tab, href: routeHash(playersRoute(tab.id)) }))} active={playerSection} onNavigate={id => goPlayers(id as PlayerSection)} label={playerName ? `${playerName}: sections` : 'Player sections'} />
          <div className="page-panel" hidden={playerSection !== 'props'}>
            {historyError && <ErrorNotice message={historyError} onRetry={mode === 'league' ? () => { void loadHistory(selectedId, historyWindow, seasonType); } : () => { void loadSaved(savedId); }} />}
            <ForecastPanel form={form} onChange={setForm} players={players} playersLoading={historyLoading || (mode === 'league' && rosterLoading)} applied={visibleApplied} busy={forecastBusy} dirty={dirty} error={forecastError} onRun={run} onRevert={() => { if (visibleApplied) setForm(formFromRequest(visibleApplied.request)); }}
              onValidation={() => go({ page: 'how', section: 'accuracy' }, { focus: 'backtest-title' })} onDetails={section => goPlayers(section)} onSaveLine={saveLine}
              showPlayerSelect={mode === 'saved'} historyNotice={historyNotice} currentTeam={currentTeam}
              slipLegs={parlay.legs} slipFull={parlay.full} onAddLeg={addLeg} onOpenSlip={openSlip} />
          </div>
          <div className="page-panel" hidden={playerSection !== 'log' && playerSection !== 'career'}>
            {mode === 'league'
              ? <PlayerIntelligence section={playerSection === 'career' ? 'career' : 'log'} player={selectedPlayer} profile={profile.data} profileLoading={profile.loading} profileError={profile.error} career={career.data} careerLoading={career.loading} careerError={career.error} history={history} historyLoading={historyLoading} historyError={historyError} />
              : <div className="card"><EmptyState title={playerSection === 'career' ? 'Profiles and careers come from the league directory.' : 'Full game logs come from the league directory.'} icon={<IdCard size={32} />}
                action={<button type="button" className="primary-button" onClick={openInLeague}>{studyPlayerInLeague && studyPlayer ? `Open ${studyPlayer.name} in the league` : 'Back to the league'} <ArrowUpRight size={15} aria-hidden="true" /></button>}>
                This saved study holds game rows for forecasting in Props. Game logs, splits, shooting, profiles and careers load from the live NBA directory.
              </EmptyState></div>}
          </div>
          <div className="page-panel" hidden={playerSection !== 'lines'}>
            <LineNotebook applied={visibleApplied} onForecast={() => goPlayers('props')} prefill={prefill} />
          </div>
        </section>

        <section className="view-section" hidden={page !== 'parlay'} aria-label="Parlay"><ParlayView parlay={parlay} onForecast={() => goPlayers('props')} onGames={() => go({ page: 'news', section: 'games', team: null })} onTracked={onTracked} /></section>

        <section className="view-section" hidden={page !== 'bets'} aria-label="My Bets"><MyBets active={page === 'bets'} version={ledgerVersion} learning={learning} onParlay={() => go({ page: 'parlay' })} onCoach={openCoach} /></section>

        <section className="view-section" hidden={page !== 'news'} aria-label="News">
          <SubNav items={newsTabs.map(tab => ({ ...tab, href: routeHash({ page: 'news', section: tab.id, team: null }) }))} active={newsSection} onNavigate={id => go({ page: 'news', section: id as NewsSection, team: null })} label="News sections" />
          <NewsPage active={page === 'news'} section={newsSection} team={newsTeam} teams={newsTeams} slip={gameSlip} />
        </section>

        {page === 'coach' && <section className="view-section coach-page" aria-label="Coach">
          <div className="coach-layout">
            <ChatPage {...conversation} />
            <aside className="card coach-playbook" aria-labelledby="playbook-title">
              <span className="eyebrow">DRAW IT UP</span>
              <h2 id="playbook-title">Playbook</h2>
              <p>Tap a play to put it in the message box. Edit it, then press Enter.</p>
              {playbook.map(group => <div className="playbook-group" key={group.title}>
                <h3>{group.title}</h3>
                <ul>{group.plays.map(play => <li key={play}><button type="button" className="playbook-play" onClick={() => putPlay(play)}><CornerDownLeft size={14} aria-hidden="true" /><span>{play}</span></button></li>)}</ul>
              </div>)}
              <div className="playbook-links">
                <RouteLink className="text-button" to={{ page: 'how', section: 'method' }}>How Coach works <ArrowUpRight size={14} aria-hidden="true" /></RouteLink>
                <RouteLink className="text-button" to={{ page: 'settings', section: 'coach' }}>Coach settings <ArrowUpRight size={14} aria-hidden="true" /></RouteLink>
              </div>
            </aside>
          </div>
        </section>}

        <section className="view-section" hidden={page !== 'how'} aria-label="How it works">
          <SubNav items={howTabs.map(tab => ({ ...tab, href: routeHash({ page: 'how', section: tab.id }) }))} active={howSection} onNavigate={id => go({ page: 'how', section: id as HowSection })} label="How it works sections" />
          <div className="page-panel" hidden={howSection !== 'method'}><Methodology /></div>
          <div className="page-panel accuracy-page" hidden={howSection !== 'accuracy'}>
            <LearningPanel learning={learning} />
            <ModelReportCard />
            <ValidationPanel applied={visibleApplied} datasetId={datasetId} onForecast={() => goPlayers('props')} />
          </div>
        </section>

        <section className="view-section" hidden={page !== 'settings'} aria-label="Settings">
          <SubNav items={settingsTabs.map(tab => ({ ...tab, href: routeHash({ page: 'settings', section: tab.id }) }))} active={settingsSection} onNavigate={id => go({ page: 'settings', section: id as SettingsSection })} label="Settings sections" />
          <div className="page-panel" hidden={settingsSection !== 'appearance'}><AppearanceSettings theme={theme} onTheme={chooseTheme} /></div>
          <div className="page-panel" hidden={settingsSection !== 'model'}><ModelSettings simulations={simulations} onSimulations={setSimulations} /></div>
          <div className="page-panel" hidden={settingsSection !== 'data'}>
            {datasetError && <ErrorNotice message={datasetError} onRetry={() => { void loadDatasets(); }} />}
            <DataWorkspace datasets={datasets} datasetId={datasetId} onSelect={selectSaved} onBrowseLeague={browseLeague} onRefresh={() => { void loadDatasets(); }} refreshing={datasetsLoading} />
          </div>
          <div className="page-panel" hidden={settingsSection !== 'coach'}><CoachSettings chat={chat} learning={learning} onOpenCoach={openCoach} /></div>
        </section>

        <footer className="page-footer"><div className="footer-brand"><Ball /><strong>COURTSIDE</strong><span>Built for better questions.</span></div><span>Probabilities, not certainties <span className="slash">/</span> Research only, not betting advice</span></footer>
      </main>
      <CourtNav variant="bottom" items={tabs} active={activeTab} onNavigate={goTab} label="Main" />
      <SlipButton count={legCount} open={slipOpen} bounce={bounce} onOpen={openSlip} />
      {page !== 'coach' && <ChatLauncher open={chatOpen} busy={chat.pending !== null} unread={chatUnread} onOpen={openChat} />}
    </div>
    <SlipDrawer parlay={parlay} open={slipOpen} onClose={closeSlip} onViewParlay={() => leaveDrawerFor({ page: 'parlay' })} onTracked={onTracked} toast={slipOpen ? toastNode : null} />
    {page !== 'coach' && <ChatPanel {...conversation} open={chatOpen} onClose={closeChat} />}
    {drawer === null && toastNode}
    <div className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</div>
  </GoContext.Provider>;
}
