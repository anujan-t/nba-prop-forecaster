import { useEffect, useId, useMemo, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { ArrowLeft, ArrowRight, BookOpen, CalendarDays, CheckCircle2, Database, Info, Search, ShieldCheck, TriangleAlert, UserRound } from 'lucide-react';
import { dateLabel, decimal, pct, signed, venueLabel } from './api';
import { Basketball, Busy, EmptyState, ErrorNotice, Field, SectionHeading } from './components';
import type {
  CareerLine, OverviewStat, PlayerGame, PlayerHistory, PlayerIntelligenceProps,
  PlayerSplit, PlayerSummary, ShootingSummary,
} from './league-types';

type IntelTab = 'overview' | 'games' | 'career';
type CareerMode = 'regular_season' | 'playoffs';
const tabs: { id: IntelTab; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'games', label: 'Game log' },
  { id: 'career', label: 'Career' },
];
const overviewStats: { key: OverviewStat; label: string; short: string }[] = [
  { key: 'points', label: 'Points', short: 'PTS' },
  { key: 'rebounds', label: 'Rebounds', short: 'REB' },
  { key: 'assists', label: 'Assists', short: 'AST' },
  { key: 'minutes', label: 'Minutes', short: 'MIN' },
  { key: 'steals', label: 'Steals', short: 'STL' },
  { key: 'blocks', label: 'Blocks', short: 'BLK' },
  { key: 'three_pointers_made', label: '3-pointers made', short: '3PM' },
  { key: 'turnovers', label: 'Turnovers', short: 'TOV' },
];

function reported(value: string | number | null | undefined): string {
  return value === null || value === undefined || String(value).trim() === '' ? 'Not reported' : String(value);
}

function numberLabel(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined || !Number.isFinite(value) ? 'Not reported' : decimal(value, digits);
}

function percentageLabel(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value) ? 'Not reported' : pct(value);
}

function attempts(made: number | null | undefined, attempted: number | null | undefined): string {
  if (made == null && attempted == null) return 'Not reported';
  return `${numberLabel(made, 0)} / ${numberLabel(attempted, 0)}`;
}

function timestamp(value: string): string {
  if (!value) return 'Not reported';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short',
  });
}

function normalize(value: string): string {
  return value.normalize('NFD').replace(/\p{Diacritic}/gu, '').toLocaleLowerCase().trim();
}

function historyLabel(history: PlayerHistory): string {
  const seasons = history.queried_seasons;
  if (!seasons.length) return 'Requested history window';
  return `${seasons.length > 1 ? 'Pooled history' : 'Single season'} · ${seasons.join(' + ')}`;
}

function HistoryProvenance({ history }: { history: PlayerHistory }) {
  return <div className="intel-source-meta">
    <span><Database size={12} aria-hidden="true" />{history.cache_hit ? 'Served from local cache' : 'Freshly retrieved history'}</span>
    <span>Source fetched {timestamp(history.fetched_at)}</span>
    {history.history && <span className="intel-source-dataset">{history.history.name}</span>}
  </div>;
}

function SectionStatus({ label, loading, error, loaded }: { label: string; loading: boolean; error: string; loaded: boolean }) {
  return <div className={`intel-section-status${error ? ' intel-section-status-error' : ''}`}>
    <span>{label}</span>
    {loading ? <Busy text="Loading" /> : error ? <strong><TriangleAlert size={13} aria-hidden="true" />Request failed</strong>
      : loaded ? <strong><CheckCircle2 size={13} aria-hidden="true" />Available</strong>
        : <strong className="intel-status-muted">Not loaded</strong>}
  </div>;
}

function ShootingCard({ shooting }: { shooting: ShootingSummary }) {
  const rates: { key: keyof ShootingSummary; label: string; short: string }[] = [
    { key: 'field_goal', label: 'Field goals', short: 'FG%' },
    { key: 'three_point', label: '3-pointers', short: '3P%' },
    { key: 'free_throw', label: 'Free throws', short: 'FT%' },
  ];
  return <article className="card intel-shooting-card">
    <SectionHeading eyebrow="THE SHOT PROFILE" title="Makes over attempts"><span className="pill">Selected history</span></SectionHeading>
    <div className="intel-shooting-rates">{rates.map(rate => {
      const value = shooting[rate.key];
      return <div className="intel-shooting-rate" key={rate.key}>
        <div><span>{rate.label}</span><strong className={value === null ? 'intel-missing-value' : ''}>{percentageLabel(value)}</strong></div>
        <div className="intel-shooting-track" role="img" aria-label={`${rate.label}: ${percentageLabel(value)}`}>
          {value !== null && Number.isFinite(value) && <span style={{ width: pct(Math.max(0, Math.min(1, value))) }} />}
        </div>
        <span className="intel-shooting-short">{rate.short}</span>
      </div>;
    })}</div>
    <p className="intel-note">Rates use total makes ÷ total attempts across the window, not an average of game percentages. “Not reported” means unavailable data or no attempts—not a 0% shooting rate.</p>
  </article>;
}

function AverageCards({ summary }: { summary: PlayerSummary }) {
  return <div className="intel-averages-grid">{overviewStats.map((stat, index) => {
    const value = summary.averages[stat.key];
    return <article className={`intel-average-card${index === 0 ? ' intel-average-featured' : ''}`} key={stat.key}>
      <div className="intel-average-label"><span>{stat.label}</span><span>{stat.short}</span></div>
      <strong className={value === null ? 'intel-missing-value' : ''}>{numberLabel(value)}</strong>
      <span className="intel-average-unit">per game</span>
    </article>;
  })}</div>;
}

function SplitsTable({ splits }: { splits: PlayerSplit[] }) {
  return <article className="card intel-splits-card">
    <SectionHeading eyebrow="SAME PLAYER. DIFFERENT CONTEXT." title="The split sheet"><span className="pill">{splits.length} groups</span></SectionHeading>
    <p className="intel-note intel-note-top">Every group is provided by the backend, including recent games, venue, season, and opponents where available. Groups overlap; do not add their sample counts together.</p>
    {splits.length ? <div className="intel-table-wrap intel-splits-scroll" role="region" aria-label="All available player splits" tabIndex={0}>
      <table className="data-table intel-table"><caption className="visually-hidden">Per-game statistics and pooled shooting ratios for every returned split. Games is the sample size.</caption>
        <thead><tr><th scope="col">Split</th><th scope="col">Games</th><th scope="col">MIN/G</th><th scope="col">PTS/G</th><th scope="col">REB/G</th><th scope="col">AST/G</th><th scope="col">STL/G</th><th scope="col">BLK/G</th><th scope="col">3PM/G</th><th scope="col">TOV/G</th><th scope="col">FG%</th><th scope="col">3P%</th><th scope="col">FT%</th></tr></thead>
        <tbody>{splits.map((split, index) => <tr key={`${split.label}-${index}`}>
          <th scope="row">{split.label}</th><td>{split.games.toLocaleString()}</td>
          <td>{numberLabel(split.averages.minutes)}</td><td>{numberLabel(split.averages.points)}</td><td>{numberLabel(split.averages.rebounds)}</td>
          <td>{numberLabel(split.averages.assists)}</td><td>{numberLabel(split.averages.steals)}</td><td>{numberLabel(split.averages.blocks)}</td>
          <td>{numberLabel(split.averages.three_pointers_made)}</td><td>{numberLabel(split.averages.turnovers)}</td>
          <td>{percentageLabel(split.shooting.field_goal)}</td><td>{percentageLabel(split.shooting.three_point)}</td><td>{percentageLabel(split.shooting.free_throw)}</td>
        </tr>)}</tbody>
      </table>
    </div> : <div className="intel-inline-empty">No split groups were returned for this history window.</div>}
    <p className="intel-note">Small samples can swing sharply. These are descriptive splits—not a forecast, a matchup guarantee, or injury-adjusted projections.</p>
  </article>;
}

function CareerRow({ line, total = false }: { line: CareerLine; total?: boolean }) {
  return <tr className={total ? 'intel-career-total-row' : ''}>
    <th scope="row">{total ? 'Career · per game' : line.season}</th>
    <td>{reported(line.team)}</td><td>{line.games.toLocaleString()}</td><td>{numberLabel(line.starts, 0)}</td>
    <td>{numberLabel(line.averages.minutes)}</td><td>{numberLabel(line.averages.points)}</td><td>{numberLabel(line.averages.rebounds)}</td>
    <td>{numberLabel(line.averages.assists)}</td><td>{numberLabel(line.averages.steals)}</td><td>{numberLabel(line.averages.blocks)}</td>
    <td>{numberLabel(line.averages.three_pointers_made)}</td><td>{numberLabel(line.averages.turnovers)}</td>
    <td>{percentageLabel(line.shooting.field_goal)}</td><td>{percentageLabel(line.shooting.three_point)}</td><td>{percentageLabel(line.shooting.free_throw)}</td>
  </tr>;
}

function CareerTable({ lines, total, label }: { lines: CareerLine[]; total: CareerLine | null; label: string }) {
  return <div className="intel-table-wrap intel-career-scroll" role="region" aria-label={`${label} career statistics`} tabIndex={0}>
    <table className="data-table intel-table"><caption className="visually-hidden">{label} per-season statistics, with career per-game rates. Games and starts are counts, not per-game averages.</caption>
      <thead><tr><th scope="col">Season</th><th scope="col">Team</th><th scope="col">Games</th><th scope="col">Starts</th><th scope="col">MIN/G</th><th scope="col">PTS/G</th><th scope="col">REB/G</th><th scope="col">AST/G</th><th scope="col">STL/G</th><th scope="col">BLK/G</th><th scope="col">3PM/G</th><th scope="col">TOV/G</th><th scope="col">FG%</th><th scope="col">3P%</th><th scope="col">FT%</th></tr></thead>
      <tbody>{lines.map((line, index) => <CareerRow key={`${line.season}-${line.team}-${index}`} line={line} />)}</tbody>
      {total && <tfoot><CareerRow line={total} total /></tfoot>}
    </table>
  </div>;
}

function GameRow({ game, full }: { game: PlayerGame; full: boolean }) {
  return <tr>
    <th scope="row">{dateLabel(game.date, true)}</th>
    {full && <td>{reported(game.season)}</td>}
    <td>{reported(game.team)}</td>
    <td><strong>{reported(game.opponent)}</strong><span className="intel-game-venue">{venueLabel(game.home)}</span></td>
    <td>{numberLabel(game.minutes)}</td><td className="intel-game-points">{numberLabel(game.points, 0)}</td>
    <td>{numberLabel(game.rebounds, 0)}</td><td>{numberLabel(game.assists, 0)}</td><td>{numberLabel(game.steals, 0)}</td><td>{numberLabel(game.blocks, 0)}</td>
    <td>{numberLabel(game.three_pointers_made, 0)}</td><td>{numberLabel(game.turnovers, 0)}</td>
    {full && <><td>{attempts(game.field_goals_made, game.field_goals_attempted)}</td><td>{attempts(game.three_pointers_made, game.three_pointers_attempted)}</td><td>{attempts(game.free_throws_made, game.free_throws_attempted)}</td>
      <td>{numberLabel(game.offensive_rebounds, 0)}</td><td>{numberLabel(game.defensive_rebounds, 0)}</td><td>{numberLabel(game.fouls, 0)}</td></>}
    <td>{game.plus_minus == null ? 'Not reported' : signed(game.plus_minus, 0)}</td>
    <td><span className={game.win === 'W' ? 'intel-game-win' : ''}>{reported(game.win)}</span></td>
    {full && <><td>{numberLabel(game.rest_days, 0)}</td><td><details className="intel-record-details"><summary aria-label={`Record identifiers for ${dateLabel(game.date, true)}`}>Record IDs</summary>
      <dl><dt>Game ID</dt><dd>{reported(game.game_id)}</dd><dt>Player ID</dt><dd>{reported(game.player_id)}</dd><dt>Recorded name</dt><dd>{reported(game.player_name)}</dd></dl>
    </details></td></>}
  </tr>;
}

export default function PlayerIntelligence({
  player, profile, profileLoading, profileError, career, careerLoading, careerError,
  history, historyLoading, historyError,
}: PlayerIntelligenceProps) {
  const [tab, setTab] = useState<IntelTab>('overview');
  const [careerMode, setCareerMode] = useState<CareerMode>('regular_season');
  const [gameColumns, setGameColumns] = useState<'compact' | 'full'>('compact');
  const [gameSearch, setGameSearch] = useState('');
  const [page, setPage] = useState(1);
  const instanceId = useId();
  const playerId = player?.id ?? profile?.id ?? history?.player_id ?? career?.player_id ?? '';
  const profileData = profile?.id === playerId ? profile : null;
  const historyData = history?.player_id === playerId ? history : null;
  const careerData = career?.player_id === playerId ? career : null;
  const playerName = profileData?.name || player?.name || 'Player intelligence';
  const currentTeam = player?.team_name || player?.team || profileData?.team_name || profileData?.team || 'Not reported';
  const monogram = playerName.split(/\s+/).filter(Boolean).slice(0, 2).map(part => part.charAt(0)).join('').toLocaleUpperCase();
  const games = useMemo(() => [...(historyData?.games ?? [])].sort((a, b) =>
    b.date.localeCompare(a.date) || (b.game_id ?? '').localeCompare(a.game_id ?? '')), [historyData?.games]);
  const filteredGames = useMemo(() => {
    const terms = normalize(gameSearch).split(/\s+/).filter(Boolean);
    return games.filter(game => terms.every(term =>
      normalize(`${game.date} ${game.team} ${game.opponent} ${game.season ?? ''} ${venueLabel(game.home)} ${game.win ?? ''}`).includes(term)));
  }, [games, gameSearch]);
  const careerLines = useMemo(() => [...(careerData?.[careerMode] ?? [])].sort((a, b) =>
    b.season.localeCompare(a.season)), [careerData, careerMode]);
  const careerTotal = careerData?.totals[careerMode] ?? null;
  const maxPages = Math.max(1, Math.ceil(filteredGames.length / 25));
  const visiblePage = Math.min(page, maxPages);
  const pageGames = filteredGames.slice((visiblePage - 1) * 25, visiblePage * 25);
  const historyCount = historyData?.summary?.games ?? games.length;
  const full = gameColumns === 'full';

  useEffect(() => { setPage(1); }, [history]);
  useEffect(() => {
    setTab('overview');
    setPage(1);
    setGameSearch('');
    setCareerMode('regular_season');
  }, [playerId]);

  function handleTabKey(event: KeyboardEvent<HTMLButtonElement>, current: IntelTab) {
    const index = tabs.findIndex(item => item.id === current);
    let next: IntelTab;
    if (event.key === 'ArrowRight') next = tabs[(index + 1) % tabs.length].id;
    else if (event.key === 'ArrowLeft') next = tabs[(index - 1 + tabs.length) % tabs.length].id;
    else if (event.key === 'Home') next = tabs[0].id;
    else if (event.key === 'End') next = tabs[tabs.length - 1].id;
    else return;
    event.preventDefault();
    setTab(next);
    document.getElementById(`${instanceId}-tab-${next}`)?.focus();
  }

  if (!playerId && !profileLoading && !careerLoading && !historyLoading && !profileError && !careerError && !historyError) {
    return <section className="card intel-no-selection"><EmptyState icon={<UserRound size={32} />} title="Every player starts with a question.">Select any player from the league directory above. Profile, career, and completed game history load independently—there is no five-player shortlist.</EmptyState></section>;
  }

  const weight = profileData?.weight?.trim();
  const weightLabel = weight ? /\b(?:lb|lbs|pounds)\b/i.test(weight) ? weight : `${weight} lb` : 'Not reported';
  const experience = profileData?.experience;
  const experienceLabel = experience == null ? 'Not reported' : `${numberLabel(experience, 0)} ${experience === 1 ? 'season' : 'seasons'}`;
  const profileFacts = [
    { label: 'Position', value: reported(profileData?.position) },
    { label: 'Jersey', value: reported(profileData?.jersey) },
    { label: 'Age', value: profileData?.age == null ? 'Not reported' : `${numberLabel(profileData.age, 0)} years` },
    { label: 'Date of birth', value: profileData?.birth_date ? dateLabel(profileData.birth_date, true) : 'Not reported' },
    { label: 'Height', value: reported(profileData?.height) },
    { label: 'Weight', value: weightLabel },
    { label: 'NBA experience', value: experienceLabel },
    { label: 'Country', value: reported(profileData?.country) },
    { label: 'School / prior program', value: reported(profileData?.school) },
    { label: 'Draft year', value: reported(profileData?.draft_year) },
    { label: 'Draft round', value: reported(profileData?.draft_round) },
    { label: 'Draft pick', value: reported(profileData?.draft_pick) },
    { label: 'Roster status', value: reported(profileData?.roster_status) },
    { label: 'NBA profile team', value: reported(profileData?.team_name || profileData?.team) },
  ];

  return <section className="intel-workspace" aria-label={`${playerName} player intelligence`}>
    <article className="card intel-profile-hero">
      <Basketball className="intel-profile-watermark" />
      <div className="intel-profile-identity">
        <div className="intel-player-monogram" aria-hidden="true">{playerId ? monogram : <UserRound size={34} />}</div>
        <div className="intel-profile-name">
          <span className="eyebrow">PLAYER INTELLIGENCE{playerId && <> <span className="slash">/</span> NBA ID {playerId}</>}</span>
          <h2>{playerName}</h2>
          <p><span className="intel-current-team-label">Current roster team</span><strong>{currentTeam}</strong></p>
        </div>
      </div>
      <div className="intel-profile-highlights">
        <div><span>Position</span><strong>{reported(profileData?.position)}</strong></div>
        <div><span>Jersey</span><strong>{profileData?.jersey == null || profileData.jersey.trim() === '' ? 'Not reported' : `#${profileData.jersey.replace(/^#/, '')}`}</strong></div>
        <div><span>Experience</span><strong>{experienceLabel}</strong></div>
      </div>
      <div className="intel-profile-disclosure"><ShieldCheck size={14} aria-hidden="true" /><span>Roster status is not injury status. This app does not provide medical availability, confirmed lineups, or a live odds feed.</span></div>
    </article>

    <div className="intel-status-grid" aria-label="Independent player data sources">
      <SectionStatus label="Player profile" loading={profileLoading} error={profileError} loaded={Boolean(profileData)} />
      <SectionStatus label="Completed game history" loading={historyLoading} error={historyError} loaded={Boolean(historyData)} />
      <SectionStatus label="NBA career" loading={careerLoading} error={careerError} loaded={Boolean(careerData)} />
    </div>
    {profileError && <div className="intel-request-error"><ErrorNotice message={`Player profile: ${profileError}`} /></div>}
    {historyError && <div className="intel-request-error"><ErrorNotice message={`Game history: ${historyError}`} /></div>}
    {careerError && <div className="intel-request-error"><ErrorNotice message={`NBA career: ${careerError}`} /></div>}

    {historyData?.status === 'limited' && <div className="intel-history-notice intel-limited-notice" role="status">
      <TriangleAlert size={18} aria-hidden="true" /><div><strong>A young sample deserves an honest limit.</strong>
        <p>{historyCount} completed {historyCount === 1 ? 'game is' : 'games are'} available in the requested window. All reported history and profile details remain available; forecasting needs at least 10 completed games. No forecast is inferred from an undersized sample.</p></div>
    </div>}
    {historyData?.status === 'empty' && <div className="intel-history-notice" role="status"><CalendarDays size={18} aria-hidden="true" />
      <div><strong>No completed games in this requested window.</strong><p>{historyLabel(historyData)}. This is not an invalid player or a claim that the player has no NBA career. Profile and career information remain available independently.</p></div>
    </div>}
    {historyData && historyData.warnings.length > 0 && <div className="intel-history-notice intel-provider-warnings" role="status"><Info size={18} aria-hidden="true" /><div><strong>History notes & limitations</strong><ul>{historyData.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></div></div>}

    <div className="intel-tabs" role="tablist" aria-label="Player intelligence sections">
      {tabs.map(item => <button type="button" role="tab" key={item.id} id={`${instanceId}-tab-${item.id}`}
        aria-controls={`${instanceId}-panel-${item.id}`} aria-selected={tab === item.id} tabIndex={tab === item.id ? 0 : -1}
        onClick={() => setTab(item.id)} onKeyDown={event => handleTabKey(event, item.id)} className={tab === item.id ? 'intel-tab-active' : ''}>
        {item.label}{item.id === 'games' && historyData && <span>{games.length.toLocaleString()}</span>}
      </button>)}
    </div>

    <div id={`${instanceId}-panel-overview`} role="tabpanel" aria-labelledby={`${instanceId}-tab-overview`} hidden={tab !== 'overview'} tabIndex={0} className="intel-tab-panel">
      <div className="intel-overview-grid">
        <div className="intel-overview-performance">
          <article className="card intel-history-summary">
            <SectionHeading eyebrow="COMPLETED GAMES, NOT PREDICTIONS" title="The player, by the numbers">
              {historyData && <span className="pill">{historyCount.toLocaleString()} {historyCount === 1 ? 'game' : 'games'}</span>}
            </SectionHeading>
            {historyLoading && <div className="intel-inline-loading"><Busy text={historyData ? 'Refreshing history; the displayed snapshot has not changed yet.' : 'Loading the requested game history…'} /></div>}
            {historyData && <div className="intel-history-window"><span><CalendarDays size={14} aria-hidden="true" />{historyLabel(historyData)}</span>
              {games.length > 0 && <p>{dateLabel(games[games.length - 1].date, true)} — {dateLabel(games[0].date, true)}</p>}
            </div>}
            {historyData?.summary && historyData.summary.games > 0 ? <>
              <AverageCards summary={historyData.summary} />
              <p className="intel-note">{historyData.queried_seasons.length > 1 ? 'These are pooled per-game averages across the listed seasons, not current-season averages.' : 'These are per-game averages for the single requested season.'} A loaded sample is not proof of predictive accuracy.</p>
            </> : !historyLoading && <EmptyState icon={<BookOpen size={29} />} title={historyData?.status === 'empty' ? 'No game sample for this window.' : historyError ? 'Game history could not be loaded.' : 'No summary has been returned.'}>
              {historyData?.status === 'empty' ? 'Choose another season or the pooled Recent window above. The player’s profile and career are separate records.' : historyError ? 'The error above applies only to this request. Available profile and career information are still accessible.' : 'Averages appear only when the provider returns completed game data. Missing figures are not replaced with zero.'}
            </EmptyState>}
            {historyData && <HistoryProvenance history={historyData} />}
          </article>
          {historyData?.summary && historyData.summary.games > 0 && <ShootingCard shooting={historyData.summary.shooting} />}
        </div>

        <article className="card intel-biography">
          <SectionHeading eyebrow="THE PERSON BEHIND THE BOX SCORE" title="Player file"><UserRound size={19} className="subtle" aria-hidden="true" /></SectionHeading>
          {profileLoading && <div className="intel-inline-loading"><Busy text={profileData ? 'Refreshing the profile snapshot…' : 'Loading biographical details…'} /></div>}
          {!profileData && !profileLoading && <p className="intel-note intel-note-top">Biographical details are not available in the current response. The name and current team above may come from the roster directory; they do not supply missing profile facts.</p>}
          <dl className="intel-facts">{profileFacts.map(fact => <div key={fact.label}><dt>{fact.label}</dt><dd className={fact.value === 'Not reported' ? 'intel-unreported' : ''}>{fact.value}</dd></div>)}</dl>
          {profileData && <details className="intel-season-catalog"><summary>Provider season catalog <span>{profileData.available_seasons.length}</span></summary>
            {profileData.available_seasons.length ? <div className="intel-season-pills">{[...new Set(profileData.available_seasons)].map(season => <span key={season}>{season}</span>)}</div> : <p className="intel-note">No season catalog was returned. This does not establish that the player has no career history.</p>}
            <p className="intel-note">Catalog availability does not imply a full game sample in every competition or season.</p>
          </details>}
          {profileData && <div className="intel-source-meta"><span>{profileData.provider || 'Provider not reported'}</span><span>Profile fetched {timestamp(profileData.fetched_at)}</span></div>}
        </article>
      </div>
      {historyData && <SplitsTable splits={historyData.splits} />}
    </div>

    <div id={`${instanceId}-panel-games`} role="tabpanel" aria-labelledby={`${instanceId}-tab-games`} hidden={tab !== 'games'} tabIndex={0} className="intel-tab-panel">
      <article className="card intel-game-log">
        <SectionHeading eyebrow="EVERY RETURNED GAME, FULLY ACCESSIBLE" title="The completed-game ledger"><span className="pill">Newest first</span></SectionHeading>
        {historyLoading && <div className="intel-inline-loading"><Busy text={historyData ? 'Refreshing history; the displayed ledger is the previous snapshot.' : 'Loading completed game logs…'} /></div>}
        {historyData && <div className="intel-history-window"><span><CalendarDays size={14} aria-hidden="true" />{historyLabel(historyData)}</span><p>Team columns reflect each historical game—not necessarily the player’s current roster team.</p></div>}
        {games.length > 0 && <>
          <div className="intel-log-toolbar">
            <Field label="Filter game logs" className="intel-log-search-field"><div className="intel-log-search"><Search size={15} aria-hidden="true" /><input type="search" value={gameSearch} placeholder="Opponent, team, season, or date…" onChange={event => { setGameSearch(event.target.value); setPage(1); }} /></div></Field>
            <div className="intel-column-filter"><span>Box score detail</span><div className="segmented" aria-label="Game log columns">
              <button type="button" aria-pressed={!full} className={!full ? 'selected' : ''} onClick={() => setGameColumns('compact')}>Core stats</button>
              <button type="button" aria-pressed={full} className={full ? 'selected' : ''} onClick={() => setGameColumns('full')}>Full box score</button>
            </div></div>
          </div>
          <div className="intel-log-count" role="status"><span>{filteredGames.length.toLocaleString()} matching / {games.length.toLocaleString()} returned games</span>{gameSearch && <button type="button" className="text-button" onClick={() => { setGameSearch(''); setPage(1); }}>Clear game filters</button>}</div>
          {filteredGames.length ? <>
            <div className="intel-table-wrap intel-game-table-wrap" role="region" aria-label={`${full ? 'Full box scores' : 'Core game statistics'}, page ${visiblePage} of ${maxPages}`} tabIndex={0}>
              <table className="data-table intel-table intel-game-table"><caption className="visually-hidden">Completed games, newest first. Missing optional values are not reported, not zero. Home or away refers to the venue in that game.</caption>
                <thead><tr><th scope="col">Date</th>{full && <th scope="col">Season</th>}<th scope="col">Team</th><th scope="col">Opponent / venue</th>
                  <th scope="col">MIN</th><th scope="col">PTS</th><th scope="col">REB</th><th scope="col">AST</th><th scope="col">STL</th><th scope="col">BLK</th><th scope="col">3PM</th><th scope="col">TOV</th>
                  {full && <><th scope="col">FGM / FGA</th><th scope="col">3PM / 3PA</th><th scope="col">FTM / FTA</th><th scope="col">OREB</th><th scope="col">DREB</th><th scope="col">PF</th></>}
                  <th scope="col">+/−</th><th scope="col">W / L</th>{full && <><th scope="col">Rest days</th><th scope="col">Source record</th></>}
                </tr></thead>
                <tbody>{pageGames.map(game => <GameRow game={game} full={full} key={game.game_id || `${game.player_id}-${game.date}`} />)}</tbody>
              </table>
            </div>
            <div className="intel-pagination">
              <span>Showing {((visiblePage - 1) * 25 + 1).toLocaleString()}–{Math.min(visiblePage * 25, filteredGames.length).toLocaleString()} of {filteredGames.length.toLocaleString()}</span>
              <div><button type="button" className="icon-button" disabled={visiblePage <= 1} onClick={() => setPage(visiblePage - 1)} aria-label="Previous page of game logs" title="Previous page"><ArrowLeft size={15} /></button>
                <label><span className="visually-hidden">Game log page</span><select value={visiblePage} onChange={event => setPage(Number(event.target.value))}>{Array.from({ length: maxPages }, (_, index) => <option value={index + 1} key={index + 1}>Page {index + 1} of {maxPages}</option>)}</select></label>
                <button type="button" className="icon-button" disabled={visiblePage >= maxPages} onClick={() => setPage(visiblePage + 1)} aria-label="Next page of game logs" title="Next page"><ArrowRight size={15} /></button>
              </div>
            </div>
          </> : <EmptyState icon={<Search size={28} />} title="No games match these filters." action={<button type="button" className="secondary-button" onClick={() => { setGameSearch(''); setPage(1); }}>Clear game filters</button>}>The full returned history is still available. Clear the filter to see every game.</EmptyState>}
          <p className="intel-note">25 games per page; no rows are discarded. Full box score exposes shooting attempts, offensive/defensive rebounds, fouls, rest, and source identifiers. Rest counts days between games: 0 means back-to-back.</p>
        </>}
        {!games.length && !historyLoading && <EmptyState icon={<CalendarDays size={30} />} title={historyError ? 'This game-history request failed.' : 'No completed games in the requested window.'}>{historyError ? 'Profile and career requests are independent. Refresh the player above to retry the game history.' : 'Try Recent or a different season. An empty window does not imply that this player has no NBA career.'}</EmptyState>}
        {historyData && <HistoryProvenance history={historyData} />}
      </article>
    </div>

    <div id={`${instanceId}-panel-career`} role="tabpanel" aria-labelledby={`${instanceId}-tab-career`} hidden={tab !== 'career'} tabIndex={0} className="intel-tab-panel">
      <article className="card intel-career-card">
        <SectionHeading eyebrow="THE LONGER VIEW" title="A career in context"><BookOpen size={19} className="subtle" aria-hidden="true" /></SectionHeading>
        <div className="intel-career-toolbar">
          <div className="segmented" aria-label="Career competition">
            <button type="button" aria-pressed={careerMode === 'regular_season'} className={careerMode === 'regular_season' ? 'selected' : ''} onClick={() => setCareerMode('regular_season')}>Regular season{careerData && <span>{careerData.regular_season.length}</span>}</button>
            <button type="button" aria-pressed={careerMode === 'playoffs'} className={careerMode === 'playoffs' ? 'selected' : ''} onClick={() => setCareerMode('playoffs')}>Playoffs{careerData && <span>{careerData.playoffs.length}</span>}</button>
          </div>
          <span className="intel-career-rate-note">Counting stats below are per game.</span>
        </div>
        {careerLoading && <div className="intel-inline-loading"><Busy text={careerData ? 'Refreshing career data; the displayed snapshot is unchanged.' : 'Loading NBA career records…'} /></div>}
        {careerTotal && <div className="intel-career-summary">
          <div><span>Career games</span><strong>{careerTotal.games.toLocaleString()}</strong><small>{careerMode === 'regular_season' ? 'Regular season' : 'Playoffs'} count</small></div>
          <div><span>Career points</span><strong className={careerTotal.averages.points === null ? 'intel-missing-value' : ''}>{numberLabel(careerTotal.averages.points)}</strong><small>per game · not total points</small></div>
          <div><span>Career rebounds</span><strong className={careerTotal.averages.rebounds === null ? 'intel-missing-value' : ''}>{numberLabel(careerTotal.averages.rebounds)}</strong><small>per game</small></div>
          <div><span>Career assists</span><strong className={careerTotal.averages.assists === null ? 'intel-missing-value' : ''}>{numberLabel(careerTotal.averages.assists)}</strong><small>per game</small></div>
        </div>}
        {(careerLines.length > 0 || careerTotal) && <>
          <CareerTable lines={careerLines} total={careerTotal} label={careerMode === 'regular_season' ? 'Regular season' : 'Playoffs'} />
          <p className="intel-note">The backend uses one aggregated TOT row for a traded season where available, rather than counting team stints twice. Season teams are historical. The career aggregate’s averages remain per-game rates; only games and starts are counts.</p>
        </>}
        {!careerLines.length && !careerTotal && !careerLoading && <EmptyState icon={<BookOpen size={30} />} title={careerError ? 'Career records could not be loaded.' : `No ${careerMode === 'regular_season' ? 'regular-season' : 'playoff'} career rows returned.`}>{careerError ? 'This failure does not invalidate the player or remove any game history already available.' : 'The provider returned no rows for this competition. This is not a claim about eligibility, health, or a complete absence of NBA experience.'}</EmptyState>}
        {careerData && <div className="intel-source-meta"><span>{careerData.provider || 'Provider not reported'}</span><span>Career fetched {timestamp(careerData.fetched_at)}</span></div>}
      </article>
    </div>
    <div className="intel-workspace-disclaimer"><Info size={14} aria-hidden="true" /><p>Profile, roster, game history, and career records can have different fetch times. These are historical research records—not live performance, injury, lineup, or wagering guidance.</p></div>
  </section>;
}
