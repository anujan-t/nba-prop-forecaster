import { ArrowLeft, ArrowUpRight, Clapperboard, ClipboardList, ExternalLink, HeartPulse, Newspaper, RefreshCw, Repeat2, Search, Shield, Stethoscope, Users, X } from 'lucide-react';
import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { dateLabel, timeAgo } from './api';
import { Busy, EmptyState, ErrorNotice, SectionHeading } from './components';
import type { CoachEntry, Feed, Headlines, InjuryReport, NewsArticle, NewsKind, PlayType, SeasonProfile, TeamDetail, TeamMetric, TeamProfile, TeamStyles, Transaction, Transactions } from './news-types';
import { RouteLink } from './RouteLink';
import type { NewsSection } from './routes';
import type { InjuryEntry } from './types';
import { useFeed } from './useFeed';

export const kindNames: Record<NewsKind, string> = {
  injury: 'Injuries & availability', transaction: 'Roster moves', coaching: 'Coaches & front office', game: 'Games',
  betting: 'Betting & fantasy', other: 'Around the league',
};
const kindShort: Record<NewsKind, string> = {
  injury: 'Injury', transaction: 'Roster move', coaching: 'Coaching', game: 'Game', betting: 'Betting', other: 'League',
};
const statusNames: Record<InjuryEntry['status_key'], string> = {
  out: 'Out', suspended: 'Suspended', doubtful: 'Doubtful', questionable: 'Questionable', day_to_day: 'Day-to-day', probable: 'Probable', other: 'Listed',
};
const statusFilters: { id: 'all' | 'out' | 'uncertain' | 'day_to_day'; label: string; keys: InjuryEntry['status_key'][] }[] = [
  { id: 'all', label: 'All', keys: ['out', 'suspended', 'doubtful', 'questionable', 'day_to_day', 'probable', 'other'] },
  { id: 'out', label: 'Out', keys: ['out', 'suspended'] },
  { id: 'uncertain', label: 'Doubtful / questionable', keys: ['doubtful', 'questionable'] },
  { id: 'day_to_day', label: 'Day-to-day', keys: ['day_to_day', 'probable', 'other'] },
];
/** Opponent metrics where rank 1 means the team allows the least. */
const allowedMetrics = ['opponent_points', 'opponent_threes', 'opponent_rebounds', 'opponent_assists', 'opponent_efg'];
const styleMetrics = ['pace', 'offense', 'defense', 'net', 'three_rate', 'paint_share', 'fast_break_share', 'assisted', 'free_throw_rate',
  'turnover_rate', 'offensive_rebounding', 'forced_turnovers'];

function ordinal(value: number): string {
  const suffix = value % 100 >= 11 && value % 100 <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[value % 10] ?? 'th';
  return `${value}${suffix}`;
}

/** A date with its year when it is not this year (an estimated return next July should say so). */
function dayLabel(value: string): string {
  const year = Number(value.slice(0, 4));
  return dateLabel(value, Number.isFinite(year) && year !== new Date().getFullYear());
}

function metricValue(metric: TeamMetric): string {
  return metric.percent ? `${(metric.value * 100).toFixed(1)}%` : metric.value.toFixed(1);
}

export function InjuryBadge({ entry, compact = false }: { entry: InjuryEntry; compact?: boolean }) {
  const what = [entry.side, entry.injury].filter(Boolean).join(' ');
  return <span className={`injury-badge status-${entry.status_key}`} title={entry.short_comment ?? undefined}>
    <HeartPulse size={13} aria-hidden="true" /><strong>{statusNames[entry.status_key] ?? entry.status}</strong>{!compact && what && <span>{what}</span>}
  </span>;
}

function FeedMeta({ feed, loading, onRefresh, label }: { feed: Feed | null; loading: boolean; onRefresh: () => void; label: string }) {
  return <div className="news-meta">
    <span>{feed ? <>Updated {timeAgo(feed.fetched_at)} · {feed.provider}</> : 'Connecting to the feed…'}</span>
    <button type="button" className="text-button" onClick={onRefresh} aria-disabled={loading || undefined} aria-label={`Refresh ${label}`}>
      <RefreshCw size={14} aria-hidden="true" className={loading ? 'spin' : ''} />{loading ? 'Refreshing…' : 'Refresh'}
    </button>
  </div>;
}

function StaleNotice({ feed }: { feed: Feed | null }) {
  if (!feed?.stale) return null;
  return <div className="notice warning-notice" role="status"><HeartPulse size={18} aria-hidden="true" /><div><strong>Showing the last saved copy</strong><p>{feed.error}</p></div></div>;
}

function TeamChip({ team }: { team: string }) {
  return <RouteLink className="team-chip" to={{ page: 'news', section: 'teams', team }} aria-label={`${team} team page`}>{team}</RouteLink>;
}

function PlayerLink({ name, playerId, children }: { name: string; playerId: string | null; children?: ReactNode }) {
  if (!playerId) return <>{children ?? name}</>;
  return <RouteLink className="player-link" to={{ page: 'players', playerId, section: 'props' }} title={`Open ${name} in Players`}>{children ?? name}</RouteLink>;
}

function ArticleCard({ article, lead = false }: { article: NewsArticle; lead?: boolean }) {
  const headline = article.url
    ? <a href={article.url} target="_blank" rel="noopener noreferrer">{article.headline}<ExternalLink size={14} aria-hidden="true" /><span className="sr-only"> (opens ESPN in a new tab)</span></a>
    : article.headline;
  return <article className={`news-card kind-${article.kind}${lead ? ' lead' : ''}${article.image ? ' has-image' : ''}`}>
    {article.image && <div className="news-image"><img src={article.image} alt="" loading="lazy" referrerPolicy="no-referrer" />{article.video && <span className="news-video"><Clapperboard size={14} aria-hidden="true" />Video</span>}</div>}
    <div className="news-body">
      <div className="news-kicker"><span className={`kind-pill kind-${article.kind}`}>{kindShort[article.kind]}</span><time dateTime={article.published ?? undefined}>{timeAgo(article.published)}</time>{article.byline && <span className="news-byline">{article.byline}</span>}</div>
      <h3>{headline}</h3>
      {article.description && <p>{article.description}</p>}
      {(article.teams.length > 0 || article.athletes.length > 0) && <div className="news-tags">
        {article.teams.map(team => <TeamChip key={team} team={team} />)}
        {article.athletes.map(athlete => athlete.player_id
          ? <PlayerLink key={athlete.name} name={athlete.name} playerId={athlete.player_id}><span className="athlete-chip">{athlete.name}<ArrowUpRight size={12} aria-hidden="true" /></span></PlayerLink>
          : <span key={athlete.name} className="athlete-chip muted">{athlete.name}</span>)}
      </div>}
    </div>
  </article>;
}

function HeadlinesSection({ active, teams }: { active: boolean; teams: string[] }) {
  const feed = useFeed<Headlines>('/api/news/headlines', active);
  const [kind, setKind] = useState<NewsKind | 'all'>('all');
  const [team, setTeam] = useState('');
  const [videos, setVideos] = useState(true);
  const data = feed.data;
  const shown = useMemo(() => (data?.articles ?? []).filter(article => (kind === 'all' || article.kind === kind)
    && (!team || article.teams.includes(team)) && (videos || !article.video)), [data, kind, team, videos]);
  const lead = shown.find(article => article.image && !article.video) ?? null;
  return <section className="card news-panel" aria-labelledby="news-headlines-title">
    <SectionHeading id="news-headlines-title" eyebrow="THE WIRE" title="Latest NBA news"><Newspaper size={21} className="subtle" aria-hidden="true" /></SectionHeading>
    <FeedMeta feed={data} loading={feed.loading} onRefresh={() => feed.refresh(true)} label="headlines" />
    <StaleNotice feed={data} />
    {feed.error && <ErrorNotice message={feed.error} onRetry={() => feed.refresh()} />}
    <div className="news-toolbar">
      <div className="segmented small news-kinds" role="group" aria-label="Kind of news">
        <button type="button" aria-pressed={kind === 'all'} onClick={() => setKind('all')}>All{data ? ` ${data.articles.length}` : ''}</button>
        {(Object.keys(kindNames) as NewsKind[]).map(id => <button key={id} type="button" aria-pressed={kind === id} onClick={() => setKind(id)}>{kindNames[id]}{data ? ` ${data.counts[id] ?? 0}` : ''}</button>)}
      </div>
      <label className="news-select"><span>Team</span><select value={team} onChange={event => setTeam(event.target.value)}><option value="">All teams</option>{teams.map(code => <option key={code} value={code}>{code}</option>)}</select></label>
      <label className="news-check"><input type="checkbox" checked={videos} onChange={event => setVideos(event.target.checked)} />Videos</label>
    </div>
    {!data && feed.loading && <Busy text="Reading the wire…" />}
    {data && !shown.length && <EmptyState title="Nothing matches these filters" icon={<Newspaper size={30} />}>Try another kind of news or all teams.</EmptyState>}
    {lead && <ArticleCard article={lead} lead />}
    <div className="news-list">{shown.filter(article => article !== lead).map(article => <ArticleCard key={article.id} article={article} />)}</div>
  </section>;
}

function InjuryRow({ entry }: { entry: InjuryEntry }) {
  const what = [entry.side, entry.injury, entry.detail && entry.detail !== entry.injury ? entry.detail.toLowerCase() : null].filter(Boolean).join(' ');
  return <tr className={`status-row status-${entry.status_key}`}>
    <th scope="row"><PlayerLink name={entry.player} playerId={entry.player_id} />{entry.position && <span className="injury-position">{entry.position}</span>}</th>
    <td><InjuryBadge entry={entry} compact /></td>
    <td>{what || '—'}</td>
    <td>{entry.return_date ? dayLabel(entry.return_date) : '—'}</td>
    <td className="injury-note">{entry.short_comment ?? '—'}{entry.long_comment && <details><summary>More</summary><p>{entry.long_comment}</p></details>}<small>{entry.date ? `Updated ${timeAgo(entry.date)}` : ''}</small></td>
  </tr>;
}

function InjuryTable({ entries, caption }: { entries: InjuryEntry[]; caption: string }) {
  return <div className="table-scroll" tabIndex={0} role="region" aria-label={caption}>
    <table className="data-table injury-table"><caption className="sr-only">{caption}</caption>
      <thead><tr><th scope="col">Player</th><th scope="col">Status</th><th scope="col">Injury</th><th scope="col">Est. return</th><th scope="col">Latest</th></tr></thead>
      <tbody>{entries.map(entry => <InjuryRow key={`${entry.player}-${entry.team}`} entry={entry} />)}</tbody>
    </table>
  </div>;
}

function InjuriesSection({ active }: { active: boolean }) {
  const feed = useFeed<InjuryReport>('/api/news/injuries', active);
  const [status, setStatus] = useState<(typeof statusFilters)[number]['id']>('all');
  const [query, setQuery] = useState('');
  const data = feed.data;
  const keys = statusFilters.find(filter => filter.id === status)?.keys ?? [];
  const needle = query.trim().toLocaleLowerCase();
  const teams = useMemo(() => (data?.teams ?? []).map(team => ({
    ...team, injuries: team.injuries.filter(entry => keys.includes(entry.status_key)
      && (!needle || entry.player.toLocaleLowerCase().includes(needle) || team.team.toLocaleLowerCase() === needle || team.team_name.toLocaleLowerCase().includes(needle))),
  })).filter(team => team.injuries.length), [data, keys, needle]);
  const count = (ids: InjuryEntry['status_key'][]) => ids.reduce((sum, id) => sum + (data?.counts[id] ?? 0), 0);
  return <section className="card news-panel" aria-labelledby="news-injuries-title">
    <SectionHeading id="news-injuries-title" eyebrow="THE TRAINER'S ROOM" title="League injury report"><Stethoscope size={21} className="subtle" aria-hidden="true" /></SectionHeading>
    <FeedMeta feed={data} loading={feed.loading} onRefresh={() => feed.refresh(true)} label="injury report" />
    <StaleNotice feed={data} />
    {feed.error && <ErrorNotice message={feed.error} onRetry={() => feed.refresh()} />}
    {data && <div className="injury-summary">
      {statusFilters.map(filter => <button key={filter.id} type="button" className={`injury-count count-${filter.id}${status === filter.id ? ' selected' : ''}`} aria-pressed={status === filter.id} onClick={() => setStatus(filter.id)}>
        <strong>{filter.id === 'all' ? data.total : count(filter.keys)}</strong><span>{filter.id === 'all' ? 'players listed' : filter.label}</span>
      </button>)}
    </div>}
    <div className="news-toolbar">
      <label className="league-search-control news-search"><Search size={16} aria-hidden="true" /><input type="search" value={query} onChange={event => setQuery(event.target.value)} placeholder="Search a player or team" aria-label="Search the injury report" />
        {query && <button type="button" onClick={() => setQuery('')} aria-label="Clear search"><X size={14} aria-hidden="true" /></button>}</label>
    </div>
    <p className="news-footnote">Statuses come from team and media reports and change through the day. COURTSIDE's forecasts assume the player plays; check here before you build a slip.</p>
    {!data && feed.loading && <Busy text="Checking the trainer's room…" />}
    {data && !teams.length && <EmptyState title="No listed players match" icon={<Stethoscope size={30} />}>Try another status or clear the search.</EmptyState>}
    <div className="injury-teams">{teams.map(team => <section key={team.team} className="injury-team" aria-labelledby={`injury-${team.team}`}>
      <div className="injury-team-head"><h3 id={`injury-${team.team}`}>{team.team_name}</h3><span className="pill">{team.injuries.length} listed</span><RouteLink className="text-button" to={{ page: 'news', section: 'teams', team: team.team }}>Team page <ArrowUpRight size={13} aria-hidden="true" /></RouteLink></div>
      <InjuryTable entries={team.injuries} caption={`${team.team_name} injuries`} />
    </section>)}</div>
  </section>;
}

function rankText(key: string, metric: TeamMetric, count: number): string {
  if (allowedMetrics.includes(key)) return metric.rank <= count / 2 ? `${ordinal(metric.rank)} fewest` : `${ordinal(count + 1 - metric.rank)} most`;
  return ordinal(metric.rank);
}

function RankBar({ rank, count }: { rank: number; count: number }) {
  const strength = (count - rank + 1) / count;
  return <span className="rank-bar" aria-hidden="true"><span style={{ width: `${Math.max(4, strength * 100)}%` }} /></span>;
}

function TeamCard({ team, count }: { team: TeamProfile; count: number }) {
  const metric = (key: string) => team.metrics[key];
  return <RouteLink className="team-card" to={{ page: 'news', section: 'teams', team: team.team }} aria-label={`${team.name}: ${team.wins}-${team.losses}. Open the team page.`}>
    <span className="team-card-head"><span className="team-code">{team.team}</span><span className="team-card-name">{team.name}<small>{team.wins}–{team.losses}{team.head_coach ? ` · ${team.head_coach}` : ''}</small></span></span>
    <span className="team-card-stats">
      {(['net', 'pace', 'offense', 'defense'] as const).map(key => metric(key) && <span key={key}><small>{{ net: 'Net', pace: 'Pace', offense: 'Off', defense: 'Def' }[key]}</small><strong>{ordinal(metric(key).rank)}</strong></span>)}
    </span>
    {team.tags.length > 0 && <span className="style-tags">{team.tags.slice(0, 4).map(tag => <span key={tag}>{tag}</span>)}</span>}
    {(team.injuries_out ?? 0) > 0 && <span className="team-card-injuries"><HeartPulse size={13} aria-hidden="true" />{team.injuries_out} out</span>}
    <span className="sr-only">League ranks out of {count}.</span>
  </RouteLink>;
}

type Season = 'current' | 'previous';
const sorts: { id: string; label: string; rank: (team: TeamProfile) => number }[] = [
  { id: 'name', label: 'Name', rank: () => 0 },
  { id: 'net', label: 'Net rating', rank: team => team.metrics.net?.rank ?? 99 },
  { id: 'pace', label: 'Pace', rank: team => team.metrics.pace?.rank ?? 99 },
  { id: 'three_rate', label: '3-point rate', rank: team => team.metrics.three_rate?.rank ?? 99 },
  { id: 'defense', label: 'Defense', rank: team => team.metrics.defense?.rank ?? 99 },
];

function SeasonToggle({ season, onSeason, label }: { season: Season; onSeason: (season: Season) => void; label: string }) {
  return <div className="segmented small" role="group" aria-label={label}>
    <button type="button" aria-pressed={season === 'current'} onClick={() => onSeason('current')}>This season</button>
    <button type="button" aria-pressed={season === 'previous'} onClick={() => onSeason('previous')}>Last season</button>
  </div>;
}

function TeamsSection({ active }: { active: boolean }) {
  const [season, setSeason] = useState<Season>('current');
  const [sort, setSort] = useState('name');
  const feed = useFeed<TeamStyles>(`/api/news/teams?season=${season}`, active);
  const data = feed.data;
  const order = sorts.find(item => item.id === sort) ?? sorts[0];
  const teams = useMemo(() => [...(data?.teams ?? [])].sort((a, b) => order.rank(a) - order.rank(b) || a.name.localeCompare(b.name)), [data, order]);
  return <section className="card news-panel" aria-labelledby="news-teams-title">
    <SectionHeading id="news-teams-title" eyebrow="SCOUTING REPORTS" title="Teams, coaches and playing styles"><Shield size={21} className="subtle" aria-hidden="true" /></SectionHeading>
    <FeedMeta feed={data} loading={feed.loading} onRefresh={() => feed.refresh(true)} label="team styles" />
    <StaleNotice feed={data} />
    {feed.error && <ErrorNotice message={feed.error} onRetry={() => feed.refresh()} />}
    <div className="news-toolbar">
      <SeasonToggle season={season} onSeason={setSeason} label="Season" />
      <label className="news-select"><span>Sort by</span><select value={sort} onChange={event => setSort(event.target.value)}>{sorts.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
    </div>
    {data && season === 'current' && data.median_games < 15 && <p className="news-footnote emphasis">Teams have played a median of {data.median_games} games in {data.season}, so this season's ranks are still noisy. Last season is the steadier read on a coach's system.</p>}
    <p className="news-footnote">Ranks are out of {data?.team_count ?? 30}: 1st is the fastest pace, best offense or defense, or highest rate. Styles are measured from NBA Stats team and play-type data, not opinion.</p>
    {!data && feed.loading && <Busy text="Pulling scouting reports… the first load reads 17 NBA Stats tables and can take a few seconds." />}
    <div className="team-grid">{teams.map(team => <TeamCard key={team.team} team={team} count={data?.team_count ?? 30} />)}</div>
  </section>;
}

function PlayTypes({ plays }: { plays: PlayType[] }) {
  if (!plays.length) return <p className="news-footnote">Play-type data isn't available for this season yet.</p>;
  const top = Math.max(...plays.map(play => play.frequency));
  return <ul className="play-types">{plays.map(play => <li key={play.play_type}>
    <span className="play-label">{play.label}<small>{ordinal(play.frequency_rank)} most used</small></span>
    <span className="play-bar" aria-hidden="true"><span style={{ width: `${(play.frequency / top) * 100}%` }} /></span>
    <span className="play-value"><strong>{(play.frequency * 100).toFixed(1)}%</strong><small>{play.points_per_possession.toFixed(2)} pts/poss{play.percentile !== null ? ` · ${Math.round(play.percentile * 100)}th pct` : ''}</small></span>
  </li>)}</ul>;
}

function MetricTable({ profile, keys, caption }: { profile: SeasonProfile; keys: string[]; caption: string }) {
  const rows = keys.filter(key => profile.metrics[key]);
  return <div className="table-scroll" tabIndex={0} role="region" aria-label={caption}>
    <table className="data-table metric-table"><caption className="sr-only">{caption}</caption>
      <thead><tr><th scope="col">Measure</th><th scope="col">Value</th><th scope="col">League rank</th></tr></thead>
      <tbody>{rows.map(key => {
        const metric = profile.metrics[key];
        return <tr key={key}><th scope="row">{metric.label}</th><td className="mono">{metricValue(metric)}</td>
          <td><span className="rank-cell"><RankBar rank={metric.rank} count={profile.team_count} />{rankText(key, metric, profile.team_count)}</span></td></tr>;
      })}</tbody>
    </table>
  </div>;
}

function Staff({ coaches, season }: { coaches: CoachEntry[]; season: string | null }) {
  if (!coaches.length) return <p className="news-footnote">The coaching staff couldn't be loaded right now.</p>;
  const head = coaches.filter(coach => coach.head);
  const rest = coaches.filter(coach => !coach.head);
  return <div className="staff">
    {head.map(coach => <div key={coach.name} className="head-coach"><span className="eyebrow">HEAD COACH</span><strong>{coach.name}</strong></div>)}
    <ul>{rest.map(coach => <li key={`${coach.name}-${coach.role}`}><span>{coach.name}</span><small>{coach.role}</small></li>)}</ul>
    {season && <p className="news-footnote">{season} staff · NBA Stats</p>}
  </div>;
}

function MovesList({ items }: { items: Transaction[] }) {
  if (!items.length) return <p className="news-footnote">No recent roster moves.</p>;
  return <ul className="moves-list">{items.map((item, index) => <li key={`${item.date}-${index}`}>
    <time dateTime={item.date ?? undefined}>{item.date ? dayLabel(item.date) : '—'}</time>
    {item.team && <TeamChip team={item.team} />}<span>{item.description}</span>
  </li>)}</ul>;
}

function TeamDetailView({ code, active }: { code: string; active: boolean }) {
  const feed = useFeed<TeamDetail>(`/api/news/teams/${code.toLowerCase()}`, active);
  const [season, setSeason] = useState<Season>('current');
  const data = feed.data;
  const profile = data?.profiles[season] ?? data?.profiles[season === 'current' ? 'previous' : 'current'] ?? null;
  const record = data?.profiles.current ?? data?.profiles.previous;
  const head = data?.coaches.find(coach => coach.head);
  return <div className="team-detail">
    <RouteLink className="text-button back-link" to={{ page: 'news', section: 'teams', team: null }}><ArrowLeft size={15} aria-hidden="true" />All teams</RouteLink>
    {feed.error && <ErrorNotice message={feed.error} onRetry={() => feed.refresh()} />}
    {!data && feed.loading && <div className="card"><Busy text={`Scouting ${code}…`} /></div>}
    {data && <>
      <section className="card team-hero" aria-labelledby="team-detail-title">
        <span className="team-code large" aria-hidden="true">{data.team.abbreviation}</span>
        <div>
          <span className="eyebrow">SCOUTING REPORT</span>
          <h2 id="team-detail-title">{data.team.name}</h2>
          <p>{record ? `${record.wins}–${record.losses} in ${record.season}` : ''}{head ? ` · Head coach ${head.name}` : ''}{data.injuries.length ? ` · ${data.injuries.length} on the injury report` : ''}</p>
        </div>
        <button type="button" className="text-button" onClick={() => feed.refresh(true)} aria-disabled={feed.loading || undefined}><RefreshCw size={14} aria-hidden="true" className={feed.loading ? 'spin' : ''} />Refresh</button>
      </section>
      {data.errors.length > 0 && <div className="notice warning-notice"><HeartPulse size={18} aria-hidden="true" /><div><strong>Some sections couldn't load</strong>{data.errors.map(error => <p key={error}>{error}</p>)}</div></div>}
      <div className="team-detail-grid">
        <section className="card" aria-labelledby="team-style-title">
          <SectionHeading id="team-style-title" eyebrow="HOW THEY PLAY" title="Playing style"><SeasonToggle season={season} onSeason={setSeason} label="Style season" /></SectionHeading>
          {profile ? <>
            <p className="style-summary">{profile.season}: {profile.summary}</p>
            {profile.tags.length > 0 && <div className="style-tags large">{profile.tags.map(tag => <span key={tag}>{tag}</span>)}</div>}
            {season === 'current' && profile.games < 15 && <p className="news-footnote emphasis">Only {profile.games} games played this season; compare with last season.</p>}
            <MetricTable profile={profile} keys={styleMetrics} caption={`${data.team.name} playing style, ${profile.season}`} />
          </> : <p className="news-footnote">No team statistics for this season yet.</p>}
        </section>
        <section className="card" aria-labelledby="team-plays-title">
          <SectionHeading id="team-plays-title" eyebrow="THE PLAYBOOK" title="Play types" />
          {profile ? <PlayTypes plays={profile.play_types} /> : <p className="news-footnote">No play-type data yet.</p>}
          <p className="news-footnote">Share of offensive possessions by Synergy play type, with points per possession and the league efficiency percentile.</p>
        </section>
        <section className="card" aria-labelledby="team-allowed-title">
          <SectionHeading id="team-allowed-title" eyebrow="PROP ANGLE" title="What they give up" />
          {profile ? <MetricTable profile={profile} keys={allowedMetrics} caption={`${data.team.name} opponent averages, ${profile.season}`} /> : null}
          <p className="news-footnote">Per game against this team. A defense that allows the most threes or rebounds is a friendlier matchup for those props; COURTSIDE's model does not apply these ranks automatically.</p>
        </section>
        <section className="card" aria-labelledby="team-staff-title">
          <SectionHeading id="team-staff-title" eyebrow="THE BENCH" title="Coaching staff"><Users size={20} className="subtle" aria-hidden="true" /></SectionHeading>
          <Staff coaches={data.coaches} season={data.coaches_season} />
        </section>
      </div>
      <section className="card" aria-labelledby="team-injuries-title">
        <SectionHeading id="team-injuries-title" eyebrow="AVAILABILITY" title="Injury report"><Stethoscope size={20} className="subtle" aria-hidden="true" /></SectionHeading>
        {data.injuries.length ? <InjuryTable entries={data.injuries} caption={`${data.team.name} injury report`} /> : <p className="news-footnote">Nobody is listed right now.</p>}
      </section>
      <div className="team-detail-grid">
        <section className="card" aria-labelledby="team-news-title">
          <SectionHeading id="team-news-title" eyebrow="THE WIRE" title="Latest news" />
          {data.headlines.length ? <div className="news-list compact">{data.headlines.map(article => <ArticleCard key={article.id} article={article} />)}</div> : <p className="news-footnote">No recent headlines tagged to this team.</p>}
        </section>
        <section className="card" aria-labelledby="team-moves-title">
          <SectionHeading id="team-moves-title" eyebrow="TRANSACTIONS" title="Roster moves"><Repeat2 size={20} className="subtle" aria-hidden="true" /></SectionHeading>
          <MovesList items={data.transactions} />
        </section>
      </div>
    </>}
  </div>;
}

function MovesSection({ active, teams }: { active: boolean; teams: string[] }) {
  const feed = useFeed<Transactions>('/api/news/transactions', active);
  const [team, setTeam] = useState('');
  const data = feed.data;
  const items = (data?.transactions ?? []).filter(item => !team || item.team === team);
  return <section className="card news-panel" aria-labelledby="news-moves-title">
    <SectionHeading id="news-moves-title" eyebrow="TRANSACTIONS" title="Roster moves"><Repeat2 size={21} className="subtle" aria-hidden="true" /></SectionHeading>
    <FeedMeta feed={data} loading={feed.loading} onRefresh={() => feed.refresh(true)} label="roster moves" />
    <StaleNotice feed={data} />
    {feed.error && <ErrorNotice message={feed.error} onRetry={() => feed.refresh()} />}
    <div className="news-toolbar">
      <label className="news-select"><span>Team</span><select value={team} onChange={event => setTeam(event.target.value)}><option value="">All teams</option>{teams.map(code => <option key={code} value={code}>{code}</option>)}</select></label>
    </div>
    <p className="news-footnote">Signings, waivers, trades, options and assignments. A new arrival or departure changes who takes the minutes and shots.</p>
    {!data && feed.loading && <Busy text="Reading the transaction wire…" />}
    {data && <MovesList items={items} />}
  </section>;
}

export const newsTabs: { id: NewsSection; label: string; icon: ReactNode }[] = [
  { id: 'headlines', label: 'Headlines', icon: <Newspaper size={16} aria-hidden="true" /> },
  { id: 'injuries', label: 'Injury report', icon: <Stethoscope size={16} aria-hidden="true" /> },
  { id: 'teams', label: 'Teams & coaches', icon: <Shield size={16} aria-hidden="true" /> },
  { id: 'moves', label: 'Roster moves', icon: <ClipboardList size={16} aria-hidden="true" /> },
];

/** The newsroom: every section loads its feed the first time it is opened, then keeps it until refreshed. */
export default function NewsPage({ active, section, team, teams }: { active: boolean; section: NewsSection; team: string | null; teams: string[] }) {
  const [visited, setVisited] = useState<Set<NewsSection>>(() => new Set());
  if (active && !visited.has(section)) setVisited(new Set(visited).add(section));
  const on = (id: NewsSection) => active && section === id;
  return <>
    <div className="page-panel" hidden={section !== 'headlines'}>{visited.has('headlines') && <HeadlinesSection active={on('headlines')} teams={teams} />}</div>
    <div className="page-panel" hidden={section !== 'injuries'}>{visited.has('injuries') && <InjuriesSection active={on('injuries')} />}</div>
    <div className="page-panel" hidden={section !== 'teams'}>
      {visited.has('teams') && (team ? <TeamDetailView key={team} code={team} active={on('teams')} /> : <TeamsSection active={on('teams')} />)}
    </div>
    <div className="page-panel" hidden={section !== 'moves'}>{visited.has('moves') && <MovesSection active={on('moves')} teams={teams} />}</div>
  </>;
}
