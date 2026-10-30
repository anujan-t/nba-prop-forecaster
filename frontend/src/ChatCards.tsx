import { ArrowRight, BrainCircuit, CircleCheck, CircleX, Clock3, ExternalLink, HeartPulse, History, Scale } from 'lucide-react';
import { Component, Fragment, useId, useState } from 'react';
import type { ReactNode } from 'react';
import { dateLabel, decimal, edgePoints, modelNames, pct, signed, statNames, timeAgo } from './api';
import type {
  BetStatus, CalibrationStatus, ChatCard, CoachesCard, ForecastCard, GamesCard, HouseComparison, InjuriesCard, LeagueCard, LearningStatus, MatchupCard,
  MovesCard, NewsCard, ParlayCard, PlayerCard, RankingCard, StandingsCard, TeamCard, TrackRecord, TrackedBet,
} from './assistant-types';
import { parseRichText } from './chat';
import type { InlineRun } from './chat';
import { ScoreboardDigits } from './components';
import { betStatusNames, legLine, legResult, legStatusNames, legsDetail, profitLabel, recordDetail, roiLabel, signedUnits, units } from './ledger';
import { allowedMetrics, dayLabel, GameCard, gameDayLabel, InjuryBadge, injuryCounts, kindShort, metricValue, newsTabs, ordinal, PlayerLink, TeamChip } from './NewsPage';
import { americanLabel, chance, chanceDigits, decimalOdds, isGameLeg, legNumber, sideNames, spreadLabel, statLabel } from './parlay';
import { GoContext, RouteLink } from './RouteLink';
import type { Go } from './RouteLink';
import { parseHash } from './routes';
import type { PlayerSection, Route, SettingsSection } from './routes';

// ---------- Reply text ----------

function Runs({ runs }: { runs: InlineRun[] }) {
  return <>{runs.map((run, index) => {
    const text = run.bold ? <strong>{run.text}</strong> : run.text;
    return run.italic ? <em key={index}>{text}</em> : <Fragment key={index}>{text}</Fragment>;
  })}</>;
}

/** **bold**, line breaks, "- " bullets and whole-line _asides_ as React nodes. No HTML is ever injected. */
export function RichText({ text, className = '' }: { text: string; className?: string }) {
  return <div className={`rich-text ${className}`}>
    {parseRichText(text).map((block, index) => block.type === 'list'
      ? <ul key={index}>{block.items.map((item, position) => <li key={position}><Runs runs={item} /></li>)}</ul>
      : <p key={index}>{block.lines.map((line, position) => <Fragment key={position}>{position > 0 && <br />}<Runs runs={line} /></Fragment>)}</p>)}
  </div>;
}

// ---------- Shared bits (also used by My Bets and the learning panel) ----------

const minus = (text: string) => text.replace(/^-/, '−');

const statusIcons: Record<BetStatus, ReactNode> = {
  pending: <Clock3 size={13} aria-hidden="true" />,
  won: <CircleCheck size={13} aria-hidden="true" />,
  lost: <CircleX size={13} aria-hidden="true" />,
  void: null,
  reduced: null,
};

export function BetStatusChip({ status }: { status: BetStatus }) {
  return <span className={`bet-status ${status}`}>{statusIcons[status]}{betStatusNames[status]}</span>;
}

export function CalibrationBadge({ calibration }: { calibration: CalibrationStatus }) {
  const markets = (calibration.stats ?? []).filter(entry => entry.active).length;
  const on = calibration.active || markets > 0;
  const text = calibration.stats?.length ? (markets ? `Calibrated · ${markets} of ${calibration.stats.length} markets` : 'Calibration off') : calibration.active ? 'Calibration on' : 'Calibration off';
  return <span className={`calibration-badge ${on ? 'on' : 'off'}`}><span aria-hidden="true" />{text}</span>;
}

const verdictNames: Record<HouseComparison['verdict'], string> = { value: 'Better than fair', fair: 'Close to fair', house: 'House edge' };

/** An offered price set against the model. 'standard' is an assumption, never presented as a book's price. */
export function HouseBlock({ house }: { house: HouseComparison }) {
  return <section className={`house-block verdict-${house.verdict}`} aria-label="Compared with the house">
    <div className="house-head">
      <span className="house-title"><Scale size={14} aria-hidden="true" />Vs the house</span>
      <span className={`house-verdict ${house.verdict}`}>{verdictNames[house.verdict]}</span>
    </div>
    <p className="house-basis">{house.basis === 'entered'
      ? <><strong>Your price: {americanLabel(house.odds)}</strong> (decimal {decimalOdds(house.decimal)})</>
      : <><strong>Standard pricing, not a real book price.</strong> Assumes −110 on every leg, priced as independent: {americanLabel(house.odds)} (decimal {decimalOdds(house.decimal)}).</>}</p>
    <dl className="house-stats">
      <div><dt>Price implies</dt><dd>{pct(house.implied)}</dd></div>
      <div><dt>Model</dt><dd>{pct(house.model)}</dd></div>
      <div><dt>Model fair odds</dt><dd>{house.fair_odds ? americanLabel(house.fair_odds.american) : 'N/A'}</dd></div>
      <div><dt>Model − implied</dt><dd className={house.edge >= 0 ? 'positive' : 'negative'}>{minus(edgePoints(house.edge))}</dd></div>
      <div><dt>Return per unit</dt><dd className={house.expected_roi >= 0 ? 'positive' : 'negative'}>{minus(signed(house.expected_roi, 2))}</dd></div>
      <div><dt>House take per unit</dt><dd>{minus(signed(house.house_edge, 2))}</dd></div>
      {house.vig !== null && <div><dt>Vig, both sides</dt><dd>{pct(house.vig)}</dd></div>}
    </dl>
    {house.note && <p className="house-note">{house.note}</p>}
  </section>;
}

export function amount(value: number): string {
  return Number.isInteger(value) ? String(value) : decimal(value);
}

function OpenButton({ children, onClick }: { children: ReactNode; onClick: () => void }) {
  return <button type="button" className="secondary-button chat-card-action" onClick={onClick}>{children}<ArrowRight size={15} aria-hidden="true" /></button>;
}

// ---------- Cards ----------

function ForecastCardView({ card }: { card: ForecastCard }) {
  const over = Math.max(0, card.over);
  const under = Math.max(0, card.under);
  const push = Math.max(0, card.push);
  const total = over + under + push || 1;
  const games = card.hit_rate.games;
  const fair = (value: ForecastCard['fair']['over']) => value ? americanLabel(value.american) : 'N/A';
  return <article className="chat-card" aria-label={`Forecast: ${card.player.name}, ${statLabel(card.stat)}, line ${card.line}`}>
    <header className="chat-card-head">
      <span className="chat-card-kicker">Forecast</span>
      <h4>{card.player.name}</h4>
      <span className="chat-card-sub">{[card.player.team, statNames[card.stat], `line ${card.line}`].filter(Boolean).join(' · ')}</span>
    </header>
    <div className="ou-split" aria-hidden="true">
      <span className="ou-over" style={{ flexGrow: over / total }} />
      {push > 0 && <span className="ou-push" style={{ flexGrow: push / total }} />}
      <span className="ou-under" style={{ flexGrow: under / total }} />
    </div>
    <dl className="ou-values">
      <div className={`ou-value over${card.side === 'over' ? ' picked' : ''}`}><dt>Over {card.line}</dt><dd>{chance(card.over)}</dd>{card.calibrated && <dd className="cal">cal. {chance(card.calibrated.over)}</dd>}</div>
      {push > 0 && <div className="ou-value push"><dt>Push</dt><dd>{chance(card.push)}</dd></div>}
      <div className={`ou-value under${card.side === 'under' ? ' picked' : ''}`}><dt>Under {card.line}</dt><dd>{chance(card.under)}</dd>{card.calibrated && <dd className="cal">cal. {chance(card.calibrated.under)}</dd>}</div>
    </dl>
    <dl className="chat-facts">
      <div><dt>Model mean</dt><dd>{decimal(card.mean)}</dd></div>
      <div><dt>{Math.round(card.interval.coverage * 100)}% range</dt><dd>{amount(card.interval.low)}–{amount(card.interval.high)}</dd></div>
      <div><dt>Hit rate</dt><dd>{games ? `Over ${card.hit_rate.over}/${games} · Under ${card.hit_rate.under}/${games}` : 'No games'}</dd></div>
      <div><dt>Fair odds</dt><dd>O {fair(card.fair.over)} · U {fair(card.fair.under)}</dd></div>
      <div className="wide"><dt>Model</dt><dd>{modelNames[card.resolved_model]}{card.model === 'auto' ? ' (auto)' : ''} · {card.sample_size} games</dd></div>
    </dl>
    {card.calibrated && <p className="chat-card-note"><BrainCircuit size={13} aria-hidden="true" /><span>“cal.” is after this market’s calibration, which is on because it beat the raw model on recent held-out games.</span></p>}
    {card.house && <HouseBlock house={card.house} />}
  </article>;
}

function ParlayCardView({ card, onOpenParlay }: { card: ParlayCard; onOpenParlay: () => void }) {
  const { result, house } = card;
  const count = result.legs.length;
  const calibration = result.calibration;
  return <article className="chat-card chat-parlay" aria-label={`Parlay, ${count} ${count === 1 ? 'leg' : 'legs'}`}>
    <div className="chat-scoreboard">
      <div className="chat-scoreboard-top">
        <span className="scoreboard-label">ALL {count} {count === 1 ? 'LEG HITS' : 'LEGS HIT'}</span>
        <span className="scoreboard-label">FAIR {result.fair_odds ? americanLabel(result.fair_odds.american) : 'N/A'}</span>
      </div>
      <div className="chat-scoreboard-value">
        <ScoreboardDigits text={chanceDigits(result.probability)} label={`${chance(result.probability)} model probability that all ${count} legs hit`} className="chat-sb-digits" />
        <span className="sb-unit" aria-hidden="true">%</span>
      </div>
      <dl className="chat-scoreboard-odds">
        <div><dt>Linked</dt><dd>{chance(result.probability)}</dd></div>
        <div><dt>Independent</dt><dd>{chance(result.independent_probability)}</dd></div>
        <div><dt>Lift</dt><dd>{result.correlation_lift === null ? 'N/A' : `×${result.correlation_lift.toFixed(2)}`}</dd></div>
      </dl>
    </div>
    {calibration && <p className="chat-card-note"><BrainCircuit size={13} aria-hidden="true" /><span><strong>Calibrated {chance(calibration.probability)}.</strong> {calibration.note}</span></p>}
    <ol className="chat-legs">{result.legs.map((leg, position) => isGameLeg(leg)
      ? <li key={leg.index} className="side-game">
        <span className="chat-leg-no" aria-hidden="true">{legNumber(leg.index)}</span>
        <span className="chat-leg-who"><span className="sr-only">Leg {leg.index + 1}: </span>{leg.team} <span className="chat-leg-pick">{leg.market === 'moneyline' ? 'moneyline' : spreadLabel(leg.line ?? 0)} {leg.home ? 'vs' : 'at'} {leg.opponent}</span></span>
        <span className="chat-leg-p">{chance(leg.probability)}</span>
      </li>
      : <li key={leg.index} className={`side-${leg.side}`}>
        <span className="chat-leg-no" aria-hidden="true">{legNumber(leg.index)}</span>
        <span className="chat-leg-who"><span className="sr-only">Leg {leg.index + 1}: </span>{leg.player.name} <span className="chat-leg-pick">{sideNames[leg.side]} {leg.line} {statLabel(leg.stat)}</span></span>
        <span className="chat-leg-p">{chance(leg.probability)}{calibration && typeof calibration.legs[position] === 'number' && <small>cal. {chance(calibration.legs[position])}</small>}</span>
      </li>)}</ol>
    <HouseBlock house={house} />
    <OpenButton onClick={onOpenParlay}>Open Parlay view</OpenButton>
  </article>;
}

const averageKeys = [['points', 'PTS'], ['rebounds', 'REB'], ['assists', 'AST'], ['steals', 'STL'], ['blocks', 'BLK'], ['minutes', 'MIN']] as const;

function PlayerCardView({ card }: { card: PlayerCard }) {
  return <article className="chat-card" aria-label={`${card.player.name}, recent form`}>
    <header className="chat-card-head">
      <span className="chat-card-kicker">Player · last {card.games} {card.games === 1 ? 'game' : 'games'}</span>
      <h4>{card.player.name}</h4>
      {card.player.team && <span className="chat-card-sub">{card.player.team}</span>}
    </header>
    <dl className="chat-averages">{averageKeys.map(([key, short]) => <div key={key}>
      <dt><abbr title={key === 'minutes' ? 'Minutes' : statNames[key]}>{short}</abbr></dt><dd>{typeof card.averages[key] === 'number' ? decimal(card.averages[key]) : '—'}</dd>
    </div>)}</dl>
    {card.last_games.length > 0 && <div className="table-scroll" tabIndex={0} role="region" aria-label={`${card.player.name}, last ${card.last_games.length} games`}>
      <table className="chat-games">
        <caption className="sr-only">Most recent first</caption>
        <thead><tr><th scope="col">Date</th><th scope="col">Opp</th><th scope="col">MIN</th><th scope="col">PTS</th><th scope="col">REB</th><th scope="col">AST</th><th scope="col">STL</th><th scope="col">BLK</th></tr></thead>
        <tbody>{card.last_games.slice(0, 5).map((game, index) => <tr key={`${game.date}-${index}`}>
          <th scope="row">{dateLabel(game.date)}</th><td>{game.opponent}</td><td>{Math.round(game.minutes)}</td><td>{game.points}</td><td>{game.rebounds}</td><td>{game.assists}</td><td>{game.steals}</td><td>{game.blocks}</td>
        </tr>)}</tbody>
      </table>
    </div>}
  </article>;
}

function RecordCardView({ record, onOpenRecord }: { record: TrackRecord; onOpenRecord: () => void }) {
  return <article className="chat-card" aria-label="My Bets record">
    <header className="chat-card-head">
      <span className="chat-card-kicker">My Bets · {record.bets} {record.bets === 1 ? 'bet' : 'bets'}</span>
      <h4><span className="record-score" aria-label={`${record.won} won, ${record.lost} lost`}>{record.won}–{record.lost}</span></h4>
      <span className="chat-card-sub">{recordDetail(record)}</span>
    </header>
    <dl className="chat-facts">
      <div><dt>Profit</dt><dd className={record.profit > 0 ? 'positive' : record.profit < 0 ? 'negative' : ''}>{signedUnits(record.profit)}</dd></div>
      <div><dt>ROI</dt><dd>{roiLabel(record.roi)} <small>on {units(record.staked)} staked</small></dd></div>
      <div className="wide"><dt>Legs</dt><dd>{legsDetail(record)}</dd></div>
      <div><dt>Brier score</dt><dd>{record.brier === null ? 'N/A' : record.brier.toFixed(3)}</dd></div>
    </dl>
    <OpenButton onClick={onOpenRecord}>Open My Bets</OpenButton>
  </article>;
}

function LearningCardView({ status, onOpenAccuracy }: { status: LearningStatus; onOpenAccuracy: () => void }) {
  const { calibration, memory } = status;
  return <article className="chat-card" aria-label="Learning loop">
    <header className="chat-card-head chat-card-head-row">
      <span className="chat-card-kicker">Learning loop</span>
      <CalibrationBadge calibration={calibration} />
    </header>
    <p className="chat-card-text">{calibration.message}</p>
    <dl className="chat-facts">
      <div><dt>Graded predictions</dt><dd>{calibration.samples} <small>({calibration.ledger_samples} from your bets)</small></dd></div>
      <div><dt>Players</dt><dd>{calibration.players}</dd></div>
      {calibration.holdout && <div className="wide"><dt>Held-out log loss</dt><dd>{calibration.holdout.raw_log_loss.toFixed(3)} raw → {calibration.holdout.calibrated_log_loss.toFixed(3)} calibrated <small>({calibration.holdout.samples} games)</small></dd></div>}
      <div><dt>Nicknames learned</dt><dd>{memory.aliases.length}</dd></div>
      <div><dt>Ratings</dt><dd>{memory.liked} helpful · {memory.disliked} not</dd></div>
    </dl>
    {status.training && <p className="chat-card-note"><Clock3 size={13} aria-hidden="true" /><span>A retrain is running.</span></p>}
    <OpenButton onClick={onOpenAccuracy}>Open Accuracy</OpenButton>
  </article>;
}

function BetCardView({ bet, onOpenRecord }: { bet: TrackedBet; onOpenRecord: () => void }) {
  return <article className="chat-card" aria-label={`Tracked bet, ${bet.legs.length} ${bet.legs.length === 1 ? 'leg' : 'legs'}`}>
    <header className="chat-card-head chat-card-head-row">
      <span className="chat-card-kicker">Tracked bet · {dateLabel(bet.event_date, true)}</span>
      <BetStatusChip status={bet.status} />
    </header>
    <ol className="chat-legs">{bet.legs.map((leg, index) => <li key={`${leg.player_id}-${leg.stat}`} className={`side-${leg.side} leg-${leg.status}`}>
      <span className="chat-leg-no" aria-hidden="true">{legNumber(index)}</span>
      <span className="chat-leg-who"><span className="sr-only">Leg {index + 1}: </span>{legLine(leg)}</span>
      <span className="chat-leg-p">{chance(leg.probability)}{leg.status !== 'pending' && <small>{legStatusNames[leg.status]}{leg.actual !== null ? ` · ${legResult(leg, amount)}` : ''}</small>}</span>
    </li>)}</ol>
    <dl className="chat-facts">
      <div><dt>Model, all legs</dt><dd>{chance(bet.probability)}</dd></div>
      <div><dt>Book odds</dt><dd>{bet.book_odds === null ? 'None' : americanLabel(bet.book_odds)}</dd></div>
      <div><dt>Stake</dt><dd>{bet.stake === null ? 'None' : units(bet.stake)}</dd></div>
      {bet.status !== 'pending' && <div><dt>Profit</dt><dd>{profitLabel(bet)}</dd></div>}
    </dl>
    {bet.note && <p className="chat-card-text">“{bet.note}”</p>}
    <OpenButton onClick={onOpenRecord}>Open My Bets</OpenButton>
  </article>;
}

// ---------- League and news cards ----------

const playerRouteLabels: Record<PlayerSection, string> = { props: 'Open in Players', log: 'Open game log', career: 'Open profile & career', lines: 'Open lines' };
const settingsRouteLabels: Record<SettingsSection, string> = {
  appearance: 'Open Appearance settings', model: 'Open Model settings', data: 'Open Data settings', coach: 'Open Coach settings',
};

/** What a card's link opens: "Open Games", "Open LAL team page", "Open Accuracy". */
function routeLabel(route: Route): string {
  switch (route.page) {
    case 'news': return route.section === 'teams' && route.team ? `Open ${route.team} team page` : `Open ${newsTabs.find(tab => tab.id === route.section)?.label ?? 'News'}`;
    case 'players': return route.playerId ? playerRouteLabels[route.section] : 'Open Players';
    case 'how': return route.section === 'accuracy' ? 'Open Accuracy' : 'Open How it works';
    case 'settings': return settingsRouteLabels[route.section];
    case 'parlay': return 'Open Parlay view';
    case 'bets': return 'Open My Bets';
    case 'coach': return 'Open Coach';
    default: return 'Open';
  }
}

/** An in-app link to the page a card summarizes. Nothing when the route isn't one the app knows. */
function RouteButton({ route, label }: { route: unknown; label?: string }) {
  const to = typeof route === 'string' ? parseHash(route) : null;
  if (!to) return null;
  return <RouteLink className="secondary-button chat-card-action" to={to}>{label ?? routeLabel(to)}<ArrowRight size={15} aria-hidden="true" /></RouteLink>;
}

/** Long lists stay short: the first `limit` items, and a toggle for the rest. */
function useCapped<T>(items: T[], limit: number) {
  const [expanded, setExpanded] = useState(false);
  const listId = useId();
  const capped = items.length > limit;
  const toggle = capped
    ? <button type="button" className="text-button chat-card-more" aria-expanded={expanded} aria-controls={listId} onClick={() => setExpanded(value => !value)}>{expanded ? 'Show fewer' : `Show all ${items.length}`}</button>
    : null;
  return { shown: capped && !expanded ? items.slice(0, limit) : items, listId, toggle };
}

/** A feed that failed and fell back to the copy saved on this computer. */
function SavedCopy({ stale, fetchedAt }: { stale: boolean; fetchedAt: string | null }) {
  if (!stale) return null;
  const when = timeAgo(fetchedAt);
  return <p className="chat-card-stale"><History size={13} aria-hidden="true" /><span>Saved copy{when ? ` from ${when}` : ''}; the live feed didn’t answer.</span></p>;
}

function updated(fetchedAt: string | null, stale: boolean): string {
  const when = !stale ? timeAgo(fetchedAt) : '';
  return when ? `Updated ${when}` : '';
}

function NewsCardView({ card }: { card: NewsCard }) {
  const { shown, listId, toggle } = useCapped(card.articles, 5);
  const sub = updated(card.fetched_at, card.stale);
  return <article className="chat-card" aria-label={card.title}>
    <header className="chat-card-head">
      <span className="chat-card-kicker">News</span>
      <h4>{card.title}</h4>
      {sub && <span className="chat-card-sub">{sub}</span>}
    </header>
    <SavedCopy stale={card.stale} fetchedAt={card.fetched_at} />
    {card.articles.length > 0
      ? <ul id={listId} className="chat-news">{shown.map(article => <li key={article.id} className={`kind-${article.kind}`}>
        <span className="chat-news-meta"><span className={`kind-pill kind-${article.kind}`}>{kindShort[article.kind] ?? 'News'}</span>{article.published && <time dateTime={article.published}>{timeAgo(article.published)}</time>}</span>
        <h5>{article.url
          ? <a href={article.url} target="_blank" rel="noopener noreferrer">{article.headline}<ExternalLink size={12} aria-hidden="true" /><span className="sr-only"> (opens in a new tab)</span></a>
          : article.headline}</h5>
        {article.description && <p>{article.description}</p>}
        {(article.teams.length > 0 || article.byline) && <span className="chat-news-foot">{article.teams.slice(0, 4).map(team => <TeamChip key={team} team={team} />)}{article.byline && <span>{article.byline}</span>}</span>}
      </li>)}</ul>
      : <p className="chat-card-text">No stories match right now.</p>}
    {toggle}
    <RouteButton route={card.route} />
  </article>;
}

function InjuriesCardView({ card }: { card: InjuriesCard }) {
  const { shown, listId, toggle } = useCapped(card.entries, 8);
  const complete = card.total <= card.entries.length;
  const sub = [`${card.total} ${card.total === 1 ? 'player' : 'players'} listed`, complete && card.entries.length ? injuryCounts(card.entries) : '', updated(card.fetched_at, card.stale)].filter(Boolean).join(' · ');
  return <article className="chat-card" aria-label={card.title}>
    <header className="chat-card-head">
      <span className="chat-card-kicker">Injury report</span>
      <h4>{card.title}</h4>
      <span className="chat-card-sub">{sub}</span>
    </header>
    <SavedCopy stale={card.stale} fetchedAt={card.fetched_at} />
    {card.entries.length > 0
      ? <ul id={listId} className="chat-injuries">{shown.map(entry => {
        const what = [entry.side, entry.injury, entry.detail && entry.detail !== entry.injury ? entry.detail.toLowerCase() : null].filter(Boolean).join(' ');
        const back = (entry.status_key === 'out' || entry.status_key === 'suspended') && entry.return_date ? `est. return ${dayLabel(entry.return_date)}` : '';
        return <li key={`${entry.team}-${entry.player}`} className={`status-${entry.status_key}`}>
          <span className="chat-injury-head">
            <span className="chat-injury-who"><PlayerLink name={entry.player} playerId={entry.player_id} /><small>{[entry.team, entry.position].filter(Boolean).join(' · ')}</small></span>
            <InjuryBadge entry={entry} compact />
          </span>
          {(what || back) && <span className="chat-injury-what">{[what, back].filter(Boolean).join(' · ')}</span>}
          {entry.short_comment && <p>{entry.short_comment}</p>}
        </li>;
      })}</ul>
      : <p className="chat-card-text">Nobody is listed right now.</p>}
    {toggle}
    {!complete && <p className="chat-card-note"><HeartPulse size={13} aria-hidden="true" /><span>Showing {card.entries.length} of {card.total}; the full report has everyone.</span></p>}
    <RouteButton route={card.route} />
  </article>;
}

function GamesCardView({ card }: { card: GamesCard }) {
  const { shown, listId, toggle } = useCapped(card.games, 6);
  const providers = [...new Set(card.games.map(game => game.odds?.provider).filter((name): name is string => !!name))];
  const live = card.games.filter(game => game.state === 'in').length;
  const lines = card.games.some(game => game.odds);
  return <article className="chat-card" aria-label={card.title}>
    <header className="chat-card-head">
      <span className="chat-card-kicker">Games · {gameDayLabel(card.date)}</span>
      <h4>{card.title}</h4>
      <span className="chat-card-sub">{card.games.length} {card.games.length === 1 ? 'game' : 'games'}{live ? ` · ${live} live` : ''}</span>
    </header>
    {card.games.length > 0
      ? <ul id={listId} className="chat-slate">{shown.map(game => <li key={game.id}><GameCard game={game} compact showProvider={false} /></li>)}</ul>
      : <p className="chat-card-text">No games on this date.</p>}
    {toggle}
    {lines && <p className="chat-card-note"><Scale size={13} aria-hidden="true" /><span>Lines: {providers.length ? providers.join(', ') : 'the listed sportsbook'}, for reference. COURTSIDE models player props, not sides or totals.</span></p>}
    <RouteButton route={card.route} />
  </article>;
}

function gamesBack(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value) || value === 0) return '—';
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

/** "41–12 · 1st in the West · 2.5 GB" */
function recordLine(record: TeamCard['record']): string {
  if (!record) return '';
  const place = record.seed ? (record.conference ? `${ordinal(record.seed)} in the ${record.conference}` : `${ordinal(record.seed)} seed`) : '';
  const behind = record.games_back ? `${gamesBack(record.games_back)} GB` : '';
  return [`${record.wins}–${record.losses}`, place, behind].filter(Boolean).join(' · ');
}

function TeamInjuries({ injuries }: { injuries: TeamCard['injuries'] }) {
  const { shown, listId, toggle } = useCapped(injuries, 5);
  return <section className="chat-card-section" aria-label="Injury report">
    <h5 className="chat-card-label">Injuries · {injuryCounts(injuries)}</h5>
    <ul id={listId} className="chat-team-injuries">{shown.map(entry => <li key={entry.player}>
      <span>{entry.player}</span><InjuryBadge entry={entry} compact />{entry.injury && <small>{entry.injury}</small>}
    </li>)}</ul>
    {toggle}
  </section>;
}

function TeamCardView({ card }: { card: TeamCard }) {
  const count = card.team_count ?? 30;
  const metrics = card.metrics.slice(0, 8);
  const allowed = metrics.some(metric => allowedMetrics.includes(metric.key));
  const sub = [recordLine(card.record), card.head_coach ? `Coach ${card.head_coach}` : ''].filter(Boolean).join(' · ');
  return <article className="chat-card" aria-label={`${card.name}, scouting report`}>
    <header className="chat-team-head">
      <span className="team-code" aria-hidden="true">{card.team}</span>
      <div className="chat-card-head">
        <span className="chat-card-kicker">Team{card.season ? ` · ${card.season}` : ''}</span>
        <h4>{card.name}</h4>
        {sub && <span className="chat-card-sub">{sub}</span>}
      </div>
    </header>
    {card.summary && <p className="chat-card-text">{card.summary}</p>}
    {card.tags.length > 0 && <div className="style-tags">{card.tags.slice(0, 5).map(tag => <span key={tag}>{tag}</span>)}</div>}
    {metrics.length > 0 && <dl className="chat-facts">{metrics.map(metric => <div key={metric.key}>
      <dt>{metric.label}</dt><dd>{metricValue(metric)} <small>{ordinal(metric.rank)} of {count}</small></dd>
    </div>)}</dl>}
    {allowed && <p className="chat-card-note"><Scale size={13} aria-hidden="true" /><span>For what opponents score against them, 1st means they give up the fewest.</span></p>}
    {card.play_types.length > 0 && <section className="chat-card-section" aria-label="Play types">
      <h5 className="chat-card-label">Most-used plays</h5>
      <ul className="chat-plays">{card.play_types.slice(0, 4).map(play => <li key={play.label}>
        <span>{play.label}</span><span className="chat-plays-value">{pct(play.frequency)} <small>· {decimal(play.points_per_possession, 2)} pts/poss</small></span>
      </li>)}</ul>
    </section>}
    {card.injuries.length > 0 && <TeamInjuries injuries={card.injuries} />}
    {card.next_game && <section className="chat-card-section" aria-label="Next game">
      <h5 className="chat-card-label">{card.next_game.state === 'in' ? 'Live now' : 'Next game'}</h5>
      <GameCard game={card.next_game} compact showDate />
    </section>}
    <RouteButton route={card.route} />
  </article>;
}

const rankKeys = [['net', 'Net'], ['pace', 'Pace'], ['offense', 'Off'], ['defense', 'Def']] as const;

function MatchupTeam({ team }: { team: TeamCard }) {
  const count = team.team_count ?? 30;
  const ranks = rankKeys.flatMap(([key, label]) => {
    const metric = team.metrics.find(item => item.key === key);
    return metric ? [{ key, label, metric }] : [];
  });
  const record = recordLine(team.record);
  return <section className="chat-matchup-team" aria-label={team.name}>
    <div className="chat-team-head">
      <span className="team-code" aria-hidden="true">{team.team}</span>
      <div className="chat-card-head"><h5>{team.name}</h5>{record && <span className="chat-card-sub">{record}</span>}</div>
    </div>
    {ranks.length > 0 && <dl className="chat-ranks">{ranks.map(({ key, label, metric }) => <div key={key}>
      <dt>{label}</dt><dd>{ordinal(metric.rank)}<span className="sr-only"> of {count}</span></dd>
    </div>)}</dl>}
    {team.head_coach && <p className="chat-card-sub">Coach {team.head_coach}</p>}
    {team.tags.length > 0 && <div className="style-tags">{team.tags.slice(0, 3).map(tag => <span key={tag}>{tag}</span>)}</div>}
    {team.injuries.length > 0 && <p className="chat-matchup-injuries"><HeartPulse size={13} aria-hidden="true" /><span>{injuryCounts(team.injuries)}</span></p>}
    <RouteButton route={team.route} />
  </section>;
}

function MatchupCardView({ card }: { card: MatchupCard }) {
  const { a, b, game } = card;
  return <article className="chat-card" aria-label={`${a.name} vs ${b.name}`}>
    <header className="chat-card-head">
      <span className="chat-card-kicker">Matchup{a.season ? ` · ${a.season}` : ''}</span>
      <h4>{a.team} vs {b.team}</h4>
      <span className="chat-card-sub">League ranks out of {a.team_count ?? 30}; 1st is the best net rating, fastest pace, best offense or defense.</span>
    </header>
    {game && <GameCard game={game} compact showDate />}
    <div className="chat-matchup">{[a, b].map(team => <MatchupTeam key={team.team} team={team} />)}</div>
  </article>;
}

function RankingCardView({ card }: { card: RankingCard }) {
  return <article className="chat-card" aria-label={card.title}>
    <header className="chat-card-head">
      <span className="chat-card-kicker">Ranking{card.season ? ` · ${card.season}` : ''}</span>
      <h4>{card.title}</h4>
      <span className="chat-card-sub">Ranks are out of {card.team_count}</span>
    </header>
    {card.rows.length > 0 && <div className="table-scroll" tabIndex={0} role="region" aria-label={card.title}>
      <table className="chat-games chat-ranking">
        <caption className="sr-only">{card.label}, ranks out of {card.team_count}</caption>
        <thead><tr><th scope="col">Rank</th><th scope="col">Team</th><th scope="col">{card.label}</th></tr></thead>
        <tbody>{card.rows.map(row => <tr key={row.team}>
          <td>{ordinal(row.rank)}</td>
          <th scope="row"><span className="chat-team-cell"><TeamChip team={row.team} /><span>{row.name}</span></span></th>
          <td>{metricValue(row)}</td>
        </tr>)}</tbody>
      </table>
    </div>}
    <RouteButton route={card.route} />
  </article>;
}

type StandingRow = StandingsCard['teams'][number];

function StandingsTable({ title, teams }: { title: string; teams: StandingRow[] }) {
  const { shown, listId, toggle } = useCapped(teams, 10);
  return <section className="chat-standings-group" aria-label={title}>
    <h5 className="chat-card-label">{title}</h5>
    <div className="table-scroll" tabIndex={0} role="region" aria-label={`${title} standings`}>
      <table id={listId} className="chat-games chat-standings">
        <thead><tr><th scope="col"><abbr title="Seed">#</abbr></th><th scope="col">Team</th><th scope="col">W–L</th><th scope="col"><abbr title="Games back">GB</abbr></th></tr></thead>
        <tbody>{shown.map((team, index) => <tr key={team.team}>
          <td>{team.seed ?? index + 1}</td>
          <th scope="row"><TeamChip team={team.team} /><span className="sr-only"> {team.name}</span></th>
          <td>{team.wins}–{team.losses}</td>
          <td>{gamesBack(team.games_back)}</td>
        </tr>)}</tbody>
      </table>
    </div>
    {toggle}
  </section>;
}

function StandingsCardView({ card }: { card: StandingsCard }) {
  const bySeed = (x: StandingRow, y: StandingRow) => (x.seed ?? 99) - (y.seed ?? 99) || y.pct - x.pct;
  const groups = (['East', 'West'] as const)
    .map(conference => ({ title: `${conference}ern Conference`, teams: card.teams.filter(team => team.conference === conference).sort(bySeed).slice(0, 15) }))
    .filter(group => group.teams.length > 0);
  const unplaced = card.teams.filter(team => team.conference !== 'East' && team.conference !== 'West');
  if (!groups.length && unplaced.length) groups.push({ title: 'League', teams: unplaced.slice(0, 15) });
  return <article className="chat-card" aria-label={`Standings, ${card.season}`}>
    <header className="chat-card-head">
      <span className="chat-card-kicker">Standings</span>
      <h4>{card.season} standings</h4>
    </header>
    {groups.length > 0 ? <div className="chat-standings-grid">{groups.map(group => <StandingsTable key={group.title} title={group.title} teams={group.teams} />)}</div>
      : <p className="chat-card-text">No standings yet.</p>}
    <RouteButton route={card.route} />
  </article>;
}

function CoachesCardView({ card }: { card: CoachesCard }) {
  const head = card.coaches.filter(coach => coach.head);
  const { shown, listId, toggle } = useCapped(card.coaches.filter(coach => !coach.head), 8);
  return <article className="chat-card" aria-label={`${card.name} coaching staff`}>
    <header className="chat-card-head">
      <span className="chat-card-kicker">Coaching staff{card.season ? ` · ${card.season}` : ''}</span>
      <h4>{card.name}</h4>
    </header>
    {card.coaches.length > 0 ? <div className="staff chat-staff">
      {head.map(coach => <div key={coach.name} className="head-coach"><span className="eyebrow">HEAD COACH</span><strong>{coach.name}</strong></div>)}
      {shown.length > 0 && <ul id={listId}>{shown.map(coach => <li key={`${coach.name}-${coach.role}`}><span>{coach.name}</span><small>{coach.role}</small></li>)}</ul>}
    </div> : <p className="chat-card-text">The coaching staff couldn’t be loaded right now.</p>}
    {toggle}
    <RouteButton route={card.route} />
  </article>;
}

function MovesCardView({ card }: { card: MovesCard }) {
  const { shown, listId, toggle } = useCapped(card.transactions, 8);
  const sub = updated(card.fetched_at, card.stale);
  return <article className="chat-card" aria-label={card.title}>
    <header className="chat-card-head">
      <span className="chat-card-kicker">Roster moves</span>
      <h4>{card.title}</h4>
      {sub && <span className="chat-card-sub">{sub}</span>}
    </header>
    <SavedCopy stale={card.stale} fetchedAt={card.fetched_at} />
    {card.transactions.length > 0
      ? <ul id={listId} className="moves-list chat-moves">{shown.map((item, index) => <li key={`${item.date}-${index}`}>
        <time dateTime={item.date ?? undefined}>{item.date ? dayLabel(item.date) : '—'}</time>
        {item.team && <TeamChip team={item.team} />}<span>{item.description}</span>
      </li>)}</ul>
      : <p className="chat-card-text">No recent roster moves.</p>}
    {toggle}
    <RouteButton route={card.route} />
  </article>;
}

function LeagueCardView({ card }: { card: LeagueCard }) {
  switch (card.type) {
    case 'news': return <NewsCardView card={card} />;
    case 'injuries': return <InjuriesCardView card={card} />;
    case 'games': return <GamesCardView card={card} />;
    case 'team': return <TeamCardView card={card} />;
    case 'matchup': return <MatchupCardView card={card} />;
    case 'ranking': return <RankingCardView card={card} />;
    case 'standings': return <StandingsCardView card={card} />;
    case 'coaches': return <CoachesCardView card={card} />;
    case 'moves': return <MovesCardView card={card} />;
    case 'link': return <RouteButton route={card.route} label={card.label} />;
    default: return null;
  }
}

export function ChatCardView({ card, onOpenParlay, onOpenRecord, onOpenAccuracy, onOpenRoute }: {
  card: ChatCard; onOpenParlay: () => void; onOpenRecord: () => void; onOpenAccuracy: () => void;
  /** Where in-app links inside league and news cards go, so the drawer can close first. */
  onOpenRoute: Go;
}) {
  switch (card.type) {
    case 'forecast': return <ForecastCardView card={card} />;
    case 'parlay': return <ParlayCardView card={card} onOpenParlay={onOpenParlay} />;
    case 'player': return <PlayerCardView card={card} />;
    case 'track_record': return <RecordCardView record={card.record} onOpenRecord={onOpenRecord} />;
    case 'learning': return <LearningCardView status={card.status} onOpenAccuracy={onOpenAccuracy} />;
    case 'bet': return <BetCardView bet={card.bet} onOpenRecord={onOpenRecord} />;
    default: return <GoContext.Provider value={onOpenRoute}><LeagueCardView card={card} /></GoContext.Provider>;
  }
}

/** One malformed card should not take the conversation down with it. */
export class CardBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    return this.state.failed ? <p className="chat-card-error">This card couldn’t be shown.</p> : this.props.children;
  }
}
