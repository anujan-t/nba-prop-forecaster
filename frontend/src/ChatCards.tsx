import { ArrowRight, BrainCircuit, CircleCheck, CircleX, Clock3, Scale } from 'lucide-react';
import { Component, Fragment } from 'react';
import type { ReactNode } from 'react';
import { dateLabel, decimal, edgePoints, modelNames, pct, signed, statNames } from './api';
import type { BetStatus, CalibrationStatus, ChatCard, ForecastCard, HouseComparison, LearningStatus, ParlayCard, PlayerCard, TrackRecord, TrackedBet } from './assistant-types';
import { parseRichText } from './chat';
import type { InlineRun } from './chat';
import { ScoreboardDigits } from './components';
import { betStatusNames, legLine, legStatusNames, legsDetail, profitLabel, recordDetail, roiLabel, signedUnits, units } from './ledger';
import { americanLabel, chance, chanceDigits, decimalOdds, legNumber, sideNames, statLabel } from './parlay';

// ---------- Reply text ----------

function Runs({ runs }: { runs: InlineRun[] }) {
  return <>{runs.map((run, index) => run.bold ? <strong key={index}>{run.text}</strong> : <Fragment key={index}>{run.text}</Fragment>)}</>;
}

/** **bold**, line breaks and "- " bullets as React nodes. No HTML is ever injected. */
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
    <ol className="chat-legs">{result.legs.map((leg, position) => <li key={leg.index} className={`side-${leg.side}`}>
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
      <span className="chat-leg-p">{chance(leg.probability)}{leg.status !== 'pending' && <small>{legStatusNames[leg.status]}{leg.actual !== null ? ` · ${amount(leg.actual)}` : ''}</small>}</span>
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

export function ChatCardView({ card, onOpenParlay, onOpenRecord, onOpenAccuracy }: { card: ChatCard; onOpenParlay: () => void; onOpenRecord: () => void; onOpenAccuracy: () => void }) {
  switch (card.type) {
    case 'forecast': return <ForecastCardView card={card} />;
    case 'parlay': return <ParlayCardView card={card} onOpenParlay={onOpenParlay} />;
    case 'player': return <PlayerCardView card={card} />;
    case 'track_record': return <RecordCardView record={card.record} onOpenRecord={onOpenRecord} />;
    case 'learning': return <LearningCardView status={card.status} onOpenAccuracy={onOpenAccuracy} />;
    case 'bet': return <BetCardView bet={card.bet} onOpenRecord={onOpenRecord} />;
    default: return null;
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
