import { ArrowUpRight, BrainCircuit, CircleCheck, CircleDot, CircleMinus, CircleX, ClipboardCheck, Clock3, Gavel, RefreshCw, ThumbsDown, ThumbsUp, Trash2, TriangleAlert, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode, Ref } from 'react';
import { dateLabel, isAbort, messageOf, pct } from './api';
import type { CalibrationStatus, LearningStatus, LegStatus, SettleResult, TrackRecord as RecordSummary, TrackedBet } from './assistant-types';
import { BetStatusChip, CalibrationBadge, amount } from './ChatCards';
import { CoachMark } from './ChatPanel';
import { Busy, CourtLines, EmptyState, ErrorNotice, StatCard } from './components';
import { deleteBet, fetchLearning, fetchLedger, forgetAlias, legLine, legStatusNames, profitLabel, recordDetail, retrainLearning, roiLabel, settleBets, signedUnits, units } from './ledger';
import type { Ledger } from './ledger';
import { americanLabel, chance } from './parlay';

const legIcons: { [status in LegStatus]: ReactNode } = {
  pending: <Clock3 size={15} aria-hidden="true" />,
  won: <CircleCheck size={15} aria-hidden="true" />,
  lost: <CircleX size={15} aria-hidden="true" />,
  push: <CircleMinus size={15} aria-hidden="true" />,
  void: <CircleMinus size={15} aria-hidden="true" />,
};

function timestampLabel(value: string | null): string {
  if (!value) return 'Never';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit' });
}

const fixedMinus = (value: number) => `${value < 0 ? '−' : ''}${Math.abs(value).toFixed(2)}`;

function Heading({ id, eyebrow, title, children, headingRef }: { id: string; eyebrow: string; title: string; children?: ReactNode; headingRef?: Ref<HTMLHeadingElement> }) {
  return <div className="section-heading">
    <div className="section-heading-copy"><span className="eyebrow"><CourtLines />{eyebrow}</span><h2 id={id} ref={headingRef} tabIndex={-1}>{title}</h2></div>
    {children}
  </div>;
}

function SummaryTiles({ record }: { record: RecordSummary }) {
  const profitTone = record.profit > 0 ? 'positive' : record.profit < 0 ? 'negative' : '';
  return <div className="record-stat-grid">
    <StatCard accent label="RECORD" value={<strong className="record-score-value"><span aria-hidden="true">{record.won}–{record.lost}</span><span className="sr-only">{record.won} won, {record.lost} lost</span></strong>}
      detail={<><strong>{record.bets} tracked {record.bets === 1 ? 'parlay' : 'parlays'}</strong><span>{recordDetail(record)}</span></>} />
    <StatCard label="PROFIT" value={<strong className={profitTone}>{signedUnits(record.profit)}</strong>} unit="units"
      detail={<><strong>ROI {roiLabel(record.roi)}</strong><span>on {units(record.staked)} staked with a known result</span></>} />
    <StatCard label="LEGS WON" value={record.legs_graded ? <>{record.legs_won}<span className="record-of">/{record.legs_graded}</span></> : '—'}
      detail={<><strong>Model expected {record.expected_legs_won.toFixed(1)}</strong><span>{record.legs_graded ? 'Won vs the sum of the model’s leg probabilities' : 'No graded legs yet'}</span></>} />
    <StatCard label="BRIER SCORE" value={record.brier === null ? '—' : record.brier.toFixed(3)}
      detail={<><strong>Lower is better</strong><span>Always guessing 50% scores 0.250</span></>} />
  </div>;
}

function BetItem({ bet, confirming, deleting, error, onAsk, onCancel, onDelete }: { bet: TrackedBet; confirming: boolean; deleting: boolean; error: string; onAsk: () => void; onCancel: () => void; onDelete: () => void }) {
  const keepButton = useRef<HTMLButtonElement>(null);
  const askButton = useRef<HTMLButtonElement>(null);
  const restore = useRef(false);
  useEffect(() => {
    if (confirming) keepButton.current?.focus();
    else if (restore.current) { restore.current = false; askButton.current?.focus(); }
  }, [confirming]);
  const legCount = `${bet.legs.length} ${bet.legs.length === 1 ? 'leg' : 'legs'}`;
  const profitTone = bet.profit !== null && bet.status !== 'pending' ? bet.profit > 0 ? 'positive' : bet.profit < 0 ? 'negative' : '' : '';
  const titleId = `bet-${bet.id}-title`;
  return <li className={`record-bet status-${bet.status}`} data-bet-id={bet.id} tabIndex={-1} aria-labelledby={titleId}>
    <div className="record-bet-head">
      <BetStatusChip status={bet.status} />
      <h3 id={titleId}>{legCount} · game date {dateLabel(bet.event_date, true)}</h3>
      <span className={`record-bet-profit ${profitTone}`}><span className="sr-only">Profit: </span>{profitLabel(bet)}</span>
    </div>
    <ol className="record-legs">{bet.legs.map((leg, index) => <li key={`${leg.player_id}-${leg.stat}-${index}`} className={`leg-${leg.status}`}>
      <span className="record-leg-mark">{legIcons[leg.status]}<span className="sr-only">{legStatusNames[leg.status]}:</span></span>
      <span className="record-leg-line">{legLine(leg)}{leg.team && <small> · {leg.team}</small>}</span>
      <span className="record-leg-result">{leg.actual !== null ? <>Actual <strong>{amount(leg.actual)}</strong></> : leg.status === 'void' ? 'Did not play' : leg.status === 'pending' ? 'Not graded' : legStatusNames[leg.status]}</span>
      <span className="record-leg-p">Model {chance(leg.probability)}</span>
    </li>)}</ol>
    <dl className="record-bet-facts">
      <div><dt>Model, all legs</dt><dd>{chance(bet.probability)} <small>(independent {chance(bet.independent_probability)})</small></dd></div>
      <div><dt>Book odds</dt><dd>{bet.book_odds === null ? 'None' : americanLabel(bet.book_odds)}</dd></div>
      <div><dt>Stake</dt><dd>{bet.stake === null ? 'None' : units(bet.stake)}</dd></div>
      <div><dt>Profit</dt><dd className={profitTone}>{profitLabel(bet)}</dd></div>
    </dl>
    {bet.note && <p className="record-bet-note">“{bet.note}”</p>}
    <div className="record-bet-foot">
      <span>Tracked {timestampLabel(bet.created_at)}</span>
      {confirming
        ? <div className="clear-confirm record-delete-confirm" role="group" aria-label="Confirm delete"
          onKeyDown={event => { if (event.key === 'Escape' && !deleting) { event.stopPropagation(); restore.current = true; onCancel(); } }}>
          <span>Delete this tracked bet?</span>
          <button type="button" className="danger-button" aria-disabled={deleting || undefined} onClick={() => { if (!deleting) onDelete(); }}>{deleting ? 'Deleting…' : 'Delete'}</button>
          <button ref={keepButton} type="button" className="text-button" aria-disabled={deleting || undefined} onClick={() => { if (deleting) return; restore.current = true; onCancel(); }}>Keep</button>
        </div>
        : <button ref={askButton} type="button" className="text-button record-delete" onClick={onAsk} aria-label={`Delete tracked bet: ${legCount}, game date ${dateLabel(bet.event_date, true)}`}><Trash2 size={14} aria-hidden="true" />Delete</button>}
    </div>
    {error && <p className="field-error" role="alert">{error}</p>}
  </li>;
}

function ReliabilityChart({ calibration }: { calibration: CalibrationStatus }) {
  const [active, setActive] = useState<number | null>(null);
  const bins = calibration.curve
    .filter(bin => bin.count > 0 && [bin.predicted, bin.observed, bin.calibrated].every(value => Number.isFinite(value)))
    .sort((a, b) => a.predicted - b.predicted);
  const width = 480;
  const height = 300;
  const left = 52;
  const top = 16;
  const plotWidth = width - left - 22;
  const plotHeight = height - top - 50;
  const clamp = (value: number) => Math.min(1, Math.max(0, value));
  const x = (value: number) => left + clamp(value) * plotWidth;
  const y = (value: number) => top + (1 - clamp(value)) * plotHeight;
  const selected = active === null ? undefined : bins[active];
  if (!bins.length) return <EmptyState title="No graded predictions yet" icon={<BrainCircuit size={30} />}>The chart fills in once tracked legs are graded or a retrain scores recent games.</EmptyState>;
  const calibratedLabel = calibration.active ? 'Calibrated' : 'Calibrated (not in use)';
  return <div className="chart-component reliability">
    <div className="chart-toolbar"><div className="chart-legend">
      <span><i className="legend-dot over-dot" />Raw model</span>
      <span><i className="legend-dot calibrated-dot" />{calibratedLabel}</span>
      <span><i className="legend-dash" />Perfect calibration</span>
    </div></div>
    <svg viewBox={`0 0 ${width} ${height}`} className="data-chart calibration-chart reliability-chart" role="group"
      aria-label="Reliability chart. Predicted probability on the horizontal axis, how often it happened on the vertical axis. The diagonal is perfect calibration. Raw model points are orange; calibrated points are blue." onMouseLeave={() => setActive(null)}>
      {[0, 0.25, 0.5, 0.75, 1].map(value => <g key={value} className="chart-grid">
        <line x1={left} x2={left + plotWidth} y1={y(value)} y2={y(value)} />
        <text x={left - 12} y={y(value) + 4} textAnchor="end">{pct(value, 0)}</text>
        <text x={x(value)} y={height - 27} textAnchor="middle">{pct(value, 0)}</text>
      </g>)}
      <path d={`M${x(0)},${y(0)} L${x(1)},${y(1)}`} className="ideal-line" />
      <path d={bins.map((bin, index) => `${index === 0 ? 'M' : 'L'}${x(bin.calibrated)},${y(bin.observed)}`).join(' ')} className={`calibrated-curve${calibration.active ? '' : ' inactive'}`} />
      <path d={bins.map((bin, index) => `${index === 0 ? 'M' : 'L'}${x(bin.predicted)},${y(bin.observed)}`).join(' ')} className="chart-curve" />
      {bins.map((bin, index) => <rect key={`c-${index}`} x={x(bin.calibrated) - 4.5} y={y(bin.observed) - 4.5} width={9} height={9} rx={1.5}
        transform={`rotate(45 ${x(bin.calibrated)} ${y(bin.observed)})`} className={`calibrated-point${index === active ? ' is-active' : ''}${calibration.active ? '' : ' inactive'}`} aria-hidden="true" />)}
      {bins.map((bin, index) => <circle key={`r-${index}`} cx={x(bin.predicted)} cy={y(bin.observed)} r={index === active ? 8 : 5} className="chart-point" tabIndex={0} role="img"
        aria-label={`Predicted ${pct(bin.predicted)}, calibrated ${pct(bin.calibrated)}, happened ${pct(bin.observed)}, ${bin.count} legs`}
        onFocus={() => setActive(index)} onBlur={() => setActive(null)} onMouseEnter={() => setActive(index)}>
        <title>{pct(bin.predicted)} raw · {pct(bin.calibrated)} calibrated · {pct(bin.observed)} happened · n={bin.count}</title>
      </circle>)}
      <text className="axis-title" x={left + plotWidth / 2} y={height - 5} textAnchor="middle">PREDICTED PROBABILITY</text>
      <text className="axis-title" transform={`translate(12 ${top + plotHeight / 2}) rotate(-90)`} textAnchor="middle">HOW OFTEN IT HAPPENED</text>
    </svg>
    <div className="chart-readout" aria-live="polite"><CircleDot size={13} aria-hidden="true" />{selected
      ? <><strong>{pct(selected.predicted)} raw → {pct(selected.calibrated)} calibrated → {pct(selected.observed)} happened</strong><span>n = {selected.count}</span></>
      : <><span>Closer to the diagonal is better calibrated</span><span>{bins.length} {bins.length === 1 ? 'bin' : 'bins'}</span></>}</div>
  </div>;
}

function Holdout({ calibration }: { calibration: CalibrationStatus }) {
  const holdout = calibration.holdout;
  if (!holdout) return <p className="learning-holdout-empty">No held-out test yet. Calibration stays off until it has recent games to prove itself on.</p>;
  const better = (raw: number, calibrated: number) => calibrated < raw;
  return <dl className="learning-holdout">
    <div><dt>Held-out games</dt><dd>{holdout.samples}</dd></div>
    <div><dt>Log loss</dt><dd>{holdout.raw_log_loss.toFixed(3)} <span aria-hidden="true">→</span><span className="sr-only">raw, then</span> <span className={better(holdout.raw_log_loss, holdout.calibrated_log_loss) ? 'positive' : 'negative'}>{holdout.calibrated_log_loss.toFixed(3)}</span> <small>calibrated</small></dd></div>
    <div><dt>Brier</dt><dd>{holdout.raw_brier.toFixed(3)} <span aria-hidden="true">→</span><span className="sr-only">raw, then</span> <span className={better(holdout.raw_brier, holdout.calibrated_brier) ? 'positive' : 'negative'}>{holdout.calibrated_brier.toFixed(3)}</span> <small>calibrated</small></dd></div>
  </dl>;
}

interface Props {
  active: boolean;
  version: number;
  onParlay: () => void;
  onCoach: () => void;
}

export default function TrackRecord({ active, version, onParlay, onCoach }: Props) {
  const [ledger, setLedger] = useState<Ledger | null>(null);
  const [ledgerLoading, setLedgerLoading] = useState(false);
  const [ledgerError, setLedgerError] = useState('');
  const [learning, setLearning] = useState<LearningStatus | null>(null);
  const [learningLoading, setLearningLoading] = useState(false);
  const [learningError, setLearningError] = useState('');
  const [settling, setSettling] = useState(false);
  const [settled, setSettled] = useState<SettleResult | null>(null);
  const [settleError, setSettleError] = useState('');
  const [retraining, setRetraining] = useState(false);
  const [retrainError, setRetrainError] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<{ id: string; message: string } | null>(null);
  const [forgetting, setForgetting] = useState<string | null>(null);
  const [aliasError, setAliasError] = useState('');
  const [announcement, setAnnouncement] = useState('');
  const ledgerController = useRef<AbortController | null>(null);
  const learningController = useRef<AbortController | null>(null);
  const betsHeading = useRef<HTMLHeadingElement>(null);
  const memoryHeading = useRef<HTMLHeadingElement>(null);
  const list = useRef<HTMLOListElement>(null);
  const aliasList = useRef<HTMLUListElement>(null);

  const announce = (text: string) => setAnnouncement(current => current === text ? `${text}\u00a0` : text);

  const loadLedger = useCallback(async () => {
    ledgerController.current?.abort();
    const controller = new AbortController();
    ledgerController.current = controller;
    setLedgerLoading(true);
    setLedgerError('');
    try {
      const data = await fetchLedger(controller.signal);
      if (!controller.signal.aborted) setLedger({ bets: Array.isArray(data?.bets) ? data.bets : [], record: data.record });
    } catch (reason) {
      if (!isAbort(reason) && !controller.signal.aborted) setLedgerError(messageOf(reason));
    } finally {
      if (ledgerController.current === controller) setLedgerLoading(false);
    }
  }, []);

  const loadLearning = useCallback(async (quiet = false) => {
    learningController.current?.abort();
    const controller = new AbortController();
    learningController.current = controller;
    if (!quiet) { setLearningLoading(true); setLearningError(''); }
    try {
      const data = await fetchLearning(controller.signal);
      if (!controller.signal.aborted) { setLearning(data); setLearningError(''); }
    } catch (reason) {
      if (!isAbort(reason) && !controller.signal.aborted && !quiet) setLearningError(messageOf(reason));
    } finally {
      if (learningController.current === controller) setLearningLoading(false);
    }
  }, []);

  // Refetch whenever the view opens or something elsewhere (the chat) changed the ledger.
  useEffect(() => {
    if (!active) return;
    void loadLedger();
    if (!retraining) void loadLearning();
  }, [active, version, loadLedger, loadLearning]);

  useEffect(() => () => { ledgerController.current?.abort(); learningController.current?.abort(); }, []);

  // A background retrain (started by grading or by Coach) finishes on its own; check back while the view is open.
  useEffect(() => {
    if (!active || retraining || !learning?.training) return;
    const timer = window.setTimeout(() => { void loadLearning(true); }, 4000);
    return () => window.clearTimeout(timer);
  }, [active, retraining, learning, loadLearning]);

  useEffect(() => {
    if (!retraining) return;
    const started = Date.now();
    setElapsed(0);
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [retraining]);

  async function grade() {
    if (settling) return;
    setSettling(true);
    setSettleError('');
    setSettled(null);
    try {
      const result = await settleBets();
      setSettled(result);
      setLedger(current => current ? { ...current, record: result.record } : current);
      announce(result.settled_legs
        ? `Graded ${result.settled_legs} ${result.settled_legs === 1 ? 'leg' : 'legs'} across ${result.updated_bets} ${result.updated_bets === 1 ? 'bet' : 'bets'}.`
        : 'Nothing new to grade yet.');
      void loadLedger();
      void loadLearning(true);
    } catch (reason) {
      setSettleError(messageOf(reason));
    } finally {
      setSettling(false);
    }
  }

  async function retrain() {
    if (retraining || learning?.training) return;
    learningController.current?.abort();
    setRetraining(true);
    setRetrainError('');
    try {
      const status = await retrainLearning();
      setLearning(status);
      announce(`Retrained. ${status.calibration.active ? 'Calibration is on.' : 'Calibration is off.'} ${status.calibration.message}`);
    } catch (reason) {
      setRetrainError(messageOf(reason));
      announce('The retrain did not finish.');
    } finally {
      setRetraining(false);
    }
  }

  /** Focus moves to the neighbouring item (or the section heading) so it is never lost with the removed row. */
  function focusAfterRemoval(container: HTMLElement | null, attribute: string, removed: string, fallback: HTMLElement | null, pick: (item: HTMLElement) => HTMLElement | null) {
    const items = container ? Array.from(container.querySelectorAll<HTMLElement>(`[${attribute}]`)) : [];
    const index = items.findIndex(item => item.getAttribute(attribute) === removed);
    const next = index < 0 ? undefined : items[index + 1] ?? items[index - 1];
    window.requestAnimationFrame(() => {
      const target = next && document.contains(next) ? pick(next) : fallback;
      (target ?? fallback)?.focus();
    });
  }

  async function remove(bet: TrackedBet) {
    setDeleting(bet.id);
    setDeleteError(null);
    try {
      await deleteBet(bet.id);
      focusAfterRemoval(list.current, 'data-bet-id', bet.id, betsHeading.current, item => item);
      setConfirming(null);
      setLedger(current => current ? { ...current, bets: current.bets.filter(item => item.id !== bet.id) } : current);
      announce('Deleted the tracked bet.');
      void loadLedger();
      void loadLearning(true);
    } catch (reason) {
      setDeleteError({ id: bet.id, message: messageOf(reason) });
    } finally {
      setDeleting(null);
    }
  }

  async function forget(phrase: string, player: string) {
    if (forgetting) return;
    setForgetting(phrase);
    setAliasError('');
    try {
      await forgetAlias(phrase);
      focusAfterRemoval(aliasList.current, 'data-alias', phrase, memoryHeading.current, item => item.querySelector<HTMLElement>('button'));
      setLearning(current => current ? { ...current, memory: { ...current.memory, aliases: current.memory.aliases.filter(alias => alias.phrase !== phrase) } } : current);
      announce(`Forgot that “${phrase}” means ${player}.`);
    } catch (reason) {
      setAliasError(messageOf(reason));
    } finally {
      setForgetting(null);
    }
  }

  const record = ledger?.record ?? learning?.record ?? null;
  const bets = ledger ? [...ledger.bets].sort((a, b) => b.created_at.localeCompare(a.created_at)) : [];
  const calibration = learning?.calibration;
  const memory = learning?.memory;
  const training = retraining || Boolean(learning?.training);

  return <div className="record-view">
    <div className="notice research-notice record-intro">
      <ClipboardCheck size={18} aria-hidden="true" />
      <div><strong>Keep score honestly.</strong><p>Tracked parlays are graded against final box scores. Results feed the learning loop. A research record, not betting advice.</p></div>
    </div>
    {ledgerError && <ErrorNotice message={ledgerError} onRetry={() => { void loadLedger(); }} />}
    {record ? <SummaryTiles record={record} /> : ledgerLoading && <div className="record-loading"><Busy text="Pulling up the scoreboard…" /></div>}
    <div className="record-grid">
      <section className="card record-bets" aria-labelledby="record-bets-title">
        <Heading id="record-bets-title" eyebrow="THE LEDGER" title="Tracked bets" headingRef={betsHeading}>
          <button type="button" className="primary-button" onClick={() => { void grade(); }} disabled={!settling && !ledger?.bets.length} aria-disabled={settling || undefined} aria-describedby="grade-hint"><Gavel size={17} aria-hidden="true" />{settling ? 'Grading…' : 'Grade finished games'}</button>
        </Heading>
        <p id="grade-hint" className="record-hint">Grading checks final NBA box scores for every pending leg. A player who didn’t play voids his leg.</p>
        {settling && <div className="record-progress"><Busy text="Checking final box scores…" /></div>}
        {settleError && <ErrorNotice message={settleError} onRetry={() => { void grade(); }} />}
        {settled && <div className={`notice ${settled.warnings.length ? 'warning-notice' : 'research-notice'} settle-notice`}>
          {settled.warnings.length ? <TriangleAlert size={18} aria-hidden="true" /> : <CircleCheck size={18} aria-hidden="true" />}
          <div>
            <strong>{settled.settled_legs ? `Graded ${settled.settled_legs} ${settled.settled_legs === 1 ? 'leg' : 'legs'} across ${settled.updated_bets} ${settled.updated_bets === 1 ? 'bet' : 'bets'}.` : 'Nothing new to grade yet.'}</strong>
            {settled.warnings.length > 0 && <ul className="settle-warnings">{settled.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>}
          </div>
          <button type="button" className="icon-button settle-dismiss" aria-label="Dismiss grading summary" onClick={() => setSettled(null)}><X size={16} aria-hidden="true" /></button>
        </div>}
        {!ledger && ledgerLoading && <Busy text="Loading tracked bets…" />}
        {ledger && !bets.length && <EmptyState title="Nothing tracked yet" icon={<ClipboardCheck size={30} />}
          action={<div className="record-empty-actions">
            <button type="button" className="primary-button" onClick={onParlay}>Open the Parlay builder <ArrowUpRight size={15} aria-hidden="true" /></button>
            <button type="button" className="secondary-button" onClick={onCoach}><CoachMark className="inline-mark" />Ask Coach</button>
          </div>}>
          Build a parlay and press <strong>Track this parlay</strong>, or tell Coach “track this for $10”. Once the games are final, grade them here.
        </EmptyState>}
        {bets.length > 0 && <ol className="record-bet-list" ref={list}>{bets.map(bet => <BetItem key={bet.id} bet={bet}
          confirming={confirming === bet.id} deleting={deleting === bet.id} error={deleteError?.id === bet.id ? deleteError.message : ''}
          onAsk={() => { setConfirming(bet.id); setDeleteError(null); }} onCancel={() => setConfirming(null)} onDelete={() => { void remove(bet); }} />)}</ol>}
      </section>
      <section className="card learning-panel" aria-labelledby="learning-title">
        <Heading id="learning-title" eyebrow="FILM ROOM" title="Learning loop">
          {calibration && <CalibrationBadge calibration={calibration} />}
        </Heading>
        <p className="learning-explainer">Coach gets smarter from graded results: a calibration layer retrains on them and only switches on when it beats the raw model on recent held-out games, and it remembers the nicknames you teach it and how you rate its answers.</p>
        {learningError && <ErrorNotice message={learningError} onRetry={() => { void loadLearning(); }} />}
        {!learning && learningLoading && <Busy text="Opening the film room…" />}
        {calibration && <>
          <p className={`learning-message ${calibration.active ? 'on' : 'off'}`}><BrainCircuit size={16} aria-hidden="true" /><span>{calibration.message}</span></p>
          <ReliabilityChart calibration={calibration} />
          <Holdout calibration={calibration} />
          <dl className="learning-samples">
            <div><dt>Graded predictions</dt><dd>{calibration.samples}</dd></div>
            <div><dt>From your bets</dt><dd>{calibration.ledger_samples}</dd></div>
            <div><dt>Players</dt><dd>{calibration.players}</dd></div>
            <div><dt>Last trained</dt><dd>{timestampLabel(calibration.trained_at)}</dd></div>
          </dl>
          {calibration.intercept !== null && calibration.slope !== null && <p className="learning-fit">Correction: logit(q) = {fixedMinus(calibration.intercept)} + {fixedMinus(calibration.slope)} · logit(p)</p>}
          <div className="retrain-row">
            <button type="button" className="secondary-button" aria-disabled={training || undefined} onClick={() => { void retrain(); }} aria-describedby="retrain-hint">
              <RefreshCw size={16} aria-hidden="true" className={training ? 'spin' : ''} />{retraining ? 'Retraining…' : learning?.training ? 'Retrain running…' : 'Retrain now'}
            </button>
            {retraining && <span className="retrain-elapsed" aria-hidden="true">{elapsed}s</span>}
            <span id="retrain-hint" className="record-hint">{retraining ? 'Scoring recent games and testing the correction. This can take up to a minute; you can keep using COURTSIDE.' : 'Refits the calibration on every graded result. Usually 5–40 seconds.'}</span>
          </div>
          {retraining && <div className="record-progress" role="status"><Busy text="Retraining on graded games…" /></div>}
          {retrainError && <ErrorNotice message={retrainError} onRetry={() => { void retrain(); }} />}
        </>}
        {memory && <div className="learning-memory">
          <h3 id="memory-title" ref={memoryHeading} tabIndex={-1}>Memory</h3>
          <dl className="memory-counts">
            <div><dt><ThumbsUp size={14} aria-hidden="true" />Helpful</dt><dd>{memory.liked}</dd></div>
            <div><dt><ThumbsDown size={14} aria-hidden="true" />Not helpful</dt><dd>{memory.disliked}</dd></div>
            <div><dt>Turns</dt><dd>{memory.turns}</dd></div>
          </dl>
          <h4 className="alias-title">Nicknames learned</h4>
          {memory.aliases.length
            ? <ul className="alias-list" ref={aliasList}>{memory.aliases.map(alias => <li key={alias.phrase} data-alias={alias.phrase}>
              <span className="alias-text"><span className="alias-phrase">“{alias.phrase}”</span> <span aria-hidden="true">→</span><span className="sr-only">means</span> <strong>{alias.player}</strong>
                <small>{alias.source === 'taught' ? 'You taught it' : 'From a clarification'} · used {alias.uses} {alias.uses === 1 ? 'time' : 'times'}</small></span>
              <button type="button" className="text-button" aria-disabled={forgetting === alias.phrase || undefined} onClick={() => { void forget(alias.phrase, alias.player); }} aria-label={`Forget that “${alias.phrase}” means ${alias.player}`}>{forgetting === alias.phrase ? 'Forgetting…' : 'Forget'}</button>
            </li>)}</ul>
            : <p className="record-hint">None yet. Tell Coach “joker means Nikola Jokić”, or pick an answer when it asks which player you meant.</p>}
          {aliasError && <p className="field-error" role="alert">{aliasError}</p>}
        </div>}
      </section>
    </div>
    <div className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</div>
  </div>;
}
