import { ArrowRight, ArrowUpRight, BrainCircuit, CircleCheck, CircleMinus, CircleX, ClipboardCheck, Clock3, Gavel, Trash2, TriangleAlert, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { dateLabel, isAbort, messageOf } from './api';
import type { LegStatus, SettleResult, TrackRecord as RecordSummary, TrackedBet } from './assistant-types';
import { BetStatusChip, CalibrationBadge, amount } from './ChatCards';
import { CoachMark } from './ChatPanel';
import { Busy, EmptyState, ErrorNotice, SectionHeading, StatCard } from './components';
import { deleteBet, fetchLedger, focusAfterRemoval, legLine, legResult, legStatusNames, legWhere, profitLabel, recordDetail, roiLabel, settleBets, signedUnits, timestampLabel, units } from './ledger';
import type { Ledger } from './ledger';
import { americanLabel, chance } from './parlay';
import { RouteLink } from './RouteLink';
import type { LearningState } from './useLearning';

const legIcons: { [status in LegStatus]: ReactNode } = {
  pending: <Clock3 size={15} aria-hidden="true" />,
  won: <CircleCheck size={15} aria-hidden="true" />,
  lost: <CircleX size={15} aria-hidden="true" />,
  push: <CircleMinus size={15} aria-hidden="true" />,
  void: <CircleMinus size={15} aria-hidden="true" />,
};

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
      <span className="record-leg-line">{legLine(leg)}{legWhere(leg) && <small> · {legWhere(leg)}</small>}</span>
      <span className="record-leg-result">{leg.actual !== null
        ? leg.kind === 'game' ? <>Final <strong>{legResult(leg, amount)}</strong></> : <>Actual <strong>{amount(leg.actual)}</strong></>
        : leg.status === 'void' ? (leg.kind === 'game' ? 'Postponed' : 'Did not play') : leg.status === 'pending' ? 'Not graded' : legStatusNames[leg.status]}</span>
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

/** The short version of How it works › Accuracy: is the calibration on, and did it beat the raw model? */
function AccuracySummary({ learning }: { learning: LearningState }) {
  const { status, loading, error, load } = learning;
  const calibration = status?.calibration;
  const holdout = calibration?.holdout ?? null;
  return <section className="card accuracy-summary" aria-labelledby="accuracy-summary-title">
    <SectionHeading id="accuracy-summary-title" eyebrow="FILM ROOM" title="Model accuracy">
      {calibration && <CalibrationBadge calibration={calibration} />}
    </SectionHeading>
    {error && <ErrorNotice message={error} onRetry={() => load()} />}
    {!status && loading && <Busy text="Checking the film…" />}
    {calibration && <>
      <p className={`learning-message ${calibration.active ? 'on' : 'off'}`}><BrainCircuit size={16} aria-hidden="true" /><span>{calibration.message}</span></p>
      <dl className="accuracy-facts">
        <div className="wide"><dt>Held-out log loss</dt><dd>{holdout
          ? <><span className="accuracy-before">{holdout.raw_log_loss.toFixed(3)}</span> <span aria-hidden="true">→</span><span className="sr-only">raw model, then</span> <span className={holdout.calibrated_log_loss < holdout.raw_log_loss ? 'positive' : 'negative'}>{holdout.calibrated_log_loss.toFixed(3)}</span> <small>calibrated · lower is better · {holdout.samples.toLocaleString()} recent games</small></>
          : <small>Not tested yet</small>}</dd></div>
        <div><dt>Graded predictions</dt><dd>{calibration.samples.toLocaleString()}</dd></div>
        <div><dt>From your bets</dt><dd>{calibration.ledger_samples.toLocaleString()}</dd></div>
      </dl>
    </>}
    <p className="record-hint">Graded results retrain a calibration layer, and it only switches on when it beats the raw model on recent held-out games.</p>
    <RouteLink to={{ page: 'how', section: 'accuracy' }} focus="learning-title" className="secondary-button accuracy-link">Full accuracy report<ArrowRight size={15} aria-hidden="true" /></RouteLink>
  </section>;
}

interface Props {
  active: boolean;
  version: number;
  learning: LearningState;
  onParlay: () => void;
  onCoach: () => void;
}

export default function MyBets({ active, version, learning, onParlay, onCoach }: Props) {
  const [ledger, setLedger] = useState<Ledger | null>(null);
  const [ledgerLoading, setLedgerLoading] = useState(false);
  const [ledgerError, setLedgerError] = useState('');
  const [settling, setSettling] = useState(false);
  const [settled, setSettled] = useState<SettleResult | null>(null);
  const [settleError, setSettleError] = useState('');
  const [confirming, setConfirming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<{ id: string; message: string } | null>(null);
  const [announcement, setAnnouncement] = useState('');
  const ledgerController = useRef<AbortController | null>(null);
  const betsHeading = useRef<HTMLHeadingElement>(null);
  const list = useRef<HTMLOListElement>(null);
  const reloadLearning = learning.load;

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

  // Refetch whenever the page opens or something elsewhere (Coach, the slip) changed the ledger.
  useEffect(() => {
    if (active) void loadLedger();
  }, [active, version, loadLedger]);

  useEffect(() => () => ledgerController.current?.abort(), []);

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
      reloadLearning(true);
    } catch (reason) {
      setSettleError(messageOf(reason));
    } finally {
      setSettling(false);
    }
  }

  async function remove(bet: TrackedBet) {
    setDeleting(bet.id);
    setDeleteError(null);
    const refocus = focusAfterRemoval(list.current, 'data-bet-id', bet.id, betsHeading.current, item => item);
    try {
      await deleteBet(bet.id);
      refocus();
      setConfirming(null);
      setLedger(current => current ? { ...current, bets: current.bets.filter(item => item.id !== bet.id) } : current);
      announce('Deleted the tracked bet.');
      void loadLedger();
      reloadLearning(true);
    } catch (reason) {
      setDeleteError({ id: bet.id, message: messageOf(reason) });
    } finally {
      setDeleting(null);
    }
  }

  const record = ledger?.record ?? learning.status?.record ?? null;
  const bets = ledger ? [...ledger.bets].sort((a, b) => b.created_at.localeCompare(a.created_at)) : [];

  return <div className="record-view">
    <div className="notice research-notice record-intro">
      <ClipboardCheck size={18} aria-hidden="true" />
      <div><strong>Keep score honestly.</strong><p>Tracked parlays are graded against final box scores. Results feed the learning loop. A research record, not betting advice.</p></div>
    </div>
    {ledgerError && <ErrorNotice message={ledgerError} onRetry={() => { void loadLedger(); }} />}
    {record ? <SummaryTiles record={record} /> : ledgerLoading && <div className="record-loading"><Busy text="Pulling up the scoreboard…" /></div>}
    <div className="record-grid">
      <section className="card record-bets" aria-labelledby="record-bets-title">
        <SectionHeading id="record-bets-title" eyebrow="THE LEDGER" title="Tracked bets" headingRef={betsHeading}>
          <button type="button" className="primary-button" onClick={() => { void grade(); }} disabled={!settling && !ledger?.bets.length} aria-disabled={settling || undefined} aria-describedby="grade-hint"><Gavel size={17} aria-hidden="true" />{settling ? 'Grading…' : 'Grade finished games'}</button>
        </SectionHeading>
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
      <AccuracySummary learning={learning} />
    </div>
    <div className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</div>
  </div>;
}
