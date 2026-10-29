import { BrainCircuit, CircleDot, RefreshCw, Settings2 } from 'lucide-react';
import { useState } from 'react';
import { pct, statNames } from './api';
import type { CalibrationStatus, StatCalibration } from './assistant-types';
import { CalibrationBadge } from './ChatCards';
import { Busy, EmptyState, ErrorNotice, SectionHeading } from './components';
import { timestampLabel } from './ledger';
import { RouteLink } from './RouteLink';
import type { Stat } from './types';
import type { LearningState } from './useLearning';

const fixedMinus = (value: number) => `${value < 0 ? '−' : ''}${Math.abs(value).toFixed(2)}`;

export const learningSentence = 'Coach gets smarter from graded results: every market (points, threes, PRA, double-doubles…) gets its own calibration, which switches on only when it beats the raw model on recent held-out games, and Coach remembers the nicknames you teach it and how you rate its answers.';

interface CurveSource { curve: CalibrationStatus['curve']; active: boolean }

function ReliabilityChart({ calibration, label }: { calibration: CurveSource; label: string }) {
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
      aria-label={`Reliability chart for ${label}. Predicted probability on the horizontal axis, how often it happened on the vertical axis. The diagonal is perfect calibration. Raw model points are orange; calibrated points are blue.`} onMouseLeave={() => setActive(null)}>
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

export function Holdout({ calibration, brier = true }: { calibration: CalibrationStatus; brier?: boolean }) {
  const holdout = calibration.holdout;
  if (!holdout) return <p className="learning-holdout-empty">No held-out test yet. Calibration stays off until it has recent games to prove itself on.</p>;
  const better = (raw: number, calibrated: number) => calibrated < raw;
  return <dl className="learning-holdout">
    <div><dt>Held-out games</dt><dd>{holdout.samples}</dd></div>
    <div><dt>Log loss</dt><dd>{holdout.raw_log_loss.toFixed(3)} <span aria-hidden="true">→</span><span className="sr-only">raw, then</span> <span className={better(holdout.raw_log_loss, holdout.calibrated_log_loss) ? 'positive' : 'negative'}>{holdout.calibrated_log_loss.toFixed(3)}</span> <small>calibrated</small></dd></div>
    {brier && <div><dt>Brier</dt><dd>{holdout.raw_brier.toFixed(3)} <span aria-hidden="true">→</span><span className="sr-only">raw, then</span> <span className={better(holdout.raw_brier, holdout.calibrated_brier) ? 'positive' : 'negative'}>{holdout.calibrated_brier.toFixed(3)}</span> <small>calibrated</small></dd></div>}
  </dl>;
}

const choiceNames: Record<StatCalibration['choice'], string> = { own: 'Own correction', shared: 'Shared correction', none: 'Raw model' };

/** Every market's calibration: what it learned from, how each option scored on recent held-out games, what is applied. */
function MarketTable({ stats, selected, onSelect }: { stats: StatCalibration[]; selected: Stat | 'all'; onSelect: (stat: Stat | 'all') => void }) {
  return <div className="table-scroll" tabIndex={0} role="region" aria-label="Calibration by market">
    <table className="data-table market-calibration">
      <thead><tr><th scope="col">Market</th><th scope="col">Graded</th><th scope="col">Held-out log loss<small>raw → applied</small></th><th scope="col">Correction</th><th scope="col">Applied</th></tr></thead>
      <tbody>{stats.map(entry => {
        const holdout = entry.holdout;
        const applied = holdout ? entry.choice === 'own' ? holdout.own_log_loss : entry.choice === 'shared' ? holdout.shared_log_loss : holdout.raw_log_loss : null;
        return <tr key={entry.stat} className={selected === entry.stat ? 'selected-row' : ''}>
          <th scope="row"><button type="button" className="text-button market-pick" aria-pressed={selected === entry.stat} onClick={() => onSelect(selected === entry.stat ? 'all' : entry.stat)} title="Show this market in the chart">{statNames[entry.stat] ?? entry.label}</button></th>
          <td className="mono">{entry.samples.toLocaleString()}</td>
          <td className="mono">{holdout && applied !== null ? <>{holdout.raw_log_loss.toFixed(3)} → <span className={applied < holdout.raw_log_loss ? 'positive' : ''}>{applied.toFixed(3)}</span></> : <span className="subtle">too few</span>}</td>
          <td className="mono">{entry.active && entry.intercept !== null && entry.slope !== null ? `${fixedMinus(entry.intercept)} + ${fixedMinus(entry.slope)}·x` : '—'}</td>
          <td><span className={`choice-pill choice-${entry.choice}`} title={entry.message}>{choiceNames[entry.choice]}</span></td>
        </tr>;
      })}</tbody>
    </table>
  </div>;
}

/** The live learning loop: how well the calibration layer is doing on recent held-out games, and a manual retrain. */
export default function LearningPanel({ learning }: { learning: LearningState }) {
  const { status, loading, error, retraining, retrainError, elapsed, load, retrain } = learning;
  const [announcement, setAnnouncement] = useState('');
  const [market, setMarket] = useState<Stat | 'all'>('all');
  const calibration = status?.calibration;
  const stats = calibration?.stats ?? [];
  const chosen = market === 'all' ? null : stats.find(entry => entry.stat === market) ?? null;
  const training = retraining || Boolean(status?.training);
  const announce = (text: string) => setAnnouncement(current => current === text ? `${text}\u00a0` : text);
  const own = stats.filter(entry => entry.choice === 'own').length;
  const shared = stats.filter(entry => entry.choice === 'shared').length;

  async function run() {
    if (training) return;
    const next = await retrain();
    announce(next ? `Retrained. ${next.calibration.message}` : 'The retrain did not finish.');
  }

  return <section className="card learning-panel" aria-labelledby="learning-title">
    <SectionHeading id="learning-title" eyebrow="FILM ROOM · LIVE" title="Learning loop">
      {calibration && <CalibrationBadge calibration={calibration} />}
    </SectionHeading>
    <p className="learning-explainer">{learningSentence}</p>
    {error && <ErrorNotice message={error} onRetry={() => load()} />}
    {!status && loading && <Busy text="Opening the film room…" />}
    {calibration && <div className="learning-layout">
      <div className="learning-main">
        <p className={`learning-message ${calibration.active || own + shared > 0 ? 'on' : 'off'}`}><BrainCircuit size={16} aria-hidden="true" /><span>{chosen ? chosen.message : calibration.message}</span></p>
        {stats.length > 0 && <label className="market-select"><span>Chart</span><select value={market} onChange={event => setMarket(event.target.value as Stat | 'all')}>
          <option value="all">All markets together</option>
          {stats.map(entry => <option key={entry.stat} value={entry.stat}>{statNames[entry.stat] ?? entry.label} · {choiceNames[entry.choice].toLowerCase()}</option>)}
        </select></label>}
        <ReliabilityChart calibration={chosen ?? calibration} label={chosen ? statNames[chosen.stat] : 'all markets'} />
      </div>
      <div className="learning-side">
        <h3 className="learning-subtitle">Held-out test{chosen ? ` · ${statNames[chosen.stat]}` : ' · all markets'}</h3>
        {chosen
          ? chosen.holdout ? <dl className="learning-holdout per-market">
            <div><dt>Held-out predictions</dt><dd>{chosen.holdout.samples}</dd></div>
            <div><dt>Log loss</dt><dd>{chosen.holdout.raw_log_loss.toFixed(3)} raw · {chosen.holdout.shared_log_loss.toFixed(3)} shared · <span className={chosen.holdout.own_log_loss < chosen.holdout.raw_log_loss ? 'positive' : ''}>{chosen.holdout.own_log_loss.toFixed(3)}</span> own</dd></div>
            <div><dt>Brier</dt><dd>{chosen.holdout.raw_brier.toFixed(3)} raw · {chosen.holdout.shared_brier.toFixed(3)} shared · {chosen.holdout.own_brier.toFixed(3)} own</dd></div>
          </dl> : <p className="learning-holdout-empty">Too few graded {statNames[chosen.stat].toLowerCase()} predictions for a separate test yet.</p>
          : <Holdout calibration={calibration} />}
        <h3 className="learning-subtitle">What it learned from</h3>
        <dl className="learning-samples">
          <div><dt>Graded predictions</dt><dd>{(chosen ?? calibration).samples}</dd></div>
          <div><dt>From your bets</dt><dd>{(chosen ?? calibration).ledger_samples}</dd></div>
          <div><dt>Players</dt><dd>{calibration.players}</dd></div>
          <div><dt>Last trained</dt><dd>{timestampLabel(calibration.trained_at)}</dd></div>
        </dl>
        {!chosen && calibration.intercept !== null && calibration.slope !== null && <p className="learning-fit">Shared correction: logit(q) = {fixedMinus(calibration.intercept)} + {fixedMinus(calibration.slope)} · logit(p)</p>}
        {chosen && chosen.intercept !== null && chosen.slope !== null && <p className="learning-fit">{choiceNames[chosen.choice]}: logit(q) = {fixedMinus(chosen.intercept)} + {fixedMinus(chosen.slope)} · logit(p)</p>}
        <div className="retrain-row">
          <button type="button" className="secondary-button" aria-disabled={training || undefined} onClick={() => { void run(); }} aria-describedby="retrain-hint">
            <RefreshCw size={16} aria-hidden="true" className={training ? 'spin' : ''} />{retraining ? 'Retraining…' : status?.training ? 'Retrain running…' : 'Retrain now'}
          </button>
          {retraining && <span className="retrain-elapsed" aria-hidden="true">{elapsed}s</span>}
          <span id="retrain-hint" className="record-hint">{retraining ? 'Simulating every saved game and testing each market’s correction. This can take a minute; you can keep using COURTSIDE.' : 'Refits every market on every graded result. Usually 10–60 seconds.'}</span>
        </div>
        {retraining && <div className="record-progress" role="status"><Busy text="Retraining on graded games…" /></div>}
        {retrainError && <ErrorNotice message={retrainError} onRetry={() => { void run(); }} />}
        <p className="learning-memory-link"><Settings2 size={15} aria-hidden="true" /><span>The nicknames and ratings Coach remembers are in <RouteLink to={{ page: 'settings', section: 'coach' }} focus="coach-memory-title">Settings › Coach</RouteLink>.</span></p>
      </div>
    </div>}
    {stats.length > 0 && <div className="market-calibration-block">
      <h3 className="learning-subtitle">Calibration by market · {own} own, {shared} shared, {stats.length - own - shared} raw</h3>
      <MarketTable stats={stats} selected={market} onSelect={setMarket} />
      <p className="record-hint">Each market's correction is fitted on its own graded predictions (shrunk toward the shared one) and kept only if it beat both the raw model and the shared correction on that market's most recent quarter. Pick a market to see its curve.</p>
    </div>}
    <div className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</div>
  </section>;
}
