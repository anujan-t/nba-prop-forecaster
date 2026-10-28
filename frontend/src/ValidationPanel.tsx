import { useEffect, useRef, useState } from 'react';
import { ArrowRight, FlaskConical, Info, Play, ShieldCheck } from 'lucide-react';
import { api, dateLabel, decimal, isAbort, messageOf, modelNames, pct, signed, sourceLabel, statNames } from './api';
import { CalibrationChart } from './Charts';
import { Busy, EmptyState, ErrorNotice, Field, SectionHeading, StatCard } from './components';
import type { AppliedBacktest, AppliedForecast, Backtest, BacktestRequest } from './types';

export default function ValidationPanel({ applied, datasetId, onForecast }: { applied: AppliedForecast | null; datasetId: string; onForecast: () => void }) {
  const [scope, setScope] = useState('player');
  const [lineMode, setLineMode] = useState('fixed');
  const [result, setResult] = useState<AppliedBacktest | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const controller = useRef<AbortController | null>(null);
  useEffect(() => {
    controller.current?.abort();
    setResult(null);
    setBusy(false);
    setError('');
    return () => controller.current?.abort();
  }, [datasetId, applied?.data.dataset.fetched_at]);
  async function run() {
    if (!applied) return;
    controller.current?.abort();
    const current = new AbortController();
    controller.current = current;
    setBusy(true);
    setError('');
    const request: BacktestRequest = {
      dataset_id: applied.request.dataset_id,
      player_id: scope === 'all' ? null : applied.request.player_id,
      stat: applied.request.stat, model: applied.request.model, lookback: applied.request.lookback,
      line: lineMode === 'adaptive' ? null : applied.request.line,
      use_context: applied.request.use_context,
    };
    try {
      const data = await api<Backtest>('/api/backtest', { method: 'POST', body: JSON.stringify(request), signal: current.signal });
      if (!current.signal.aborted) setResult({ data, request, playerName: scope === 'all' ? 'All dataset players' : applied.data.player.name, dataset: applied.data.dataset });
    } catch (error) { if (!isAbort(error)) setError(messageOf(error)); }
    finally { if (!current.signal.aborted) setBusy(false); }
  }
  const data = result?.data;
  return <>
    <div className="notice research-notice"><ShieldCheck size={20} /><div><strong>A forecast is only as useful as its evaluation.</strong><p>Each prediction uses strictly older game rows. The first 15 games are warm-up, and minutes are estimated before each held-out game. No in-sample accuracy claims.</p></div></div>
    <article className="card evaluation-controls"><SectionHeading eyebrow="CHRONOLOGICAL HOLDOUT" title="Put the assumptions to work" />
      {applied ? <><div className="evaluation-scenario"><span className="pill">{statNames[applied.data.stat]}</span><span>{modelNames[applied.request.model]}</span><span>{applied.request.lookback}-game lookback</span><span>Context {applied.request.use_context ? 'on' : 'off'}</span><button className="text-button" onClick={onForecast}>Edit in forecast lab <ArrowRight size={13} /></button></div>
        <div className="evaluation-form"><Field label="Evaluation scope"><select value={scope} onChange={event => setScope(event.target.value)} disabled={busy}><option value="player">{applied.data.player.name}</option><option value="all">All players in this dataset</option></select></Field>
          <Field label="Line strategy"><select value={lineMode} onChange={event => setLineMode(event.target.value)} disabled={busy}><option value="fixed">Fixed scenario · {applied.data.line}</option><option value="adaptive">Adaptive · prior rolling median + half point</option></select></Field>
          <button className="primary-button" onClick={run} disabled={busy}>{busy ? <Busy text="Evaluating" /> : <><Play size={14} fill="currentColor" />Run walk-forward test</>}</button>
        </div>
        <p className="chart-footnote">{lineMode === 'adaptive' ? 'Adaptive lines are derived from each prior rolling median. They are synthetic research thresholds, not historical bookmaker prices.' : `Every held-out game is evaluated against the same ${applied.data.line} ${applied.data.stat} line. This is a fixed research scenario, not a historical betting simulation.`} {scope === 'all' && 'All-player evaluation can take a little longer; it runs only when requested.'}</p>
      </> : <EmptyState title="Start with an applied forecast" icon={<FlaskConical size={32} />} action={<button className="secondary-button" onClick={onForecast}>Open forecast lab <ArrowRight size={14} /></button>}>A completed forecast supplies the dataset, statistic, model, and lookback for evaluation.</EmptyState>}
    </article>
    {error && <ErrorNotice message={error} onRetry={run} />}
    {!data && !busy && applied && <article className="card evaluation-empty"><EmptyState title="Evidence, not a victory lap" icon={<FlaskConical size={34} />}><span>Run an evaluation to reveal held-out Brier scores, calibration, and sample sizes. No pre-filled performance numbers. No cherry-picked win rate.</span></EmptyState><div className="evaluation-principles"><div><span>01</span><strong>Train on the past</strong><p>Only older rows enter each fitting window.</p></div><div><span>02</span><strong>Predict the next game</strong><p>Score a probability against an unseen outcome.</p></div><div><span>03</span><strong>Check the reliability</strong><p>Compare calibration and a constant baseline.</p></div></div></article>}
    {busy && <div className="card progress-card" role="status"><Busy text="Walking forward, one completed game at a time…" /><p>Fitting models to prior rows and evaluating each later outcome. Existing results, if shown, are from the last completed run.</p></div>}
    {data && result && <div className="enter">
      <div className="result-context"><div><span className="eyebrow">LAST COMPLETED EVALUATION</span><h2>{result.playerName}</h2></div><div><span className="pill">{sourceLabel(result.dataset.source)}</span><p>{statNames[result.request.stat]} · {modelNames[result.request.model]} · {result.request.lookback} games · {result.request.line === null ? 'Adaptive line' : `Fixed ${result.request.line} line`}</p></div></div>
      {result.dataset.source === 'synthetic' && <div className="notice warning-notice"><Info size={18} /><div><strong>Simulation results, not demonstrated real-world performance.</strong><p>These held-out rows are synthetic. Calibration and Brier scores describe behavior within this generator, not predictive accuracy on future NBA games.</p></div></div>}
      <div className="metrics-grid">
        <StatCard label="BRIER SCORE" value={decimal(data.brier_score, 3)} detail={<><span>Mean squared probability error</span><strong>Lower is better</strong></>} accent />
        <StatCard label="BASELINE BRIER" value={decimal(data.baseline_brier, 3)} detail={<><span>Reference forecast</span><strong>Compare, don’t cherry-pick</strong></>} />
        <StatCard label="SKILL VS BASELINE" value={signed(data.skill_score * 100)} unit="%" detail={<><span>1 − Brier / baseline Brier</span><strong>{data.skill_score > 0 ? 'Improves on this baseline' : data.skill_score < 0 ? 'Below this baseline' : 'Matches this baseline'}</strong></>} />
        <StatCard label="SCORED PREDICTIONS" value={data.evaluations.toLocaleString()} detail={<><span>Non-push outcomes</span><strong>{data.pushes_excluded} pushes excluded</strong></>} />
      </div>
      <div className="two-column">
        <article className="card chart-card"><SectionHeading eyebrow="DOES 70% REALLY MEAN 70%?" title="Probability, checked" /><CalibrationChart calibration={data.calibration} /><p className="chart-footnote">Bins group held-out over probabilities. Small bins are noisy; inspect their counts before drawing conclusions.</p></article>
        <article className="card model-table-card"><SectionHeading eyebrow="SAME HOLDOUT, DIFFERENT LENSES" title="Model scorecard" /><div className="table-scroll" tabIndex={0} role="region" aria-label="Model scorecard"><table className="data-table"><thead><tr><th>Model</th><th>Brier ↓</th><th>Log loss ↓</th><th>Scored</th></tr></thead><tbody>{data.model_scores.map(score => <tr key={score.model}><td><strong>{modelNames[score.model]}</strong></td><td>{decimal(score.brier_score, 3)}</td><td>{decimal(score.log_loss, 3)}</td><td>{score.evaluations.toLocaleString()}</td></tr>)}</tbody></table></div><div className="score-explainer"><span className="eyebrow">READ THE SCORE, NOT A RANKING</span><p>Brier measures probability error; log loss penalizes confident mistakes more heavily. Both favor lower values. A favorable result here is not a guarantee of future performance.</p><strong>Selected-model log loss <span className="mono">{decimal(data.log_loss, 3)}</span></strong></div></article>
      </div>
      <article className="card model-table-card"><SectionHeading eyebrow="LOOK UNDER THE HOOD" title="Calibration bins" /><div className="table-scroll" tabIndex={0} role="region" aria-label="Calibration bins"><table className="data-table"><thead><tr><th>Probability bin</th><th>Mean predicted</th><th>Observed over rate</th><th>Count</th></tr></thead><tbody>{data.calibration.map(bin => <tr key={bin.lower}><td>{pct(bin.lower, 0)}–{pct(bin.upper, 0)}</td><td>{bin.count ? pct(bin.predicted) : '—'}</td><td>{bin.count ? pct(bin.observed) : '—'}</td><td>{bin.count}</td></tr>)}</tbody></table></div></article>
      <article className="card model-table-card"><SectionHeading eyebrow="AN AUDIT TRAIL, NOT AN ACCURACY CLAIM" title="Latest held-out predictions"><span className="pill">Up to 12 recent rows</span></SectionHeading><div className="table-scroll" tabIndex={0} role="region" aria-label="Latest held-out predictions"><table className="data-table"><thead><tr><th>Game date</th>{result.request.player_id === null && <th>Player ID</th>}<th>Research line</th><th>Actual {result.request.stat}</th><th>P(over | no push)</th><th>Outcome</th></tr></thead><tbody>{[...data.predictions].sort((a, b) => b.date.localeCompare(a.date)).slice(0, 12).map((prediction, index) => <tr key={`${prediction.date}-${prediction.player_id}-${index}`}><td>{dateLabel(prediction.date, true)}</td>{result.request.player_id === null && <td>{prediction.player_id}</td>}<td>{prediction.line}</td><td>{prediction.actual}</td><td>{pct(prediction.probability)}</td><td><span className={`pill ${prediction.outcome === 1 ? 'accent-pill' : ''}`}>{prediction.outcome === 1 ? 'Over' : 'Under'}</span></td></tr>)}</tbody></table></div><p className="chart-footnote">{data.description}</p></article>
    </div>}
  </>;
}
