import { ArrowDownToLine, ArrowRight, BookmarkPlus, BrainCircuit, ChartNoAxesCombined, ChevronDown, CircleHelp, Cpu, Info, Play, RotateCcw, SlidersHorizontal, Sparkles, TriangleAlert } from 'lucide-react';
import type { FormEvent } from 'react';
import { decimal, downloadJson, edgePoints, isMilestone, modelNames, odds, pct, signed, simulationsLabel, sourceLabel, statGroups, statNames, statUnits, suggestedLine } from './api';
import { DistributionChart, RecentGamesChart } from './Charts';
import { Busy, EmptyState, ErrorNotice, Field, Jersey, SectionHeading, StatCard } from './components';
import { InjuryBadge } from './NewsPage';
import type { SlipLeg } from './parlay';
import type { Side } from './parlay-types';
import { ForecastParlayActions } from './ParlayLegs';
import { RouteLink } from './RouteLink';
import type { AppliedForecast, ForecastForm, Model, Player, Stat } from './types';

interface Props {
  form: ForecastForm;
  onChange: (form: ForecastForm) => void;
  players: Player[];
  playersLoading: boolean;
  applied: AppliedForecast | null;
  busy: boolean;
  dirty: boolean;
  error: string;
  onRun: (event: FormEvent<HTMLFormElement>) => void;
  onRevert: () => void;
  onValidation: () => void;
  /** Opens Players › Game log or Players › Profile & career. */
  onDetails: (section: 'log' | 'career') => void;
  /** Opens Players › Lines with this forecast's line and odds filled in. */
  onSaveLine: () => void;
  showPlayerSelect?: boolean;
  historyNotice?: string;
  currentTeam?: string;
  slipLegs: SlipLeg[];
  slipFull: boolean;
  onAddLeg: (side: Side, origin: HTMLElement) => void;
  onOpenSlip: () => void;
}

const modelOrder: Model[] = ['auto', 'simulation', 'normal', 'poisson', 'negative_binomial', 'empirical'];

export default function ForecastPanel({ form, onChange, players, playersLoading, applied, busy, dirty, error, onRun, onRevert, onValidation, onDetails, onSaveLine, showPlayerSelect = true, historyNotice = '', currentTeam, slipLegs, slipFull, onAddLeg, onOpenSlip }: Props) {
  const change = <K extends keyof ForecastForm>(key: K, value: ForecastForm[K]) => onChange({ ...form, [key]: value });
  const player = players.find(item => item.id === form.player_id);
  const forecast = applied?.data;
  const milestone = isMilestone(form.stat);
  const chooseStat = (stat: Stat) => {
    const line = isMilestone(stat) ? '0.5' : isMilestone(form.stat) || form.stat !== stat ? suggestedLine(stat, player) ?? form.line : form.line;
    const model = isMilestone(stat) && !['auto', 'simulation', 'empirical'].includes(form.model) ? 'auto' : form.model;
    onChange({ ...form, stat, line, model });
  };
  const yesNo = forecast ? isMilestone(forecast.stat) : false;
  const simulated = forecast?.simulation ?? null;
  const calibrated = forecast?.calibrated ?? null;
  return <>
    <form className="card controls-card" onSubmit={onRun} onInvalidCapture={event => {
      const target = event.target;
      if (target instanceof HTMLElement) {
        const details = target.closest('details');
        if (details) details.open = true;
      }
    }}>
      <div className="controls-heading"><div><span className="eyebrow">SET THE SCENARIO</span><span className="subtle controls-subtitle">Your assumptions. In plain sight.</span></div>
        {dirty && applied && <span className="pending-chip"><span />Changes not applied</span>}
      </div>
      <div className={`forecast-fields ${showPlayerSelect ? '' : 'full-roster-fields'}`}>
        {showPlayerSelect && <Field label="Player in saved study" className="player-field">
          <select value={form.player_id} onChange={event => change('player_id', event.target.value)} disabled={playersLoading || !players.length} required aria-label="Select player">
            {playersLoading ? <option>Loading players…</option> : !players.length ? <option value="">No players available</option> : players.map(item => <option value={item.id} key={item.id}>{item.name} · {item.team}</option>)}
          </select>
        </Field>}
        <Field label="Market"><select value={form.stat} onChange={event => chooseStat(event.target.value as Stat)}>
          {statGroups.map(group => <optgroup key={group.label} label={group.label}>{group.stats.map(id => <option value={id} key={id}>{statNames[id]}</option>)}</optgroup>)}
        </select></Field>
        <Field label={milestone ? 'Line (yes / no)' : 'Prop line'}>{milestone
          ? <input type="text" value="Yes or no" readOnly aria-readonly="true" title="A yes/no market: Over means yes, Under means no." />
          : <input type="number" min="0" max="200" step="0.5" required value={form.line} onChange={event => change('line', event.target.value)} />}</Field>
        <Field label="Lookback"><div className="input-with-unit"><input type="number" min="10" max="82" step="1" required value={form.lookback} onChange={event => change('lookback', event.target.value)} /><span>games</span></div></Field>
        <Field label="Model"><select value={form.model} onChange={event => change('model', event.target.value as Model)}>{modelOrder.map(id => <option value={id} key={id} disabled={milestone && !['auto', 'simulation', 'empirical'].includes(id)}>{modelNames[id]}</option>)}</select></Field>
        <button className="primary-button run-button" type="submit" disabled={busy || playersLoading || !player || player.games < 10}>{busy ? <Busy text="Simulating" /> : <><Play size={15} fill="currentColor" />Run forecast</>}</button>
      </div>
      <div className="controls-footer">
        <details className="advanced-controls">
          <summary><SlidersHorizontal size={14} /><span>Market & context</span><ChevronDown size={13} /><span className="context-preview">{form.use_context ? 'Context adjusted' : 'Context off'}</span></summary>
          <div className="advanced-grid">
            <Field label="Over odds" hint="American format"><input type="number" min="-10000" max="10000" step="1" required value={form.odds_over} onChange={event => change('odds_over', event.target.value)} /></Field>
            <Field label="Under odds" hint="American format"><input type="number" min="-10000" max="10000" step="1" required value={form.odds_under} onChange={event => change('odds_under', event.target.value)} /></Field>
            <Field label="Expected minutes" hint="Blank = estimate from history"><input type="number" min="1" max="60" step="0.5" placeholder="Auto estimate" value={form.expected_minutes} onChange={event => change('expected_minutes', event.target.value)} /></Field>
            <Field label="Days of rest" hint="0 = back-to-back; blank = unspecified"><input type="number" min="0" max="30" step="1" placeholder="Not specified" value={form.rest_days} onChange={event => change('rest_days', event.target.value)} /></Field>
            <Field label="Opponent" hint="Team abbreviation, e.g. BOS"><input type="text" maxLength={5} pattern="[A-Z0-9]{2,5}" placeholder="Not specified" value={form.opponent} onChange={event => change('opponent', event.target.value.toUpperCase())} /></Field>
            <label className="toggle-field"><input type="checkbox" checked={form.use_context} onChange={event => change('use_context', event.target.checked)} /><span><strong>Context adjustments</strong><small>Minutes, rest & opponent</small></span></label>
          </div>
          <p className="field-hint advanced-note">Inputs are research assumptions, not live lineup or bookmaker feeds. The injury report is in News; the model assumes the player plays. Opponent effects require sufficient historical support.</p>
        </details>
        <div className="sample-hint">{player && <><span className="tiny-dot" />{player.games} available player games</>}{dirty && applied && <button type="button" className="text-button" onClick={onRevert}><RotateCcw size={12} />Reset changes</button>}</div>
      </div>
    </form>
    {error && <ErrorNotice message={error} />}
    {historyNotice && <div className="notice research-notice"><Info size={18} /><div><strong>Historical coverage</strong><p>{historyNotice}</p><button className="text-button" onClick={() => onDetails('log')}>Open the game log <ArrowRight size={13} /></button></div></div>}
    {!forecast && (busy || playersLoading) && <div className="card loading-dashboard" role="status"><Busy text="Finding the story in the sample…" /><div className="skeleton-row"><span /><span /><span /><span /></div><div className="skeleton-chart" /><p>Reading completed games and fitting the selected probability model.</p></div>}
    {!forecast && !busy && !playersLoading && <div className="card"><EmptyState title={player && player.games < 10 ? 'This player is accessible. A forecast needs more history.' : players.length ? 'Your next question starts here' : 'A forecast needs a little history'} icon={<ChartNoAxesCombined size={34} />}
      action={<button className="secondary-button" onClick={() => onDetails('career')}>Profile & career <ArrowRight size={15} /></button>}>
      {player && player.games < 10 ? `${player.games} completed games are available; at least 10 are required. You can still inspect the player's profile, career and every available game.` : players.length ? 'Choose a statistic and a line. Run a forecast to explore a full probability distribution—not just a point estimate.' : 'Choose a current NBA player above. A valid roster player can have no completed games in the selected window.'}
    </EmptyState></div>}
    {forecast && applied && <div className="forecast-results enter" aria-busy={busy}>
      <div className="applied-heading"><div className="player-identity"><Jersey className="player-jersey" label={forecast.player.name.split(' ').map(part => part.charAt(0)).slice(0, 2).join('').toLocaleUpperCase()} /><div><div className="eyebrow">APPLIED FORECAST <span className="slash">/</span> {sourceLabel(forecast.dataset.source)}</div><h2>{forecast.player.name}<span title={currentTeam ? 'Current roster team' : 'Latest historical team'}>{currentTeam ?? forecast.player.team}</span></h2></div></div>
        <div className="applied-actions">{forecast.injury && <InjuryBadge entry={forecast.injury} />}<span className="pill">{statNames[forecast.stat]} · {forecast.sample_size} games</span><button className="icon-button" title="Download applied forecast as JSON" aria-label="Download applied forecast as JSON" onClick={() => downloadJson({ ...applied, exported_at: new Date().toISOString(), note: 'Research only. Not a wagering recommendation.' }, `courtside-${forecast.player.id}-${forecast.stat}-forecast.json`)}><ArrowDownToLine size={17} /></button></div>
      </div>
      {busy && <div className="updating-bar"><Busy text="Updating forecast. Previous applied results remain below." /></div>}
      {forecast.warnings.length > 0 && <div className="notice warning-notice" role="status"><TriangleAlert size={18} aria-hidden="true" /><div><strong>Keep the limitations in view</strong>{forecast.warnings.map((warning, index) => <p key={index}>{warning}</p>)}</div></div>}
      <div className="metrics-grid">
        {yesNo
          ? <StatCard label="SIMULATED RATE" value={<>{decimal(forecast.summary.mean * 100)}<span>%</span></>} detail={<><span>Share of simulated games with a {statNames[forecast.stat].toLowerCase()}</span><strong>Recent rate {pct(forecast.summary.observed_mean, 0)}</strong></>} />
          : <StatCard label="PROJECTED AVERAGE" value={decimal(forecast.summary.mean)} unit={statUnits[forecast.stat]} detail={<><span>{pct(forecast.interval.coverage, 0)} predictive interval</span><strong>{decimal(forecast.interval.low, 0)}—{decimal(forecast.interval.high, 0)} <span>{statUnits[forecast.stat]}</span></strong></>} />}
        <StatCard label={yesNo ? 'YES' : `OVER ${forecast.line}`} value={<>{decimal(forecast.probabilities.over * 100)}<span>%</span></>} accent detail={<><span>Model probability</span><strong>{yesNo ? `Gets a ${statNames[forecast.stat].toLowerCase()}` : 'Strictly above the line'} <ArrowRight size={15} /></strong></>} />
        <StatCard label={yesNo ? 'NO' : `UNDER ${forecast.line}`} value={<>{decimal(forecast.probabilities.under * 100)}<span>%</span></>} detail={<><span>Model probability</span><strong>{yesNo ? 'Falls short' : 'Strictly below the line'}</strong></>} />
        <StatCard label={yesNo ? 'YES · NO-VIG EDGE' : 'OVER · NO-VIG EDGE'} value={edgePoints(forecast.market.edge_over, false)} unit={forecast.market.edge_over === null ? undefined : 'pp'} detail={<><span>Model vs. normalized market</span><strong>{forecast.market.edge_over === null ? 'Undefined: all outcomes push' : 'Conditional on no push'} <CircleHelp size={14} aria-hidden="true" /></strong></>} />
      </div>
      {(simulated || calibrated) && <div className="model-strips">
        {simulated && <p className="sim-strip"><Cpu size={16} aria-hidden="true" /><span><strong>{simulationsLabel(simulated.runs)} simulated games</strong> · simulation error ±{(simulated.standard_error * 100).toFixed(2)} pp · minutes {decimal(simulated.minutes.low, 0)}–{decimal(simulated.minutes.high, 0)} in 80% of games{simulated.box_score === 'basic' ? ' · five-count box scores (no shot splits)' : ' · built from attempts, makes, rebounds, turnovers and fouls'}</span>
          <RouteLink className="text-button" to={{ page: 'settings', section: 'model' }}>Simulations</RouteLink></p>}
        {calibrated && <p className="calibration-strip"><BrainCircuit size={16} aria-hidden="true" /><span><strong>Calibrated: {yesNo ? 'yes' : 'over'} {pct(calibrated.over)} · {yesNo ? 'no' : 'under'} {pct(calibrated.under)}</strong> · {calibrated.choice === 'own' ? `its own ${statNames[forecast.stat].toLowerCase()} correction` : 'the shared correction'}, learned from graded predictions</span>
          <RouteLink className="text-button" to={{ page: 'how', section: 'accuracy' }} focus="learning-title">How</RouteLink></p>}
      </div>}
      <div className="forecast-caption"><span><Info size={14} /><strong>Push {pct(forecast.probabilities.push)}</strong> · {yesNo ? 'a yes/no market never pushes' : `exactly ${forecast.line}, stake returned`}</span><span>{modelNames[forecast.resolved_model]}{forecast.model === 'auto' ? ' · auto-selected' : ''} <span className="slash">/</span> Predictive range ≠ chance of winning</span></div>
      <ForecastParlayActions applied={applied} legs={slipLegs} full={slipFull} dirty={dirty} onAdd={onAddLeg} onOpenSlip={onOpenSlip} />
      <div className="two-column chart-grid-layout">
        <article className="card chart-card"><SectionHeading eyebrow="THE FULL PICTURE" title="A range, not a promise"><span className="pill">{modelNames[forecast.resolved_model]}</span></SectionHeading><DistributionChart forecast={forecast} /><p className="chart-footnote">{simulated ? `Share of ${simulationsLabel(simulated.runs)} simulated games at each ${statNames[forecast.stat].toLowerCase()} total.` : `Modeled probability of each ${statNames[forecast.stat].toLowerCase()} outcome.`} Orange marks outcomes above {forecast.line}.</p></article>
        <article className="card chart-card"><SectionHeading eyebrow="THE RECENT RECORD" title="Game by game"><span className="mono subtle">n = {forecast.sample_size}</span></SectionHeading><RecentGamesChart forecast={forecast} /><p className="chart-footnote">Actual rows from {sourceLabel(forecast.dataset.source).toLowerCase()}. The dashed line is today’s research scenario—not historical market lines.</p></article>
      </div>
      <div className="two-column analysis-grid">
        <article className="card market-card"><SectionHeading eyebrow="STRIP OUT THE MARGIN" title="Model meets market"><span className="pill">Manual odds</span></SectionHeading>
          <div className="market-probability"><div><span>Model over, no push</span><strong>{pct(forecast.market.model_over_conditional)}</strong></div><span className="versus">vs</span><div><span>No-vig market over</span><strong>{pct(forecast.market.no_vig_over)}</strong></div></div>
          {forecast.market.model_over_conditional === null
            ? <p className="conditional-unavailable">No conditional comparison: all modeled outcomes push. N/A is not a 0% probability or zero edge.</p>
            : <div className="comparison-track" role="img" aria-label={`Model conditional over probability ${pct(forecast.market.model_over_conditional)}, no-vig market ${pct(forecast.market.no_vig_over)}`}><span style={{ width: pct(Math.min(1, Math.max(0, forecast.market.model_over_conditional))) }} /><i style={{ left: pct(Math.min(1, Math.max(0, forecast.market.no_vig_over))) }} /></div>}
          <div className="table-scroll" tabIndex={0} role="region" aria-label="Model versus market by side"><table className="data-table market-table"><thead><tr><th>Side / odds</th><th>Implied</th><th>No-vig</th><th>Edge</th><th>Exp. ROI</th></tr></thead><tbody>
            <tr><td><strong>Over</strong> <span className="subtle mono">{odds(applied.request.odds_over)}</span></td><td>{pct(forecast.market.over_implied)}</td><td>{pct(forecast.market.no_vig_over)}</td><td>{edgePoints(forecast.market.edge_over)}</td><td>{signed(forecast.market.expected_roi_over * 100)}%</td></tr>
            <tr><td><strong>Under</strong> <span className="subtle mono">{odds(applied.request.odds_under)}</span></td><td>{pct(forecast.market.under_implied)}</td><td>{pct(forecast.market.no_vig_under)}</td><td>{edgePoints(forecast.market.edge_under)}</td><td>{signed(forecast.market.expected_roi_under * 100)}%</td></tr>
          </tbody></table></div>
          <p className="chart-footnote">Overround: <strong>{pct(forecast.market.overround)}</strong>. Edge uses probabilities conditional on no push. Expected ROI includes push refunds and is a model calculation, not realized profit.</p>
          <div className="market-save">
            <button type="button" className="secondary-button" onClick={onSaveLine} aria-describedby="market-save-hint"><BookmarkPlus size={16} aria-hidden="true" />Save this line</button>
            <span id="market-save-hint">Opens Lines with {forecast.line} {forecast.stat} at {odds(applied.request.odds_over)} / {odds(applied.request.odds_under)} filled in.</span>
          </div>
        </article>
        <article className="card context-card"><SectionHeading eyebrow="BEHIND THE ESTIMATE" title="Context, made explicit"><SlidersHorizontal size={19} className="subtle" /></SectionHeading>
          <div className="context-stats"><div><span>Observed mean</span><strong>{decimal(forecast.summary.observed_mean)}</strong></div><div><span>Projected minutes</span><strong>{decimal(forecast.summary.projected_minutes)}</strong></div><div><span>Dispersion · σ</span><strong>{decimal(forecast.summary.std)}</strong></div></div>
          <div className="factor-row"><span>Rest factor<strong>×{decimal(forecast.summary.rest_factor, 3)}</strong></span><span>Opponent factor<strong>×{decimal(forecast.summary.opponent_factor, 3)}</strong></span><span>Combined context<strong>×{decimal(forecast.summary.context_factor, 3)}</strong></span></div>
          <ul className="context-notes">{forecast.context_notes.length ? forecast.context_notes.map((note, index) => <li key={index}><span className="tiny-dot" />{note}</li>) : <li><span className="tiny-dot" />{applied.request.use_context ? 'No additional context notes were returned for this sample.' : 'Context adjustments are disabled for this forecast.'}</li>}</ul>
          <div className="quiet-note"><Info size={14} /><span>Estimated effects are bounded historical heuristics, not causal claims. They cannot account for unobserved injuries or rotation changes.</span></div>
        </article>
      </div>
      <article className="card model-table-card"><SectionHeading eyebrow="DON’T TRUST JUST ONE LENS" title="The model bench"><span className="subtle">Same sample. Different assumptions.</span></SectionHeading>
        <div className="table-scroll" tabIndex={0} role="region" aria-label="Model bench"><table className="data-table"><thead><tr><th>Distribution</th><th>Projected mean</th><th>Over {forecast.line}</th><th>Under {forecast.line}</th><th>Push</th><th>Role</th></tr></thead><tbody>{forecast.comparisons.map(comparison => <tr key={comparison.model} className={comparison.model === forecast.resolved_model ? 'selected-row' : ''}>
          <td><span className={`model-dot ${comparison.model === forecast.resolved_model ? 'accent-dot' : 'muted-dot'}`} /><strong>{modelNames[comparison.model]}</strong></td><td>{decimal(comparison.mean)}</td><td>{pct(comparison.over)}</td><td>{pct(comparison.under)}</td><td>{pct(comparison.push)}</td><td>{comparison.model === forecast.resolved_model ? <span className="pill accent-pill">Applied model</span> : <span className="subtle">Sensitivity check</span>}</td>
        </tr>)}</tbody></table></div><p className="chart-footnote">Agreement is not validation. Evaluate forecasts on later, unseen game rows before interpreting these probabilities.</p>
      </article>
      <div className="validation-cta"><div className="cta-icon"><Sparkles size={23} /></div><div><span className="eyebrow">THE NEXT QUESTION</span><h3>Does the model hold up out of sample?</h3><p>Walk forward through history. Inspect calibration, not a made-up win rate.</p></div><button className="secondary-button" onClick={onValidation}>Test the evidence <ArrowRight size={16} /></button></div>
    </div>}
  </>;
}
