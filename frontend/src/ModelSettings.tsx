import { Check, Cpu, FlaskConical, Gauge } from 'lucide-react';
import { useState } from 'react';
import { simulationsLabel, statNames } from './api';
import { SectionHeading } from './components';
import { MODEL_REPORT } from './modelReport';
import { RouteLink } from './RouteLink';
import { DEFAULT_SIMULATIONS, SIMULATION_CHOICES, simulationError } from './simulations';

const speed: Record<number, string> = { 100_000: 'Fastest', [DEFAULT_SIMULATIONS]: 'Default', 500_000: 'Slower', 1_000_000: 'Slowest · most exact' };

/** Settings › Model: how many games the simulator plays per forecast and parlay. */
export function ModelSettings({ simulations, onSimulations }: { simulations: number; onSimulations: (value: number) => void }) {
  return <div className="settings-stack">
    <section className="card settings-card" aria-labelledby="model-settings-title">
      <SectionHeading id="model-settings-title" eyebrow="THE SIMULATOR" title="Simulations per forecast"><Cpu size={21} className="subtle" aria-hidden="true" /></SectionHeading>
      <p className="settings-lede">Every forecast replays the player's next game this many times: minutes, shot attempts and makes, rebounds, assists, steals, blocks, turnovers and fouls, all drawn together. Every market (points, threes, PRA, double-doubles…) is counted from the same games, so they always agree with each other.</p>
      <fieldset className="simulation-options">
        <legend>Simulated games</legend>
        {SIMULATION_CHOICES.map(value => {
          const selected = value === simulations;
          return <label key={value} className={`simulation-option${selected ? ' selected' : ''}`}>
            <input type="radio" name="courtside-simulations" value={value} checked={selected} onChange={() => onSimulations(value)} />
            <span className="simulation-count">{simulationsLabel(value)}{selected && <Check size={16} aria-hidden="true" />}</span>
            <span className="simulation-error">±{simulationError(value).toFixed(2)} pp</span>
            <small>{speed[value]}</small>
          </label>;
        })}
      </fieldset>
      <p className="record-hint">± is the largest simulation error on any probability (at 50%). Saved on this device; applies to Props and the parlay builder.</p>
      <div className="notice research-notice"><Gauge size={18} aria-hidden="true" /><div><strong>More simulations make the answer exact, not more right.</strong><p>At 250,000 games the simulation noise is already a tenth of a percentage point. What decides accuracy is what gets simulated: recent-form weighting, minutes, shooting and dispersion measured on real box scores, then per-market calibration. That is what the walk-forward test below measures.</p></div></div>
    </section>
    <ModelReportCard />
  </div>;
}

/** The walk-forward test that chose the simulator as the default model. */
export function ModelReportCard({ compact = false }: { compact?: boolean }) {
  const [metric, setMetric] = useState<'log_loss' | 'brier'>('log_loss');
  const report = MODEL_REPORT;
  const rows = report.markets.map(row => ({ ...row, rule: metric === 'log_loss' ? row.rule_ll : row.rule_brier, sim: metric === 'log_loss' ? row.sim_ll : row.sim_brier }));
  const overall = metric === 'log_loss' ? [report.overall.rule_ll, report.overall.sim_ll] : [report.overall.rule_brier, report.overall.sim_brier];
  const better = rows.filter(row => row.sim < row.rule).length;
  return <section className="card model-report" aria-labelledby="model-report-title">
    <SectionHeading id="model-report-title" eyebrow="WALK-FORWARD TEST" title="Game simulator vs. the old Auto rule"><FlaskConical size={21} className="subtle" aria-hidden="true" /></SectionHeading>
    <p className="settings-lede">{report.predictions.toLocaleString()} graded predictions from {report.games.toLocaleString()} real games ({report.players} players, {report.period}). Each forecast used only the 60 games before it. The simulator scored better on <strong>{better} of {rows.length}</strong> markets by {metric === 'log_loss' ? 'log loss' : 'Brier score'}; lower is better.</p>
    <div className="segmented small" role="group" aria-label="Score"><button type="button" aria-pressed={metric === 'log_loss'} onClick={() => setMetric('log_loss')}>Log loss</button><button type="button" aria-pressed={metric === 'brier'} onClick={() => setMetric('brier')}>Brier score</button></div>
    <div className="report-overall"><span>All markets</span><strong className="mono">{overall[0].toFixed(4)} → {overall[1].toFixed(4)}</strong><span className="positive">{((1 - overall[1] / overall[0]) * 100).toFixed(1)}% lower</span></div>
    {!compact && <div className="table-scroll" tabIndex={0} role="region" aria-label="Walk-forward scores by market">
      <table className="data-table report-table">
        <thead><tr><th scope="col">Market</th><th scope="col">Graded</th><th scope="col">Old Auto</th><th scope="col">Simulator</th><th scope="col">Change</th></tr></thead>
        <tbody>{rows.map(row => {
          const change = (row.sim / row.rule - 1) * 100;
          return <tr key={row.stat}><th scope="row">{statNames[row.stat]}</th><td className="mono">{row.n.toLocaleString()}</td><td className="mono">{row.rule.toFixed(4)}</td><td className="mono">{row.sim.toFixed(4)}</td>
            <td className={`mono ${change < 0 ? 'positive' : change > 0 ? 'negative' : ''}`}>{change > 0 ? '+' : ''}{change.toFixed(1)}%</td></tr>;
        })}</tbody>
      </table>
    </div>}
    <p className="record-hint">Same-player parlay legs: priced jointly from the simulated games, "both legs win" scored {report.joint.simulator.toFixed(4)} log loss on {report.joint.pairs.toLocaleString()} held-out pairs, vs {report.joint.history_copula.toFixed(4)} for the shared-games correlation and {report.joint.independent.toFixed(4)} if treated as independent. {compact && <RouteLink to={{ page: 'how', section: 'accuracy' }} focus="model-report-title">Full table</RouteLink>}</p>
  </section>;
}
