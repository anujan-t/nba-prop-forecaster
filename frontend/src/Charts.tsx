import { useMemo, useState } from 'react';
import { BarChart3, CircleDot } from 'lucide-react';
import { dateLabel, decimal, pct, statNames, venueLabel } from './api';
import { EmptyState } from './components';
import type { Backtest, Forecast, LineSnapshot } from './types';

const dimensions = { width: 640, height: 248, left: 42, right: 20, top: 18, bottom: 38 };
const innerWidth = dimensions.width - dimensions.left - dimensions.right;
const innerHeight = dimensions.height - dimensions.top - dimensions.bottom;

function HorizontalGrid({ max, probability = false }: { max: number; probability?: boolean }) {
  return <g className="chart-grid">
    {[0, 1, 2, 3, 4].map(tick => {
      const value = max * tick / 4;
      const y = dimensions.top + innerHeight * (1 - tick / 4);
      return <g key={tick}><line x1={dimensions.left} x2={dimensions.width - dimensions.right} y1={y} y2={y} />
        <text x={dimensions.left - 10} y={y + 4} textAnchor="end">{probability ? pct(value, 0) : decimal(value, Number.isInteger(value) ? 0 : 1)}</text>
      </g>;
    })}
  </g>;
}

export function DistributionChart({ forecast }: { forecast: Forecast }) {
  const [mode, setMode] = useState<'mass' | 'cumulative'>('mass');
  const [active, setActive] = useState<number | null>(null);
  const points = useMemo(() => {
    let cumulative = 0;
    return [...forecast.distribution].sort((a, b) => a.value - b.value).map(point => {
      cumulative += point.probability;
      return { ...point, cumulative };
    });
  }, [forecast.distribution]);
  if (!points.length) return <EmptyState title="No distribution available" icon={<BarChart3 />}>The engine returned no probability bins for this sample.</EmptyState>;
  const minimum = points[0].value - 0.5;
  const maximum = points[points.length - 1].value + 0.5;
  const x = (value: number) => dimensions.left + (value - minimum) / (maximum - minimum) * innerWidth;
  const maxProbability = mode === 'cumulative' ? 1 : Math.max(...points.map(point => point.probability), 0.01) * 1.12;
  const y = (value: number) => dimensions.top + innerHeight * (1 - value / maxProbability);
  const width = Math.max(1, innerWidth / points.length - 2);
  const selected = active === null ? undefined : points[active];
  const lineVisible = forecast.line >= minimum && forecast.line <= maximum;
  const lineX = x(forecast.line);
  const step = Math.max(1, Math.ceil(points.length / 8));
  const curve = points.map((point, index) => `${index === 0 ? 'M' : 'L'}${x(point.value)},${y(point.cumulative)}`).join(' ');
  return <div className="chart-component">
    <div className="chart-toolbar"><div className="chart-legend"><span><i className="legend-dot neutral-dot" />Under</span><span><i className="legend-dot rose-dot" />Over</span>{forecast.probabilities.push > 0 && <span><i className="legend-dot push-dot" />Push</span>}</div>
      <div className="segmented small" aria-label="Distribution display">
        <button type="button" aria-pressed={mode === 'mass'} className={mode === 'mass' ? 'selected' : ''} onClick={() => setMode('mass')}>Probability</button>
        <button type="button" aria-pressed={mode === 'cumulative'} className={mode === 'cumulative' ? 'selected' : ''} onClick={() => setMode('cumulative')}>Cumulative</button>
      </div>
    </div>
    <svg viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} className="data-chart" role="group" aria-label={`${statNames[forecast.stat]} probability distribution for ${forecast.player.name}. Forecast line ${forecast.line}. Focus a bar to inspect its probability.`} onMouseLeave={() => setActive(null)}>
      <HorizontalGrid max={maxProbability} probability />
      {mode === 'cumulative' && <path d={curve} className="chart-curve" />}
      {points.map((point, index) => {
        const height = innerHeight - (y(point.probability) - dimensions.top);
        return mode === 'mass'
          ? <rect key={point.value} x={x(point.value) - width / 2} y={y(point.probability)} width={width} height={Math.max(0, height)} rx={Math.min(width / 3, 3)}
            className={`chart-bar ${point.value > forecast.line ? 'over-bar' : point.value === forecast.line ? 'push-bar' : 'under-bar'}${index === active ? ' active-bar' : ''}`}
            tabIndex={0} role="img" aria-label={`${point.value} ${forecast.stat}: ${pct(point.probability)}`} onFocus={() => setActive(index)} onBlur={() => setActive(null)} onMouseEnter={() => setActive(index)}>
            <title>{point.value} {forecast.stat}: {pct(point.probability)}</title>
          </rect>
          : <circle key={point.value} cx={x(point.value)} cy={y(point.cumulative)} r={index === active ? 5 : 3} className="chart-point"
            tabIndex={0} role="img" aria-label={`At most ${point.value} ${forecast.stat}: ${pct(point.cumulative)}`} onFocus={() => setActive(index)} onBlur={() => setActive(null)} onMouseEnter={() => setActive(index)}><title>At most {point.value}: {pct(point.cumulative)}</title></circle>;
      })}
      {lineVisible && <g className="threshold"><line x1={lineX} x2={lineX} y1={dimensions.top - 2} y2={dimensions.top + innerHeight} /><rect x={Math.min(Math.max(lineX - 35, dimensions.left), dimensions.width - 88)} y={2} width={70} height={22} rx={4} /><text x={Math.min(Math.max(lineX - 35, dimensions.left), dimensions.width - 88) + 35} y={17} textAnchor="middle">LINE {forecast.line}</text></g>}
      <g className="chart-axis">{points.filter((_, index) => index % step === 0).map(point => <text key={point.value} x={x(point.value)} y={dimensions.height - 14} textAnchor="middle">{point.value}</text>)}</g>
    </svg>
    <div className="chart-readout" aria-live="polite"><CircleDot size={13} aria-hidden="true" />{selected
      ? <><strong>{mode === 'cumulative' ? 'At most ' : ''}{selected.value} {forecast.stat}</strong><span>{pct(mode === 'cumulative' ? selected.cumulative : selected.probability)} probability</span></>
      : <><span>Hover or focus {mode === 'mass' ? 'a bar' : 'a point'} to explore</span><span>{lineVisible ? `${points.length} modeled outcomes` : `Line ${forecast.line} lies outside displayed support`}</span></>}</div>
  </div>;
}

export function RecentGamesChart({ forecast }: { forecast: Forecast }) {
  const [count, setCount] = useState('12');
  const [metric, setMetric] = useState<'value' | 'minutes'>('value');
  const [active, setActive] = useState<number | null>(null);
  const history = useMemo(() => [...forecast.history].sort((a, b) => a.date.localeCompare(b.date)), [forecast.history]);
  const shown = count === 'all' ? history : history.slice(-Number(count));
  if (!shown.length) return <EmptyState title="No games in this sample">Select a player with completed game logs.</EmptyState>;
  const maximum = Math.max(...shown.map(game => game[metric]), metric === 'value' ? forecast.line : 0, 1) * 1.15;
  const y = (value: number) => dimensions.top + innerHeight * (1 - value / maximum);
  const columnWidth = innerWidth / shown.length;
  const x = (index: number) => dimensions.left + (index + 0.5) * columnWidth;
  const selected = shown[active ?? shown.length - 1] ?? shown[shown.length - 1];
  return <div className="chart-component">
    <div className="chart-toolbar"><div className="segmented small" aria-label="Game chart metric"><button type="button" aria-pressed={metric === 'value'} className={metric === 'value' ? 'selected' : ''} onClick={() => setMetric('value')}>{statNames[forecast.stat]}</button><button type="button" aria-pressed={metric === 'minutes'} className={metric === 'minutes' ? 'selected' : ''} onClick={() => setMetric('minutes')}>Minutes</button></div>
      <select className="compact-select" aria-label="Number of recent games" value={count} onChange={event => { setCount(event.target.value); setActive(null); }}><option value="12">Last 12</option><option value="24">Last 24</option><option value="all">All in sample</option></select>
    </div>
    <svg viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} className="data-chart" role="group" aria-label={`${metric === 'value' ? statNames[forecast.stat] : 'Minutes'} in ${shown.length} recent completed games. Focus a bar for game details.`} onMouseLeave={() => setActive(null)}>
      <HorizontalGrid max={maximum} />
      {shown.map((game, index) => <g key={`${game.date}-${index}`}>
        <rect x={x(index) - columnWidth * 0.32} y={y(game[metric])} width={columnWidth * 0.64} height={Math.max(1, dimensions.top + innerHeight - y(game[metric]))} rx={3}
          className={`chart-bar ${metric === 'value' && game.value > forecast.line ? 'over-bar' : 'under-bar'}${index === active ? ' active-bar' : ''}`}
          tabIndex={0} role="img" aria-label={`${dateLabel(game.date)} versus ${game.opponent}, ${venueLabel(game.home)}: ${game.value} ${forecast.stat}, ${decimal(game.minutes)} minutes, ${game.rest_days} rest days`}
          onMouseEnter={() => setActive(index)} onFocus={() => setActive(index)} onBlur={() => setActive(null)}><title>{dateLabel(game.date)} · {game.opponent} · {game[metric]}</title></rect>
        {(shown.length <= 12 || index % Math.ceil(shown.length / 8) === 0) && <text className="chart-axis" x={x(index)} y={dimensions.height - 14} textAnchor="middle">{game.opponent}</text>}
      </g>)}
      {metric === 'value' && <g className="threshold"><line x1={dimensions.left} x2={dimensions.width - dimensions.right} y1={y(forecast.line)} y2={y(forecast.line)} /><rect x={dimensions.width - 86} y={y(forecast.line) - 19} width={66} height={20} rx={4} /><text x={dimensions.width - 53} y={y(forecast.line) - 5} textAnchor="middle">{forecast.line} LINE</text></g>}
    </svg>
    <div className="chart-readout" aria-live="polite"><CircleDot size={13} aria-hidden="true" /><strong>{dateLabel(selected.date)} · {selected.opponent} · {venueLabel(selected.home)}</strong><span>{selected.value} {forecast.stat} / {decimal(selected.minutes)} min / {selected.rest_days}d rest</span></div>
  </div>;
}

export function CalibrationChart({ calibration }: { calibration: Backtest['calibration'] }) {
  const [active, setActive] = useState<number | null>(null);
  const bins = calibration.filter(bin => bin.count > 0).sort((a, b) => a.predicted - b.predicted);
  const width = 560;
  const height = 300;
  const left = 52;
  const top = 18;
  const plotWidth = width - left - 24;
  const plotHeight = height - top - 48;
  const x = (value: number) => left + value * plotWidth;
  const y = (value: number) => top + (1 - value) * plotHeight;
  const selected = active === null ? undefined : bins[active];
  if (!bins.length) return <EmptyState title="Not enough scored predictions" icon={<BarChart3 />}>Calibration needs non-push, out-of-sample evaluations. Try more game logs or all players.</EmptyState>;
  return <div className="chart-component">
    <div className="chart-toolbar"><div className="chart-legend"><span><i className="legend-dot rose-dot" />Observed calibration</span><span><i className="legend-dash" />Perfect calibration</span></div></div>
    <svg viewBox={`0 0 ${width} ${height}`} className="data-chart calibration-chart" role="group" aria-label="Reliability diagram. Predicted probability on the horizontal axis, observed frequency on the vertical axis. A diagonal line denotes perfect calibration." onMouseLeave={() => setActive(null)}>
      {[0, 0.25, 0.5, 0.75, 1].map(value => <g key={value} className="chart-grid"><line x1={left} x2={width - 24} y1={y(value)} y2={y(value)} /><text x={left - 12} y={y(value) + 4} textAnchor="end">{pct(value, 0)}</text><text x={x(value)} y={height - 25} textAnchor="middle">{pct(value, 0)}</text></g>)}
      <path d={`M${x(0)},${y(0)} L${x(1)},${y(1)}`} className="ideal-line" />
      <path d={bins.map((bin, index) => `${index === 0 ? 'M' : 'L'}${x(bin.predicted)},${y(bin.observed)}`).join(' ')} className="chart-curve" />
      {bins.map((bin, index) => <circle key={bin.lower} cx={x(bin.predicted)} cy={y(bin.observed)} r={index === active ? 8 : 5} className="chart-point" tabIndex={0} role="img"
        aria-label={`${pct(bin.predicted)} predicted, ${pct(bin.observed)} observed, ${bin.count} evaluations`}
        onFocus={() => setActive(index)} onBlur={() => setActive(null)} onMouseEnter={() => setActive(index)}><title>{pct(bin.predicted)} predicted · {pct(bin.observed)} observed · n={bin.count}</title></circle>)}
      <text className="axis-title" x={left + plotWidth / 2} y={height - 4} textAnchor="middle">PREDICTED PROBABILITY</text>
      <text className="axis-title" transform={`translate(12 ${top + plotHeight / 2}) rotate(-90)`} textAnchor="middle">OBSERVED FREQUENCY</text>
    </svg>
    <div className="chart-readout" aria-live="polite"><CircleDot size={13} />{selected ? <><strong>{pct(selected.predicted)} predicted → {pct(selected.observed)} observed</strong><span>n = {selected.count}</span></> : <><span>Closer to the diagonal is better calibrated</span><span>{bins.length} populated bins</span></>}</div>
  </div>;
}

export function LineHistoryChart({ lines }: { lines: LineSnapshot[] }) {
  const [active, setActive] = useState<number | null>(null);
  const sorted = [...lines].sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  if (!sorted.length) return <EmptyState title="A clean slate, not a fake feed" icon={<BarChart3 size={32} />}>Save a manual snapshot to begin. Movement appears only for the same player, statistic, event date, and book.</EmptyState>;
  const min = Math.max(0, Math.min(...sorted.map(line => line.line)) - 1);
  const max = Math.max(...sorted.map(line => line.line)) + 1;
  const x = (index: number) => dimensions.left + (sorted.length === 1 ? innerWidth / 2 : index / (sorted.length - 1) * innerWidth);
  const y = (value: number) => dimensions.top + innerHeight * (1 - (value - min) / (max - min));
  const selected = sorted[active ?? sorted.length - 1] ?? sorted[sorted.length - 1];
  return <div className="chart-component">
    <svg viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} className="data-chart" role="group" aria-label={`Manual line snapshots for ${sorted[0].book} on ${sorted[0].event_date}. Snapshot order on the horizontal axis.`} onMouseLeave={() => setActive(null)}>
      {[0, 1, 2, 3, 4].map(tick => {
        const value = min + (max - min) * tick / 4;
        return <g key={tick} className="chart-grid"><line x1={dimensions.left} x2={dimensions.width - dimensions.right} y1={y(value)} y2={y(value)} /><text x={dimensions.left - 10} y={y(value) + 4} textAnchor="end">{decimal(value)}</text></g>;
      })}
      <path d={sorted.map((line, index) => `${index ? 'L' : 'M'}${x(index)},${y(line.line)}`).join(' ')} className="chart-curve" />
      {sorted.map((line, index) => <g key={line.id}><circle cx={x(index)} cy={y(line.line)} r={active === index ? 7 : 5} className="chart-point"
        tabIndex={0} role="img" aria-label={`Snapshot ${index + 1}, ${new Date(line.timestamp).toLocaleString()}, line ${line.line}`} onMouseEnter={() => setActive(index)} onFocus={() => setActive(index)} onBlur={() => setActive(null)}><title>{new Date(line.timestamp).toLocaleString()} · {line.line}</title></circle>
        {(sorted.length < 10 || index % Math.ceil(sorted.length / 8) === 0) && <text className="chart-axis" x={x(index)} y={dimensions.height - 14} textAnchor="middle">#{index + 1}</text>}
      </g>)}
    </svg>
    <div className="chart-readout" aria-live="polite"><CircleDot size={13} /><strong>Line {selected.line}</strong><span>{new Date(selected.timestamp).toLocaleString()}</span></div>
    {sorted.length === 1 && <p className="chart-footnote">One snapshot saved. Add another for this book and event to trace movement.</p>}
  </div>;
}
