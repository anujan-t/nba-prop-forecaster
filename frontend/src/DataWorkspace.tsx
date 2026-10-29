import { ArrowRight, Check, Database, Globe2, RefreshCw } from 'lucide-react';
import { dateLabel, historySyncLabel, sourceLabel } from './api';
import { EmptyState, SectionHeading, StatCard } from './components';
import type { Dataset } from './types';

interface Props {
  datasets: Dataset[];
  datasetId: string;
  onSelect: (id: string) => void;
  onBrowseLeague: () => void;
  onRefresh: () => void;
  refreshing: boolean;
}

export default function DataWorkspace({ datasets, datasetId, onSelect, onBrowseLeague, onRefresh, refreshing }: Props) {
  const active = datasets.find(dataset => dataset.id === datasetId);
  const history = datasets.filter(dataset => dataset.source === 'nba_api');
  const demo = datasets.find(dataset => dataset.source === 'synthetic');
  return <>
    <div className="data-hero card"><div><span className="eyebrow">THE NBA IS THE SOURCE</span><h2>Game logs, not spreadsheets.</h2><p>Pull historical player performances straight from NBA Stats. COURTSIDE handles the requests, local caching and analysis. No upload step, API key or HTML scraping.</p></div><div className="data-hero-icon"><Globe2 size={62} strokeWidth={1} /></div></div>
    <article className="card"><SectionHeading eyebrow="THE DIRECTORY IS THE WHOLE LEAGUE" title="Your cache is not the player list." /><p className="chart-footnote">Every current NBA roster player is available in the league selector. Choosing one automatically fetches their profile, game history and career details. The histories below are only the studies already saved on this computer.</p><button className="primary-button" onClick={onBrowseLeague}>Browse every current NBA player <ArrowRight size={15} /></button></article>
    {active && <div className="data-metrics">
      <StatCard label="COMPLETED GAMES" value={active.rows.toLocaleString()} unit="rows" detail={<><span>{active.name}</span><strong>{sourceLabel(active.source)}</strong></>} />
      <StatCard label="PLAYER COVERAGE" value={active.players.toLocaleString()} unit="players" detail={<><span>22 modeled markets</span><strong>Counts, combos, shooting and milestones</strong></>} />
      <article className="card stat-card range-card"><span className="eyebrow">HISTORICAL COVERAGE</span><div className="date-range"><strong>{dateLabel(active.start_date, true)}</strong><ArrowRight size={18} /><strong>{dateLabel(active.end_date, true)}</strong></div><p className="field-hint">{historySyncLabel(active)}</p></article>
    </div>}
    <article className="card dataset-library"><SectionHeading eyebrow="LOCAL CACHE OF NBA RESPONSES" title="Saved NBA history"><button className="icon-button" title="Reload history library" aria-label="Reload history library" disabled={refreshing} onClick={onRefresh}><RefreshCw size={17} className={refreshing ? 'spin' : ''} /></button></SectionHeading>
      {!history.length ? <EmptyState title="Connect your first NBA history" icon={<Database size={30} />}>Choose any current player in the league selector. Their completed games are fetched directly from NBA Stats.</EmptyState> : <div className="table-scroll" tabIndex={0} role="region" aria-label="Saved NBA histories"><table className="data-table"><thead><tr><th>NBA history</th><th>Games</th><th>Players</th><th>Last fetched</th><th><span className="visually-hidden">Actions</span></th></tr></thead><tbody>{history.map(dataset => <tr key={dataset.id} className={dataset.id === datasetId ? 'selected-row' : ''}>
        <td><strong>{dataset.name}</strong><p className="field-hint">{dateLabel(dataset.start_date, true)} to {dateLabel(dataset.end_date, true)}</p></td><td>{dataset.rows.toLocaleString()}</td><td>{dataset.players}</td><td>{dataset.fetched_at ? new Date(dataset.fetched_at).toLocaleString() : 'Legacy saved history'}</td><td><button className={dataset.id === datasetId ? 'text-button active-dataset-button' : 'secondary-button small-button'} onClick={() => onSelect(dataset.id)} disabled={dataset.id === datasetId}>{dataset.id === datasetId ? <><Check size={14} />Active</> : <>Use history <ArrowRight size={13} /></>}</button></td>
      </tr>)}</tbody></table></div>}
      <p className="chart-footnote">Reusing a saved study does not contact NBA. Selecting a current player checks the six-hour cache; Refresh player requests updated data and replaces the same history without losing notebook entries. Upstream failures are shown and never masquerade as successful refreshes.</p>
    </article>
    {demo && <details className="card development-fixture"><summary>Developer fixture / optional synthetic data</summary><p className="chart-footnote">The {demo.rows}-game seeded sample is retained only for reproducible model development. It is never used as a fallback when NBA is unavailable.</p><button className="secondary-button small-button" disabled={datasetId === demo.id} onClick={() => onSelect(demo.id)}>{datasetId === demo.id ? 'Synthetic fixture active' : 'Use synthetic development fixture'}</button></details>}
    <div className="quiet-note"><Globe2 size={16} /><span>The <a href="https://github.com/swar/nba_api" target="_blank" rel="noreferrer">nba_api client</a> wraps the endpoints behind NBA.com; it is not an NBA-supported commercial API. Historical availability varies by player and season. Public endpoints may time out or be blocked. This connector does not fetch live odds or injuries.</span></div>
  </>;
}
