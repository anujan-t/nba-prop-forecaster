import { useId, useMemo, useState } from 'react';
import { ArrowRight, Check, Globe2, Info, RefreshCw, Search, UsersRound, X } from 'lucide-react';
import { Busy, ErrorNotice, Field } from './components';
import type { LeagueSelectorProps } from './league-types';

function normalize(value: string): string {
  return value.normalize('NFD').replace(/\p{Diacritic}/gu, '').toLocaleLowerCase().trim();
}

function seasonFromYear(year: number): string {
  return `${year}-${String((year + 1) % 100).padStart(2, '0')}`;
}

function timestamp(value: string): string {
  if (!value) return 'Not reported';
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short',
  });
}

export default function LeagueSelector({
  roster, loading, error, selectedId, onSelect, onRefresh,
  historyWindow, onHistoryWindowChange, seasonType, onSeasonTypeChange,
  busy, onReload, onDetails,
}: LeagueSelectorProps) {
  const [search, setSearch] = useState('');
  const [teamFilter, setTeamFilter] = useState('');
  const controlId = useId();
  const selected = roster?.players.find(player => player.id === selectedId);
  const matches = useMemo(() => {
    const terms = normalize(search).split(/\s+/).filter(Boolean);
    return (roster?.players ?? [])
      .filter(player => (!teamFilter || player.team_id === teamFilter)
        && terms.every(term => normalize(`${player.name} ${player.team} ${player.team_name}`).includes(term)))
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
  }, [roster, search, teamFilter]);
  const teams = useMemo(() => [...(roster?.teams ?? [])].sort((a, b) =>
    a.abbreviation.localeCompare(b.abbreviation)), [roster]);
  const firstYear = Number(roster?.season.split('-')[0]);
  const seasons = useMemo(() => {
    if (!Number.isInteger(firstYear) || firstYear < 1946) return [];
    return Array.from({ length: firstYear - 1946 + 1 }, (_, index) => seasonFromYear(firstYear - index));
  }, [firstYear]);
  const selectionIsVisible = matches.some(player => player.id === selectedId);
  const filtered = Boolean(search.trim() || teamFilter);
  const needsCurrentWindowOption = historyWindow !== 'recent' && !seasons.includes(historyWindow);
  const previousSeason = Number.isInteger(firstYear) && firstYear >= 1946 ? seasonFromYear(firstYear - 1) : null;
  const recentLabel = roster && previousSeason
    ? `Recent · ${roster.season} + ${previousSeason}`
    : 'Recent · current + previous NBA seasons';

  function clearFilters() {
    setSearch('');
    setTeamFilter('');
  }

  return <section className="card league-selector" aria-label="League-wide player research selector">
    <div className="league-selector-heading">
      <div className="league-heading-copy">
        <span className="league-heading-icon"><Globe2 size={21} strokeWidth={1.5} aria-hidden="true" /></span>
        <div><span className="eyebrow">THE LEAGUE, NOT A SHORTLIST</span><h2>Find your next player.</h2></div>
      </div>
      <div className="league-coverage-actions">
        {roster && <div className="league-coverage" aria-label={`${roster.total_players} rostered players across ${roster.total_teams} teams`}>
          <span><strong>{roster.total_players.toLocaleString()}</strong> players</span>
          <span className="league-coverage-divider" aria-hidden="true" />
          <span><strong>{roster.total_teams.toLocaleString()}</strong> teams</span>
        </div>}
        <button type="button" className="icon-button league-refresh-roster" onClick={onRefresh}
          disabled={loading || busy} title="Refresh the NBA roster directory" aria-label="Refresh the NBA roster directory">
          <RefreshCw size={16} className={loading ? 'spin' : ''} aria-hidden="true" />
        </button>
      </div>
    </div>

    {error && <div className="league-roster-error"><ErrorNotice message={error} onRetry={!loading && !busy ? onRefresh : undefined} /></div>}
    {loading && <div className="league-loading"><Busy text={roster ? 'Refreshing the roster snapshot. Existing players remain selectable.' : 'Loading the NBA roster directory…'} /></div>}
    {!loading && !roster && !error && <p className="league-directory-empty">The roster directory has not been loaded. Refresh it to browse the provider’s current player list.</p>}

    <div className="league-primary-fields">
      <Field label="Search the league" className="league-search-field">
        <div className="league-search-control">
          <Search size={16} aria-hidden="true" />
          <input type="search" aria-label="Search the league" placeholder="Player name or team…" value={search}
            onChange={event => setSearch(event.target.value)} aria-describedby={`${controlId}-matches`} autoComplete="off" />
          {search && <button type="button" onClick={() => setSearch('')} aria-label="Clear player search" title="Clear player search"><X size={14} /></button>}
        </div>
      </Field>
      <Field label="Team" className="league-team-field">
        <select value={teamFilter} onChange={event => setTeamFilter(event.target.value)} disabled={!roster}>
          <option value="">All teams</option>
          {teamFilter && !teams.some(team => team.id === teamFilter) && <option value={teamFilter}>Team absent from this snapshot</option>}
          {teams.map(team => <option value={team.id} key={team.id}>{team.abbreviation} · {team.name} ({team.count})</option>)}
        </select>
      </Field>
      <Field label="Player" className="league-player-field">
        <select value={selectionIsVisible ? selectedId : ''} disabled={!matches.length}
          onChange={event => { if (event.target.value) onSelect(event.target.value); }}
          aria-describedby={`${controlId}-matches ${controlId}-selection`}>
          <option value="" disabled>{!roster ? 'Waiting for roster…' : !matches.length ? 'No players match these filters' : 'Choose any matching player'}</option>
          {matches.map(player => <option value={player.id} key={player.id}>{player.name} · {player.team || player.team_name || 'Team not reported'}</option>)}
        </select>
      </Field>
    </div>
    <div className="league-filter-status" id={`${controlId}-matches`} role="status">
      <span>{roster ? `${matches.length.toLocaleString()} matching ${matches.length === 1 ? 'player' : 'players'} · every match is selectable` : 'Search is local to the loaded directory.'}</span>
      {filtered && <button type="button" className="text-button" onClick={clearFilters}><X size={12} aria-hidden="true" />Clear filters</button>}
    </div>

    <div className="league-secondary-fields">
      <Field label="History window" className="league-window-field">
        <select value={historyWindow} onChange={event => onHistoryWindowChange(event.target.value)} aria-describedby={`${controlId}-window-note`}>
          <option value="recent">{recentLabel}</option>
          {needsCurrentWindowOption && <option value={historyWindow}>{historyWindow} · selected season</option>}
          {seasons.map(season => <option key={season} value={season}>{season} · single season</option>)}
        </select>
      </Field>
      <Field label="Competition" className="league-competition-field">
        <select value={seasonType} onChange={event => onSeasonTypeChange(event.target.value === 'Playoffs' ? 'Playoffs' : 'Regular Season')}>
          <option value="Regular Season">Regular Season</option>
          <option value="Playoffs">Playoffs</option>
        </select>
      </Field>
      <div className="league-player-actions">
        <button type="button" className="secondary-button" onClick={onReload} disabled={busy || !selected}>
          <RefreshCw size={14} className={busy ? 'spin' : ''} aria-hidden="true" />Refresh player
        </button>
        <button type="button" className="primary-button" onClick={onDetails} disabled={!selected}>Player details <ArrowRight size={15} aria-hidden="true" /></button>
      </div>
    </div>
    <p className="league-window-note" id={`${controlId}-window-note`}>
      {historyWindow === 'recent'
        ? 'Recent pools the current and previous NBA seasons. Its averages are pooled history—not current-season-only performance.'
        : `Single-season history: ${historyWindow}. A player with no logs in this window may still have games in other seasons.`}
    </p>

    <div className="league-selection-status" id={`${controlId}-selection`}>
      <div className="league-current-player">
        <span className="league-selection-icon">{selected ? <Check size={15} aria-hidden="true" /> : <UsersRound size={16} aria-hidden="true" />}</span>
        <div>
          <span className="league-selection-caption">CURRENT RESEARCH PLAYER</span>
          <strong>{selected?.name ?? (selectedId ? `Player ID ${selectedId}` : 'Choose a player above')}</strong>
          {selected && <span className="league-current-team">{selected.team_name || selected.team || 'Team not reported'}</span>}
          {selected && !selectionIsVisible && <p className="league-filter-exclusion">Outside the current search filters. Your selected player has not changed.</p>}
          {selectedId && roster && !selected && <p className="league-filter-exclusion">This ID is not in the loaded roster snapshot. Select a directory entry to research a current rostered player.</p>}
        </div>
      </div>
      {busy && <div className="league-player-loading"><Busy text="Loading player data…" /><span>You can select another player while this loads.</span></div>}
    </div>
    <div className="league-roster-provenance">
      <div><Info size={13} aria-hidden="true" /><span>Roster membership is not medical clearance. No injury or confirmed-lineup feed.</span></div>
      {roster && <p>{roster.provider || 'Provider not reported'} · {roster.season} roster · as of {timestamp(roster.fetched_at)}</p>}
    </div>
  </section>;
}
