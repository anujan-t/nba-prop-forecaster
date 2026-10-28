import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ArrowRight, BookOpen, Info, Save } from 'lucide-react';
import { api, dateLabel, isAbort, messageOf, odds, signed, statNames, tomorrow, validateOdds } from './api';
import { LineHistoryChart } from './Charts';
import { Busy, EmptyState, ErrorNotice, Field, SectionHeading } from './components';
import type { AppliedForecast, LineSnapshot } from './types';

export default function LineNotebook({ applied, onForecast }: { applied: AppliedForecast | null; onForecast: () => void }) {
  const [eventDate, setEventDate] = useState(tomorrow);
  const [book, setBook] = useState('Manual entry');
  const [line, setLine] = useState('');
  const [overOdds, setOverOdds] = useState('-110');
  const [underOdds, setUnderOdds] = useState('-110');
  const [lines, setLines] = useState<LineSnapshot[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [readError, setReadError] = useState('');
  const [saveError, setSaveError] = useState('');
  const [saved, setSaved] = useState('');
  const [reload, setReload] = useState(0);
  const saveController = useRef<AbortController | null>(null);
  const datasetId = applied?.request.dataset_id ?? '';
  const playerId = applied?.request.player_id ?? '';
  const stat = applied?.request.stat ?? 'points';
  const appliedLine = applied?.request.line;
  const appliedOver = applied?.request.odds_over;
  const appliedUnder = applied?.request.odds_under;
  useEffect(() => {
    setLine(appliedLine === undefined ? '' : String(appliedLine));
    setOverOdds(String(appliedOver ?? -110));
    setUnderOdds(String(appliedUnder ?? -110));
    setSaved('');
  }, [appliedLine, appliedOver, appliedUnder, datasetId, playerId, stat]);
  useEffect(() => {
    saveController.current?.abort();
    setSaving(false);
    setSaved('');
    setSaveError('');
    return () => saveController.current?.abort();
  }, [datasetId, playerId, stat]);
  useEffect(() => {
    setLines([]);
    setReadError('');
    if (!datasetId || !playerId || !eventDate) { setLoading(false); return; }
    const controller = new AbortController();
    setLoading(true);
    const params = new URLSearchParams({ dataset_id: datasetId, player_id: playerId, stat, event_date: eventDate });
    api<{ lines: LineSnapshot[] }>(`/api/lines?${params}`, { signal: controller.signal })
      .then(data => { if (!controller.signal.aborted) setLines(data.lines); })
      .catch((error: unknown) => { if (!isAbort(error)) setReadError(messageOf(error)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [datasetId, playerId, stat, eventDate, reload]);
  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!applied) return;
    const oddsError = validateOdds(Number(overOdds), Number(underOdds));
    if (oddsError) { setSaveError(oddsError); return; }
    if (!book.trim()) { setSaveError('Give this snapshot a book or source name.'); return; }
    saveController.current?.abort();
    const controller = new AbortController();
    saveController.current = controller;
    setSaving(true);
    setSaveError('');
    setSaved('');
    try {
      await api<LineSnapshot>('/api/lines', {
        method: 'POST', signal: controller.signal,
        body: JSON.stringify({ dataset_id: datasetId, player_id: playerId, stat, event_date: eventDate, book: book.trim(), line: Number(line), odds_over: Number(overOdds), odds_under: Number(underOdds) }),
      });
      if (!controller.signal.aborted) { setReload(value => value + 1); setSaved(`Snapshot saved for ${dateLabel(eventDate, true)} at ${book.trim()}.`); }
    } catch (error) { if (!isAbort(error)) setSaveError(messageOf(error)); }
    finally { if (!controller.signal.aborted) setSaving(false); }
  }
  if (!applied) return <article className="card"><EmptyState title="Give your notebook a scenario" icon={<BookOpen size={32} />} action={<button className="primary-button" onClick={onForecast}>Open forecast lab <ArrowRight size={15} /></button>}>Run a forecast first to select a dataset, player, and statistic. Then capture manual lines for a specific game and book.</EmptyState></article>;
  const grouped = lines.filter(item => item.book === book.trim()).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  const movement = grouped.length > 1 ? grouped[grouped.length - 1].line - grouped[0].line : null;
  return <>
    <div className="notice research-notice"><BookOpen size={20} /><div><strong>A notebook, not an odds feed.</strong><p>Record manually observed prices. Snapshots persist on your local server. Charts never join different games, books, players, statistics, or datasets.</p></div></div>
    <article className="card notebook-controls"><SectionHeading eyebrow="CAPTURE WHAT YOU OBSERVE" title="Save a market snapshot"><span className="pill">{applied.data.player.name} · {statNames[stat]}</span></SectionHeading>
      <form onSubmit={save}>
        <div className="snapshot-fields"><Field label="Event date"><input type="date" value={eventDate} onChange={event => { setEventDate(event.target.value); setSaved(''); }} required disabled={saving} /></Field>
          <Field label="Book / source"><input list="known-books" maxLength={60} value={book} onChange={event => { setBook(event.target.value); setSaved(''); }} required disabled={saving} /><datalist id="known-books">{[...new Set(lines.map(item => item.book))].map(name => <option value={name} key={name} />)}</datalist></Field>
          <Field label="Line"><input type="number" min="0" max="200" step="0.5" required value={line} onChange={event => setLine(event.target.value)} disabled={saving} /></Field>
          <Field label="Over odds"><input type="number" min="-10000" max="10000" step="1" required value={overOdds} onChange={event => setOverOdds(event.target.value)} disabled={saving} /></Field>
          <Field label="Under odds"><input type="number" min="-10000" max="10000" step="1" required value={underOdds} onChange={event => setUnderOdds(event.target.value)} disabled={saving} /></Field>
          <button className="primary-button" type="submit" disabled={saving}>{saving ? <Busy text="Saving" /> : <><Save size={15} />Save snapshot</>}</button>
        </div>
        <p className="chart-footnote">Line and odds start from the applied forecast; edit them here as the market changes. Saving a snapshot does not rerun or relabel your forecast. Choose the actual future event date you’re researching.</p>
      </form>
      {saveError && <ErrorNotice message={saveError} />}
      {saved && <p className="saved-message" role="status"><Save size={14} />{saved}</p>}
    </article>
    <div className="two-column notebook-grid"><article className="card chart-card"><SectionHeading eyebrow="ONE GAME. ONE BOOK." title="The line over time"><span className="pill">{book.trim() || 'Choose a book'}</span></SectionHeading>
      <div className="notebook-chart-meta"><span>{eventDate ? dateLabel(eventDate, true) : 'Choose an event date'} · {grouped.length} snapshots</span>{movement !== null && <strong className="mono">{signed(movement)} net movement</strong>}</div>
      {loading ? <div className="chart-loading"><Busy text="Reading saved snapshots…" /></div> : readError ? <ErrorNotice message={readError} onRetry={() => setReload(value => value + 1)} /> : <LineHistoryChart lines={grouped} />}
      <p className="chart-footnote">Horizontal spacing represents snapshot order, not elapsed time. The timestamp for every point is shown on focus or hover.</p>
    </article><article className="card notebook-guide"><span className="eyebrow">SMALL DETAILS, BETTER RESEARCH</span><h2>Keep the comparisons honest.</h2><ol><li><span>01</span><div><strong>Match the event</strong><p>Today’s and tomorrow’s lines are different questions—even for the same player.</p></div></li><li><span>02</span><div><strong>Match the source</strong><p>A change of book is not line movement. Use a consistent source name.</p></div></li><li><span>03</span><div><strong>Record both prices</strong><p>A 26.5 line at −110 is not the same proposition as 26.5 at −150.</p></div></li></ol><div className="quiet-note"><Info size={15} /><span>Manual entries are user-supplied and unverified. No API imports bookmaker odds automatically.</span></div></article></div>
    <article className="card model-table-card"><SectionHeading eyebrow="PERSISTED MANUAL ENTRIES" title="Snapshot ledger"><span className="subtle">All books · selected event only</span></SectionHeading>
      {!lines.length && !loading && !readError ? <div className="ledger-empty">No snapshots saved for this event yet. Your first entry will appear here.</div> : <div className="table-scroll"><table className="data-table"><thead><tr><th>Captured at</th><th>Event date</th><th>Book / source</th><th>Line</th><th>Over odds</th><th>Under odds</th></tr></thead><tbody>{[...lines].sort((a, b) => b.timestamp.localeCompare(a.timestamp)).map(item => <tr key={item.id}><td className="nowrap">{new Date(item.timestamp).toLocaleString()}</td><td>{dateLabel(item.event_date, true)}</td><td><strong>{item.book}</strong></td><td>{item.line}</td><td>{odds(item.odds_over)}</td><td>{odds(item.odds_under)}</td></tr>)}</tbody></table></div>}
    </article>
  </>;
}
