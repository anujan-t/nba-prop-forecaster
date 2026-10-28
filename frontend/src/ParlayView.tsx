import { ArrowRight, BrainCircuit, ClipboardCheck, Info, Link2, ShieldAlert, TriangleAlert } from 'lucide-react';
import { useRef } from 'react';
import { edgePoints, pct, signed } from './api';
import type { TrackedBet } from './assistant-types';
import { CourtLines, ErrorNotice, Field, ScoreboardDigits, SectionHeading, SwishHoop } from './components';
import { useCountUp } from './motion';
import { MAX_LEGS, americanLabel, basisNames, chance, chanceDigits, decimalOdds, effectStrength, legNumber, liftSentence, pairEffect } from './parlay';
import type { SlipLeg } from './parlay';
import type { ParlayLegRequest, ParlayPair, ParlayResult } from './parlay-types';
import { LegEditor, focusNearestLeg } from './ParlayLegs';
import { TrackBetForm } from './TrackBet';
import { computedLeg } from './useParlay';
import type { ParlayState } from './useParlay';

function legList(indices: number[]): string {
  const names = indices.map(index => String(index + 1));
  if (names.length <= 1) return `Leg ${names.join('')}`;
  return `Legs ${names.slice(0, -1).join(', ')} & ${names[names.length - 1]}`;
}

function Scoreboard({ data, count, stale, computing, blocked }: { data: ParlayResult | null; count: number; stale: boolean; computing: boolean; blocked: boolean }) {
  const finalText = data ? chanceDigits(data.probability) : '';
  const target = finalText && Number.isFinite(Number(finalText)) ? Number(finalText) : null;
  const value = useCountUp(target);
  const decimals = finalText.includes('.') ? finalText.split('.')[1].length : 1;
  const shown = value === null ? finalText : value.toFixed(decimals).padStart(finalText.length, ' ');
  const status = computing ? 'Calculating' : blocked ? 'Needs attention' : stale ? 'Out of date' : data ? 'Up to date' : 'Waiting';
  return <article className={`scoreboard-card${stale || computing ? ' is-stale' : ''}`} aria-busy={computing}>
    <div className="scoreboard-head">
      <span className="scoreboard-label">ALL {count} {count === 1 ? 'LEG HITS' : 'LEGS HIT'}</span>
      <span className={`scoreboard-status${computing ? ' live' : blocked ? ' alert' : ''}`}><span aria-hidden="true" />{status}</span>
    </div>
    <div className="scoreboard-main">
      {data ? <div className="scoreboard-value">
        <ScoreboardDigits text={shown} label={`${chance(data.probability)} model probability that all ${count} legs hit`} className="scoreboard-big" />
        <span className="sb-unit" aria-hidden="true">%</span>
      </div> : computing ? null : <span className="scoreboard-placeholder">--.-</span>}
      {computing && <div className={data ? 'swish-corner' : 'swish-center'}><SwishHoop /><span className="sr-only">Calculating the parlay</span></div>}
    </div>
    {data && <>
      <div className="scoreboard-odds">
        <div><span>Fair odds</span><strong>{data.fair_odds ? americanLabel(data.fair_odds.american) : 'N/A'}</strong></div>
        <div><span>Decimal</span><strong>{data.fair_odds ? decimalOdds(data.fair_odds.decimal) : 'N/A'}</strong></div>
        <div><span>Independent</span><strong>{chance(data.independent_probability)}</strong></div>
        <div><span>Correlation lift</span><strong>{data.correlation_lift === null ? 'N/A' : `×${data.correlation_lift.toFixed(2)}`}</strong></div>
      </div>
      <p className="lift-sentence">{liftSentence(data)}</p>
      {data.calibration && <p className="scoreboard-cal"><BrainCircuit size={14} aria-hidden="true" /><span>Calibrated <strong>{chance(data.calibration.probability)}</strong> <span className="scoreboard-cal-note">· {data.calibration.note}</span></span></p>}
      {data.no_loss_probability !== null && <p className="scoreboard-note">No leg loses (a push on a whole-number line refunds that leg): <strong>{chance(data.no_loss_probability)}</strong>.</p>}
      <p className="scoreboard-foot">Fair odds are the price with no margin for this probability. Calculated at {new Date(data.computed_at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</p>
    </>}
  </article>;
}

function BookOdds({ parlay, data }: { parlay: ParlayState; data: ParlayResult | null }) {
  const book = data?.book ?? null;
  const pending = parlay.stale || parlay.computing;
  return <article className="card book-card">
    <SectionHeading eyebrow="COMPARE A PRICE" title="Book odds" />
    <Field label="Parlay odds offered (American, optional)" hint="For example +450 or −120. Leave blank to skip.">
      <input type="text" inputMode="numeric" autoComplete="off" placeholder="+450" value={parlay.bookOdds} maxLength={12}
        aria-invalid={parlay.bookOddsError ? true : undefined} aria-describedby={parlay.bookOddsError ? 'book-odds-error' : undefined}
        onChange={event => parlay.setBookOdds(event.target.value)} />
    </Field>
    {parlay.bookOddsError && <p id="book-odds-error" className="field-error" role="alert">{parlay.bookOddsError}</p>}
    {book ? <dl className={`book-stats${pending ? ' is-stale' : ''}`}>
      <div><dt>Implied by {americanLabel(book.odds)}</dt><dd>{pct(book.implied)}</dd></div>
      <div><dt>Model minus implied</dt><dd className={book.edge >= 0 ? 'positive' : 'negative'}>{edgePoints(book.edge)}</dd></div>
      <div><dt>Expected return per 1 unit</dt><dd className={book.expected_roi >= 0 ? 'positive' : 'negative'}>{signed(book.expected_roi, 2)} <small>({signed(book.expected_roi * 100)}%)</small></dd></div>
    </dl> : <p className="book-empty">{parlay.bookOdds.trim() && !parlay.bookOddsError && pending ? 'Comparing…' : 'Enter a price to compare it with the model’s joint probability.'}</p>}
    <p className="field-hint">Pushes count as losses in this comparison. A positive model edge is not a recommendation: the model can be wrong, and every leg must win.</p>
  </article>;
}

function Linkage({ data, requests, legs }: { data: ParlayResult; requests: ParlayLegRequest[]; legs: SlipLeg[] }) {
  const count = data.legs.length;
  const pairAt = (a: number, b: number): ParlayPair | undefined => data.pairs.find(pair => pair.a === a && pair.b === b);
  const linked = data.pairs.filter(pair => pair.basis !== 'independent');
  const independent = data.pairs.length - linked.length;
  const tone = (effect: number) => effect > 0.0001 ? 'together' : effect < -0.0001 ? 'against' : 'neutral';
  return <article className="card linkage-card">
    <SectionHeading eyebrow="HOW THE LEGS MOVE" title="Linked legs"><Link2 size={19} className="subtle" aria-hidden="true" /></SectionHeading>
    <p className="card-intro">Same-player and teammate legs share games, so they tend to rise or fall together. Players on different current teams are treated as independent.</p>
    <div className="table-scroll" tabIndex={0} role="region" aria-label="Leg correlation matrix"><table className="link-matrix">
      <caption className="sr-only">Correlation between legs, oriented by side: positive values mean the two legs tend to win together.</caption>
      <thead><tr><th scope="col"><span className="sr-only">Leg</span></th>{data.legs.slice(1).map(leg => <th scope="col" key={leg.index}>{legNumber(leg.index)}</th>)}</tr></thead>
      <tbody>{data.legs.slice(0, -1).map(row => <tr key={row.index}>
        <th scope="row">{legNumber(row.index)}</th>
        {data.legs.slice(1).map(column => {
          if (column.index <= row.index) return <td key={column.index} className="link-empty" />;
          const pair = pairAt(row.index, column.index);
          if (!pair) return <td key={column.index} className="link-empty" />;
          const effect = pairEffect(pair, requests);
          const independentPair = pair.basis === 'independent';
          return <td key={column.index}><span className={`link-cell ${independentPair ? 'neutral' : tone(effect)} strength-${effectStrength(effect)}`} title={pair.note}>
            {independentPair ? '·' : signed(effect, 2)}<span className="sr-only">, {basisNames[pair.basis]}{independentPair ? '' : `, ${pair.games} shared games`}</span>
          </span></td>;
        })}
      </tr>)}</tbody>
    </table></div>
    <div className="link-legend" aria-hidden="true"><span><i className="together" />Win together</span><span><i className="against" />Work against each other</span><span><i className="neutral" />Independent</span><span className="legend-note">Values are signed for the sides you picked.</span></div>
    {linked.length > 0 && <ul className="pair-list">{linked.map(pair => {
      const effect = pairEffect(pair, requests);
      const first = legs[pair.a]?.label.player;
      const second = legs[pair.b]?.label.player;
      return <li key={`${pair.a}-${pair.b}`} className={tone(effect)}>
        <div className="pair-head"><strong>{legList([pair.a, pair.b])}</strong><span className="pill">{basisNames[pair.basis]}</span><span className="pair-strength">{effectStrength(effect) === 'none' ? 'negligible' : effectStrength(effect)} · {effect >= 0 ? 'together' : 'against'}</span></div>
        <p>{pair.note}</p>
        <span className="pair-stats">{pair.games} shared games · rank correlation {pair.rank_correlation === null ? 'N/A' : signed(pair.rank_correlation, 2)} · model uses {signed(pair.correlation, 2)}{Math.abs(effect - pair.correlation) > 0.0001 ? ` (${signed(effect, 2)} for these sides)` : ''}</span>
        {first && second && <span className="sr-only">{first === second ? `Both legs are ${first}.` : `${first} and ${second}.`}</span>}
      </li>;
    })}</ul>}
    {count > 1 && independent > 0 && <p className="field-hint">{independent === data.pairs.length ? 'Every pair' : `${independent} of ${data.pairs.length} pairs`} {independent === 1 ? 'is' : 'are'} treated as independent (different current teams).</p>}
  </article>;
}

function History({ data }: { data: ParlayResult }) {
  return <article className="card history-card">
    <SectionHeading eyebrow="BEEN HERE BEFORE" title="Linked legs, in past games" />
    <ul className="history-groups">{data.history.map(group => {
      const rate = group.games ? group.hits / group.games : null;
      return <li key={group.legs.join('-')}>
        <p><strong>{legList(group.legs)}:</strong> {group.games
          ? <>all hit in <strong>{group.hits} of {group.games}</strong> shared games ({pct(rate, 0)}) vs model {chance(group.model_probability)}.</>
          : <>no shared games to check yet. Model {chance(group.model_probability)}.</>}</p>
        {rate !== null && <div className="history-bars" aria-hidden="true">
          <span className="history-bar actual"><i style={{ width: pct(rate) }} /></span>
          <span className="history-bar model"><i style={{ width: pct(Math.min(1, group.model_probability)) }} /></span>
        </div>}
      </li>;
    })}</ul>
    <p className="field-hint">Shared games are past games in which every player in the group played. Small samples swing widely, so read these as a sanity check, not proof.</p>
  </article>;
}

export default function ParlayView({ parlay, onForecast, onTracked }: { parlay: ParlayState; onForecast: () => void; onTracked: (bet: TrackedBet) => void }) {
  const { legs, result, stale, computing, error } = parlay;
  const data = result?.data ?? null;
  const calibration = data?.calibration ?? null;
  const fresh = data !== null && !stale && !computing;
  const legErrors = error ? Object.keys(error.legs).length : 0;
  const view = useRef<HTMLDivElement>(null);
  const removeLeg = (index: number) => { parlay.removeLeg(index); focusNearestLeg(view.current, index, () => view.current?.querySelector<HTMLElement>('.parlay-empty .primary-button')); };
  return <div className="parlay-view" ref={view}>
    <div className="notice research-notice parlay-disclaimer"><ShieldAlert size={18} aria-hidden="true" /><div><strong>Probabilities, not certainties.</strong><p>Research only, not betting advice. A parlay only cashes if every leg wins, so even a strong-looking combination misses often.</p></div></div>
    {!legs.length ? <div className="card parlay-empty">
      <SwishHoop className="is-static" />
      <h3>Your slip is empty</h3>
      <ol className="parlay-steps">
        <li><span>1</span>Run a forecast for any player in the Forecast lab.</li>
        <li><span>2</span>Add its Over or Under as a leg. Up to {MAX_LEGS} legs, one per player and stat.</li>
        <li><span>3</span>Come back here for the chance that every leg hits, with links between same-player and teammate legs.</li>
      </ol>
      <button type="button" className="primary-button" onClick={onForecast}>Go to the Forecast lab <ArrowRight size={16} aria-hidden="true" /></button>
    </div> : <>
      <p className="sr-only" aria-live="polite">{fresh && data ? `All ${legs.length} legs hit: ${chance(data.probability)} model probability${data.fair_odds ? `, fair odds ${americanLabel(data.fair_odds.american)}` : ''}.` : ''}</p>
      {error && error.general.length > 0 && <ErrorNotice message={error.general.join(' ')} onRetry={parlay.retry} />}
      {legErrors > 0 && <div className="notice warning-notice" role="status"><TriangleAlert size={18} aria-hidden="true" /><div><strong>{legErrors === 1 ? 'One leg needs attention.' : `${legErrors} legs need attention.`}</strong><p>The details are shown on the {legErrors === 1 ? 'leg' : 'legs'} below. Adjust or remove {legErrors === 1 ? 'it' : 'them'} to recalculate.</p></div></div>}
      <div className="parlay-top">
        <Scoreboard data={data} count={legs.length} stale={stale} computing={computing} blocked={!computing && error !== null && (error.general.length > 0 || legErrors > 0)} />
        <BookOdds parlay={parlay} data={data} />
      </div>
      <section className="card track-card" aria-labelledby="track-card-title">
        <span className="track-card-icon" aria-hidden="true"><ClipboardCheck size={22} /></span>
        <div className="track-card-copy">
          <span className="eyebrow">KEEP SCORE</span>
          <h2 id="track-card-title">Grade it when the games end</h2>
          <p>Tracking saves these legs and the model’s probability now. Once the games are final, grade it on Track record. Graded results feed the learning loop.</p>
        </div>
        <TrackBetForm parlay={parlay} onTracked={onTracked} context="view" />
      </section>
      <section className="parlay-legs" aria-labelledby="parlay-legs-title">
        <div className="section-heading"><div className="section-heading-copy"><span className="eyebrow"><CourtLines />THE LEGS</span><h2 id="parlay-legs-title">Leg by leg</h2></div><span className="pill">{legs.length} of {MAX_LEGS}</span></div>
        <ol className="leg-grid">{legs.map((leg, index) => {
          const computed = computedLeg(result, leg, index);
          return <LegEditor key={`${leg.request.player_id}:${leg.request.stat}`} leg={leg} index={index} detailed computed={computed} computing={computing}
            calibrated={computed && calibration ? calibration.legs[index] ?? null : null}
            errors={error?.legs[index]} onSide={parlay.setSide} onLine={parlay.setLine} onRemove={removeLeg} />;
        })}</ol>
        {calibration && <p className="calibration-legend"><BrainCircuit size={14} aria-hidden="true" /><span>“cal.” is each leg after the learning loop’s calibration, based on {calibration.samples} graded predictions. It is on only because it beat the raw model on recent held-out games.</span></p>}
      </section>
      {data && result && <div className={`parlay-details${stale ? ' is-stale' : ''}`}>
        {data.legs.length > 1 && <Linkage data={data} requests={result.request.legs} legs={legs} />}
        {data.history.length > 0 && <History data={data} />}
        <article className="card method-card"><SectionHeading eyebrow="HOW IT’S CALCULATED" title="The method" /><p>{data.method}</p>
          <p className="quiet-note"><Info size={14} aria-hidden="true" /><span>Each leg uses exactly the same model as the Forecast lab. A whole-number line can push; a push does not count as a win here.</span></p></article>
        {data.warnings.length > 0 && <div className="notice warning-notice" role="status"><TriangleAlert size={18} aria-hidden="true" /><div><strong>Keep the limitations in view</strong>{data.warnings.map((warning, index) => <p key={index}>{warning}</p>)}</div></div>}
      </div>}
    </>}
  </div>;
}
