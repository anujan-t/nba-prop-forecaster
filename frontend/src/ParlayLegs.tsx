import { Check, Minus, Plus, Replace, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { decimal, modelNames, pct, statNames } from './api';
import { MAX_LEGS, americanLabel, chance, decimalOdds, legDescription, legFromForecast, legKey, legNumber, normalizeLine, sameLeg, sideNames, statLabel } from './parlay';
import type { SlipLeg } from './parlay';
import type { ParlayLeg, Side } from './parlay-types';
import type { AppliedForecast } from './types';

export function SideToggle({ side, onChange, label }: { side: Side; onChange: (side: Side) => void; label: string }) {
  return <div className="side-toggle" role="group" aria-label={label}>
    {(['over', 'under'] as const).map(option => <button key={option} type="button" className={side === option ? 'selected' : ''} aria-pressed={side === option} onClick={() => { if (option !== side) onChange(option); }}>{sideNames[option]}</button>)}
  </div>;
}

/** Line editor in 0.5 steps. Typed values are committed on Enter or blur and snapped to the nearest whole or half point. */
export function LineStepper({ value, onChange, label }: { value: number; onChange: (value: number) => void; label: string }) {
  const [draft, setDraft] = useState(String(value));
  const [invalid, setInvalid] = useState(false);
  useEffect(() => { setDraft(String(value)); setInvalid(false); }, [value]);
  const step = (delta: number) => {
    const next = normalizeLine(value + delta);
    if (next !== null && next !== value) onChange(next);
  };
  const commit = () => {
    const text = draft.trim().replace(',', '.');
    const next = text === '' ? null : normalizeLine(Number(text));
    if (next === null) { setDraft(String(value)); setInvalid(true); return; }
    setInvalid(false);
    setDraft(String(next));
    if (next !== value) onChange(next);
  };
  return <div className={`line-stepper${invalid ? ' invalid' : ''}`} role="group" aria-label={label}>
    <button type="button" aria-label="Lower the line by 0.5" disabled={value <= 0} onClick={() => step(-0.5)}><Minus size={14} aria-hidden="true" /></button>
    <input type="text" inputMode="decimal" autoComplete="off" spellCheck={false} aria-label="Line" aria-invalid={invalid} value={draft} title="Whole or half points from 0 to 200"
      onChange={event => setDraft(event.target.value.slice(0, 6))} onBlur={commit}
      onKeyDown={event => {
        if (event.key === 'Enter') { event.preventDefault(); commit(); }
        else if (event.key === 'ArrowUp') { event.preventDefault(); step(0.5); }
        else if (event.key === 'ArrowDown') { event.preventDefault(); step(-0.5); }
      }} />
    <button type="button" aria-label="Raise the line by 0.5" disabled={value >= 200} onClick={() => step(0.5)}><Plus size={14} aria-hidden="true" /></button>
    {invalid && <span className="line-stepper-error" role="alert">Use a number from 0 to 200.</span>}
  </div>;
}

/** A removed leg takes its focused button with it; move keyboard focus to the nearest remaining leg, or to a fallback. */
export function focusNearestLeg(container: HTMLElement | null, index: number, fallback: () => HTMLElement | null | undefined) {
  requestAnimationFrame(() => {
    const buttons = container ? Array.from(container.querySelectorAll<HTMLButtonElement>('.leg-remove')) : [];
    (buttons[Math.min(index, buttons.length - 1)] ?? fallback())?.focus();
  });
}

interface LegEditorProps {
  leg: SlipLeg;
  index: number;
  computed: { leg: ParlayLeg; fresh: boolean } | null;
  computing: boolean;
  errors?: string[];
  detailed?: boolean;
  /** The learning loop's recalibrated probability for this leg, when calibration is active. */
  calibrated?: number | null;
  onSide: (index: number, side: Side) => void;
  onLine: (index: number, line: number) => void;
  onRemove: (index: number) => void;
}

export function LegEditor({ leg, index, computed, computing, errors, detailed = false, calibrated = null, onSide, onLine, onRemove }: LegEditorProps) {
  const description = legDescription(leg);
  const shown = computed?.leg ?? null;
  const fresh = computed?.fresh ?? false;
  const probability = shown?.probability ?? null;
  const games = shown?.hit_rate.games ?? 0;
  return <li className={`leg-card${detailed ? ' leg-card-detailed' : ''}${errors?.length ? ' has-error' : ''} side-${leg.request.side}`}>
    <span className="leg-number" aria-hidden="true">{legNumber(index)}</span>
    <div className="leg-body">
      <div className="leg-head">
        <div className="leg-who">
          <h3><span className="sr-only">Leg {index + 1}: </span>{leg.label.player}</h3>
          <span>{[leg.label.team, statNames[leg.request.stat]].filter(Boolean).join(' · ')}</span>
        </div>
        <button type="button" className="icon-button leg-remove" aria-label={`Remove leg ${index + 1}: ${description}`} title="Remove leg" onClick={() => onRemove(index)}><X size={16} aria-hidden="true" /></button>
      </div>
      <div className="leg-controls">
        <SideToggle side={leg.request.side} onChange={side => onSide(index, side)} label={`Side for leg ${index + 1}`} />
        <LineStepper value={leg.request.line} onChange={line => onLine(index, line)} label={`Line for leg ${index + 1}, ${statLabel(leg.request.stat)}`} />
      </div>
      <div className={`leg-probability${fresh ? '' : ' is-stale'}`}>
        <div className="leg-bar" aria-hidden="true"><span style={{ width: probability === null ? '0%' : pct(Math.min(1, Math.max(0, probability))) }} /></div>
        <span className="leg-percent">
          {probability === null ? (computing ? 'Calculating…' : '—') : <>{chance(probability)}<span className="sr-only"> chance this leg wins</span></>}
          {probability !== null && calibrated !== null && Number.isFinite(calibrated) && <span className="cal-tag" title="After the learning loop’s calibration">cal. {chance(calibrated)}<span className="sr-only"> after calibration</span></span>}
          {probability !== null && !fresh && <span className="stale-tag">{computing ? 'updating' : 'out of date'}</span>}
        </span>
      </div>
      {detailed && shown && <dl className={`leg-facts${fresh ? '' : ' is-stale'}`}>
        <div><dt>Hit rate</dt><dd>{shown.hit_rate.hits}/{games}{games ? ` · ${pct(shown.hit_rate.hits / games, 0)}` : ''}</dd></div>
        <div><dt>Fair odds</dt><dd>{shown.fair_odds ? `${americanLabel(shown.fair_odds.american)} · ${decimalOdds(shown.fair_odds.decimal)}` : 'N/A'}</dd></div>
        <div><dt>Push</dt><dd>{pct(shown.push)}</dd></div>
        <div><dt>Model mean</dt><dd>{decimal(shown.mean)}</dd></div>
        <div className="leg-fact-wide"><dt>Model</dt><dd>{modelNames[shown.resolved_model]}{shown.model === 'auto' ? ' (auto)' : ''} · {shown.sample_size} games</dd></div>
      </dl>}
      {errors?.map((message, position) => <p key={position} className="leg-error" role="alert">{message}</p>)}
    </div>
  </li>;
}

interface ActionProps {
  applied: AppliedForecast;
  legs: SlipLeg[];
  full: boolean;
  dirty: boolean;
  onAdd: (side: Side, origin: HTMLElement) => void;
  onOpenSlip: () => void;
}

/** "Add Over / Under" buttons under a Forecast lab result. */
export function ForecastParlayActions({ applied, legs, full, dirty, onAdd, onOpenSlip }: ActionProps) {
  const forecast = applied.data;
  const key = legKey(applied.request);
  const index = legs.findIndex(leg => legKey(leg.request) === key);
  const existing = index >= 0 ? legs[index] : null;
  const button = (side: Side) => {
    const candidate = legFromForecast(applied, side);
    const probability = side === 'over' ? forecast.probabilities.over : forecast.probabilities.under;
    const name = `${sideNames[side]} ${forecast.line}`;
    if (existing && sameLeg(existing.request, candidate)) {
      return <button type="button" className={`leg-action ${side}-action in-slip`} aria-disabled="true"><Check size={16} aria-hidden="true" /><span>{name} is leg {index + 1}</span><span className="leg-action-prob">{pct(probability)}</span></button>;
    }
    const label = existing ? `Replace leg ${index + 1} with ${name}` : `Add ${name}`;
    return <button type="button" className={`leg-action ${side}-action`} disabled={!existing && full} onClick={event => onAdd(side, event.currentTarget)}>
      {existing ? <Replace size={16} aria-hidden="true" /> : <Plus size={16} aria-hidden="true" />}<span>{label}</span><span className="leg-action-prob">{pct(probability)}</span>
    </button>;
  };
  const message = existing
    ? `Your slip already has ${forecast.player.name} ${statLabel(forecast.stat)} as leg ${index + 1} (${sideNames[existing.request.side]} ${existing.request.line}). One leg per player and stat.`
    : full ? `The slip is full: ${MAX_LEGS} legs is the maximum. Remove a leg to add this one.`
      : 'Back the Over or the Under as a parlay leg. The Parlay view shows the chance that every leg hits.';
  return <div className="card parlay-actions">
    <div className="parlay-actions-copy">
      <span className="eyebrow">BUILD A PARLAY</span>
      <p>{message}</p>
      {dirty && <p className="field-hint">Adds the applied forecast shown here, not the unapplied changes in the form.</p>}
    </div>
    <div className="parlay-action-buttons">{button('over')}{button('under')}</div>
    <button type="button" className="text-button slip-link" onClick={onOpenSlip}>Open slip · {legs.length} {legs.length === 1 ? 'leg' : 'legs'}</button>
  </div>;
}
