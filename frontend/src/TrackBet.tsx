import { ArrowUpRight, CircleCheck, ClipboardCheck, X } from 'lucide-react';
import { useEffect, useId, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import { messageOf, todayEastern } from './api';
import type { TrackedBet } from './assistant-types';
import { Field } from './components';
import { trackBet } from './ledger';
import { americanLabel, chatSlip, parseBookOdds } from './parlay';
import type { ParlayState } from './useParlay';

const MAX_STAKE = 1_000_000;

function parseStake(text: string): { value: number | null; error: string } {
  const cleaned = text.trim().replace(/^\$/, '').replace(/,/g, '');
  if (!cleaned) return { value: null, error: '' };
  const value = Number(cleaned);
  if (!/^\d*\.?\d+$/.test(cleaned) || !Number.isFinite(value) || value <= 0 || value > MAX_STAKE) {
    return { value: null, error: 'Enter a stake above 0 and up to 1,000,000, or leave it blank.' };
  }
  return { value: Math.round(value * 100) / 100, error: '' };
}

/** "Track this parlay": a small inline form that saves the current slip to the Track record. */
export function TrackBetForm({ parlay, onTracked, context }: { parlay: ParlayState; onTracked: (bet: TrackedBet) => void; context: 'view' | 'slip' }) {
  const [open, setOpen] = useState(false);
  const [stake, setStake] = useState('');
  const [date, setDate] = useState(todayEastern);
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [stakeError, setStakeError] = useState('');
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const toggle = useRef<HTMLButtonElement>(null);
  const firstField = useRef<HTMLInputElement>(null);
  const restoreFocus = useRef(false);
  const id = useId();
  const formId = `${id}-track-form`;
  const legs = parlay.legs.length;
  const price = parseBookOdds(parlay.bookOdds);
  const blocked = !legs ? 'Add at least one leg first.' : price.error ? 'Fix the book odds before tracking this parlay.' : '';

  useEffect(() => {
    if (open) { setDate(todayEastern()); firstField.current?.focus(); }
    else if (restoreFocus.current) { restoreFocus.current = false; toggle.current?.focus(); }
  }, [open]);

  useEffect(() => { if (!legs) setOpen(false); }, [legs]);

  const close = () => { restoreFocus.current = true; setOpen(false); setError(''); setStakeError(''); };

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy || blocked) return;
    const parsed = parseStake(stake);
    if (parsed.error) { setStakeError(parsed.error); firstField.current?.focus(); return; }
    if (date && !/^\d{4}-\d{2}-\d{2}$/.test(date)) { setError('Pick the game date.'); return; }
    setBusy(true);
    setError('');
    try {
      const bet = await trackBet({ slip: chatSlip(parlay.legs, parlay.bookOdds), stake: parsed.value, event_date: date || null, note: note.trim() || null });
      setStake('');
      setNote('');
      setStatus(`Tracked the ${legs}-leg parlay.`);
      restoreFocus.current = true;
      setOpen(false);
      onTracked(bet);
    } catch (reason) {
      setError(messageOf(reason));
    } finally {
      setBusy(false);
    }
  }

  const onKeyDown = (event: KeyboardEvent<HTMLFormElement>) => {
    if (event.key === 'Escape' && !busy) { event.stopPropagation(); close(); }
  };

  return <>
    <button ref={toggle} type="button" className={`${context === 'slip' ? 'text-button' : 'secondary-button'} track-toggle`} aria-expanded={open} aria-controls={formId}
      disabled={!legs} onClick={() => open ? close() : setOpen(true)}>
      <ClipboardCheck size={15} aria-hidden="true" />Track this parlay
    </button>
    {open && <form id={formId} className={`track-form ${context}`} onSubmit={event => { void submit(event); }} onKeyDown={onKeyDown} aria-label="Track this parlay" aria-busy={busy || undefined} noValidate>
      <div className="track-fields">
        <Field label="Stake (optional)" className="track-stake">
          <input ref={firstField} type="text" inputMode="decimal" autoComplete="off" placeholder="10" maxLength={14} value={stake}
            aria-invalid={stakeError ? true : undefined} aria-describedby={stakeError ? `${id}-stake-error` : undefined}
            onChange={event => { setStake(event.target.value); setStakeError(''); }} />
        </Field>
        <Field label="Game date (US Eastern)" className="track-date">
          <input type="date" value={date} onChange={event => setDate(event.target.value)} />
        </Field>
        <Field label="Note (optional)" className="track-note">
          <input type="text" maxLength={200} autoComplete="off" placeholder="Why you like it" value={note} onChange={event => setNote(event.target.value)} />
        </Field>
      </div>
      {stakeError && <p id={`${id}-stake-error`} className="field-error" role="alert">{stakeError}</p>}
      {blocked && <p className="field-error" role="alert">{blocked}</p>}
      {error && <p className="field-error" role="alert">{error}</p>}
      <div className="track-actions">
        <button type="submit" className="primary-button" aria-disabled={busy || Boolean(blocked) || undefined}>{busy ? 'Tracking…' : `Track ${legs} ${legs === 1 ? 'leg' : 'legs'}`}</button>
        <button type="button" className="text-button" disabled={busy} onClick={close}>Cancel</button>
      </div>
      <p className="track-fine">Saves the legs, the model’s probability right now{price.value !== null ? ` and your book odds (${americanLabel(price.value)})` : ''}. Grade it in Track record once the games are final.</p>
    </form>}
    <span className="sr-only" role="status">{status}</span>
  </>;
}

export interface ToastData { id: number; title: string; detail: string }

/** Confirmation that floats above the page and links to the Track record. Pauses while hovered or focused. */
export function Toast({ toast, onClose, onOpenRecord }: { toast: ToastData; onClose: () => void; onOpenRecord: () => void }) {
  const [paused, setPaused] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (paused) return;
    const timer = window.setTimeout(onClose, 10000);
    return () => window.clearTimeout(timer);
  }, [toast.id, paused, onClose]);
  const dismiss = () => {
    const scope = root.current?.closest<HTMLElement>('.slip-panel') ?? document.querySelector<HTMLElement>('.view-section:not([hidden])');
    scope?.querySelector<HTMLElement>('.track-toggle:not(:disabled)')?.focus();
    onClose();
  };
  return <div ref={root} className="toast" role="status" aria-live="polite" onMouseEnter={() => setPaused(true)} onMouseLeave={() => setPaused(false)}
    onFocus={() => setPaused(true)} onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setPaused(false); }}>
    <CircleCheck size={20} aria-hidden="true" className="toast-icon" />
    <div className="toast-copy"><strong>{toast.title}</strong><span>{toast.detail}</span></div>
    <div className="toast-actions">
      <button type="button" className="text-button" onClick={onOpenRecord}>Open Track record <ArrowUpRight size={14} aria-hidden="true" /></button>
      <button type="button" className="icon-button toast-close" aria-label="Dismiss" onClick={dismiss}><X size={16} aria-hidden="true" /></button>
    </div>
  </div>;
}
