import { ArrowRight, Trash2, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { Ball, Busy, ScoreboardDigits } from './components';
import { MAX_LEGS, americanLabel, chance, chanceDigits } from './parlay';
import { LegEditor, focusNearestLeg } from './ParlayLegs';
import { computedLeg } from './useParlay';
import type { ParlayState } from './useParlay';

export function SlipButton({ count, open, bounce, onOpen }: { count: number; open: boolean; bounce: number; onOpen: () => void }) {
  return <button id="parlay-slip-button" type="button" className={`slip-fab${count ? ' has-legs' : ''}`} aria-haspopup="dialog" aria-expanded={open} aria-controls="parlay-slip"
    aria-label={`Parlay slip, ${count} ${count === 1 ? 'leg' : 'legs'}`} title="Parlay slip" onClick={onOpen}>
    <Ball className="slip-fab-ball" />
    <span key={bounce} className={`slip-badge${bounce ? ' bounce' : ''}`} aria-hidden="true">{count}</span>
  </button>;
}

const focusableSelector = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

interface DrawerProps { parlay: ParlayState; open: boolean; onClose: () => void; onViewParlay: () => void }

export function SlipDrawer({ parlay, open, onClose, onViewParlay }: DrawerProps) {
  const panel = useRef<HTMLDivElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const clearButton = useRef<HTMLButtonElement>(null);
  const keepButton = useRef<HTMLButtonElement>(null);
  const confirmFocus = useRef<'keep' | 'clear' | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const { legs, result, stale, computing, error } = parlay;
  const data = result?.data ?? null;

  useEffect(() => {
    const target = confirmFocus.current === 'keep' ? keepButton.current : confirmFocus.current === 'clear' ? clearButton.current : null;
    confirmFocus.current = null;
    target?.focus();
  }, [confirmClear]);

  const askClear = (confirm: boolean) => { confirmFocus.current = confirm ? 'keep' : 'clear'; setConfirmClear(confirm); };
  const removeLeg = (index: number) => { parlay.removeLeg(index); focusNearestLeg(panel.current, index, () => closeButton.current); };

  useEffect(() => {
    if (!open) { setConfirmClear(false); return; }
    closeButton.current?.focus();
    const onKey = (event: globalThis.KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  useEffect(() => { if (!legs.length) setConfirmClear(false); }, [legs.length]);

  const trapFocus = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Tab' || !panel.current) return;
    const focusable = Array.from(panel.current.querySelectorAll<HTMLElement>(focusableSelector)).filter(element => element.getClientRects().length > 0);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && (document.activeElement === first || !panel.current.contains(document.activeElement))) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  };

  const fresh = data !== null && !stale && !computing;
  const legErrors = error ? Object.keys(error.legs).length : 0;
  return <div className={`slip-layer${open ? ' open' : ''}`} inert={!open}>
    <div className="slip-backdrop" onClick={onClose} aria-hidden="true" />
    <div className="slip-panel" id="parlay-slip" role="dialog" aria-modal="true" aria-labelledby="slip-title" ref={panel} onKeyDown={trapFocus}>
      <header className="slip-header">
        <div className="slip-heading"><span className="eyebrow">PARLAY SLIP</span><h2 id="slip-title">{legs.length ? `${legs.length} ${legs.length === 1 ? 'leg' : 'legs'}` : 'No legs yet'}<span className="slip-capacity"> of {MAX_LEGS}</span></h2></div>
        <button ref={closeButton} type="button" className="icon-button" aria-label="Close parlay slip" onClick={onClose}><X size={18} aria-hidden="true" /></button>
      </header>
      <div className="slip-scroll">
        {legs.length > 0 && <div className={`slip-summary${stale || computing ? ' is-stale' : ''}`}>
          <div className="slip-summary-score">
            <span className="scoreboard-label">ALL {legs.length} {legs.length === 1 ? 'LEG HITS' : 'LEGS HIT'}</span>
            {data ? <ScoreboardDigits text={chanceDigits(data.probability)} label={`${chance(data.probability)} chance`} className="slip-digits" /> : <span className="scoreboard-placeholder">--.-</span>}
            <span className="sb-unit" aria-hidden="true">%</span>
          </div>
          <div className="slip-summary-meta">
            {computing ? <Busy text="Calculating…" /> : stale ? <span className="status-chip">Out of date</span> : data ? <span className="status-chip ready">Up to date</span> : null}
            {data?.fair_odds && <span>Fair odds <strong>{americanLabel(data.fair_odds.american)}</strong></span>}
          </div>
          <p className="sr-only" aria-live="polite">{fresh && data ? `All ${legs.length} legs hit: ${chance(data.probability)} model probability.` : ''}</p>
        </div>}
        {error && (error.general.length > 0 || legErrors > 0) && <div className="notice error-notice slip-error" role="alert">
          <div><strong>The parlay couldn’t be calculated.</strong>
            {error.general.map((message, index) => <p key={index}>{message}</p>)}
            {legErrors > 0 && <p>Check the {legErrors === 1 ? 'highlighted leg' : `${legErrors} highlighted legs`} below.</p>}
            <button type="button" className="text-button" onClick={parlay.retry}>Try again</button>
          </div>
        </div>}
        {legs.length ? <ol className="leg-list">
          {legs.map((leg, index) => <LegEditor key={`${leg.request.player_id}:${leg.request.stat}`} leg={leg} index={index} computed={computedLeg(result, leg, index)} computing={computing}
            errors={error?.legs[index]} onSide={parlay.setSide} onLine={parlay.setLine} onRemove={removeLeg} />)}
        </ol> : <div className="slip-empty">
          <Ball className="slip-empty-ball" />
          <p><strong>Your slip is empty.</strong> Run a forecast in the Forecast lab, then add its Over or Under as a leg.</p>
        </div>}
      </div>
      <footer className="slip-footer">
        <button type="button" className="primary-button" onClick={onViewParlay}>{legs.length ? 'Open full breakdown' : 'Go to the Parlay view'} <ArrowRight size={16} aria-hidden="true" /></button>
        {legs.length > 0 && (confirmClear
          ? <div className="clear-confirm" role="group" aria-label="Confirm clearing the slip"><span>Remove all {legs.length} legs?</span><button type="button" className="danger-button" onClick={() => { parlay.clear(); setConfirmClear(false); closeButton.current?.focus(); }}>Clear slip</button><button ref={keepButton} type="button" className="text-button" onClick={() => askClear(false)}>Keep</button></div>
          : <button ref={clearButton} type="button" className="text-button clear-button" onClick={() => askClear(true)}><Trash2 size={14} aria-hidden="true" />Clear slip</button>)}
        <p className="slip-disclaimer">Probabilities, not certainties. Research only, not betting advice.</p>
      </footer>
    </div>
  </div>;
}
