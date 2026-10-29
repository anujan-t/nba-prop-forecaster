import { AlertCircle, ArrowUpRight } from 'lucide-react';
import type { ReactNode, Ref } from 'react';
import { ballSeams } from './motion';

export function Basketball({ className = '' }: { className?: string }) {
  return <svg className={className} viewBox="0 0 48 48" fill="none" aria-hidden="true">
    <circle cx="24" cy="24" r="21" stroke="currentColor" strokeWidth="1.7" />
    <path d="M3 24h42M24 3v42M9.2 9.2c15 3.5 26.1 15.9 29.6 29.6M38.8 9.2C23.9 12.7 12.7 23.9 9.2 38.8" stroke="currentColor" strokeWidth="1.7" />
  </svg>;
}

/** A shaded basketball with seams; sized by CSS. */
export function Ball({ className = '' }: { className?: string }) {
  return <span className={`ball ${className}`} aria-hidden="true"><svg viewBox="0 0 32 32"><path d={ballSeams} /></svg></span>;
}

/** Half-court diagram for the page hero: lines draw on, then a shot drops through the rim. */
export function CourtArt() {
  return <svg className="court-art" viewBox="0 0 420 220" fill="none" aria-hidden="true">
    <path className="court-paint" d="M290 78h110v64H290z" />
    <g className="court-art-lines">
      <path pathLength={1} d="M400 20H20v180h380z" />
      <path pathLength={1} d="M20 78a32 32 0 0 1 0 64" />
      <path pathLength={1} d="M400 78H290v64h110" />
      <path pathLength={1} d="M290 78a32 32 0 0 0 0 64" />
      <path pathLength={1} d="M400 34h-54a80.3 80.3 0 0 0 0 152h54" />
      <path pathLength={1} d="M352 92a20 20 0 0 0 0 36" />
    </g>
    <path className="court-shot" pathLength={1} d="M96 168Q248 -24 368 104" />
    <path className="court-board" d="M385 92v36" />
    <g className="court-ball-x"><g className="court-ball-y">
      <circle className="court-ball-skin" r="7" />
      <path className="court-ball-seams" transform="translate(-7 -7) scale(.4375)" d={ballSeams} />
    </g></g>
    <g className="court-net"><path d="M365 111l2.5 11M372 112v11.5M379 111l-2.5 11M366.5 117h11" /></g>
    <ellipse className="court-rim" cx="372" cy="110" rx="8" ry="2.6" />
  </svg>;
}

/** Centre-circle ornament used in front of section eyebrows; strokes draw on. */
export function CourtLines() {
  return <svg className="court-lines" viewBox="0 0 40 14" fill="none" aria-hidden="true">
    <path pathLength={1} d="M1 7h12M27 7h12" />
    <path pathLength={1} d="M14 7a6 6 0 1 0 12 0a6 6 0 1 0 -12 0" />
  </svg>;
}

export function ErrorNotice({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return <div className="notice error-notice" role="alert">
    <AlertCircle size={18} aria-hidden="true" /><div><strong>We couldn’t complete that request.</strong><p>{message}</p>
      {onRetry && <button className="text-button" onClick={onRetry}>Try again <ArrowUpRight size={14} /></button>}
    </div>
  </div>;
}

export function DribbleBall({ size = 'small' }: { size?: 'small' | 'large' }) {
  return <span className={`dribble dribble-${size}`} aria-hidden="true"><span className="dribble-ball"><Ball /></span><span className="dribble-shadow" /></span>;
}

export function Busy({ text = 'Crunching the numbers…' }: { text?: string }) {
  return <span className="busy" role="status"><DribbleBall />{text}</span>;
}

/** Computing state for the parlay: the ball arcs in and the net swishes. */
export function SwishHoop({ className = '' }: { className?: string }) {
  return <svg className={`swish ${className}`} viewBox="0 0 120 120" fill="none" aria-hidden="true">
    <rect className="swish-board" x="28" y="6" width="64" height="40" rx="4" />
    <rect className="swish-square" x="47" y="22" width="26" height="18" rx="1.5" />
    <path className="swish-rim-back" d="M40 50a20 4.5 0 0 1 40 0" />
    <g className="swish-ball-x"><g className="swish-ball-y">
      <circle className="swish-ball-skin" r="9" />
      <path className="swish-ball-seams" transform="translate(-9 -9) scale(.5625)" d={ballSeams} />
    </g></g>
    <g className="swish-net"><path d="M41 51l7 28M50 53.5l3.5 26.5M60 54.5V81M70 53.5l-3.5 26.5M79 51l-7 28M44 63h32M47 73h26" /></g>
    <path className="swish-rim-front" d="M40 50a20 4.5 0 0 0 40 0" />
  </svg>;
}

export function EmptyState({ icon, title, children, action }: { icon?: ReactNode; title: string; children: ReactNode; action?: ReactNode }) {
  return <div className="empty-state"><div className="empty-icon">{icon ?? <Basketball />}</div><h3>{title}</h3><p>{children}</p>{action}</div>;
}

export function SectionHeading({ eyebrow, title, children, id, headingRef }: { eyebrow?: string; title: string; children?: ReactNode; id?: string; headingRef?: Ref<HTMLHeadingElement> }) {
  return <div className="section-heading"><div className="section-heading-copy">{eyebrow && <span className="eyebrow"><CourtLines />{eyebrow}</span>}<h2 id={id} ref={headingRef} tabIndex={id ? -1 : undefined}>{title}</h2></div>{children}</div>;
}

export function Field({ label, children, hint, className = '' }: { label: string; children: ReactNode; hint?: string; className?: string }) {
  return <label className={`field ${className}`}><span className="field-label">{label}</span>{children}{hint && <span className="field-hint">{hint}</span>}</label>;
}

export function StatCard({ label, value, unit, detail, accent = false }: { label: string; value: ReactNode; unit?: string; detail: ReactNode; accent?: boolean }) {
  return <article className={`card stat-card${accent ? ' accent-card' : ''}`}>
    <span className="eyebrow">{label}</span><div className="stat-number">{value}{unit && <span>{unit}</span>}</div><div className="stat-detail">{detail}</div>
  </article>;
}

/** Scoreboard digits: each numeral sits in a fixed cell over an unlit "8". Screen readers get the label instead. */
export function ScoreboardDigits({ text, label, className = '' }: { text: string; label: string; className?: string }) {
  return <span className={`scoreboard-digits ${className}`} role="img" aria-label={label}>
    {Array.from(text).map((character, index) => /[\d ]/.test(character)
      ? <span key={index} className="sb-digit" data-ghost="8">{character === ' ' ? '' : character}</span>
      : <span key={index} className="sb-mark">{character}</span>)}
  </span>;
}

/** Jersey silhouette carrying a number or initials. Decorative. */
export function Jersey({ label, className = '' }: { label: string; className?: string }) {
  const size = label.length >= 3 ? 'jersey-long' : label.length === 2 ? 'jersey-mid' : 'jersey-short';
  return <span className={`jersey ${size} ${className}`} aria-hidden="true">
    <svg viewBox="0 0 64 72"><path d="M17 2h8c1 6 3.5 9 7 9s6-3 7-9h8c1 9 5.5 15.5 13 17v51H4V19c7.5-1.5 12-8 13-17z" /></svg>
    <span className="jersey-label">{label}</span>
  </span>;
}
