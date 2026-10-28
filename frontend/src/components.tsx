import { AlertCircle, ArrowUpRight, LoaderCircle } from 'lucide-react';
import type { ReactNode } from 'react';

export function Basketball({ className = '' }: { className?: string }) {
  return <svg className={className} viewBox="0 0 48 48" fill="none" aria-hidden="true">
    <circle cx="24" cy="24" r="21" stroke="currentColor" strokeWidth="1.7" />
    <path d="M3 24h42M24 3v42M9.2 9.2c15 3.5 26.1 15.9 29.6 29.6M38.8 9.2C23.9 12.7 12.7 23.9 9.2 38.8" stroke="currentColor" strokeWidth="1.7" />
  </svg>;
}
export function CourtMotif() {
  return <svg className="court-motif" viewBox="0 0 420 210" fill="none" aria-hidden="true">
    <path d="M24 18h368v178H24zM208 18v178" />
    <circle cx="208" cy="107" r="35" /><circle cx="208" cy="107" r="5" />
    <path d="M24 68h62v78H24M392 68h-62v78h62M24 47h14a78 78 0 0 1 0 120H24M392 47h-14a78 78 0 0 0 0 120h14" />
    <circle cx="86" cy="107" r="24" /><circle cx="330" cy="107" r="24" />
    <path d="M40 95v24M376 95v24" /><circle cx="46" cy="107" r="5" /><circle cx="370" cy="107" r="5" />
    <path className="court-trajectory" d="M157 155q26-72 94-63t86-39" strokeDasharray="4 7" />
    <circle className="court-dot" cx="157" cy="155" r="5" /><circle className="court-dot" cx="251" cy="92" r="5" /><circle className="court-dot" cx="337" cy="53" r="5" />
  </svg>;
}
export function ErrorNotice({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return <div className="notice error-notice" role="alert">
    <AlertCircle size={18} aria-hidden="true" /><div><strong>We couldn’t complete that request.</strong><p>{message}</p>
      {onRetry && <button className="text-button" onClick={onRetry}>Try again <ArrowUpRight size={14} /></button>}
    </div>
  </div>;
}
export function Busy({ text = 'Crunching the numbers…' }: { text?: string }) {
  return <span className="busy" role="status"><LoaderCircle size={17} className="spin" aria-hidden="true" />{text}</span>;
}
export function EmptyState({ icon, title, children, action }: { icon?: ReactNode; title: string; children: ReactNode; action?: ReactNode }) {
  return <div className="empty-state"><div className="empty-icon">{icon ?? <Basketball />}</div><h3>{title}</h3><p>{children}</p>{action}</div>;
}
export function SectionHeading({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: ReactNode }) {
  return <div className="section-heading"><div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{title}</h2></div>{children}</div>;
}
export function Field({ label, children, hint, className = '' }: { label: string; children: ReactNode; hint?: string; className?: string }) {
  return <label className={`field ${className}`}><span className="field-label">{label}</span>{children}{hint && <span className="field-hint">{hint}</span>}</label>;
}
export function StatCard({ label, value, unit, detail, accent = false }: { label: string; value: ReactNode; unit?: string; detail: ReactNode; accent?: boolean }) {
  return <article className={`card stat-card${accent ? ' accent-card' : ''}`}>
    <span className="eyebrow">{label}</span><div className="stat-number">{value}{unit && <span>{unit}</span>}</div><div className="stat-detail">{detail}</div>
  </article>;
}
