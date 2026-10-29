import { useEffect, useLayoutEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { Ball } from './components';
import { usePrefersReducedMotion } from './motion';
import { isPlainClick } from './routes';

export interface NavItem {
  id: string;
  label: string;
  icon: ReactNode;
  href: string;
  /** A count shown on the tab, with its screen-reader wording. */
  badge?: { value: number; label: string } | null;
  /** An attention dot, with its screen-reader wording. */
  dot?: string | null;
}

const travel = 420;

/**
 * The primary tabs. A ball is passed to the active tab whenever it changes: along the baseline under the
 * header tabs on wide screens, along the top edge of the bottom tab bar on phones.
 */
export default function CourtNav({ items, active, onNavigate, variant, label }: { items: NavItem[]; active: string | null; onNavigate: (id: string) => void; variant: 'top' | 'bottom'; label: string }) {
  const track = useRef<HTMLDivElement>(null);
  const marker = useRef<HTMLSpanElement>(null);
  const placed = useRef<number | null>(null);
  const idle = useRef(true);
  const current = useRef(active);
  const reduced = usePrefersReducedMotion();

  useLayoutEffect(() => {
    current.current = active;
    const container = track.current;
    const ball = marker.current;
    const arc = ball?.firstElementChild as HTMLElement | null;
    const target = active ? container?.querySelector<HTMLElement>(`[data-nav="${active}"]`) : null;
    if (!container || !ball || !arc || !target || !target.offsetWidth) { idle.current = true; return; }
    const x = target.offsetLeft + target.offsetWidth / 2 - ball.offsetWidth / 2;
    const from = idle.current ? null : placed.current;
    idle.current = false;
    placed.current = x;
    ball.style.transform = `translateX(${x}px)`;
    ball.dataset.ready = 'true';
    if (from !== null && Math.abs(from - x) > 1 && !reduced && typeof ball.animate === 'function') {
      const height = Math.min(variant === 'bottom' ? 18 : 26, 10 + Math.abs(x - from) / 24);
      ball.animate([{ transform: `translateX(${from}px)` }, { transform: `translateX(${x}px)` }], { duration: travel, easing: 'cubic-bezier(.3, .1, .3, 1)' });
      arc.animate([
        { transform: 'translateY(0) rotate(0deg)', easing: 'cubic-bezier(.5, 1, .89, 1)' },
        { transform: `translateY(${-height}px) rotate(200deg)`, offset: 0.5, easing: 'cubic-bezier(.11, 0, .5, 0)' },
        { transform: 'translateY(0) rotate(400deg)' },
      ], { duration: travel });
    }
    if (container.scrollWidth > container.clientWidth + 1) {
      const left = target.offsetLeft - (container.clientWidth - target.offsetWidth) / 2;
      container.scrollTo({ left: Math.max(0, left), behavior: reduced || from === null ? 'auto' : 'smooth' });
    }
  }, [active, reduced, variant]);

  useEffect(() => {
    const container = track.current;
    if (!container) return;
    const reposition = () => {
      const ball = marker.current;
      const target = current.current ? container.querySelector<HTMLElement>(`[data-nav="${current.current}"]`) : null;
      if (!ball || !target || !target.offsetWidth) return;
      const x = target.offsetLeft + target.offsetWidth / 2 - ball.offsetWidth / 2;
      if (!idle.current && placed.current !== null && Math.abs(placed.current - x) < 0.5) return;
      idle.current = false;
      placed.current = x;
      ball.getAnimations().forEach(animation => animation.finish());
      ball.style.transform = `translateX(${x}px)`;
      ball.dataset.ready = 'true';
    };
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(reposition) : null;
    observer?.observe(container);
    container.querySelectorAll('[data-nav]').forEach(element => observer?.observe(element));
    void document.fonts?.ready.then(reposition);
    return () => observer?.disconnect();
  }, []);

  return <nav className={`court-nav ${variant}`} aria-label={label}>
    <div className="court-nav-track" ref={track}>
      {items.map(item => {
        const selected = item.id === active;
        return <a key={item.id} href={item.href} data-nav={item.id} className={`nav-tab${selected ? ' active' : ''}`} aria-current={selected ? 'page' : undefined}
          onClick={event => { if (!isPlainClick(event)) return; event.preventDefault(); onNavigate(item.id); }}>
          <span className="nav-icon" aria-hidden="true">{item.icon}{item.dot && <span className="nav-dot" />}</span>
          <span className="nav-label">{item.label}</span>
          {item.badge && item.badge.value > 0 && <><span className="nav-count" aria-hidden="true">{item.badge.value}</span><span className="sr-only">, {item.badge.label}</span></>}
          {item.dot && <span className="sr-only">, {item.dot}</span>}
        </a>;
      })}
      <span className={`nav-ball${active ? '' : ' is-idle'}`} ref={marker} aria-hidden="true"><span className="nav-ball-arc"><Ball /></span></span>
    </div>
  </nav>;
}
