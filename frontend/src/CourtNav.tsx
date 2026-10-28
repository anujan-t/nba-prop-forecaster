import { useEffect, useLayoutEffect, useRef } from 'react';
import type { LucideIcon } from 'lucide-react';
import { Ball } from './components';
import { usePrefersReducedMotion } from './motion';
import type { View } from './types';

export interface NavItem { id: View; label: string; icon: LucideIcon }

const travel = 420;

/** Primary navigation. A ball is passed to the active tab whenever the view changes. */
export default function CourtNav({ items, view, onNavigate, legCount }: { items: NavItem[]; view: View; onNavigate: (view: View) => void; legCount: number }) {
  const track = useRef<HTMLDivElement>(null);
  const marker = useRef<HTMLSpanElement>(null);
  const placed = useRef<number | null>(null);
  const current = useRef(view);
  const reduced = usePrefersReducedMotion();

  useLayoutEffect(() => {
    current.current = view;
    const container = track.current;
    const ball = marker.current;
    const arc = ball?.firstElementChild as HTMLElement | null;
    const active = container?.querySelector<HTMLElement>(`[data-view="${view}"]`);
    if (!container || !ball || !arc || !active) return;
    const x = active.offsetLeft + active.offsetWidth / 2 - ball.offsetWidth / 2;
    const from = placed.current;
    placed.current = x;
    ball.style.transform = `translateX(${x}px)`;
    ball.dataset.ready = 'true';
    if (from !== null && Math.abs(from - x) > 1 && !reduced && typeof ball.animate === 'function') {
      const height = Math.min(26, 10 + Math.abs(x - from) / 24);
      ball.animate([{ transform: `translateX(${from}px)` }, { transform: `translateX(${x}px)` }], { duration: travel, easing: 'cubic-bezier(.3, .1, .3, 1)' });
      arc.animate([
        { transform: 'translateY(0) rotate(0deg)', easing: 'cubic-bezier(.5, 1, .89, 1)' },
        { transform: `translateY(${-height}px) rotate(200deg)`, offset: 0.5, easing: 'cubic-bezier(.11, 0, .5, 0)' },
        { transform: 'translateY(0) rotate(400deg)' },
      ], { duration: travel });
    }
    if (container.scrollWidth > container.clientWidth) {
      const left = active.offsetLeft - (container.clientWidth - active.offsetWidth) / 2;
      container.scrollTo({ left: Math.max(0, left), behavior: reduced || from === null ? 'auto' : 'smooth' });
    }
  }, [view, reduced]);

  useEffect(() => {
    const container = track.current;
    if (!container) return;
    const reposition = () => {
      const ball = marker.current;
      const active = container.querySelector<HTMLElement>(`[data-view="${current.current}"]`);
      if (!ball || !active) return;
      const x = active.offsetLeft + active.offsetWidth / 2 - ball.offsetWidth / 2;
      if (placed.current !== null && Math.abs(placed.current - x) < 0.5) return;
      placed.current = x;
      ball.getAnimations().forEach(animation => animation.finish());
      ball.style.transform = `translateX(${x}px)`;
    };
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(reposition) : null;
    observer?.observe(container);
    container.querySelectorAll('[data-view]').forEach(element => observer?.observe(element));
    void document.fonts?.ready.then(reposition);
    return () => observer?.disconnect();
  }, []);

  return <nav className="court-nav" aria-label="Research views">
    <div className="court-nav-track" ref={track}>
      {items.map(item => {
        const Icon = item.icon;
        const active = item.id === view;
        return <button key={item.id} type="button" data-view={item.id} className={`nav-tab${active ? ' active' : ''}`} aria-current={active ? 'page' : undefined} onClick={() => onNavigate(item.id)}>
          <Icon size={17} strokeWidth={1.9} aria-hidden="true" /><span>{item.label}</span>
          {item.id === 'parlay' && legCount > 0 && <><span className="nav-count" aria-hidden="true">{legCount}</span><span className="sr-only">, {legCount} {legCount === 1 ? 'leg' : 'legs'} in the slip</span></>}
        </button>;
      })}
      <span className="nav-ball" ref={marker} aria-hidden="true"><span className="nav-ball-arc"><Ball /></span></span>
    </div>
  </nav>;
}
