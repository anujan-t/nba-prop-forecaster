import { useEffect, useRef, useState, useSyncExternalStore } from 'react';

const reducedQuery = '(prefers-reduced-motion: reduce)';

export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function' && window.matchMedia(reducedQuery).matches;
}

function subscribeToMotion(callback: () => void): () => void {
  if (typeof window.matchMedia !== 'function') return () => undefined;
  const media = window.matchMedia(reducedQuery);
  media.addEventListener('change', callback);
  return () => media.removeEventListener('change', callback);
}

export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(subscribeToMotion, prefersReducedMotion, () => false);
}

/** Animates a number toward its target on the scoreboard; jumps straight there when motion is reduced. */
export function useCountUp(target: number | null, duration = 750): number | null {
  const reduced = usePrefersReducedMotion();
  const [value, setValue] = useState<number | null>(null);
  const shown = useRef(0);
  useEffect(() => {
    if (target === null) { setValue(null); return; }
    if (reduced) { shown.current = target; setValue(target); return; }
    const from = shown.current;
    const start = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / duration);
      const next = from + (target - from) * (1 - Math.pow(1 - progress, 3));
      shown.current = next;
      setValue(next);
      if (progress < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, reduced, duration]);
  return value;
}

export const ballSeams = 'M16 1v30M1 16h30M5.6 5.3c3.5 3 5.5 6.6 5.5 10.7s-2 7.7-5.5 10.7M26.4 5.3c-3.5 3-5.5 6.6-5.5 10.7s2 7.7 5.5 10.7';

/** Throws a small ball from one element into another, then calls onLand. Purely decorative and never blocks input. */
export function launchBall(from: Element, to: Element | null, onLand: () => void): void {
  if (!to || prefersReducedMotion() || typeof Element.prototype.animate !== 'function') { onLand(); return; }
  const start = from.getBoundingClientRect();
  const end = to.getBoundingClientRect();
  if (!end.width || !end.height) { onLand(); return; }
  const size = 24;
  const x0 = start.left + start.width / 2 - size / 2;
  const y0 = start.top + start.height / 2 - size / 2;
  const dx = end.left + end.width / 2 - size / 2 - x0;
  const dy = end.top + end.height / 2 - size / 2 - y0;
  const distance = Math.hypot(dx, dy);
  const duration = Math.round(Math.min(700, Math.max(420, distance * 0.55)));
  const lift = Math.min(200, Math.max(70, distance * 0.3));
  const flyer = document.createElement('div');
  flyer.className = 'ball-flight';
  flyer.setAttribute('aria-hidden', 'true');
  flyer.style.left = `${x0}px`;
  flyer.style.top = `${y0}px`;
  flyer.innerHTML = `<div class="ball-flight-arc"><span class="ball"><svg viewBox="0 0 32 32"><path d="${ballSeams}"/></svg></span></div>`;
  document.body.appendChild(flyer);
  const arc = flyer.firstElementChild as HTMLElement;
  const ball = arc.firstElementChild as HTMLElement;
  const across = flyer.animate([{ transform: 'translate(0, 0)' }, { transform: `translate(${dx}px, ${dy}px)` }], { duration, easing: 'linear', fill: 'forwards' });
  arc.animate([
    { transform: 'translateY(0)', easing: 'cubic-bezier(.5, 1, .89, 1)' },
    { transform: `translateY(${-lift}px)`, offset: 0.5, easing: 'cubic-bezier(.11, 0, .5, 0)' },
    { transform: 'translateY(0)' },
  ], { duration, fill: 'forwards' });
  ball.animate([{ transform: 'rotate(0deg) scale(1)' }, { transform: 'rotate(600deg) scale(.82)' }], { duration, easing: 'linear', fill: 'forwards' });
  let landed = false;
  const finish = () => {
    if (landed) return;
    landed = true;
    flyer.remove();
    onLand();
  };
  across.onfinish = finish;
  across.oncancel = finish;
  window.setTimeout(finish, duration + 400);
}
