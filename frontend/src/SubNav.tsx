import { useLayoutEffect, useRef } from 'react';
import type { ReactNode } from 'react';
import { isPlainClick } from './routes';

export interface SubNavItem {
  id: string;
  label: string;
  href: string;
  icon?: ReactNode;
}

/** Sections inside a page (Players › Props, How it works › Accuracy…). Each is a link, so it has its own URL. */
export default function SubNav({ items, active, onNavigate, label }: { items: SubNavItem[]; active: string; onNavigate: (id: string) => void; label: string }) {
  const list = useRef<HTMLUListElement>(null);
  useLayoutEffect(() => {
    const container = list.current;
    const link = container?.querySelector<HTMLElement>(`[data-section="${active}"]`);
    if (!container || !link || container.scrollWidth <= container.clientWidth + 1) return;
    container.scrollLeft = Math.max(0, link.offsetLeft - (container.clientWidth - link.offsetWidth) / 2);
  }, [active]);
  return <nav className="sub-nav" aria-label={label}>
    <ul className="sub-nav-list" ref={list}>
      {items.map(item => {
        const selected = item.id === active;
        return <li key={item.id}>
          <a href={item.href} data-section={item.id} className={`sub-nav-link${selected ? ' active' : ''}`} aria-current={selected ? 'page' : undefined}
            onClick={event => { if (!isPlainClick(event)) return; event.preventDefault(); onNavigate(item.id); }}>
            {item.icon}<span>{item.label}</span>
          </a>
        </li>;
      })}
    </ul>
  </nav>;
}
