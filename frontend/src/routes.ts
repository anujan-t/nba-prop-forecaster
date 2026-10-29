import { useCallback, useEffect, useState } from 'react';
import type { MouseEvent } from 'react';

/** Hash routes: #/players[/{playerId}][/{section}], #/parlay, #/bets, #/news[/{section}[/{team}]], #/coach,
 * #/how-it-works/{section}, #/settings/{section}. */
export type Tab = 'players' | 'parlay' | 'bets' | 'news' | 'coach';
export type PlayerSection = 'props' | 'log' | 'career' | 'lines';
export type NewsSection = 'headlines' | 'injuries' | 'teams' | 'moves';
export type HowSection = 'method' | 'accuracy';
export type SettingsSection = 'appearance' | 'model' | 'data' | 'coach';

export type Route =
  | { page: 'players'; playerId: string | null; section: PlayerSection }
  | { page: 'parlay' }
  | { page: 'bets' }
  | { page: 'news'; section: NewsSection; team: string | null }
  | { page: 'coach' }
  | { page: 'how'; section: HowSection }
  | { page: 'settings'; section: SettingsSection };

export type Page = Route['page'];

export const playerSections: readonly PlayerSection[] = ['props', 'log', 'career', 'lines'];
export const newsSections: readonly NewsSection[] = ['headlines', 'injuries', 'teams', 'moves'];
export const howSections: readonly HowSection[] = ['method', 'accuracy'];
export const settingsSections: readonly SettingsSection[] = ['appearance', 'model', 'data', 'coach'];

export const DEFAULT_ROUTE: Route = { page: 'players', playerId: null, section: 'props' };

const playerIdPattern = /^\d{1,12}$/;
const teamPattern = /^[a-z]{2,4}$/;

const isOneOf = <T extends string>(options: readonly T[], value: string | undefined): value is T => value !== undefined && (options as readonly string[]).includes(value);

/** The route a hash names, or null when it names nothing we know. */
export function parseHash(hash: string): Route | null {
  const parts = hash.replace(/^#\/?/, '').split('?')[0].split('/').filter(Boolean).map(part => part.toLowerCase());
  if (!parts.length) return null;
  const [head, first, second, ...extra] = parts;
  if (extra.length) return null;
  switch (head) {
    case 'players': {
      if (first === undefined) return { page: 'players', playerId: null, section: 'props' };
      if (second !== undefined) return playerIdPattern.test(first) && isOneOf(playerSections, second) ? { page: 'players', playerId: first, section: second } : null;
      if (playerIdPattern.test(first)) return { page: 'players', playerId: first, section: 'props' };
      return isOneOf(playerSections, first) ? { page: 'players', playerId: null, section: first } : null;
    }
    case 'parlay': return first === undefined ? { page: 'parlay' } : null;
    case 'bets': return first === undefined ? { page: 'bets' } : null;
    case 'news':
      if (first === undefined) return { page: 'news', section: 'headlines', team: null };
      if (first === 'teams' && second !== undefined) return teamPattern.test(second) ? { page: 'news', section: 'teams', team: second.toUpperCase() } : null;
      return second === undefined && isOneOf(newsSections, first) ? { page: 'news', section: first, team: null } : null;
    case 'coach': return first === undefined ? { page: 'coach' } : null;
    case 'how-it-works':
      if (first === undefined) return { page: 'how', section: 'method' };
      return second === undefined && isOneOf(howSections, first) ? { page: 'how', section: first } : null;
    case 'settings':
      if (first === undefined) return { page: 'settings', section: 'appearance' };
      return second === undefined && isOneOf(settingsSections, first) ? { page: 'settings', section: first } : null;
    default:
      return null;
  }
}

export function routeHash(route: Route): string {
  switch (route.page) {
    case 'players':
      if (route.playerId) return `#/players/${route.playerId}/${route.section}`;
      return route.section === 'props' ? '#/players' : `#/players/${route.section}`;
    case 'how': return `#/how-it-works/${route.section}`;
    case 'settings': return `#/settings/${route.section}`;
    case 'news':
      if (route.section === 'teams' && route.team) return `#/news/teams/${route.team.toLowerCase()}`;
      return route.section === 'headlines' ? '#/news' : `#/news/${route.section}`;
    default: return `#/${route.page}`;
  }
}

export const sameRoute = (a: Route, b: Route) => routeHash(a) === routeHash(b);

/** Let modified clicks (new tab, new window) through to the browser; everything else navigates in place. */
export function isPlainClick(event: MouseEvent<HTMLElement>): boolean {
  return event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey && !event.defaultPrevented;
}

function currentRoute(): Route {
  return parseHash(window.location.hash) ?? DEFAULT_ROUTE;
}

export type Navigate = (route: Route, options?: { replace?: boolean }) => void;

/**
 * The route lives in location.hash, so Back, Forward, reloads and deep links all work.
 * Unknown or empty hashes are replaced with #/players.
 */
export function useHashRoute(): [Route, Navigate] {
  const [route, setRoute] = useState<Route>(currentRoute);

  useEffect(() => {
    const sync = () => {
      const parsed = parseHash(window.location.hash);
      const next = parsed ?? DEFAULT_ROUTE;
      if (!parsed) window.history.replaceState(window.history.state, '', routeHash(next));
      setRoute(current => sameRoute(current, next) ? current : next);
    };
    sync();
    window.addEventListener('popstate', sync);
    window.addEventListener('hashchange', sync);
    return () => {
      window.removeEventListener('popstate', sync);
      window.removeEventListener('hashchange', sync);
    };
  }, []);

  const navigate = useCallback<Navigate>((next, options) => {
    const hash = routeHash(next);
    if (hash !== window.location.hash) {
      if (options?.replace) window.history.replaceState(window.history.state, '', hash);
      else window.history.pushState(null, '', hash);
    }
    setRoute(current => sameRoute(current, next) ? current : next);
  }, []);

  return [route, navigate];
}
