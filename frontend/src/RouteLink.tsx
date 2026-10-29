import { createContext, useContext } from 'react';
import type { AnchorHTMLAttributes } from 'react';
import { isPlainClick, routeHash } from './routes';
import type { Route } from './routes';

export interface GoOptions {
  replace?: boolean;
  /** Id of an element to focus once the page is shown, instead of the page heading. */
  focus?: string;
}
export type Go = (route: Route, options?: GoOptions) => void;

export const GoContext = createContext<Go>(() => undefined);
export const useGo = () => useContext(GoContext);

/** An in-app link: a real href (open in a new tab, copy link) that navigates in place on a plain click. */
export function RouteLink({ to, focus, onClick, ...rest }: Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'href'> & { to: Route; focus?: string }) {
  const go = useGo();
  return <a {...rest} href={routeHash(to)} onClick={event => {
    onClick?.(event);
    if (!isPlainClick(event)) return;
    event.preventDefault();
    go(to, { focus });
  }} />;
}
