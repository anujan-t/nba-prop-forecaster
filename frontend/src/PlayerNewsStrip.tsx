import { ArrowUpRight, HeartPulse, ShieldCheck } from 'lucide-react';
import { timeAgo } from './api';
import { InjuryBadge } from './NewsPage';
import type { PlayerNews } from './news-types';
import { RouteLink } from './RouteLink';
import { useFeed } from './useFeed';

/** Injury status for the selected player, and teammates who are out (their minutes and shots go somewhere). */
export default function PlayerNewsStrip({ playerId, playerName, team }: { playerId: string | null; playerName: string | undefined; team: string | undefined }) {
  const feed = useFeed<PlayerNews>(playerId ? `/api/news/players/${playerId}` : null, !!playerId);
  const data = feed.data;
  if (!data || data.player_id !== playerId || !playerName) return null;
  const out = data.teammates.filter(entry => entry.status_key === 'out' || entry.status_key === 'suspended');
  const doubtful = data.teammates.filter(entry => entry.status_key === 'doubtful' || entry.status_key === 'questionable');
  const updated = data.report_fetched_at ? ` Report updated ${timeAgo(data.report_fetched_at)}.` : '';
  const injury = data.injury;
  const what = injury ? [injury.side, injury.injury, injury.detail?.toLowerCase()].filter(Boolean).join(' ') : '';
  return <div className={`player-news ${injury ? `status-${injury.status_key}` : 'clear'}`} role="status">
    {injury ? <InjuryBadge entry={injury} /> : <ShieldCheck size={18} aria-hidden="true" className="positive" />}
    <p>
      {injury
        ? <><strong>{playerName} is listed {injury.status.toLowerCase()}{what ? ` (${what})` : ''}.</strong> {injury.short_comment}{injury.return_date ? ` Estimated return ${injury.return_date}.` : ''}</>
        : <><strong>{playerName} is not on the injury report.</strong></>}
      {out.length > 0 && <> {team ?? 'Teammates'} without {out.map(entry => entry.player).join(', ')}: more minutes and shots to share.</>}
      {doubtful.length > 0 && <> Uncertain: {doubtful.map(entry => entry.player).join(', ')}.</>}
      {data.stale && <> <HeartPulse size={13} aria-hidden="true" /> Saved report; the live feed is unreachable.</>}
      {updated}
    </p>
    {team && <RouteLink className="text-button" to={{ page: 'news', section: 'teams', team }}>{team} report <ArrowUpRight size={13} aria-hidden="true" /></RouteLink>}
  </div>;
}
