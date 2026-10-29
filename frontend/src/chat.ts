import type { ChatAction, ChatActionType, ChatCard, ChatClarification, ChatResponse, ChatSlip } from './assistant-types';

export const CHAT_STORAGE_KEY = 'courtside.chat.v1';
export const MAX_STORED_MESSAGES = 60;
export const MAX_MESSAGE_LENGTH = 1000;

/** A coach reply as kept in the conversation. Its slip is applied to the app's slip rather than stored. */
export type CoachReply = Omit<ChatResponse, 'slip'>;

export interface UserMessage { id: string; role: 'user'; text: string; at: string }
export interface CoachMessage {
  id: string;
  role: 'coach';
  at: string;
  reply: CoachReply;
  /** 'applied': the reply's slip replaced the app's slip; 'skipped': the slip changed meanwhile, so it was not applied. */
  slip: 'none' | 'applied' | 'skipped';
  slipHash: string | null; // the slip right after it was applied, to tell whether "undo" still targets it
  notes: string[]; // client-side notes, e.g. legs that could not be added
  rating: 1 | -1 | null;
}
export type ChatMessage = UserMessage | CoachMessage;
export interface Conversation { id: string; messages: ChatMessage[] }

/** Action types that change the slip, so "undo" can take them back. */
export const slipActions: ReadonlySet<ChatActionType> = new Set(['add_leg', 'replace_leg', 'remove_leg', 'update_leg', 'set_book_odds', 'clear_slip']);
const actionTypes: ReadonlySet<string> = new Set([...slipActions, 'undo', 'track_bet']);
const cardTypes: ReadonlySet<string> = new Set(['forecast', 'parlay', 'player', 'track_record', 'learning', 'bet',
  'news', 'injuries', 'games', 'team', 'matchup', 'ranking', 'standings', 'coaches', 'moves', 'link']);
const idAlphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';

export function randomId(length = 20): string {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, byte => idAlphabet[byte & 63]).join('');
}

export function newConversation(): Conversation {
  return { id: randomId(20), messages: [] };
}

/** Short fingerprint of a slip (FNV-1a), stored with replies instead of a full copy. */
export function slipHash(slip: ChatSlip): string {
  const text = JSON.stringify(slip);
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(36);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function strings(value: unknown, limit: number, length = 400): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0).map(item => item.slice(0, length)).slice(0, limit) : [];
}

function isTeamCard(value: unknown): boolean {
  return isRecord(value) && typeof value.team === 'string' && typeof value.name === 'string' && Array.isArray(value.tags)
    && Array.isArray(value.metrics) && Array.isArray(value.play_types) && Array.isArray(value.injuries);
}

function validCard(card: unknown): card is ChatCard {
  if (!isRecord(card) || typeof card.type !== 'string' || !cardTypes.has(card.type)) return false;
  switch (card.type) {
    case 'forecast': return isRecord(card.player) && typeof card.over === 'number' && typeof card.under === 'number' && isRecord(card.interval) && isRecord(card.hit_rate) && isRecord(card.fair);
    case 'parlay': return isRecord(card.result) && Array.isArray(card.result.legs) && isRecord(card.house);
    case 'player': return isRecord(card.player) && isRecord(card.averages) && Array.isArray(card.last_games);
    case 'track_record': return isRecord(card.record);
    case 'learning': return isRecord(card.status) && isRecord(card.status.calibration) && isRecord(card.status.memory) && isRecord(card.status.record);
    case 'bet': return isRecord(card.bet) && Array.isArray(card.bet.legs);
    case 'news': return typeof card.title === 'string' && Array.isArray(card.articles);
    case 'injuries': return typeof card.title === 'string' && Array.isArray(card.entries);
    case 'games': return typeof card.title === 'string' && typeof card.date === 'string' && Array.isArray(card.games);
    case 'team': return isTeamCard(card);
    case 'matchup': return isTeamCard(card.a) && isTeamCard(card.b);
    case 'ranking': return typeof card.title === 'string' && Array.isArray(card.rows);
    case 'standings': return Array.isArray(card.teams);
    case 'coaches': return typeof card.name === 'string' && Array.isArray(card.coaches);
    case 'moves': return typeof card.title === 'string' && Array.isArray(card.transactions);
    case 'link': return typeof card.label === 'string' && typeof card.route === 'string';
    default: return false;
  }
}

function clarification(value: unknown): ChatClarification | null {
  if (!isRecord(value) || !Array.isArray(value.options)) return null;
  const options = value.options
    .filter((option): option is { label: string; message: string } => isRecord(option) && typeof option.label === 'string' && typeof option.message === 'string' && option.message.trim().length > 0)
    .map(option => ({ label: option.label.slice(0, 120), message: option.message.slice(0, MAX_MESSAGE_LENGTH) }))
    .slice(0, 8);
  return options.length ? { prompt: typeof value.prompt === 'string' ? value.prompt.slice(0, 300) : '', options } : null;
}

function actions(value: unknown): ChatAction[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((action): action is ChatAction => isRecord(action) && typeof action.type === 'string' && actionTypes.has(action.type) && typeof action.description === 'string')
    .map(action => ({ type: action.type, leg: typeof action.leg === 'number' ? action.leg : null, description: action.description.slice(0, 300) }));
}

/** Defensive copy of a coach reply: unknown card types and malformed fields are dropped rather than rendered. */
export function normalizeReply(raw: unknown): CoachReply {
  const value = isRecord(raw) ? raw : {};
  const engine = isRecord(value.engine) ? value.engine : {};
  return {
    turn_id: typeof value.turn_id === 'string' ? value.turn_id.slice(0, 120) : '',
    reply: typeof value.reply === 'string' ? value.reply.slice(0, 8000) : '',
    actions: actions(value.actions),
    cards: Array.isArray(value.cards) ? value.cards.filter(validCard).slice(0, 6) : [],
    suggestions: strings(value.suggestions, 4, 200),
    clarification: clarification(value.clarification),
    learned: strings(value.learned, 6),
    engine: { mode: engine.mode === 'llm' ? 'llm' : 'built_in', model: typeof engine.model === 'string' ? engine.model.slice(0, 120) : null },
  };
}

/** The slip carried by a response, if it is shaped like one. */
export function responseSlip(raw: unknown): { legs: unknown[]; book_odds: unknown } | null {
  if (!isRecord(raw) || !isRecord(raw.slip) || !Array.isArray(raw.slip.legs)) return null;
  return { legs: raw.slip.legs, book_odds: raw.slip.book_odds ?? null };
}

function message(value: unknown): ChatMessage | null {
  if (!isRecord(value) || typeof value.id !== 'string' || typeof value.at !== 'string') return null;
  if (value.role === 'user') {
    return typeof value.text === 'string' && value.text.trim() ? { id: value.id, role: 'user', text: value.text.slice(0, MAX_MESSAGE_LENGTH), at: value.at } : null;
  }
  if (value.role !== 'coach' || !isRecord(value.reply)) return null;
  return {
    id: value.id, role: 'coach', at: value.at, reply: normalizeReply(value.reply),
    slip: value.slip === 'applied' || value.slip === 'skipped' ? value.slip : 'none',
    slipHash: typeof value.slipHash === 'string' ? value.slipHash.slice(0, 16) : null,
    notes: strings(value.notes, 4),
    rating: value.rating === 1 || value.rating === -1 ? value.rating : null,
  };
}

export function parseConversation(raw: string | null): Conversation {
  if (!raw) return newConversation();
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed) || parsed.version !== 1 || typeof parsed.conversation_id !== 'string' || !/^[A-Za-z0-9_-]{8,64}$/.test(parsed.conversation_id) || !Array.isArray(parsed.messages)) return newConversation();
    const messages = parsed.messages.map(message).filter((item): item is ChatMessage => item !== null).slice(-MAX_STORED_MESSAGES);
    return { id: parsed.conversation_id, messages };
  } catch {
    return newConversation();
  }
}

export function loadConversation(): Conversation {
  try {
    return parseConversation(window.localStorage.getItem(CHAT_STORAGE_KEY));
  } catch {
    return newConversation();
  }
}

export function saveConversation(conversation: Conversation): void {
  if (!conversation.messages.length) {
    // Nothing to keep: a fresh or reset conversation leaves no stored copy behind.
    try { window.localStorage.removeItem(CHAT_STORAGE_KEY); } catch { /* storage unavailable */ }
    return;
  }
  const messages = conversation.messages.slice(-MAX_STORED_MESSAGES);
  const write = (items: ChatMessage[]) => window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify({ version: 1, conversation_id: conversation.id, messages: items }));
  try {
    write(messages);
  } catch {
    // Cards are the bulky part: over quota, keep them only on the most recent replies.
    try {
      write(messages.map((item, index) => item.role === 'coach' && index < messages.length - 10 ? { ...item, reply: { ...item.reply, cards: [] } } : item));
    } catch {
      // Storage unavailable: the conversation still works for this page.
    }
  }
}

// ---------- Reply text: **bold**, line breaks, "- " bullet lines and whole-line _asides_ only ----------

export interface InlineRun { text: string; bold: boolean; italic?: boolean }
export type RichBlock = { type: 'paragraph'; lines: InlineRun[][] } | { type: 'list'; items: InlineRun[][] };

function inline(line: string): InlineRun[] {
  const parts = line.split('**');
  if (parts.length % 2 === 0) {
    // An unmatched ** stays literal.
    const last = parts.pop() ?? '';
    parts[parts.length - 1] += `**${last}`;
  }
  return parts.map((text, index) => ({ text, bold: index % 2 === 1 })).filter(run => run.text.length > 0);
}

/** A whole line wrapped in single underscores is an aside, e.g. "_Report updated 2 hr ago._". Underscores inside a line stay literal. */
function lineRuns(line: string): InlineRun[] {
  const aside = /^_([^_\s](?:.*[^_\s])?)_$/.exec(line);
  return aside ? inline(aside[1]).map(run => ({ ...run, italic: true })) : inline(line);
}

export function parseRichText(text: string): RichBlock[] {
  const blocks: RichBlock[] = [];
  let paragraph: InlineRun[][] | null = null;
  let list: InlineRun[][] | null = null;
  for (const raw of text.replace(/\r\n?/g, '\n').split('\n')) {
    const line = raw.trimEnd();
    const bullet = /^\s*- (.*)$/.exec(line);
    if (bullet) {
      paragraph = null;
      if (!list) { list = []; blocks.push({ type: 'list', items: list }); }
      list.push(lineRuns(bullet[1].trim()));
      continue;
    }
    list = null;
    if (!line.trim()) { paragraph = null; continue; }
    if (!paragraph) { paragraph = []; blocks.push({ type: 'paragraph', lines: paragraph }); }
    paragraph.push(lineRuns(line.trim()));
  }
  return blocks;
}

/** The reply as one plain sentence stream, for screen-reader announcements. */
export function plainText(text: string): string {
  return parseRichText(text)
    .flatMap(block => block.type === 'list' ? block.items : block.lines)
    .map(runs => runs.map(run => run.text).join('').trim())
    .filter(Boolean)
    .map(line => /[.!?:]$/.test(line) ? line : `${line}.`)
    .join(' ');
}

export function timeLabel(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.valueOf()) ? '' : date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}
