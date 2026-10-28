import { ArrowUpRight, Check, CircleAlert, Info, RotateCcw, SendHorizontal, Sparkles, Square, SquarePen, Target, ThumbsDown, ThumbsUp, TriangleAlert, Undo2, X } from 'lucide-react';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import { statNames } from './api';
import type { AssistantStatus, ChatFocus } from './assistant-types';
import { MAX_MESSAGE_LENGTH, plainText, slipActions, timeLabel } from './chat';
import type { ChatMessage, CoachMessage, UserMessage } from './chat';
import { CardBoundary, ChatCardView, RichText } from './ChatCards';
import { Ball, DribbleBall } from './components';
import { chance } from './parlay';
import type { ChatState } from './useChat';

const focusableSelector = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export const starterPrompts = [
  'Add Nikola Jokić over 26.5 points',
  'What are my parlay\'s odds?',
  'Book has my parlay at +450 — is that good?',
  'Grade my tracked bets',
  'Are you getting smarter?',
];

/** Speech bubble with an X-and-O play drawn inside: the coach's whiteboard. */
export function CoachMark({ className = '' }: { className?: string }) {
  return <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M5 3.5h14A2.5 2.5 0 0 1 21.5 6v9a2.5 2.5 0 0 1-2.5 2.5h-8.2L6.5 21v-3.5H5A2.5 2.5 0 0 1 2.5 15V6A2.5 2.5 0 0 1 5 3.5z" />
    <path d="M6.4 7.2l2.6 2.6M9 7.2l-2.6 2.6" />
    <circle cx="16.3" cy="12.2" r="1.7" />
    <path d="M10.2 10.6q1.9 3.1 4.2 1.9" strokeDasharray="1.3 1.5" />
  </svg>;
}

/** Half court with a drawn-up play, for the empty conversation. */
function HuddleArt() {
  return <svg className="huddle-art" viewBox="0 0 200 104" fill="none" aria-hidden="true">
    <path className="huddle-paint" d="M74 104V60h52v44z" />
    <g className="huddle-lines">
      <path d="M4 103.5h192" />
      <path d="M22 103.5V58a78 78 0 0 1 156 0v45.5" />
      <path d="M74 103.5V60h52v43.5" />
      <path d="M86 60a14 14 0 0 1 28 0" />
      <path d="M86 60a14 14 0 0 0 28 0" strokeDasharray="3 4" />
    </g>
    <g className="huddle-play">
      <path d="M38 22l9 9M47 22l-9 9" />
      <path d="M150 30l9 9M159 30l-9 9" />
      <circle cx="100" cy="24" r="5.5" />
      <path className="huddle-route" d="M50 36q30 26 44-6" />
      <path className="huddle-route" d="M146 44q-14 20-34 6" />
      <path d="M90 34.5l4.6-4.8 1.2 6.4M115.5 46.5l-4.2 3.1 5.2 2.4" />
    </g>
  </svg>;
}

export function ChatLauncher({ open, busy, unread, onOpen }: { open: boolean; busy: boolean; unread: boolean; onOpen: () => void }) {
  const state = busy ? ', Coach is answering' : unread ? ', new reply' : '';
  return <button id="chat-launcher" type="button" className={`chat-fab${busy ? ' is-busy' : ''}${unread ? ' has-unread' : ''}`} aria-haspopup="dialog" aria-expanded={open} aria-controls="chat-panel"
    aria-label={`Ask Coach${state}`} title="Ask Coach" onClick={onOpen}>
    <CoachMark className="chat-fab-icon" />
    <span className="chat-fab-label" aria-hidden="true">Ask Coach</span>
    {(busy || unread) && <span className="chat-fab-dot" aria-hidden="true" />}
  </button>;
}

function engineLabel(status: AssistantStatus | null, error: string): { text: string; tone: 'checking' | 'built-in' | 'llm' | 'warning' } {
  if (status) {
    if (status.mode === 'llm') return { text: `LLM · ${status.model ?? 'model'}${status.ready ? '' : ' (offline)'}`, tone: status.ready ? 'llm' : 'warning' };
    return { text: 'Built-in', tone: 'built-in' };
  }
  return error ? { text: 'Status unknown', tone: 'warning' } : { text: 'Checking…', tone: 'checking' };
}

interface TurnProps {
  message: CoachMessage;
  latest: boolean;
  undoable: boolean;
  busy: boolean;
  feedbackError: string | null;
  onSend: (text: string) => void;
  onRate: (messageId: string, rating: 1 | -1) => void;
  onOpenParlay: () => void;
  onOpenRecord: () => void;
}

function CoachTurn({ message, latest, undoable, busy, feedbackError, onSend, onRate, onOpenParlay, onOpenRecord }: TurnProps) {
  const { reply } = message;
  const skipped = message.slip === 'skipped';
  const clarification = reply.clarification;
  const engine = reply.engine.mode === 'llm' ? `LLM${reply.engine.model ? ` · ${reply.engine.model}` : ''}` : 'Built-in';
  return <li className="chat-turn coach" data-message={message.id}>
    <span className="chat-avatar" aria-hidden="true"><CoachMark /></span>
    <div className="chat-coach-body">
      {reply.reply.trim() && <div className="chat-bubble coach-bubble"><span className="sr-only">Coach said: </span><RichText text={reply.reply} /></div>}
      {reply.actions.length > 0 && <div className={`chat-receipts${skipped ? ' skipped' : ''}`}>
        <ul aria-label={skipped ? 'Slip changes, not applied' : 'Slip changes'}>
          {reply.actions.map((action, index) => <li key={index}>{skipped ? <X size={14} aria-hidden="true" /> : <Check size={14} aria-hidden="true" />}<span>{action.description}</span></li>)}
        </ul>
        {undoable && <button type="button" className="chat-chip undo-chip" disabled={busy} onClick={() => onSend('undo')}><Undo2 size={14} aria-hidden="true" />Undo</button>}
      </div>}
      {message.notes.map((note, index) => <p key={index} className="chat-note"><TriangleAlert size={14} aria-hidden="true" /><span>{note}</span></p>)}
      {reply.cards.map((card, index) => <CardBoundary key={index}><ChatCardView card={card} onOpenParlay={onOpenParlay} onOpenRecord={onOpenRecord} /></CardBoundary>)}
      {reply.learned.map((text, index) => <p key={index} className="chat-learned"><Sparkles size={13} aria-hidden="true" /><span>{/^learned\b/i.test(text) ? text : `Learned: ${text}`}</span></p>)}
      {clarification && <div className="chat-clarify" role="group" aria-labelledby={`clarify-${message.id}`}>
        <p id={`clarify-${message.id}`}>{clarification.prompt || 'Which one did you mean?'}</p>
        {latest && <div className="chat-chips">{clarification.options.map((option, index) => <button key={index} type="button" className="chat-chip option" disabled={busy} onClick={() => onSend(option.message)}>{option.label}</button>)}</div>}
      </div>}
      {latest && reply.suggestions.length > 0 && <div className="chat-chips chat-suggestions" role="group" aria-label="Suggested follow-ups">
        {reply.suggestions.map((suggestion, index) => <button key={index} type="button" className="chat-chip" disabled={busy} onClick={() => onSend(suggestion)}>{suggestion}</button>)}
      </div>}
      <div className="chat-meta">
        <span>{engine} · <time dateTime={message.at}>{timeLabel(message.at)}</time></span>
        {reply.turn_id && <div className="chat-rating" role="group" aria-label="Rate this reply">
          <button type="button" aria-pressed={message.rating === 1} aria-label="Helpful" title="Helpful" onClick={() => onRate(message.id, 1)}><ThumbsUp size={14} aria-hidden="true" /></button>
          <button type="button" aria-pressed={message.rating === -1} aria-label="Not helpful" title="Not helpful" onClick={() => onRate(message.id, -1)}><ThumbsDown size={14} aria-hidden="true" /></button>
        </div>}
      </div>
      {feedbackError && <p className="chat-feedback-error" role="alert">{feedbackError}</p>}
    </div>
  </li>;
}

function UserTurn({ message, answered }: { message: UserMessage; answered: boolean }) {
  return <li className="chat-turn user" data-message={message.id}>
    <div className="chat-bubble user-bubble"><span className="sr-only">You said: </span>{message.text}</div>
    {!answered && <span className="chat-unanswered">Not answered</span>}
  </li>;
}

interface PanelProps {
  chat: ChatState;
  open: boolean;
  onClose: () => void;
  focus: ChatFocus | null;
  slipCount: number;
  slipProbability: number | null;
  slipBusy: boolean;
  slipHash: string;
  bounce: number;
  onOpenSlip: () => void;
  onOpenParlay: () => void;
  onOpenRecord: () => void;
}

export function ChatPanel({ chat, open, onClose, focus, slipCount, slipProbability, slipBusy, slipHash, bounce, onOpenSlip, onOpenParlay, onOpenRecord }: PanelProps) {
  const { conversation, pending, failure, status, statusError } = chat;
  const messages = conversation.messages;
  const busy = pending !== null;
  const panel = useRef<HTMLDivElement>(null);
  const log = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const title = useRef<HTMLHeadingElement>(null);
  const engineButton = useRef<HTMLButtonElement>(null);
  const newChatButton = useRef<HTMLButtonElement>(null);
  const keepButton = useRef<HTMLButtonElement>(null);
  const confirmFocus = useRef<'keep' | 'new' | null>(null);
  const [draft, setDraft] = useState('');
  const [engineOpen, setEngineOpen] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [announcement, setAnnouncement] = useState('');
  const engine = engineLabel(status, statusError);
  const last = messages[messages.length - 1];
  const remaining = MAX_MESSAGE_LENGTH - draft.length;

  /** The newest reply that changed the slip can be undone while the slip still matches what it left. */
  const undoId = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message.role !== 'coach' || message.slip !== 'applied') continue;
      return message.reply.actions.some(action => slipActions.has(action.type)) && message.slipHash === slipHash ? message.id : null;
    }
    return null;
  }, [messages, slipHash]);

  useEffect(() => {
    if (!open) { setEngineOpen(false); setConfirmReset(false); return; }
    chat.refreshStatus();
    const fine = window.matchMedia('(pointer: fine)').matches;
    (fine ? input.current : title.current)?.focus({ preventScroll: true });
    const element = log.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [open, chat.refreshStatus]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Escape' || event.isComposing) return;
      if (engineOpen) { setEngineOpen(false); engineButton.current?.focus(); return; }
      if (confirmReset) { askReset(false); return; }
      onClose();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose, engineOpen, confirmReset]);

  useEffect(() => {
    const target = confirmFocus.current === 'keep' ? keepButton.current : confirmFocus.current === 'new' ? newChatButton.current : null;
    confirmFocus.current = null;
    target?.focus();
  }, [confirmReset]);

  useEffect(() => { if (!messages.length) setConfirmReset(false); }, [messages.length]);

  // Keep the newest turn in view: the bottom after a send, the top of a reply that is taller than the log.
  useLayoutEffect(() => {
    const element = log.current;
    if (!element || !open) return;
    const behavior: ScrollBehavior = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
    if (!busy && last?.role === 'coach') {
      const node = element.querySelector<HTMLElement>(`[data-message="${last.id}"]`);
      if (node && node.offsetHeight > element.clientHeight - 32) { element.scrollTo({ top: node.offsetTop - 12, behavior }); return; }
    }
    element.scrollTo({ top: element.scrollHeight, behavior });
  }, [last?.id, busy]);

  useEffect(() => {
    if (!chat.reply) return;
    const message = chat.conversation.messages.find(item => item.id === chat.reply?.messageId);
    if (!message || message.role !== 'coach') return;
    const text = [plainText(message.reply.reply), ...message.reply.actions.map(action => action.description), ...message.notes].filter(Boolean).join(' ');
    setAnnouncement(`Coach: ${text.length > 600 ? `${text.slice(0, 600)}…` : text}`);
  }, [chat.reply]);

  useLayoutEffect(() => {
    const element = input.current;
    if (!element) return;
    element.style.height = 'auto';
    if (element.scrollHeight) element.style.height = `${Math.min(element.scrollHeight, 168)}px`;
  }, [draft]);

  const askReset = (confirm: boolean) => { confirmFocus.current = confirm ? 'keep' : 'new'; setConfirmReset(confirm); };
  const afterQuickSend = () => {
    const fine = window.matchMedia('(pointer: fine)').matches;
    requestAnimationFrame(() => (fine ? input.current : log.current)?.focus({ preventScroll: true }));
  };
  const sendQuick = (text: string) => { if (!busy && chat.send(text)) afterQuickSend(); };
  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    if (busy) return;
    if (chat.send(draft)) setDraft('');
  };
  const onComposerKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); submit(); }
  };
  const trapFocus = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Tab' || !panel.current) return;
    const focusable = Array.from(panel.current.querySelectorAll<HTMLElement>(focusableSelector)).filter(element => element.getClientRects().length > 0);
    if (!focusable.length) return;
    const first = focusable[0];
    const lastFocusable = focusable[focusable.length - 1];
    if (event.shiftKey && (document.activeElement === first || !panel.current.contains(document.activeElement))) { event.preventDefault(); lastFocusable.focus(); }
    else if (!event.shiftKey && document.activeElement === lastFocusable) { event.preventDefault(); first.focus(); }
  };
  const resetChat = () => {
    chat.reset();
    setDraft('');
    setConfirmReset(false);
    requestAnimationFrame(() => input.current?.focus({ preventScroll: true }));
  };

  const answered = (index: number) => messages[index + 1]?.role === 'coach';
  const showFailure = !busy && last?.role === 'user';
  const legsText = `${slipCount} ${slipCount === 1 ? 'leg' : 'legs'}`;

  return <div className={`chat-layer${open ? ' open' : ''}`} inert={!open}>
    <div className="chat-backdrop" onClick={onClose} aria-hidden="true" />
    <div className="chat-panel" id="chat-panel" role="dialog" aria-modal="true" aria-labelledby="chat-title" aria-describedby="chat-subtitle" ref={panel} onKeyDown={trapFocus}>
      <header className="chat-header">
        <div className="chat-header-main">
          <span className="chat-header-mark" aria-hidden="true"><CoachMark /></span>
          <div className="chat-heading">
            <span className="eyebrow">THE HUDDLE</span>
            <h2 id="chat-title" ref={title} tabIndex={-1}>Coach</h2>
            <p id="chat-subtitle" className="sr-only">Ask about your bets and parlay slip, compare a price with the house, or build the slip by command.</p>
          </div>
          <button ref={engineButton} type="button" className={`engine-badge ${engine.tone}`} aria-expanded={engineOpen} aria-controls="chat-engine-detail" onClick={() => setEngineOpen(value => !value)}>
            <span className="engine-badge-dot" aria-hidden="true" /><span className="engine-badge-text"><span className="sr-only">Engine: </span>{engine.text}</span><Info size={13} aria-hidden="true" />
          </button>
          <button type="button" className="icon-button chat-close" aria-label="Close Coach" onClick={onClose}><X size={18} aria-hidden="true" /></button>
        </div>
        <div id="chat-engine-detail" className="engine-detail" hidden={!engineOpen}>
          {status ? <>
            <RichText text={status.detail} />
            {status.endpoint && <p className="engine-endpoint">Endpoint: <code>{status.endpoint}</code></p>}
          </> : <p>{statusError ? `The assistant status couldn’t be loaded. ${statusError}` : 'Checking which engine is answering…'}</p>}
          <p className="engine-promise">Every number comes from COURTSIDE’s own forecast and parlay tools.</p>
        </div>
        <div className="chat-toolbar">
          <button id="chat-slip-chip" type="button" className={`chat-slip-chip${slipCount ? ' has-legs' : ''}`} onClick={onOpenSlip}
            aria-label={`Parlay slip: ${legsText}${slipCount && slipProbability !== null && !slipBusy ? `, all hit ${chance(slipProbability)}` : ''}. Open the slip.`}>
            <span className="chat-slip-ball" aria-hidden="true"><Ball /><span key={bounce} className={`chat-slip-count${bounce ? ' bounce' : ''}`}>{slipCount}</span></span>
            <span className="chat-slip-text" aria-hidden="true">{slipCount ? legsText : 'Slip empty'}</span>
            {slipCount > 0 && <span className="chat-slip-prob" aria-hidden="true">{slipBusy ? '…' : slipProbability !== null ? chance(slipProbability) : '—'}</span>}
          </button>
          {confirmReset
            ? <div className="chat-reset-confirm" role="group" aria-label="Confirm starting a new chat"><span>Start over?</span><button type="button" className="danger-button" onClick={resetChat}>New chat</button><button ref={keepButton} type="button" className="text-button" onClick={() => askReset(false)}>Keep</button></div>
            : <button ref={newChatButton} type="button" className="text-button chat-new" disabled={!messages.length && !busy} onClick={() => askReset(true)}><SquarePen size={14} aria-hidden="true" />New chat</button>}
        </div>
      </header>
      <div className="chat-log" ref={log} tabIndex={0} role="region" aria-label="Conversation with Coach">
        {!messages.length && !busy ? <div className="chat-empty">
          <HuddleArt />
          <h3>What’s the play?</h3>
          <p>Ask about your slip, check a price against the house, or build the parlay by command. Coach runs the same models as the rest of COURTSIDE.</p>
          <ul className="chat-starters" aria-label="Try asking">{starterPrompts.map(prompt => <li key={prompt}>
            <button type="button" className="chat-starter" onClick={() => sendQuick(prompt)}><span>{prompt}</span><ArrowUpRight size={15} aria-hidden="true" /></button>
          </li>)}</ul>
        </div> : <ol className="chat-messages">
          {messages.map((message: ChatMessage, index) => message.role === 'user'
            ? <UserTurn key={message.id} message={message} answered={answered(index) || index === messages.length - 1} />
            : <CoachTurn key={message.id} message={message} latest={index === messages.length - 1} undoable={message.id === undoId} busy={busy}
              feedbackError={chat.feedbackError?.messageId === message.id ? chat.feedbackError.message : null}
              onSend={sendQuick} onRate={chat.rate} onOpenParlay={onOpenParlay} onOpenRecord={onOpenRecord} />)}
          {busy && <li className="chat-turn coach thinking">
            <span className="chat-avatar" aria-hidden="true"><CoachMark /></span>
            <div className="chat-bubble coach-bubble thinking-bubble" role="status"><DribbleBall /><span>Coach is drawing up the play…</span></div>
          </li>}
        </ol>}
        {showFailure && <div className={`chat-failure${failure?.stopped ? ' stopped' : ''}`} role={failure && !failure.stopped ? 'alert' : undefined}>
          <CircleAlert size={16} aria-hidden="true" />
          <span>{failure ? failure.message : 'Coach hasn’t answered this message yet.'}</span>
          <button type="button" className="text-button" onClick={() => { chat.retry(); afterQuickSend(); }}><RotateCcw size={14} aria-hidden="true" />Retry</button>
        </div>}
      </div>
      <form className="chat-composer" onSubmit={submit}>
        {focus && <p className="chat-context"><Target size={13} aria-hidden="true" /><span>Forecast lab: <strong>{focus.player}</strong> · {statNames[focus.stat]} {focus.line}</span></p>}
        <div className={`chat-input-row${busy ? ' is-busy' : ''}`}>
          <label htmlFor="chat-input" className="sr-only">Message Coach</label>
          <textarea id="chat-input" ref={input} rows={1} maxLength={MAX_MESSAGE_LENGTH} value={draft} placeholder={busy ? 'Coach is answering…' : 'Ask Coach, or say “add Murray over 20.5 points”'}
            readOnly={busy} aria-disabled={busy || undefined} aria-describedby="chat-hint" spellCheck
            onChange={event => setDraft(event.target.value.slice(0, MAX_MESSAGE_LENGTH))} onKeyDown={onComposerKey} />
          {busy
            ? <button type="button" className="chat-send stop" onClick={chat.stop} aria-label="Stop waiting for Coach" title="Stop"><Square size={14} aria-hidden="true" /></button>
            : <button type="submit" className="chat-send" aria-disabled={!draft.trim() || undefined} aria-label="Send message" title="Send"><SendHorizontal size={18} aria-hidden="true" /></button>}
        </div>
        <div className="chat-composer-foot">
          <span id="chat-hint">Enter to send · Shift+Enter for a new line</span>
          {draft.length >= 800 && <span className={`chat-count${remaining <= 0 ? ' at-limit' : ''}`} aria-hidden="true">{draft.length}/{MAX_MESSAGE_LENGTH}</span>}
          <span className="sr-only" aria-live="polite">{remaining <= 0 ? 'Character limit reached.' : remaining <= 100 ? 'Fewer than 100 characters left.' : ''}</span>
        </div>
        <p className="chat-disclaimer">Probabilities, not certainties. Research only, not betting advice.</p>
      </form>
      <div className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</div>
    </div>
  </div>;
}
