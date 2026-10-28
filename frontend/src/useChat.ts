import { useCallback, useEffect, useRef, useState } from 'react';
import { api, isAbort, messageOf } from './api';
import type { AssistantStatus, ChatFocus, ChatRequest, ChatSlip, FeedbackRequest } from './assistant-types';
import { CHAT_STORAGE_KEY, MAX_MESSAGE_LENGTH, MAX_STORED_MESSAGES, loadConversation, newConversation, normalizeReply, parseConversation, randomId, responseSlip, saveConversation } from './chat';
import type { ChatMessage, CoachMessage, Conversation, UserMessage } from './chat';

/** What happened to a slip carried by a reply. */
export interface SlipApplication { applied: boolean; hash: string | null; notes: string[] }

export interface ChatBindings {
  /** The slip right now, sent with every message. */
  getSlip: () => ChatSlip;
  /** The applied Forecast lab forecast, if any. */
  getFocus: () => ChatFocus | null;
  /** Applies a reply's slip, unless the slip changed since `sent` went out. */
  applySlip: (slip: { legs: unknown[]; book_odds: unknown }, sent: ChatSlip) => SlipApplication;
}

export interface ChatFailure { messageId: string; message: string; stopped: boolean }
/** A counter-stamped event, so effects fire once per occurrence. */
export interface ChatEvent { messageId: string; n: number }

export interface ChatState {
  conversation: Conversation;
  /** Id of the user message awaiting a reply. */
  pending: string | null;
  failure: ChatFailure | null;
  status: AssistantStatus | null;
  statusError: string;
  feedbackError: { messageId: string; message: string } | null;
  reply: ChatEvent | null;
  slipEvent: ChatEvent | null;
  send: (text: string) => boolean;
  retry: () => void;
  stop: () => void;
  reset: () => void;
  rate: (messageId: string, rating: 1 | -1) => void;
  refreshStatus: (force?: boolean) => void;
}

interface InFlight { controller: AbortController; reason: 'stop' | 'reset' | null }

const append = (messages: ChatMessage[], message: ChatMessage) => [...messages, message].slice(-MAX_STORED_MESSAGES);

export function useChat(bindings: ChatBindings): ChatState {
  const [conversation, setConversation] = useState<Conversation>(loadConversation);
  const [pending, setPending] = useState<string | null>(null);
  const [failure, setFailure] = useState<ChatFailure | null>(null);
  const [status, setStatus] = useState<AssistantStatus | null>(null);
  const [statusError, setStatusError] = useState('');
  const [feedbackError, setFeedbackError] = useState<ChatState['feedbackError']>(null);
  const [reply, setReply] = useState<ChatEvent | null>(null);
  const [slipEvent, setSlipEvent] = useState<ChatEvent | null>(null);
  const bindingsRef = useRef(bindings);
  bindingsRef.current = bindings;
  const conversationRef = useRef(conversation);
  conversationRef.current = conversation;
  const inFlight = useRef<InFlight | null>(null);
  const fromStorage = useRef(false);
  const statusState = useRef({ loading: false, at: 0 });
  const ratingSequence = useRef(new Map<string, number>());
  const counter = useRef(0);

  useEffect(() => {
    if (fromStorage.current) { fromStorage.current = false; return; }
    saveConversation(conversation);
  }, [conversation]);

  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key !== CHAT_STORAGE_KEY && event.key !== null) return;
      if (inFlight.current) return; // this tab's reply is on its way and will be saved after it
      fromStorage.current = true;
      setConversation(parseConversation(event.key === null ? null : event.newValue));
      setFailure(null);
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  useEffect(() => () => {
    const current = inFlight.current;
    if (current) { current.reason = 'reset'; current.controller.abort(); }
  }, []);

  const request = useCallback((message: UserMessage) => {
    const handle: InFlight = { controller: new AbortController(), reason: null };
    inFlight.current = handle;
    setPending(message.id);
    setFailure(null);
    const conversationId = conversationRef.current.id;
    const sent = bindingsRef.current.getSlip();
    const body: ChatRequest = { message: message.text, conversation_id: conversationId, slip: sent, focus: bindingsRef.current.getFocus() };
    api<unknown>('/api/assistant/chat', { method: 'POST', body: JSON.stringify(body), signal: handle.controller.signal })
      .then(raw => {
        if (handle.controller.signal.aborted || conversationRef.current.id !== conversationId) return;
        const slip = responseSlip(raw);
        const application = slip ? bindingsRef.current.applySlip(slip, sent) : null;
        const coach: CoachMessage = {
          id: randomId(12), role: 'coach', at: new Date().toISOString(), reply: normalizeReply(raw),
          slip: application ? (application.applied ? 'applied' : 'skipped') : 'none',
          slipHash: application?.hash ?? null, notes: application?.notes ?? [], rating: null,
        };
        setConversation(current => ({ ...current, messages: append(current.messages, coach) }));
        counter.current += 1;
        const event = { messageId: coach.id, n: counter.current };
        setReply(event);
        if (application?.applied) setSlipEvent(event);
      })
      .catch((reason: unknown) => {
        if (handle.reason === 'reset') return;
        if (handle.reason === 'stop' || isAbort(reason)) {
          setFailure({ messageId: message.id, message: 'Stopped. Coach did not answer this one.', stopped: true });
          return;
        }
        setFailure({ messageId: message.id, message: messageOf(reason), stopped: false });
      })
      .finally(() => {
        if (inFlight.current !== handle) return;
        inFlight.current = null;
        setPending(null);
      });
  }, []);

  const send = useCallback((text: string): boolean => {
    const message = text.trim();
    if (!message || message.length > MAX_MESSAGE_LENGTH || inFlight.current) return false;
    const user: UserMessage = { id: randomId(12), role: 'user', text: message, at: new Date().toISOString() };
    setConversation(current => ({ ...current, messages: append(current.messages, user) }));
    conversationRef.current = { ...conversationRef.current, messages: append(conversationRef.current.messages, user) };
    request(user);
    return true;
  }, [request]);

  const retry = useCallback(() => {
    if (inFlight.current) return;
    const messages = conversationRef.current.messages;
    const last = messages[messages.length - 1];
    if (last?.role === 'user') request(last);
  }, [request]);

  const stop = useCallback(() => {
    const current = inFlight.current;
    if (!current) return;
    current.reason = 'stop';
    current.controller.abort();
  }, []);

  const reset = useCallback(() => {
    const current = inFlight.current;
    if (current) {
      current.reason = 'reset';
      current.controller.abort();
      inFlight.current = null;
      setPending(null);
    }
    setFailure(null);
    setFeedbackError(null);
    const next = newConversation();
    conversationRef.current = next;
    setConversation(next);
  }, []);

  const rate = useCallback((messageId: string, rating: 1 | -1) => {
    const target = conversationRef.current.messages.find((item): item is CoachMessage => item.id === messageId && item.role === 'coach');
    if (!target || !target.reply.turn_id || target.rating === rating) return;
    const previous = target.rating;
    const sequence = (ratingSequence.current.get(messageId) ?? 0) + 1;
    ratingSequence.current.set(messageId, sequence);
    const patch = (value: 1 | -1 | null) => setConversation(current => ({
      ...current,
      messages: current.messages.map(item => item.id === messageId && item.role === 'coach' ? { ...item, rating: value } : item),
    }));
    patch(rating);
    setFeedbackError(null);
    const body: FeedbackRequest = { turn_id: target.reply.turn_id, rating, note: null };
    api<unknown>('/api/assistant/feedback', { method: 'POST', body: JSON.stringify(body) }).catch((reason: unknown) => {
      if (ratingSequence.current.get(messageId) !== sequence) return; // a newer rating superseded this one
      patch(previous);
      setFeedbackError({ messageId, message: `Your rating wasn’t saved. ${messageOf(reason)}` });
    });
  }, []);

  const refreshStatus = useCallback((force = false) => {
    const state = statusState.current;
    if (state.loading || (!force && state.at && Date.now() - state.at < 60_000)) return;
    state.loading = true;
    api<AssistantStatus>('/api/assistant/status')
      .then(data => { setStatus(data); setStatusError(''); })
      .catch((reason: unknown) => setStatusError(messageOf(reason)))
      .finally(() => { state.loading = false; state.at = Date.now(); });
  }, []);

  return { conversation, pending, failure, status, statusError, feedbackError, reply, slipEvent, send, retry, stop, reset, rate, refreshStatus };
}
