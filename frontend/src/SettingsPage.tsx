import { BrainCircuit, Check, KeyRound, MessagesSquare, Moon, RefreshCw, Sun, ThumbsDown, ThumbsUp } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { CoachMark, engineLabel } from './ChatPanel';
import { RichText } from './ChatCards';
import { Ball, Busy, ErrorNotice, SectionHeading } from './components';
import { focusAfterRemoval } from './ledger';
import type { ChatState } from './useChat';
import type { LearningState } from './useLearning';

export type Theme = 'light' | 'dark';

const themes: { id: Theme; label: string; description: string; icon: typeof Sun }[] = [
  { id: 'light', label: 'Day game', description: 'Light hardwood, navy and basketball orange.', icon: Sun },
  { id: 'dark', label: 'Night game', description: 'The dark arena floor under the lights.', icon: Moon },
];

/** Settings › Appearance. Each option previews itself in its own theme. */
export function AppearanceSettings({ theme, onTheme }: { theme: Theme; onTheme: (theme: Theme) => void }) {
  return <section className="card settings-card" aria-labelledby="appearance-title">
    <SectionHeading id="appearance-title" eyebrow="HOUSE LIGHTS" title="Appearance" />
    <fieldset className="theme-options">
      <legend>Colour theme</legend>
      {themes.map(option => {
        const Icon = option.icon;
        const selected = theme === option.id;
        return <label key={option.id} className={`theme-option${selected ? ' selected' : ''}`}>
          <input type="radio" name="courtside-theme" value={option.id} checked={selected} onChange={() => onTheme(option.id)} />
          <span className="theme-preview" data-theme={option.id} aria-hidden="true">
            <span className="preview-bar"><Ball /><span className="preview-brand">COURTSIDE</span><span className="preview-tab" /></span>
            <span className="preview-floor">
              <span className="preview-card"><span className="preview-kicker">ALL LEGS HIT</span><span className="preview-number">24.6%</span></span>
              <span className="preview-chip">Over 26.5</span>
            </span>
          </span>
          <span className="theme-option-text">
            <span className="theme-option-name"><Icon size={16} aria-hidden="true" />{option.label}{selected && <Check size={16} aria-hidden="true" className="theme-option-check" />}</span>
            <small>{option.description}</small>
          </span>
        </label>;
      })}
    </fieldset>
    <p className="record-hint">Saved on this device. The switch in the menu does the same thing.</p>
  </section>;
}

const llmOptions: { title: string; note?: string; lines: string[] }[] = [
  { title: 'OpenAI', lines: ['COURTSIDE_LLM_MODEL=gpt-4o-mini', 'COURTSIDE_LLM_API_KEY=sk-...'] },
  { title: 'A local model with Ollama', note: 'Run “ollama pull llama3.1” first. Nothing leaves this computer.', lines: ['COURTSIDE_LLM_MODEL=llama3.1', 'COURTSIDE_LLM_BASE_URL=http://127.0.0.1:11434/v1'] },
  { title: 'Azure OpenAI (v1) or another OpenAI-compatible server', lines: ['COURTSIDE_LLM_MODEL=<deployment-name>', 'COURTSIDE_LLM_BASE_URL=https://<resource>.openai.azure.com/openai/v1', 'COURTSIDE_LLM_API_KEY=<key>', 'COURTSIDE_LLM_KEY_HEADER=api-key'] },
];

/** Settings › Coach: which engine answers, how to connect a language model, and what Coach remembers. */
export function CoachSettings({ chat, learning, onOpenCoach }: { chat: ChatState; learning: LearningState; onOpenCoach: () => void }) {
  const { status, statusError, refreshStatus } = chat;
  const engine = engineLabel(status, statusError);
  const [checking, setChecking] = useState(false);
  const [announcement, setAnnouncement] = useState('');
  const memoryHeading = useRef<HTMLHeadingElement>(null);
  const aliasList = useRef<HTMLUListElement>(null);
  const memory = learning.status?.memory;
  const announce = (text: string) => setAnnouncement(current => current === text ? `${text}\u00a0` : text);

  useEffect(() => { void refreshStatus(); }, [refreshStatus]);

  async function check() {
    if (checking) return;
    setChecking(true);
    try { await refreshStatus(true); } finally { setChecking(false); }
    announce('Checked which engine is answering.');
  }

  async function forget(phrase: string, player: string) {
    if (learning.forgetting) return;
    const refocus = focusAfterRemoval(aliasList.current, 'data-alias', phrase, memoryHeading.current, item => item.querySelector<HTMLElement>('button'));
    if (await learning.forget(phrase)) {
      refocus();
      announce(`Forgot that “${phrase}” means ${player}.`);
    }
  }

  return <div className="settings-stack">
    <section className="card settings-card" aria-labelledby="coach-engine-title">
      <SectionHeading id="coach-engine-title" eyebrow="THE BENCH" title="Coach’s engine">
        <span className={`engine-badge static ${engine.tone}`}><span className="engine-badge-dot" aria-hidden="true" /><span className="engine-badge-text"><span className="sr-only">Engine: </span>{engine.text}</span></span>
      </SectionHeading>
      <div className="engine-summary">
        <span className="engine-summary-mark" aria-hidden="true"><CoachMark /></span>
        <div>
          {status ? <>
            <RichText text={status.detail} />
            {status.endpoint && <p className="engine-endpoint">Endpoint: <code>{status.endpoint}</code></p>}
          </> : statusError ? <p>The assistant status couldn’t be loaded. {statusError}</p> : <Busy text="Checking which engine is answering…" />}
          <p className="engine-promise">Either way, every number comes from COURTSIDE’s own forecast and parlay tools.</p>
        </div>
      </div>
      <div className="settings-actions">
        <button type="button" className="secondary-button" aria-disabled={checking || undefined} onClick={() => { void check(); }}><RefreshCw size={15} aria-hidden="true" className={checking ? 'spin' : ''} />{checking ? 'Checking…' : 'Check again'}</button>
        <button type="button" className="text-button" onClick={onOpenCoach}><MessagesSquare size={15} aria-hidden="true" />Open Coach</button>
      </div>
      <h3 className="settings-subtitle"><KeyRound size={17} aria-hidden="true" />{status?.mode === 'llm' ? 'Change the language model' : 'Connect a language model (optional)'}</h3>
      <ol className="llm-steps">
        <li>Copy <code>.env.example</code> in the COURTSIDE folder to <code>.env</code>. It stays on this computer and out of git.</li>
        <li>Fill in one option:
          <div className="llm-options">{llmOptions.map(option => <div key={option.title} className="llm-option">
            <strong>{option.title}</strong>
            {option.note && <small>{option.note}</small>}
            <pre><code>{option.lines.join('\n')}</code></pre>
          </div>)}</div>
        </li>
        <li>Restart COURTSIDE with <code>Start.ps1</code>, then press <strong>Check again</strong>.</li>
      </ol>
      <p className="record-hint">Without a model, the built-in coach still handles commands, slip building, price checks and grading. A model only adds open-ended answers.</p>
    </section>
    <section className="card settings-card" aria-labelledby="coach-memory-title">
      <SectionHeading id="coach-memory-title" eyebrow="MEMORY" title="What Coach remembers" headingRef={memoryHeading} />
      {learning.error && <ErrorNotice message={learning.error} onRetry={() => learning.load()} />}
      {!learning.status && learning.loading && <Busy text="Opening Coach’s notebook…" />}
      {memory && <>
        <dl className="memory-counts">
          <div><dt><ThumbsUp size={14} aria-hidden="true" />Helpful</dt><dd>{memory.liked}</dd></div>
          <div><dt><ThumbsDown size={14} aria-hidden="true" />Not helpful</dt><dd>{memory.disliked}</dd></div>
          <div><dt><BrainCircuit size={14} aria-hidden="true" />Turns</dt><dd>{memory.turns}</dd></div>
        </dl>
        <h3 className="alias-title">Nicknames learned</h3>
        {memory.aliases.length
          ? <ul className="alias-list" ref={aliasList}>{memory.aliases.map(alias => <li key={alias.phrase} data-alias={alias.phrase}>
            <span className="alias-text"><span className="alias-phrase">“{alias.phrase}”</span> <span aria-hidden="true">→</span><span className="sr-only">means</span> <strong>{alias.player}</strong>
              <small>{alias.source === 'taught' ? 'You taught it' : 'From a clarification'} · used {alias.uses} {alias.uses === 1 ? 'time' : 'times'}</small></span>
            <button type="button" className="text-button" aria-disabled={learning.forgetting === alias.phrase || undefined} onClick={() => { void forget(alias.phrase, alias.player); }} aria-label={`Forget that “${alias.phrase}” means ${alias.player}`}>{learning.forgetting === alias.phrase ? 'Forgetting…' : 'Forget'}</button>
          </li>)}</ul>
          : <p className="record-hint">None yet. Tell Coach “joker means Nikola Jokić”, or pick an answer when it asks which player you meant.</p>}
        {learning.aliasError && <p className="field-error" role="alert">{learning.aliasError}</p>}
      </>}
      <p className="record-hint">Ratings and nicknames are stored by the COURTSIDE server on this computer. The conversation itself is saved in this browser; start over with New chat in Coach.</p>
    </section>
    <div className="sr-only" aria-live="polite" aria-atomic="true">{announcement}</div>
  </div>;
}
