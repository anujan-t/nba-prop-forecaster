import { ArrowRight, Binary, BookOpen, BrainCircuit, CircleHelp, Database, FlaskConical, Layers, Scale, ShieldCheck } from 'lucide-react';
import { Basketball, SectionHeading } from './components';

export default function Methodology({ onValidation, onData, onParlay, onRecord }: { onValidation: () => void; onData: () => void; onParlay: () => void; onRecord: () => void }) {
  return <>
    <article className="card methodology-intro"><div><span className="eyebrow">TRANSPARENCY IS A FEATURE</span><h2>A model is a lens.<br />Not a crystal ball.</h2><p>Courtside turns completed NBA player-game logs into explicit probability distributions. Every forecast comes with its sample, assumptions, provenance, and limitations attached.</p><p>The player selector comes from NBA's current roster endpoint, including rookies who have not played. It is not restricted to previously cached players or the package's bundled directory. Roster classification is not medical availability.</p><div className="technology-tags">{['Python', 'Pandas', 'NumPy', 'SciPy', 'nba_api', 'FastAPI', 'React'].map(name => <span key={name}>{name}</span>)}</div></div><Basketball /></article>
    <article className="card methodology-section"><SectionHeading eyebrow="01 / THE FOUNDATION" title="Five stats. Four statistical lenses." /><p>Points, rebounds, assists, steals, and blocks are nonnegative counts. No single distribution is right for every player, statistic, or sample. Auto-selection applies backend rules to the selected historical window; it is not a claim of universally superior predictive skill.</p>
      <div className="method-model-grid">
        <div><span className="method-number">01</span><h3>Empirical</h3><p>Uses the observed sample distribution rather than a parametric shape. A useful grounded reference, but small windows give coarse probabilities and cannot describe unseen tails well.</p><span className="method-trait">Sample-driven · few shape assumptions</span></div>
        <div><span className="method-number">02</span><h3>Normal</h3><p>Fits a mean and spread, then maps continuous mass to nonnegative integer outcomes. Often a useful approximation for higher-count stats, but symmetry may miss skew and heavy tails.</p><span className="method-trait">Mean + dispersion · discretized outcomes</span></div>
        <div><span className="method-number">03</span><h3>Poisson</h3><p>A discrete count model with one rate parameter. Its variance equals its mean, an assumption that can be too restrictive when minutes, usage, or game situations vary.</p><span className="method-trait">Count-based · mean equals variance</span></div>
        <div><span className="method-number">04</span><h3>Negative binomial</h3><p>Allows greater variance than the mean, accommodating overdispersed counts. Parameter estimates can be unstable in small or near-constant samples; pay attention to model warnings.</p><span className="method-trait">Overdispersed counts · additional flexibility</span></div>
      </div>
    </article>
    <div className="two-column">
      <article className="card methodology-section"><SectionHeading eyebrow="02 / THE QUESTION" title="Over, under, and the often-forgotten push"><Binary size={21} className="subtle" /></SectionHeading><p>For a player-stat outcome X and line L, all three outcomes matter:</p><div className="formula-block"><code>Over = P(X &gt; L)</code><code>Under = P(X &lt; L)</code><code>Push = P(X = L)</code><code>Over + Under + Push = 1</code></div><p>Integer lines may push; half-point lines cannot push for integer stats. The displayed predictive interval describes a range of modeled stat outcomes. It is <strong>not</strong> a confidence level for winning a bet.</p></article>
      <article className="card methodology-section"><SectionHeading eyebrow="03 / THE COMPARISON" title="Remove the vig. Keep the caveats."><Scale size={21} className="subtle" /></SectionHeading><p>American odds are converted to implied probability, then the two sides are normalized to remove the quoted margin.</p><div className="formula-block"><code>Positive a: q = 100 / (a + 100)</code><code>Negative a: q = |a| / (|a| + 100)</code><code>No-vig over = q_over / (q_over + q_under)</code><code>Model over, no push = P(over) / (1 − P(push))</code></div><p>Edge is the conditional model probability minus the normalized market probability, expressed in percentage points. Expected ROI is win probability × net payout minus loss probability; pushes return the stake. Neither metric is realized profit or a recommendation.</p></article>
    </div>
    <article className="card methodology-section"><SectionHeading eyebrow="04 / THE CONTEXT" title="Adjust for what’s observable. Admit what isn’t." /><div className="methodology-feature-grid">
      <div><span className="feature-icon"><CircleHelp size={20} /></span><h3>Playing time</h3><p>Minutes drive opportunity. A user-specified estimate or historical minute profile adjusts the projected stat mean. Backtests use pre-game estimates, never the held-out game’s actual minutes as an input.</p></div>
      <div><span className="feature-icon"><BookOpen size={20} /></span><h3>Rest & schedule</h3><p>Rest means days between games: the calendar-date gap minus one, so 0 means back-to-back. A scenario can specify rest, with bounded adjustment based on the backend’s historical rules. Missing assumptions are not silently presented as known.</p></div>
      <div><span className="feature-icon"><ShieldCheck size={20} /></span><h3>Opponent context</h3><p>Opponent effects use available history with sample safeguards. They are limited, associational adjustments—not a possession-level defensive model or causal estimate of matchup strength.</p></div>
    </div><div className="method-warning">Injuries, confirmed lineups, trades, overtime, pace, and coaching changes can create shifts these features do not capture. Roster membership is fetched from NBA; injury availability, confirmed lineups and market odds are not.</div></article>
    <div className="two-column">
      <article className="card methodology-section"><SectionHeading eyebrow="05 / THE EVALUATION" title="Walk forward. Never peek."><FlaskConical size={21} className="subtle" /></SectionHeading><p>The first 15 games establish a warm-up history. For each later game, fit to older rows within the chosen lookback, estimate the over probability, and compare it with the held-out outcome. Pushes are excluded from binary scoring.</p><div className="formula-block"><code>Brier = mean((predicted − outcome)²)</code><code>Skill = 1 − model Brier / baseline Brier</code></div><p>Lower Brier and log loss are better. Calibration asks whether similar probability forecasts match their observed frequencies. Always inspect sample counts. Fixed scenario lines and adaptive prior-median half-point lines are <strong>not historical market lines</strong>; this is not a historical betting-profit simulation.</p><button className="text-button" onClick={onValidation}>Open model validation <ArrowRight size={15} /></button></article>
      <article className="card methodology-section"><SectionHeading eyebrow="06 / THE SOURCE & ITS LIMITS" title="NBA Stats is the starting point."><Database size={21} className="subtle" /></SectionHeading><p>Historical completed games come directly from the endpoints behind NBA.com through the community-maintained <code>nba_api</code> client. No API key or CSV upload is needed. Availability varies by player, season and statistical category.</p><p>Structured responses are stored as native records in a local SQLite cache. Load reuses history fetched within six hours; expired history is fetched again. Refresh always contacts NBA and updates the same history. NBA outages never silently produce fabricated or synthetic results.</p><p>The optional synthetic fixture is only for reproducible development, never evidence of real NBA accuracy. Source and retrieval time accompany forecasts. No live odds, injury updates or automatic market history are implied.</p><button className="text-button" onClick={onData}>Inspect NBA history <ArrowRight size={15} /></button></article>
    </div>
    <article className="card methodology-section"><SectionHeading eyebrow="07 / PARLAY BUILDER" title="Every leg has to land."><Layers size={21} className="subtle" /></SectionHeading>
      <div className="two-column method-parlay">
        <div>
          <p>Combine 1–10 legs, one per player and stat. Each leg is priced by exactly the same model and settings as the Forecast lab, and the parlay wins only if every leg wins.</p>
          <div className="formula-block"><code>Independent = p₁ × p₂ × … × pₙ</code><code>ρ = 2·sin(π·ρₛ/6) × n/(n + 10)</code><code>Joint = P(every leg wins)</code><code>Fair decimal odds = 1 / joint</code></div>
          <p className="method-footnote">ρₛ is the Spearman rank correlation of the two stats over n shared games; ρ is the copula correlation, capped at ±0.95. The joint is the multivariate-normal orthant probability (Genz algorithm, fixed seed), so identical inputs always give the same answer.</p>
        </div>
        <div>
          <ul className="method-rules">
            <li><strong>Linked legs.</strong> The same player, or current teammates, joined by a Gaussian copula fitted on shared recent games: each player’s last 30, or a longer lookback. Below 10 shared games, no link is assumed. Opposite sides flip the direction of a link.</li>
            <li><strong>Independent legs.</strong> Players on different teams, even when they meet: head-to-head linkage is not modelled.</li>
            <li><strong>Pushes.</strong> A push counts as not winning. For whole-number lines, the chance that no leg loses (each wins or pushes) is shown separately. Returns against entered odds treat a push as a loss.</li>
            <li><strong>History check.</strong> How often all linked legs hit together in shared past games, beside the model’s figure: a sanity check, not a backtest.</li>
          </ul>
          <p>Research only, not betting advice: probabilities, not certainties. Injuries, lineups and role changes the sample has not seen are not captured.</p>
          <button className="text-button" onClick={onParlay}>Open the parlay builder <ArrowRight size={15} /></button>
        </div>
      </div>
    </article>
    <article className="card methodology-section"><SectionHeading eyebrow="08 / COACH & LEARNING" title="Ask in plain words. Every number is computed."><BrainCircuit size={21} className="subtle" /></SectionHeading>
      <div className="two-column method-parlay">
        <div>
          <p><strong>Coach</strong>, the chat assistant, reads commands such as “add Jokić over 26.5 points”, “flip leg 2” or “book has it at +450” with a built-in parser wired to the same forecast and parlay code as the rest of COURTSIDE, so a change made in chat is the same change you could make by hand, and the slip stays in step.</p>
          <p>An optional LLM, any OpenAI-compatible endpoint (OpenAI, Azure OpenAI v1, or a local Ollama or LM Studio model), can take the open-ended questions. It phrases the answer; every probability, price and average still comes from the app’s own tools, never from the language model.</p>
          <p className="method-footnote">Price checks compare the offered odds’ break-even probability with the model’s. Without entered odds, Coach assumes −110 on every leg priced as independent, and says so: that is a reference point, not a real book price.</p>
        </div>
        <div>
          <ul className="method-rules">
            <li><strong>Graded results.</strong> Parlays you track are graded against final box scores; each leg becomes a labelled outcome beside the probability the model gave it.</li>
            <li><strong>Gated calibration.</strong> A recalibration layer is refitted on graded outcomes and switched on only when it beats the raw model on recent held-out games. Otherwise the raw model stands.</li>
            <li><strong>Memory.</strong> Coach remembers nicknames you teach it or confirm, and your thumbs up or down on its answers. You can make it forget a nickname at any time.</li>
          </ul>
          <p>Learning narrows calibration error; it does not see injuries or lineups the data lacks. Probabilities, not certainties. Research only, not betting advice.</p>
          <button className="text-button" onClick={onRecord}>Open the Track record <ArrowRight size={15} /></button>
        </div>
      </div>
    </article>
    <div className="methodology-disclaimer"><Basketball /><div><h3>Built for inquiry, not instructions to wager.</h3><p>Educational and portfolio research only. Probability estimates contain uncertainty, model risk, and data limitations. They are not financial advice or recommendations to gamble.</p></div></div>
  </>;
}
