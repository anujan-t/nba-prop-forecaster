import { ArrowRight, Binary, BookOpen, BrainCircuit, CircleHelp, Database, FlaskConical, Layers, Scale, ShieldCheck } from 'lucide-react';
import { Basketball, SectionHeading } from './components';
import { RouteLink } from './RouteLink';

/** How it works › Method. */
export default function Methodology() {
  return <>
    <article className="card methodology-intro"><div><span className="eyebrow">TRANSPARENCY IS A FEATURE</span><h2>A model is a lens.<br />Not a crystal ball.</h2><p>Courtside turns completed NBA player-game logs into explicit probability distributions. Every forecast comes with its sample, assumptions, provenance, and limitations attached.</p><p>The player selector comes from NBA's current roster endpoint, including rookies who have not played. It is not restricted to previously cached players or the package's bundled directory. Roster classification is not medical availability.</p><div className="technology-tags">{['Python', 'Pandas', 'NumPy', 'SciPy', 'nba_api', 'FastAPI', 'React'].map(name => <span key={name}>{name}</span>)}</div></div><Basketball /></article>
    <article className="card methodology-section"><SectionHeading eyebrow="01 / THE FOUNDATION" title="22 markets. One simulated game." /><p>Every market is a count from the box score: points, rebounds, assists, threes, steals, blocks and turnovers; the combos (PRA, pts+reb, pts+ast, reb+ast, steals+blocks); shooting makes and attempts; offensive and defensive rebounds; fouls; and the yes/no double-double and triple-double. Auto uses the <strong>game simulator</strong>, which beat the old rule on every market in a walk-forward test on 232,000 graded predictions from real games (Accuracy has the table). The other lenses stay on the model bench as sensitivity checks.</p>
      <div className="method-model-grid">
        <div><span className="method-number">00</span><h3>Game simulator</h3><p>Plays the player's next game 250,000 times (up to 1,000,000). Each game draws minutes from recent minutes, a per-minute rate for every attempt, rebound, assist, steal, block, turnover and foul (with the uncertainty of estimating it), counts as dispersed as real box scores, and makes from attempts at a shooting percentage drawn from its posterior. Recent games count more: usage and minutes have a 10-game half-life, steals and blocks 30.</p><span className="method-trait">Default · every market from the same games</span></div>
        <div><span className="method-number">01</span><h3>Empirical</h3><p>Uses the observed sample distribution rather than a parametric shape. A useful grounded reference, but small windows give coarse probabilities and cannot describe unseen tails well.</p><span className="method-trait">Sample-driven · few shape assumptions</span></div>
        <div><span className="method-number">02</span><h3>Normal</h3><p>Fits a mean and spread, then maps continuous mass to nonnegative integer outcomes. Often a useful approximation for higher-count stats, but symmetry may miss skew and heavy tails.</p><span className="method-trait">Mean + dispersion · discretized outcomes</span></div>
        <div><span className="method-number">03</span><h3>Poisson</h3><p>A discrete count model with one rate parameter. Its variance equals its mean, an assumption that can be too restrictive when minutes, usage, or game situations vary.</p><span className="method-trait">Count-based · mean equals variance</span></div>
        <div><span className="method-number">04</span><h3>Negative binomial</h3><p>Allows greater variance than the mean, accommodating overdispersed counts. Parameter estimates can be unstable in small or near-constant samples; pay attention to model warnings.</p><span className="method-trait">Overdispersed counts · additional flexibility</span></div>
      </div>
      <p className="method-footnote">More simulations shrink simulation noise (±0.1 percentage points at 250,000 games); they do not make the model more right. Accuracy comes from what is simulated, measured by walk-forward tests, then corrected per market by the learning loop. League-wide dispersion and shooting priors were measured on 8,417 real 2024-25 and 2025-26 player-games.</p>
    </article>
    <div className="two-column">
      <article className="card methodology-section"><SectionHeading eyebrow="02 / THE QUESTION" title="Over, under, and the often-forgotten push"><Binary size={21} className="subtle" /></SectionHeading><p>For a player-stat outcome X and line L, all three outcomes matter:</p><div className="formula-block"><code>Over = P(X &gt; L)</code><code>Under = P(X &lt; L)</code><code>Push = P(X = L)</code><code>Over + Under + Push = 1</code></div><p>Integer lines may push; half-point lines cannot push for integer stats. The displayed predictive interval describes a range of modeled stat outcomes. It is <strong>not</strong> a confidence level for winning a bet.</p></article>
      <article className="card methodology-section"><SectionHeading eyebrow="03 / THE COMPARISON" title="Remove the vig. Keep the caveats."><Scale size={21} className="subtle" /></SectionHeading><p>American odds are converted to implied probability, then the two sides are normalized to remove the quoted margin.</p><div className="formula-block"><code>Positive a: q = 100 / (a + 100)</code><code>Negative a: q = |a| / (|a| + 100)</code><code>No-vig over = q_over / (q_over + q_under)</code><code>Model over, no push = P(over) / (1 − P(push))</code></div><p>Edge is the conditional model probability minus the normalized market probability, expressed in percentage points. Expected ROI is win probability × net payout minus loss probability; pushes return the stake. Neither metric is realized profit or a recommendation.</p></article>
    </div>
    <article className="card methodology-section"><SectionHeading eyebrow="04 / THE CONTEXT" title="Adjust for what’s observable. Admit what isn’t." /><div className="methodology-feature-grid">
      <div><span className="feature-icon"><CircleHelp size={20} /></span><h3>Playing time</h3><p>Minutes drive opportunity. A user-specified estimate or historical minute profile adjusts the projected stat mean. Backtests use pre-game estimates, never the held-out game’s actual minutes as an input.</p></div>
      <div><span className="feature-icon"><BookOpen size={20} /></span><h3>Rest & schedule</h3><p>Rest means days between games: the calendar-date gap minus one, so 0 means back-to-back. A scenario can specify rest, with bounded adjustment based on the backend’s historical rules. Missing assumptions are not silently presented as known.</p></div>
      <div><span className="feature-icon"><ShieldCheck size={20} /></span><h3>Opponent context</h3><p>Opponent effects use available history with sample safeguards. They are limited, associational adjustments—not a possession-level defensive model or causal estimate of matchup strength.</p></div>
    </div><div className="method-warning">Injuries, confirmed lineups, trades, overtime, pace, and coaching changes can create shifts these features do not capture. News shows the league injury report, roster moves, coaching staffs and team styles, and Players flags a listed player and his teammates who are out, but the model does not adjust for them automatically: a forecast assumes the player plays his usual role.</div></article>
    <div className="two-column">
      <article className="card methodology-section"><SectionHeading eyebrow="05 / THE EVALUATION" title="Walk forward. Never peek."><FlaskConical size={21} className="subtle" /></SectionHeading><p>The first 15 games establish a warm-up history. For each later game, fit to older rows within the chosen lookback, estimate the over probability, and compare it with the held-out outcome. Pushes are excluded from binary scoring.</p><div className="formula-block"><code>Brier = mean((predicted − outcome)²)</code><code>Skill = 1 − model Brier / baseline Brier</code></div><p>Lower Brier and log loss are better. Calibration asks whether similar probability forecasts match their observed frequencies. Always inspect sample counts. Fixed scenario lines and adaptive prior-median half-point lines are <strong>not historical market lines</strong>; this is not a historical betting-profit simulation.</p><RouteLink className="text-button" to={{ page: 'how', section: 'accuracy' }} focus="backtest-title">Run a backtest in Accuracy <ArrowRight size={15} aria-hidden="true" /></RouteLink></article>
      <article className="card methodology-section"><SectionHeading eyebrow="06 / THE SOURCE & ITS LIMITS" title="NBA Stats is the starting point."><Database size={21} className="subtle" /></SectionHeading><p>Historical completed games come directly from the endpoints behind NBA.com through the community-maintained <code>nba_api</code> client. No API key or CSV upload is needed. Availability varies by player, season and statistical category.</p><p>Structured responses are stored as native records in a local SQLite cache. Load reuses history fetched within six hours; expired history is fetched again. Refresh always contacts NBA and updates the same history. NBA outages never silently produce fabricated or synthetic results.</p><p>The optional synthetic fixture is only for reproducible development, never evidence of real NBA accuracy. Source and retrieval time accompany forecasts. No live odds, injury updates or automatic market history are implied.</p><RouteLink className="text-button" to={{ page: 'settings', section: 'data' }}>Inspect saved NBA history in Settings › Data <ArrowRight size={15} aria-hidden="true" /></RouteLink></article>
    </div>
    <article className="card methodology-section"><SectionHeading eyebrow="07 / MONEYLINES & SPREADS" title="Rate the teams. Price the margin."><Scale size={21} className="subtle" /></SectionHeading>
      <div className="two-column method-parlay">
        <div>
          <p>Every completed game this season and last gives each team a rating: how many points better than an average team it is on a neutral floor. A game’s expected margin is home court plus the home rating minus the away rating, less a back-to-back penalty for a team that played the day before. Recent games count most (weights halve every 120 days; last season’s games count 0.3 as much).</p>
          <div className="formula-block"><code>μ = HCA + r_home − r_away + b2b × (away B2B − home B2B)</code><code>Home margin ~ Normal(μ, 13 points), whole points, no ties</code><code>Moneyline wins when the team’s margin &gt; 0</code><code>Spread s covers when margin + s &gt; 0, pushes when = 0</code></div>
          <p className="method-footnote">Walk-forward on 2023-24 to 2025-26 (3,680 games, each predicted only from earlier games): log loss 0.602 against 0.689 for always taking the home team, 66.7% of winners, and calibrated by decile. The book’s line knows about injuries and news the ratings don’t, so treat a big gap between the two as a reason to check the injury report, not as an edge.</p>
        </div>
        <div>
          <ul className="method-rules">
            <li><strong>Same game.</strong> A moneyline and a spread on one game are priced exactly from the same margin distribution: a team’s moneyline and −6.5 both win only when it wins by 7 or more. Legs that cannot both win are rejected.</li>
            <li><strong>Player and team.</strong> A player’s prop and his game’s moneyline or spread are linked through how his stat moved with his team’s final margin in his recent games, with the same shrinkage as teammates.</li>
            <li><strong>Grading.</strong> Tracked moneylines and spreads are graded from the final score as soon as the game is final; a postponed game is void.</li>
          </ul>
          <p>Not modeled: injuries, lineups and trades (each game card lists both teams’ injury reports), and totals. Until both teams have played 10 games, the ratings lean on last season.</p>
          <RouteLink className="text-button" to={{ page: 'news', section: 'games', team: null }}>See tonight’s games <ArrowRight size={15} aria-hidden="true" /></RouteLink>
        </div>
      </div>
    </article>
    <article className="card methodology-section"><SectionHeading eyebrow="08 / PARLAY BUILDER" title="Every leg has to land."><Layers size={21} className="subtle" /></SectionHeading>
      <div className="two-column method-parlay">
        <div>
          <p>Combine 1–10 legs: one per player and stat, and one moneyline and one spread per game. Each prop leg is priced by exactly the same model and settings as Players › Props, each game leg by the team model, and the parlay wins only if every leg wins.</p>
          <div className="formula-block"><code>Independent = p₁ × p₂ × … × pₙ</code><code>Same player (simulator) = share of simulated games where every leg wins</code><code>ρ = 2·sin(π·ρₛ/6) × n/(n + 10)</code><code>Fair decimal odds = 1 / joint</code></div>
          <p className="method-footnote">Same-player legs on the game simulator are counted together in the same simulated games, which scored better on 28,464 held-out pairs than the shared-games correlation. Teammates use ρ, from the Spearman rank correlation ρₛ of the two stats over n shared games, capped at ±0.95. The joint is the multivariate-normal orthant probability (Genz algorithm, fixed seed) with each simulated same-player block replaced by its exact simulated chance, so identical inputs always give the same answer.</p>
        </div>
        <div>
          <ul className="method-rules">
            <li><strong>Same player.</strong> Priced jointly from the simulator's games: points and PRA, or threes and points, move together exactly as they do in the simulated box scores.</li>
            <li><strong>Teammates.</strong> Joined by a Gaussian copula fitted on shared recent games: each player’s last 30, or a longer lookback. Below 10 shared games, no link is assumed. Opposite sides flip the direction of a link.</li>
            <li><strong>Independent legs.</strong> Players on different teams, even when they meet (head-to-head linkage is not modelled), and legs on different games.</li>
            <li><strong>Pushes.</strong> A push counts as not winning. For whole-number lines, the chance that no leg loses (each wins or pushes) is shown separately. Returns against entered odds treat a push as a loss.</li>
            <li><strong>History check.</strong> How often all linked legs hit together in shared past games, beside the model’s figure: a sanity check, not a backtest.</li>
          </ul>
          <p>Research only, not betting advice: probabilities, not certainties. Injuries, lineups and role changes the sample has not seen are not captured.</p>
          <RouteLink className="text-button" to={{ page: 'parlay' }}>Open the parlay builder <ArrowRight size={15} aria-hidden="true" /></RouteLink>
        </div>
      </div>
    </article>
    <article className="card methodology-section"><SectionHeading eyebrow="09 / COACH & LEARNING" title="Ask in plain words. Every number is computed."><BrainCircuit size={21} className="subtle" /></SectionHeading>
      <div className="two-column method-parlay">
        <div>
          <p><strong>Coach</strong>, the chat assistant, reads commands such as “add Jokić over 26.5 points”, “add Celtics −5.5”, “flip leg 2” or “book has it at +450” with a built-in parser wired to the same forecast and parlay code as the rest of COURTSIDE, so a change made in chat is the same change you could make by hand, and the slip stays in step.</p>
          <p>An optional LLM, any OpenAI-compatible endpoint (OpenAI, Azure OpenAI v1, or a local Ollama or LM Studio model), can take the open-ended questions. It phrases the answer; every probability, price and average still comes from the app’s own tools, never from the language model.</p>
          <p className="method-footnote">Price checks compare the offered odds’ break-even probability with the model’s. Without entered odds, Coach assumes −110 on every leg priced as independent, and says so: that is a reference point, not a real book price.</p>
        </div>
        <div>
          <ul className="method-rules">
            <li><strong>Graded results.</strong> Parlays you track are graded against final box scores and final scores; each leg becomes a labelled outcome beside the probability the model gave it. Only prop legs train the per-market calibration.</li>
            <li><strong>Calibration per market.</strong> Every market gets its own recalibration, fitted on its graded outcomes and shrunk toward a shared correction. Each market keeps whichever of its own correction, the shared one or the raw model did best on its most recent held-out quarter.</li>
            <li><strong>Memory.</strong> Coach remembers nicknames you teach it or confirm, and your thumbs up or down on its answers. You can make it forget a nickname at any time in Settings › Coach.</li>
          </ul>
          <p>Learning narrows calibration error; it does not see injuries or lineups the data lacks. Probabilities, not certainties. Research only, not betting advice.</p>
          <div className="method-links">
            <RouteLink className="text-button" to={{ page: 'bets' }}>Open My Bets <ArrowRight size={15} aria-hidden="true" /></RouteLink>
            <RouteLink className="text-button" to={{ page: 'how', section: 'accuracy' }} focus="learning-title">See the live learning loop <ArrowRight size={15} aria-hidden="true" /></RouteLink>
          </div>
        </div>
      </div>
    </article>
    <div className="methodology-disclaimer"><Basketball /><div><h3>Built for inquiry, not instructions to wager.</h3><p>Educational and portfolio research only. Probability estimates contain uncertainty, model risk, and data limitations. They are not financial advice or recommendations to gamble.</p></div></div>
  </>;
}
