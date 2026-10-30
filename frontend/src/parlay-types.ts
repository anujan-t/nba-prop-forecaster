import type { InjuryEntry, Model, Stat } from './types';

/** API contract for POST /api/parlay. Probabilities are 0–1 fractions. */

export type Side = 'over' | 'under';
export type GameMarket = 'moneyline' | 'spread';

/** One prop leg: the same settings as a Forecast lab request (minus odds), plus the side being backed. */
export interface ParlayLegRequest {
  kind?: 'player'; // omitted by slips saved before moneylines and spreads
  dataset_id: string;
  player_id: string;
  stat: Stat;
  line: number; // whole or half point, 0–200 (0.5 for yes/no markets)
  side: Side;
  model: Model;
  lookback: number; // 10–82
  expected_minutes: number | null;
  rest_days: number | null;
  opponent: string | null;
  use_context: boolean;
}

/** A moneyline or point spread on one team in one game: ESPN's event id and the game's US Eastern date. */
export interface GameLegRequest {
  kind: 'game';
  game_id: string;
  date: string; // YYYY-MM-DD
  team: string; // the team backed, e.g. "DEN"
  opponent: string;
  home: boolean; // whether `team` plays at home
  market: GameMarket;
  line: number | null; // the team's spread, negative when favored (whole or half point, ±60); null for a moneyline
}

export type AnyLegRequest = ParlayLegRequest | GameLegRequest;

export interface ParlayRequest {
  legs: AnyLegRequest[]; // 1–10 legs; one per player + stat, one moneyline and one spread per game (422 otherwise)
  book_odds: number | null; // optional American odds offered for the whole parlay, e.g. 450 or -120
  simulations?: number; // games simulated per player for legs on the game simulator
}

export interface FairOdds {
  decimal: number;
  american: number; // rounded to a whole number, e.g. 412 or -138
}

export interface ParlayLeg {
  kind?: 'player';
  index: number; // 0-based, same order as the request
  player: { id: string; name: string; team: string };
  stat: Stat;
  line: number;
  side: Side;
  model: Model; // requested
  resolved_model: Model;
  sample_size: number;
  mean: number; // model mean for the stat
  probability: number; // P(leg wins outright) — identical to the Forecast lab over/under probability
  push: number; // P(stat lands exactly on the line); 0 for half-point lines
  hit_rate: { hits: number; games: number }; // how often this leg would have won in the sample games
  fair_odds: FairOdds | null; // null when probability is 0 or 1
  simulations: number | null; // simulated games behind this leg, when its model is the game simulator
  injury?: InjuryEntry | null; // the player's entry on the last saved injury report
}

/** The book's listed price for one side of a game market, against the model. */
export interface BookSide {
  odds: number; // American
  implied: number; // break-even probability, vig included
  no_vig: number | null; // the book's own chance once the two sides' margin is removed
  vig: number | null;
  line: number | null; // the spread the price is for; null for a moneyline
  provider: string | null;
  model: number;
  edge: number; // model chance (pushes excluded) minus break-even
  expected_roi: number; // expected profit per 1 unit if the model is right; pushes refund
  verdict: 'value' | 'fair' | 'house';
}

/** A moneyline or spread leg priced by the team model. */
export interface GameParlayLeg {
  kind: 'game';
  index: number;
  game_id: string;
  date: string;
  team: string;
  opponent: string;
  home: boolean;
  market: GameMarket;
  line: number | null;
  probability: number; // P(leg wins)
  push: number; // P(margin lands exactly on a whole-number spread)
  fair_odds: FairOdds | null;
  win_probability: number; // the team's chance to win the game
  margin: number; // the team's expected margin, in points
  sigma: number;
  fair_spread: number; // the model's spread for the team, to the half point
  back_to_back: boolean;
  opponent_back_to_back: boolean;
  rating: number; // points better than an average team on a neutral floor
  opponent_rating: number;
  home_court: number;
  games: number; // games behind the ratings
  as_of: string;
  early_season: boolean; // fewer than 10 games this season for one of the teams
  book: BookSide | null; // when the book lists this market at this line
  state: 'pre' | 'in' | 'post' | null;
}

export type AnyParlayLeg = ParlayLeg | GameParlayLeg;

/** How two legs are linked. Players on different current teams, and different games, are treated as independent. */
export type PairBasis = 'same_player' | 'teammates' | 'same_game' | 'team_player' | 'independent';

export interface ParlayPair {
  a: number; // leg index, a < b
  b: number;
  basis: PairBasis;
  games: number; // shared games used for the estimate (0 when independent)
  rank_correlation: number | null; // Spearman correlation over shared games
  correlation: number; // latent correlation used in the joint model (after shrinkage); 0 if independent
  effect?: number; // the same correlation signed for the sides picked: positive means the legs tend to win together
  source: 'simulation' | 'history' | 'model' | 'none'; // simulation: same-player legs; model: legs on the same game
  simulations: number | null;
  note: string; // one plain-English sentence
}

/** Linked legs (same player or current teammates) checked against shared past games. */
export interface ParlayGroupHistory {
  legs: number[]; // leg indices in the group (2 or more)
  games: number; // shared games where every player in the group played
  hits: number; // games in which every leg of the group would have won
  model_probability: number; // the model's joint probability for this group alone
}

export interface ParlayResult {
  legs: AnyParlayLeg[];
  pairs: ParlayPair[]; // every pair of legs, a < b
  probability: number; // P(every leg wins), correlation-adjusted
  independent_probability: number; // product of leg probabilities
  correlation_lift: number | null; // probability / independent_probability; null when the product is 0
  no_loss_probability: number | null; // P(no leg loses) when a whole-number line can push; null otherwise
  fair_odds: FairOdds | null; // fair price for the correlated probability; null when probability is 0 or 1
  book: null | {
    odds: number; // the American odds supplied
    implied: number; // implied probability of those odds
    edge: number; // probability − implied
    expected_roi: number; // expected profit per 1 unit staked; pushes counted as losses
  };
  history: ParlayGroupHistory[];
  calibration: null | {
    probability: number; // joint probability with each leg recalibrated by the learning loop (same correlations)
    legs: number[]; // recalibrated leg probabilities, in leg order
    samples: number; // graded predictions behind the calibration
    note: string;
  }; // null until calibration is active
  simulations: number | null; // simulated games per player behind simulator legs; null when no leg uses the simulator
  method: string; // short plain-English description of the calculation
  warnings: string[]; // de-duplicated warnings for the whole slip
  computed_at: string; // ISO timestamp
}
