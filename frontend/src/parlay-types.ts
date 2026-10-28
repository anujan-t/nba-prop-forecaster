import type { Model, Stat } from './types';

/** API contract for POST /api/parlay. Probabilities are 0–1 fractions. */

export type Side = 'over' | 'under';

/** One leg: the same settings as a Forecast lab request (minus odds), plus the side being backed. */
export interface ParlayLegRequest {
  dataset_id: string;
  player_id: string;
  stat: Stat;
  line: number; // whole or half point, 0–200
  side: Side;
  model: Model;
  lookback: number; // 10–82
  expected_minutes: number | null;
  rest_days: number | null;
  opponent: string | null;
  use_context: boolean;
}

export interface ParlayRequest {
  legs: ParlayLegRequest[]; // 1–10 legs; each player + stat pair may appear only once (422 otherwise)
  book_odds: number | null; // optional American odds offered for the whole parlay, e.g. 450 or -120
}

export interface FairOdds {
  decimal: number;
  american: number; // rounded to a whole number, e.g. 412 or -138
}

export interface ParlayLeg {
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
}

/** How two legs are linked. Players on different current teams are treated as independent. */
export type PairBasis = 'same_player' | 'teammates' | 'independent';

export interface ParlayPair {
  a: number; // leg index, a < b
  b: number;
  basis: PairBasis;
  games: number; // shared games used for the estimate (0 when independent)
  rank_correlation: number | null; // Spearman correlation of the two stats over shared games
  correlation: number; // latent correlation used in the joint model (after shrinkage), between the two stats; 0 if independent
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
  legs: ParlayLeg[];
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
  method: string; // short plain-English description of the calculation
  warnings: string[]; // de-duplicated warnings for the whole slip
  computed_at: string; // ISO timestamp
}
