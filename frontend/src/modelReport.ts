import type { Stat } from './types';

/**
 * Walk-forward test behind the game simulator (how the default model was chosen). 70 NBA players, every 3rd game of
 * 2024-25 and 2025-26 after each player's 15th, the previous 60 games only; five lines per market around the trailing
 * median (yes/no markets: one). "Old Auto" is the previous default (normal for large counts, Poisson or negative
 * binomial for small ones, empirical for yes/no). Lower is better for both scores.
 */
export const MODEL_REPORT = {
  players: 70,
  games: 8417,
  predictions: 232288,
  period: 'Nov 2024 – Apr 2026',
  overall: { rule_ll: 0.5239, sim_ll: 0.5150, rule_brier: 0.1749, sim_brier: 0.1718 },
  markets: [
    { stat: 'points', n: 12370, rule_ll: 0.5551, sim_ll: 0.5452, rule_brier: 0.1869, sim_brier: 0.1829 },
    { stat: 'rebounds', n: 12350, rule_ll: 0.5652, sim_ll: 0.5593, rule_brier: 0.1905, sim_brier: 0.1888 },
    { stat: 'assists', n: 11655, rule_ll: 0.5352, sim_ll: 0.5219, rule_brier: 0.1766, sim_brier: 0.1730 },
    { stat: 'threes', n: 10784, rule_ll: 0.4885, sim_ll: 0.4773, rule_brier: 0.1606, sim_brier: 0.1569 },
    { stat: 'steals', n: 9053, rule_ll: 0.4295, sim_ll: 0.4286, rule_brier: 0.1384, sim_brier: 0.1381 },
    { stat: 'blocks', n: 8186, rule_ll: 0.3315, sim_ll: 0.3299, rule_brier: 0.1033, sim_brier: 0.1030 },
    { stat: 'turnovers', n: 10611, rule_ll: 0.4728, sim_ll: 0.4686, rule_brier: 0.1543, sim_brier: 0.1529 },
    { stat: 'pts_reb_ast', n: 12400, rule_ll: 0.5632, sim_ll: 0.5512, rule_brier: 0.1898, sim_brier: 0.1853 },
    { stat: 'pts_reb', n: 12393, rule_ll: 0.5633, sim_ll: 0.5539, rule_brier: 0.1899, sim_brier: 0.1863 },
    { stat: 'pts_ast', n: 12384, rule_ll: 0.5647, sim_ll: 0.5521, rule_brier: 0.1907, sim_brier: 0.1859 },
    { stat: 'reb_ast', n: 12386, rule_ll: 0.5700, sim_ll: 0.5639, rule_brier: 0.1930, sim_brier: 0.1909 },
    { stat: 'stocks', n: 10505, rule_ll: 0.4833, sim_ll: 0.4800, rule_brier: 0.1571, sim_brier: 0.1562 },
    { stat: 'field_goals_made', n: 12372, rule_ll: 0.5838, sim_ll: 0.5660, rule_brier: 0.1984, sim_brier: 0.1916 },
    { stat: 'field_goals_attempted', n: 12398, rule_ll: 0.5661, sim_ll: 0.5485, rule_brier: 0.1912, sim_brier: 0.1846 },
    { stat: 'threes_attempted', n: 11854, rule_ll: 0.5555, sim_ll: 0.5378, rule_brier: 0.1884, sim_brier: 0.1815 },
    { stat: 'free_throws_made', n: 10687, rule_ll: 0.5930, sim_ll: 0.5847, rule_brier: 0.2017, sim_brier: 0.1990 },
    { stat: 'free_throws_attempted', n: 11146, rule_ll: 0.5996, sim_ll: 0.5931, rule_brier: 0.2052, sim_brier: 0.2026 },
    { stat: 'offensive_rebounds', n: 9682, rule_ll: 0.4522, sim_ll: 0.4508, rule_brier: 0.1475, sim_brier: 0.1473 },
    { stat: 'defensive_rebounds', n: 12246, rule_ll: 0.5602, sim_ll: 0.5536, rule_brier: 0.1885, sim_brier: 0.1863 },
    { stat: 'fouls', n: 11862, rule_ll: 0.4754, sim_ll: 0.4738, rule_brier: 0.1547, sim_brier: 0.1542 },
    { stat: 'double_double', n: 2482, rule_ll: 0.3184, sim_ll: 0.2937, rule_brier: 0.0886, sim_brier: 0.0888 },
    { stat: 'triple_double', n: 2482, rule_ll: 0.0480, sim_ll: 0.0377, rule_brier: 0.0091, sim_brier: 0.0085 },
  ] as { stat: Stat; n: number; rule_ll: number; sim_ll: number; rule_brier: number; sim_brier: number }[],
  /** Same-player pairs priced jointly: log loss of "both legs win" on 28,464 held-out pairs. */
  joint: { pairs: 28464, simulator: 0.5104, history_copula: 0.5141, independent: 0.5323 },
};
