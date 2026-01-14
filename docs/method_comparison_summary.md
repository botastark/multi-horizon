# Method Comparison: Dec-MCTS vs MH-Dec-MCTS vs Greedy IG

Summary of the three planners used for the method-comparison plot saved under `plots/`.

## High-level comparison
- **Greedy IG**: Single-step, belief-driven information gain (IG) maximization. Fast, local, no multi-step lookahead. Good baseline for reactive, low-compute agents.
- **Dec-MCTS (single-level)**: Multi-step centralized/decentralized MCTS planning over a horizon. Optimizes trajectories with UCB exploration, uses overlap penalty and D-UCT staleness discounts. Balances IG and penalties via `overlap_penalty_weight`.
- **MH-Dec-MCTS (hierarchical LLP+HLP)**: Two-level planning: a short-horizon LLP (fast MCTS/greedy) that uses g1+g2 rollouts and a longer-horizon HLP that plans regions (marginal g2). Removes alignment bonus; HLP runs its own MCTS and provides HL intents to LLP. Designed to capture long-term coordination with manageable compute.

## Tunable parameters (configs used)

- Greedy IG (configs/benchmark_greedy_ig.json)
  - `action_strategy`: greedy_ig
  - `num_agents`: 4
  - `greedy_ig.overlap_penalty_weight`: 0.0 (pure belief-based IG)
  - `greedy_ig.mode_labels`: [`IG`, `IGd`, `IG_BM`, `IG_BS`, `IGd_BM`, `IGd_BS`]
  - `decentralized.radius_multiplier`: 5 (communication range calculated dynamically)
  - `correlation_types`: [`adaptive`, `equal`, `biased`]
  - `iters`: [0, 20]

- Dec-MCTS (configs/benchmark_dec_mcts.json)
  - `action_strategy`: dec_mcts
  - `num_agents`: 4
  - `dec_mcts.horizon`: 10 (steps)
  - `dec_mcts.iterations`: 100 (MCTS rollouts)
  - `dec_mcts.ucb_c`: 1.4
  - `dec_mcts.discount_factor`: 0.95
  - `dec_mcts.mode_labels`: [`IG`, `IGd`, `IG_BM`, `IG_BS`, `IGd_BM`, `IGd_BS`]
  - `dec_mcts.timeout`: 5.0 (seconds)
  - `dec_mcts.parallel`: 1
  - `decentralized.communication_range`: 15.625 (meters)
  - `decentralized.overlap_penalty_weight`: 0.3
  - `decentralized.d_uct.decay_factor`: 0.9
  - `correlation_types`: [`equal`, `biased`, `adaptive`]
  - `iters`: [0, 20]

- MH-Dec-MCTS (configs/benchmark_mh_dec_mcts.json)
  - `action_strategy`: mh_dec_mcts
  - `num_agents`: 4
  - LLP (low-level planner):
    - `llp_horizon`: 3 (steps)
    - `llp_iterations`: 50
    - `llp_ucb_c`: 1.4
    - `llp_discount_factor`: 0.95
  - HLP (high-level planner):
    - `hlp_horizon`: 10 (regions)
    - `hlp_iterations`: 30
    - `hlp_ucb_c`: 1.0
    - `hlp_discount_factor`: 0.98
    - `tile_size`: [50, 50]
    - `hlp_replan_interval`: 1.0
  - `hierarchical_dec_mcts.mode_labels`: [`IG`, `IGd`, `IG_BM`, `IG_BS`, `IGd_BM`, `IGd_BS`]
  - `decentralized.communication_range`: 15.625 (meters)
  - `decentralized.overlap_penalty_weight`: 0.3
  - `correlation_types`: [`adaptive`, `equal`, `biased`]
  - `iters`: [0, 20]

## Plot provenance (latest run)
- Plot file: `plots/method_comparison_r_5_pairwise_adaptive.png`
- CLI / selection used: `--compare-methods --radius 5 --pairwise adaptive --num-agents 4`
- Meanings:
  - `radius=5` corresponds to Gaussian radius 5 used in trial generation.
  - `pairwise=adaptive` filters trials that used adaptive pairwise correlations.
  - `num-agents=4` selects the 4-agent trials for all methods.

## Notes and recommended follow-ups
- Suggested experiments:
  1. Sweep `hlp_iterations` (e.g., 10, 30, 60) to measure HLP effectiveness vs compute.
  2. Vary `decentralized.overlap_penalty_weight` to see trade-offs between coverage and redundancy.
  3. Compare MH-Dec-MCTS vs Dec-MCTS under limited communication (lower `communication_range`).
