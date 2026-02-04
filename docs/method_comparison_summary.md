# Method Comparison: Incremental Planner Progression

Summary of the **four baselines** used for method comparison, representing an incremental progression in planning complexity.

## High-level comparison

**Incremental Planning Progression:**

1. **Greedy IG**: Single-step, belief-driven information gain (IG) maximization. Fast, local, no multi-step lookahead. Good baseline for reactive, low-compute agents.

2. **Dec-MCTS**: Single-level multi-step MCTS planning. Optimizes trajectories with UCB tree search, uses overlap penalty and D-UCT staleness discounts. Balances IG and penalties via `overlap_penalty_weight`.

3. **MH-Dec-MCTS**: Two-level hierarchical planning where both HLP and LLP use **full MCTS tree search**. HLP uses MCTS for strategic region allocation, LLP uses MCTS for tactical action selection. Uses g1+g2 reward decomposition (`use_mcts_llp=True`). Full hierarchical baseline.

4. **MH-Dec-MCTS (efficient)**: Optimized variant of (3) where LLP uses **random rollout sampling** instead of full MCTS tree search to reduce computational cost while maintaining performance. HLP still uses full MCTS (`use_mcts_llp=False`). This is the computationally efficient hierarchical baseline.

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
  - `action_strategy`: mh_dec_mcts_both
  - `num_agents`: 4
  - `use_mcts_llp`: True (Both HLP and LLP use **full MCTS tree search**)
  - LLP (low-level planner):
    - `llp_horizon`: 3 (steps)
    - `llp_iterations`: 50
    - Planning method: **UCB tree search** (full MCTS with tree building)
    - `llp_ucb_c`: 1.4
    - `llp_discount_factor`: 0.95
  - HLP (high-level planner):
    - `hlp_horizon`: 10 (regions)
    - `hlp_iterations`: 30
    - Planning method: **UCB tree search** (full MCTS)
    - `hlp_ucb_c`: 1.0
    - `hlp_discount_factor`: 0.98
    - `tile_size`: [50, 50]
    - `hlp_replan_interval`: 1.0
  - `hierarchical_dec_mcts.mode_labels`: [`IG`, `IGd`, `IG_BM`, `IG_BS`, `IGd_BM`, `IGd_BS`]
  - `decentralized.communication_range`: 15.625 (meters)
  - `decentralized.overlap_penalty_weight`: 0.3
  - `correlation_types`: [`adaptive`, `equal`, `biased`]
  - `iters`: [0, 20]

- MH-Dec-MCTS (efficient) (use `mh_dec_mcts` with `use_mcts_llp=False`)
  - `action_strategy`: mh_dec_mcts
  - `num_agents`: 4
  - `use_mcts_llp`: False (LLP uses **random rollout sampling** as optimization)
  - LLP (low-level planner):
    - `llp_horizon`: 3 (steps)
    - `llp_iterations`: 50
    - Planning method: **Random rollout sampling** (optimized, no tree building)
    - `llp_discount_factor`: 0.95
  - HLP (high-level planner):
    - Same as MH-Dec-MCTS (3) above
  - Other settings: Same as MH-Dec-MCTS (3)

## Incremental Design Rationale

The four baselines represent an **incremental progression** in planning sophistication:

| Baseline | Planning Type | Lookahead | Search Method | Relative Complexity |
|----------|---------------|-----------|---------------|---------------------|
| 1. Greedy IG | None | 1 step | Enumerate actions | Lowest |
| 2. Dec-MCTS | Single-level | Multi-step | UCB tree search | Medium |
| 3. MH-Dec-MCTS (full) | Hierarchical | HLP: Multi-region<br>LLP: Multi-step | HLP: MCTS<br>LLP: **MCTS** | Highest |
| 4. MH-Dec-MCTS (efficient) | Hierarchical | HLP: Multi-region<br>LLP: Multi-step | HLP: MCTS<br>LLP: **Random rollout** | High |

**Design Rationale (3) → (4):**
- Baseline **(3)** establishes the full hierarchical approach with both planners using MCTS tree search
- Baseline **(4)** optimizes (3) by replacing LLP's tree search with **random rollout sampling**
  - **Goal**: Reduce computational cost while maintaining similar performance
  - **Trade-off**: (4) is faster per iteration but may find slightly lower-quality short-horizon plans
  - **Hypothesis**: Random rollout is sufficient for LLP since HLP provides strategic guidance via g₂

## Plot provenance (latest run)
- Plot file: `plots/method_comparison_r_5_pairwise_adaptive.png`
- CLI / selection used: `--compare-methods --radius 5 --pairwise adaptive --num-agents 4`
- Meanings:
  - `radius=5` corresponds to Gaussian radius 5 used in trial generation.
  - `pairwise=adaptive` filters trials that used adaptive pairwise correlations.
  - `num-agents=4` selects the 4-agent trials for all methods.

## Notes and recommended follow-ups
- Suggested experiments:
  1. **Compare (3) vs (4)**: Validate that random rollout LLP (4) achieves similar performance to full MCTS LLP (3) with reduced computational cost. This tests the optimization hypothesis.
  2. Sweep `llp_iterations` in baseline (4) to find minimum iterations needed for good performance with random rollout.
  3. Sweep `hlp_iterations` (e.g., 10, 30, 60) to measure HLP effectiveness vs compute.
  4. Vary `decentralized.overlap_penalty_weight` to see trade-offs between coverage and redundancy.
  5. Compare all 4 baselines under limited communication (lower `communication_range`).
