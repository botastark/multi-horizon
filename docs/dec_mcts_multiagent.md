# Dec-MCTS (single-level) baseline

> Strategy `dec_mcts`. Code: `src/dec_mcts.py`, wired in `src/planner.py` and `src/simulator.py`.

One MCTS tree per agent, with no HLP. It is the **"MCTS without hierarchy"** baseline: it optimises the **same objective as MH-Dec-MCTS** (information gain + `g2_weight` × agent-steps saved toward observing every cell), runs under the same simulated-asynchronous scheduler, and shares intents the same way. A difference between Dec-MCTS and MH-Dec-MCTS can therefore be attributed to the hierarchy, as in Seiler et al., RA-L 2024. This page describes the current code.

## Algorithm (per agent, per step)

1. **Round start** (`begin_round`): a new tree rooted at the agent's position, altitude and (fused) belief, with `remaining_steps = horizon`.
2. **Slices** (`run_iterations`, `slices_per_step` times per step; see `docs/mh_dec_mcts_multiagent.md` §2). Each slice:
   - reads the teammate intents delivered so far and refreshes the teammate mask and g2 context;
   - runs `iterations / slices` iterations:
     - **Selection/expansion**: D-UCT (`ucb_c`, `d_uct.gamma`) down the tree, adding one untried action. Each node holds the belief after an *expected-posterior* update of its footprint.
     - **Node reward** = IG of the node's footprint − `overlap_penalty_weight` × (teammate-mask overlap ratio).
     - **Rollout**: uniformly random admissible actions until `remaining_steps` reaches 0, discounted by `discount_factor`.
     - **g2 term**: `g2_weight · (g2 if the agent stays − g2 after the tree part of the plan)`. As in the paper, only the tree part is used, not the rollout. There are no high-level plans: our g2 prices all remaining work itself, through its covering spanning tree, so the paper's random HL plans are not needed.
     - **Backpropagation of discounted returns**: walking up from the leaf, `G = immediate_reward(node) + discount_factor · G`, plus the g2 term, with D-UCT decay of older statistics.
   - publishes the current plan as its intent: the root child with the highest mean value, then its greedy path.
3. **Action**: the first action of the current plan.

### Changes in this revision

- **Backpropagation.** It used to add `leaf.immediate_reward + rollout` to every node on the path, dropping the first action's own IG after the first expansion. It now uses `DecMCTSNode.backpropagate_return`.
- **g2 term and asynchrony added** (baseline definition).
- **Intent ages** are in simulation steps.

## Coordination

- **Intents** (`DecMCTSIntent`) carry the planned footprints, positions and the `step` they were made at.
- **Delivery** (`DecMCTSCoordinator`): only to agents in communication range; in async mode at slice boundaries. Each receiver keeps the last intent per teammate.
- **Modes**: in `IG` / `IGd` modes nothing is delivered. Intents are shared only in news modes (`*_BS`, `*_BM`), which is the same rule as belief news. `position_sharing` (the "d" in IGd) has no effect on Dec-MCTS.
- **Teammate mask / g2 context**: fresh teammate intents (age ≤ 2 × `threshold_steps`), weighted by `decay_factor ** (age / threshold_steps)` in the mask. In g2, their footprints count as covered and their end positions are roots of the spanning tree.

## Configuration (`configs/strategies/dec_mcts.json` + master `decentralized`)

| Key | Current value | Meaning |
|---|---|---|
| `dec_mcts.horizon` | 5 | tree + rollout depth (steps) |
| `dec_mcts.iterations` | 50 | MCTS iterations per step (split over slices) |
| `dec_mcts.ucb_c` | 1.4 | UCT constant |
| `dec_mcts.discount_factor` | 0.95 | per-step discount |
| `dec_mcts.use_g2` | true | add the g2 coverage term (default true when not set) |
| `dec_mcts.g2_weight` | null | bits per saved step; null = auto (≈ 756) |
| `dec_mcts.timeout` | 5.0 | wall-clock seconds; **only in synchronous mode** (`async.enabled: false`) |
| `dec_mcts.parallel` | 1 | read but **not used** |
| `decentralized.overlap_penalty_weight` | 0.3 | penalty weight for teammate overlap |
| `decentralized.d_uct.gamma` | 0.99 | D-UCT tree discount per iteration |
| `decentralized.d_uct.decay_factor` / `threshold_steps` | 0.9 / 2 | intent ageing (steps) |
| `decentralized.async.enabled` / `slices_per_step` | true / 3 | simulated asynchronous planning |
| `decentralized.radius_multiplier` | 5 | communication range = 5 × 3.125 m (per axis); `-1` = unlimited |

The sweep variants in `configs/strategies/sweeps/dec_mcts_*.json` do not set `use_g2`, so they also run with g2.

## Known limitations

- **The overlap penalty is tiny relative to IG** (at most about 0.3 per overlapping teammate footprint, versus hundreds to thousands of bits). Coordination comes from belief fusion and the g2 term, which treats teammates' planned footprints as covered.
- **Rewards are cached at expansion.** A node's immediate reward uses the teammate mask from the moment it was expanded; later slices only refresh the mask for new expansions (D-UCT discounts the old statistics).
- **The mean-value root choice is noisy at low iteration counts** (7 actions, 50 iterations per step).
