# Greedy IG baseline

> Strategy `greedy_ig`. Code: `src/greedy_ig_planner.py`, `src/planner.py::greedy_ig_decision`, `src/simulator.py::_select_greedy_baseline_actions`.

Single-step lookahead. Each agent scores every admissible action by the sensor-aware information gain of the footprint it would observe next, `IG = H(p) − E[H(posterior)]` summed over cells, and takes the best one. There is no planning horizon and no intent sharing. Coordination comes from belief fusion (news) and, in IGd modes, from discounting overlap with teammates.

## Two code paths, depending on the mode

| Mode | Code path | Action choice |
|---|---|---|
| `IG_BS` | `Simulator._select_greedy_baseline_actions` | selfish argmax; ties broken by fixed priority (left, right, front, back, up, down, hover) |
| `IGd_BM` | `Simulator._select_greedy_baseline_actions` | agents decide in a random order (RNG seed 17). Each action's IG is multiplied by `1 − IoU` with every teammate in range (per-axis distance ≤ `radius_multiplier` × step), using the teammates' **already chosen next** positions where available |
| `IG`, `IGd`, `IG_BM`, `IGd_BS` | `planning.greedy_ig_decision` → `GreedyIGPlanner` | argmax; IGd variants multiply IG by Π(1 − IoU) against teammates in comm range at their **current** positions (null-policy assumption) |

The first path reproduces the reference (PA) implementation for the two modes used in the main comparison. The two paths are not equivalent for the other modes.

## What is shared

- **Belief news** (`*_BS`, `*_BM`): fused into each neighbour's belief every step; see `docs/mh_dec_mcts_multiagent.md` §3 for the BS/BM delivery rules.
- **Positions** (`IGd*`): used only for the IoU discount. Positions are refreshed every step; no wall-clock age filter is applied.

## Configuration

| Key | Value | Notes |
|---|---|---|
| `greedy_ig.overlap_penalty_weight` | 0.0 | **ignored**: the planner hardcodes 0 (pure IG, as in the paper) |
| `decentralized.radius_multiplier` | 5 | comm range 5 × 3.125 m = 15.625 m, per axis; `-1` = unlimited |
| `multi_agent.shared_observation_rng` | (baseline configs) | share one observation RNG across agents, for PA-compatible runs |

The greedy tie-breaking RNG is seeded with 0 (not the per-iteration seed).

### `limited_testing: true` (master config)

`src/main.py::_apply_mh_mode_label` overrides the radius per mode, for every strategy:

- `IG_BS` → `radius_multiplier = 16` (16 × 3.125 m = 50 m covers the whole field, so this is effectively unlimited; folders are named `commR16`)
- `IGd_BM` → `radius_multiplier = 5` (15.625 m)

## Limitations

- Myopic: no lookahead, so it can leave small uncovered spots and come back for them later.
- No explicit collision avoidance.
