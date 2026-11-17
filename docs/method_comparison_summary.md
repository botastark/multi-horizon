# Method comparison: the four baselines

The four strategies form an incremental progression. All of them run in the same simulator: 50 × 50 m Gaussian field, 0.125 m cells, 4 agents, `start_position: corner`. The world advances in synchronous steps (all agents move, then observe, then news is fused). Dec-MCTS and MH-Dec-MCTS **plan** with the same simulated-asynchronous scheduler: concurrent planners with intent exchange between `slices_per_step` slices; see `docs/mh_dec_mcts_multiagent.md` §2.

| # | Strategy | Lookahead | Search | Objective | Coordination signal |
|---|---|---|---|---|---|
| 1 | `greedy_ig` | 1 step | enumerate actions | IG | belief news; IoU discount in IGd modes |
| 2 | `dec_mcts` | `horizon` steps | one D-UCT tree | IG + g2 (coverage steps) | belief + coverage news; teammate intents |
| 3 | `mh_dec_mcts_full` | LLP `llp.horizon` steps + HLP `hlp.horizon` regions | LLP: D-UCT tree; HLP: D-UCT MCTS over region sequences | IG + g2, with HL plans | belief + coverage news; LL/HL intents |
| 4 | `mh_dec_mcts_efficient` | same as 3 | LLP: random rollouts; HLP: same as 3 | same as 3 | same as 3 |

Steps: 1 → 2 adds multi-step planning and the long-horizon coverage objective; 2 → 3 adds **only the hierarchy** (Dec-MCTS already uses the same g2, as in the paper's comparison); 3 → 4 replaces LLP tree search with random rollouts.

## What the mode label controls, per method

| Mode | Greedy IG | Dec-MCTS | MH-Dec-MCTS |
|---|---|---|---|
| `IG` | nothing shared | nothing shared | nothing shared (own coverage only) |
| `IGd` | + IoU discount vs teammates' current positions | same as `IG` | same as `IG` |
| `IG_BS` / `IG_BM` | + belief news | + belief and coverage news, intents in range | + belief and coverage news, LL/HL intents in range |
| `IGd_BS` / `IGd_BM` | + news + IoU discount | same as `IG_BS` / `IG_BM` | same as `IG_BS` / `IG_BM` |

Notes:

- The "d" (position sharing) only changes greedy IG. For Dec-MCTS and MH, `IG_BM` and `IGd_BM` are the same experiment.
- `master_config.json` currently runs `mode_labels: ["IGd_BM", "IG_BS"]` with `limited_testing: true`. That makes `IG_BS` use R16 (whole field, effectively unlimited) and `IGd_BM` use R5 (15.625 m).
- Greedy `IG_BS` and `IGd_BM` use a separate, reference-compatible code path; see `docs/greedy_ig_multiagent.md`.

## Current strategy parameters (`configs/strategies/`)

**Greedy IG** (`greedy_ig.json`): `radius_multiplier` 5. `overlap_penalty_weight` is in the file but ignored (hardcoded 0).

**Dec-MCTS** (`dec_mcts.json`):

- `horizon` 5, `iterations` 50 per step, `ucb_c` 1.4, `discount_factor` 0.95
- `use_g2` true, `g2_weight` null (auto)
- `timeout` 5.0 s (wall clock; synchronous mode only)
- `overlap_penalty_weight` 0.3
- `d_uct`: `gamma` 0.99, intent decay 0.9, `threshold_steps` 2

**MH-Dec-MCTS full / efficient** (`mh_dec_mcts_full.json`, `mh_dec_mcts_efficient.json`):

- `use_g2` true, `g2_weight` null (auto ≈ 756 bits per saved step)
- LLP: horizon 3, iterations 30, discount 0.95; `ucb_c` 1.4 (full only)
- HLP: horizon 8 regions, iterations 30 per step, `ucb_c` 1.0, `tile_size` [25, 25], `replan_interval` 2 **steps** (new search cycle), `candidate_regions` 6
- shared: `async.enabled` true, `slices_per_step` 3

Full parameter semantics: `docs/mh_dec_mcts_multiagent.md` §8 and `configs/README.md`.

## Objectives in one line each

- **Greedy**: maximises expected information gain (bits).
- **Dec-MCTS**: information gain + `g2_weight` × agent-steps saved toward observing every cell at least once (no high-level plans).
- **MH-Dec-MCTS**: the LLP maximises the same quantity, with g2 evaluated using its own and teammates' HL region plans. The HLP maximises agent-steps saved by its region sequence, starting from the end of its own LL plan. Leaving small uncovered spots behind costs the detour to come back, so the planners tend to finish areas before moving on.

## Sweeps

`sweep.json` runs the files in `configs/strategies/sweeps/`. **Several file names do not match their contents**; each run is labelled by the `experiment.name` inside the file, so saved results are correct, but don't trust the file names:

| File | Actually contains |
|---|---|
| `mh_eff_fast_replan_hlp5_rep1.json` | `mh_eff_no_g2` (HLP 8, replan 2) |
| `mh_eff_llp_boost_h4_i50.json` | LLP h3/i15, tile 50 |
| `mh_full_balanced_standard.json` | `mh_full_no_g2` |
| `mh_full_fast_tactical_llp3_i40.json` | `mh_full_standard` (i30) |
| `mh_full_ucb_tune_llp1.8.json` | LLP `ucb_c` 2.0 (`mh_full_explore_ucb2`) |

`tests/test_config_propagation.py` still expects the file-name values, so it reports failures for these files.

## Results

Results from runs before the MH/Dec-MCTS fixes in this revision were produced with a different MH objective, and with Dec-MCTS backpropagation that dropped the first step's reward. They are not comparable with new runs.

**Preliminary check of the g2 coverage objective** (1 agent, Gaussian r = 5, start at a corner, 100 steps, seeds 42–45, persistent planner, LLP h3/i30, HLP h8/i30, tiles 25×25; script-level run, not the full pipeline):

| Variant | Step at which 100 % of cells were observed | Steps with no new cell before that | MSE at step 100 |
|---|---|---|---|
| MH-efficient, g2 off | 62, 67, 79, 61 (mean 67) | 15, 31, 35, 18 | 0.012 |
| MH-efficient, g2 on | 48, 46, 55, 39 (mean 47) | 14, 13, 23, 8 | 0.019 |
| MH-full, g2 off | 67, 63, 69, 64 (mean 66) | 24, 22, 29, 21 | 0.012 |
| MH-full, g2 on | 36, 55, 52, 41 (mean 46) | 7, 17, 10, 9 | 0.019 |

g2 reaches full coverage about 20 steps earlier with fewer wasted steps. The final map error is higher, because the objective favours observing new cells over re-observing known ones at better altitudes.

After the paper-style bottom-up change (the HLP plans from the end of its own LL plan; same setup, synchronous), g2 on: MH-efficient 34, 43, 33, 53 (mean 41) and MH-full 59, 62, 32, 42 (mean 49). With 4 seeds, the MH-full change is within noise. The simulated-asynchronous scheduler has not been benchmarked over many seeds yet.
