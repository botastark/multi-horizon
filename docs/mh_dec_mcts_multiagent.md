# MH-Dec-MCTS: Multi-Horizon Decentralised MCTS

> Strategies `mh_dec_mcts_full` (LLP = MCTS tree search) and `mh_dec_mcts_efficient` (LLP = random rollouts).
> Code: `src/hierarchical_dec_mcts.py`, `src/g2_evaluator.py`; wiring in `src/planner.py` and `src/simulator.py`.
> Tests: `tests/test_mh_coverage_g2.py`.

This page describes what the code does **now**. Everything below is measured in **simulation steps** and MCTS iterations; nothing in the planners depends on wall-clock time. Planning is **simulated-asynchronous** (§2), following Seiler et al., RA-L 2024: each robot runs two concurrent planners, an LLP and an HLP, that keep exchanging intents.

---

## 1. What the planner is trying to do

Each agent runs two planners every step:

| Level | Decides | Horizon | Objective |
|---|---|---|---|
| **HLP** (high level) | which regions (tiles) to cover next, in which order | `hlp.horizon` regions | agent-steps **saved** toward "every cell observed at least once" |
| **LLP** (low level) | the next primitive actions (front/back/left/right/up/down/hover) | `llp.horizon` steps | information gain **+** `g2_weight` × agent-steps saved |

The HLP's main goal is **100 % coverage without coming back for missed small spots**:

- **Coverage** means *observed at least once*, by any agent, at any altitude. It is not "belief decided".
- **Not coming back** is a soft cost: leaving uncovered cells behind is charged the steps needed to return for them later. There is no hard rule forcing an agent to stay in a region.

---

## 2. One simulation step (simulated asynchrony)

`Simulator.run()` executes all agents together: all plan, then all move, then all observe. With `decentralized.async.enabled: true` (the default) planning uses `Simulator._select_agent_actions_async`:

```
for step in range(n_steps):
    metrics + logging
    # --- planning (simulated asynchronous) ---
    for each agent: begin_async_step()
        HLP: keep its search tree, or start a new cycle (replan due: every
             replan_interval steps / target region finished / no plan)
        LLP: start a fresh round (new tree) at the current state
    intent buses -> deferred mode
    for slice in range(slices_per_step):            # default 3
        for each agent (any order; result is order-independent):
            read intents delivered so far (teammates' LL/HL, own LL -> HLP)
            HLP: hlp.iterations / slices more MCTS iterations, publish current best HL plan
            LLP: llp.iterations / slices more iterations, publish current best LL plan
        flush buses: deliver this slice's intents (range-limited) to receivers
    each agent executes the first action of its LLP's current best plan
    # --- world ---
    all agents move; observe; update local belief (OG + LBP); mark own footprint
    news fusion (BS/BM modes only) + coverage news
```

- Within a slice every planner sees the intents published at the end of the previous slice. The planners therefore run concurrently, as in the paper, but the outcome is **deterministic** and independent of agent order (`test_async_is_deterministic_and_order_independent`).
- Search trees persist while the intents they depend on keep changing, so they use the paper's **D-UCT** (§7):
  - LLP and Dec-MCTS trees: across the slices of one step. They then restart, because the first action has been executed (receding horizon).
  - HLP tree: across its whole cycle (`hlp.replan_interval` steps), so the HLP accumulates `replan_interval × hlp.iterations` iterations and effectively runs at a slower rate.
- Budgets are per **step**: `llp.iterations`, `hlp.iterations` and `dec_mcts.iterations` are the iterations each planner gets per simulation step. Dec-MCTS's wall-clock `timeout` is not used in this mode.
- With `async.enabled: false`, each agent instead calls `HierarchicalDecMCTSPlanner.plan()` once per step, in agent order: HLP (full budget on replan), then LLP (full budget), then broadcast. Later agents then see earlier agents' intents from the same step.

---

## 3. What is shared, by mode

The mode label (`mode_labels`) controls all information exchange. The same rules apply to Dec-MCTS.

| Mode | Belief news | Coverage knowledge (observed-at-least-once) | LL/HL intents |
|---|---|---|---|
| `IG`, `IGd` | none | own observations only | **none** (IntentBus disabled) |
| `IG_BS`, `IGd_BS` | broadcast news to agents in range | same channel as news | to agents in range |
| `IG_BM`, `IGd_BM` | per-neighbour news to agents in range | same channel as news | to agents in range |

Details:

- **Range** is `MultiAgentCoordinator.get_neighbors_in_range` (`radius_multiplier` × 3.125 m, checked per axis in x/y/z; `-1` = unlimited).
- **Intents** go to the agents in range of the sender. In async mode they are delivered at the end of each planning slice (deferred bus). A receiver keeps the last intent it got from each teammate. When the teammate moves out of range, that intent stays in the inbox and ages (§7).
- **Coverage news** (`MultiAgentMapper.share_coverage_news`) mirrors belief news exactly. News carries only the sender's **own** footprints (no relaying):
  - **BS**: one pending buffer per sender, cleared once it has been delivered to anyone. Agents out of range at that moment never get it.
  - **BM**: one pending buffer per (sender, receiver), delivered and cleared when that receiver is in range. The backlog arrives later.
- **The "d" in IGd has no effect on MH-Dec-MCTS** (or Dec-MCTS). `position_sharing` is only used by greedy IG. For MH, `IG_BM` and `IGd_BM` are identical runs.
- **`limited_testing: true`** in the master config overrides the radius per mode: `IG_BS` → R16 (covers the whole 50 m field, effectively unlimited) and `IGd_BM` → R5 (15.625 m). See `src/main.py::_apply_mh_mode_label`.

---

## 4. g2: remaining coverage effort in steps (`src/g2_evaluator.py`)

`CoverageStepModel` is built from the agent's **uncovered** mask (the inverse of its coverage knowledge).

Geometry (default Gaussian field: 400×400 cells of 0.125 m; everything priced at **minimum altitude**, altitude changes ignored):

| Quantity | Value | Meaning |
|---|---|---|
| `step_cells` | 25 | one lateral move = 3.125 m = 25 cells; moves are 4-connected |
| block | 25×25 cells | the field as a 16×16 grid of one-move blocks; a block is *open* if any cell in it is unobserved |
| lattice positions | block corners | where an agent stands; at minimum altitude (5.41 m, footprint 50×50 cells) it sees the 2×2 blocks around it |

**Remaining work for a set of open blocks** is the length of a tour that visits enough lattice positions to see every open block:

1. Pick covering positions greedily (the position that sees the most open blocks first).
2. Connect them with a minimum spanning tree in true step distance (Manhattan / 25). The tree is rooted at the agents' positions (multi-root, Prim's algorithm in NumPy).

**Team estimate** g2 (sum of agent-steps; lower is better):

1. **Claimed work.** Every agent walks its HL region sequence from its start. For each still-open region it pays `reach` (moves until its footprint can see the region's open bounding box) plus `inner` (tour over the region's own covering positions; 0 for a one-block region). It then stands where it arrived. A region claimed twice is paid twice.
2. **Unclaimed work.** All open blocks in regions nobody claimed are priced by the covering-positions spanning tree, rooted at every agent's end position.

Why this form:

- **Covering the middle of an uncovered strip costs nothing extra.** Covered ground can still be crossed at its true distance.
- **Covering an end of a strip, or a whole area, lowers g2.**
- **An isolated missed spot adds the detour needed to come back for it.** That detour is the "come back later" cost that makes the planners finish areas before moving on.

Two earlier drafts were rejected:

- **Independent trip per region from the nearest agent.** This pulls agents toward the median of the remaining work.
- **Area / swath sweep + zone-to-zone spanning tree.** This prices one-tile-wide strips at half their real cost. Covering a strip's middle then *raised* g2, and the LLP learned to hover. It was found by tracing a run and is pinned by `test_g2_covering_part_of_a_strip_never_costs_more`.

Sanity values (checked in `tests/test_mh_coverage_g2.py`):

- Whole field open, one agent at a corner: **g2 = 128** (8 strips × 16 moves, the lawnmower count).
- A one-tile-wide strip along an edge: 15. Covering its middle leaves 15; covering one end gives 11.
- Half the field open plus a small hole far inside the done half: **+8** for the hole.
- Random states: covering a footprint lowers g2 on average. It raised it in 7 of 300 cases, by at most 3 moves (greedy cover and MST are not exactly monotone).

Inputs per agent:

- Its own coverage mask.
- Footprints of fresh teammate LL intents, which are treated as "will be covered".
- Teammates' end positions (last state of their LL intent) and their HL sequences.

Teammate intents older than `2 × d_uct.threshold_steps` are ignored here.

If no coverage mask is passed (standalone use, e.g. the smoke test), cells whose belief is still exactly 0.5 count as unobserved. This is only exact without LBP smoothing.

Cost: about 1–5 ms per evaluation (worst case with the whole field open).

---

## 5. HLP: region sequence (`HighLevelPlanner`)

- **Regions**: `tile_size` tiles; ids are row-major. With the default `[25, 25]` there are 256 regions of 3.125 m. Footprints are always multiples of 25 cells, so a region is usually either fully observed or untouched.
- **Bottom-up input (paper Alg. 2)**: the HLP plans from the **end of this agent's own latest LL plan**, and the cells that plan will observe count as covered. In async mode this is the LLP's current best plan at the start of each slice. At the first slice of a step it is the previous step's plan, whose first action has already been executed. Without an LL plan (step 0), it plans from the current position. If the own LL plan will finish the current target region, a new HLP cycle starts at the next step.
- **Value of a sequence**: `g2(own sequence = []) − g2(own sequence = seq)`, the agent-steps the team saves when this agent commits to `seq`. It is usually ≤ 0, because claiming a region costs its reach while leaving it to the spanning tree is a lower bound. The search therefore effectively picks the cheapest order, and the absolute value is not meaningful on its own.
- **Search**: anytime MCTS over **ordered** sequences of length `hlp.horizon`, with D-UCT statistics. The tree key is the tuple of regions chosen so far. Each `run_iterations(n)` re-reads the context (coverage, own LL plan, teammate intents), re-scores the incumbent and the greedy sequence, and grows the tree by n iterations. The published intent is the best-valued sequence under the current context.
  - Actions at a node are the `hlp.candidate_regions` uncovered regions with the fewest reach steps from the position after the sequence so far (ties: more uncovered cells first).
  - Rollout fills up the sequence greedily with the nearest uncovered region.
  - The plain greedy sequence is always evaluated too, so MCTS can only improve on it.
  - UCB uses `hlp.ucb_c`; values are in steps.
- **New cycle (fresh tree)** every `hlp.replan_interval` steps, or as soon as the current target region has no uncovered cells left in this agent's knowledge (including cells its own LL plan will cover), or when there is no plan. Between cycles the tree keeps growing, and the plan can still change from slice to slice (anytime).
- **HL intent fields**:
  - `region_sequence`
  - `eta_sequence`: cumulative steps until each region is reached
  - `completion_sequence`: cumulative steps until each region is fully observed
  - `score_sequence`: steps spent on each region
  - `value`: steps saved
- `current_region_scores` (used by the viewer) is now the **fraction of each region observed at least once**.

If `use_g2` is false, the HLP still plans and broadcasts, but its plan does not influence the agent's own actions.

---

## 6. LLP: actions (`LowLevelPlanner`)

Reward of a candidate action sequence (`_simulate_trajectory`):

```
r = Σ_k discount^k · IG_k  (×teammate mask)   +   g2_weight · (g2_baseline − g2_after)      # second term only if use_g2
```

- **g1 (IG)** is sensor-aware: `H(p) − E[H(posterior)]` per cell, using the noise of that altitude. It is evaluated on the current belief, which is **not** updated along the rollout, so a footprint visited twice counts twice.
- **Teammate mask.** For each fresh teammate LL intent of age `a`, its first `a` footprints have already been executed and are skipped. The k-th remaining footprint gets weight `w_k = intent_discount^k · d_uct.decay^(a/threshold_steps)`, and IG inside it is multiplied by `(1 − w_k)`. A teammate's next footprint from a fresh intent therefore gets zero IG; later, less certain footprints are suppressed less.
- **g2 term.** `g2_baseline` is g2 if the agent stays put, and `g2_after` is g2 after the trajectory. Their difference is the number of steps the trajectory saves: moving toward the next HL region, covering never-observed cells, and closing small leftover spots.
- **`g2_weight`** converts steps into bits. With `null` it is set automatically to the IG of one fresh step at minimum altitude: (new cells per lateral move at minimum altitude, 50 × 25 = 1250) × (1 − cH(0.5)) ≈ 1250 × 0.60 ≈ 756 bits per step.
- **Search** is an anytime round per step (`begin_round`, then `run_iterations` per slice). Each slice re-reads the context (teammate mask, own HL plan, g2 context) and re-scores the incumbent plan before searching more:
  - `efficient`: random rollouts of length `llp.horizon` over admissible actions.
  - `full`: D-UCT tree search (`llp.ucb_c`, `d_uct.gamma`).
  - In both, the published plan is the best-scoring sampled trajectory. The paper instead extracts the greedy tree path.
- The **LL intent** carries `action_sequence`, `state_sequence` (world x, y, alt), `footprint_sequence` (grid bounds), `ig_sequence` and `step`.

---

## 7. D-UCT and intent ageing

Two different mechanisms; both are configured in `decentralized.d_uct`:

1. **D-UCT tree discounting** (`gamma`, default 0.99; the paper uses 0.9999 with far larger budgets). The paper's Eq. 4–5 (`src/d_uct.py`) applies to the LLP, HLP and Dec-MCTS trees. Each episode's weight decays by `gamma` per later iteration of the same tree:

   ```
   C_t(s) = Σ_u γ^(t−u) 1_s(u)
   D-UCT(s, s') = V_t(s')/C_t(s') + c·sqrt(ln C_t(s) / C_t(s'))
   ```

   This discounts statistics collected under older intents.

2. **Intent ageing.** Intents carry the `step` at which they were made (HL intents are re-stamped when re-published):

   ```
   age        = current_step − intent.step
   weight     = decay_factor ** (age / threshold_steps)       # teammate mask
   ignored if   age > 2 × threshold_steps
   ```

   This matters when a teammate is out of range and its last intent ages in the receiver's inbox. The old key `threshold_sec` is still read and its value is interpreted as steps.

---

## 7b. Step plot: the "planner decision" panel

With `enable_plotting: true`, column 4 of each agent's row in `plots/<iter>/steps/step_<t>.png` shows what the planner saw and decided at that step (`viewer._draw_mh_decision_panel`, data from `experiment_utils.extract_mh_decision`):

- **Background**: this agent's coverage knowledge at planning time: grey = observed at least once (own or received by news), orange = never observed. Thin white lines show the region grid.
- **HLP plan**: red box "1" is the target region; boxes "2…k" are the rest of the ordered region sequence; a dashed line shows the planned route, starting where the HLP started planning (end of the previous LL plan).
- **Teammates' claims**: hatched boxes in the teammate's colour, only for HL intents this agent actually received.
- **LLP plan**: a thick line from the start triangle; the arrow marks the executed first action.
- **Text box**:
  - chosen action and full LL plan
  - g1 (IG of the plan)
  - g2 steps left if the agent stays vs. follows the plan (= steps saved)
  - whether the HLP started a new cycle this step ("replanned") or kept growing its tree ("kept plan")
  - which teammates' intents were received

---

## 8. Configuration

Nested form in `configs/strategies/mh_dec_mcts_*.json`; `config_loader.flatten_hierarchical_config` turns `llp.x` into `llp_x` and `hlp.x` into `hlp_x`.

| Key | Default in configs | Unit / meaning | Used |
|---|---|---|---|
| `use_mcts_llp` | full: true, efficient: false | LLP search type (set by the strategy name) | yes |
| `use_g2` | true | add the g2 term to the LLP reward | yes |
| `g2_weight` | null | bits per saved step (null = auto ≈ 756) | yes |
| `llp.horizon` | 3 | steps | yes |
| `llp.iterations` | 30 | rollouts / MCTS iterations per step (split over slices) | yes |
| `llp.ucb_c` | 1.4 | UCB constant (full only) | yes |
| `llp.discount_factor` | 0.95 | per-step discount of g1 | yes |
| `hlp.horizon` | 8 | regions per sequence | yes |
| `hlp.iterations` | 30 | MCTS iterations **per step** (async; split over slices; the tree accumulates over the cycle). Sync mode: per replan | yes |
| `hlp.ucb_c` | 1.0 | UCB constant (values in steps) | yes |
| `hlp.tile_size` | [25, 25] | region size in cells | yes |
| `hlp.replan_interval` | 2 | **simulation steps** (was seconds) | yes |
| `hlp.candidate_regions` | 6 | HLP branching factor | yes |
| `hlp.discount_factor` | 0.98 | — | **no** (not read) |
| `intent_sharing.max_history` | 10 | sender-side history length | yes |
| `intent_sharing.ll/hl_broadcast_interval` | 0.1 / 0.5 | — | **no** (intents are published every slice) |
| `decentralized.d_uct.gamma` | 0.99 | D-UCT tree discount per iteration, §7 | yes |
| `decentralized.d_uct.decay_factor` / `threshold_steps` | 0.9 / 2 | intent ageing, §7 | yes |
| `decentralized.async.enabled` | true | simulated asynchronous planning, §2 | yes |
| `decentralized.async.slices_per_step` | 3 | intent exchanges per step | yes |
| `decentralized.overlap_penalty_weight` | 0.3 | — | **no** for MH (Dec-MCTS only) |
| `g2_mode` | — | removed; a warning is logged if present | no |

---

## 8b. How the hierarchy compares: MH-MCTS paper, CNR slides, this code

| Aspect | MH-MCTS paper (Seiler et al., RA-L 2024) | Slides (CNR work, slides 21–24; code `threaded_dual_horizon.py`, removed) | This code (`hierarchical_dec_mcts.py`) |
|---|---|---|---|
| Planners per robot | 2 MCTS planners (LLP + HLP), each a Dec-MCTS agent; 2N agents in total | LLP + HLP worker threads behind a `ThreadedDualHorizonPlanner` | 2 anytime planners (LLP + HLP) per robot, each with its own tree |
| Execution | asynchronous; intents every 0.1 s; receding horizon | asynchronous threads | simulated asynchronous: concurrent slices with intent exchange between slices (`slices_per_step`), deterministic; receding horizon (first action executed) |
| Objective | one joint g = g1(LL intents) + g2(all intents) | heuristic HLP region score (entropy + coverage bonus + isolation bonus + distance penalty + hysteresis); LLP = IG blended with alignment | joint g = g1 (IG) + g2_weight × steps saved; g2 = remaining agent-steps to observe every cell |
| Top-down (HLP → LLP) | through g2: the LLP is rewarded for plans that shorten the mission given the HL intent; no constraints | soft alignment bonus toward the target region (never a penalty), adaptive blend weights | through g2 (paper style); soft, no constraints |
| Bottom-up (LLP → HLP) | g2 for the HLP starts **after the robot's own LL intent** (its end position and covered cells) | "local IG true vs expected, uncertainty reduction, belief updates" | same as the paper: at every slice the HLP starts from the end of its own LLP's current plan, and that plan's cells count as covered |
| Horizontal | all LL and HL intents exchanged between robots (all in range) | LLP↔LLP, HLP↔HLP, shared belief; "tree reuse" | LL + HL intents to agents in range in BS/BM modes, every slice; belief + coverage news after each step; HLP tree reused within its cycle |
| HLP state/actions | goal areas = **dynamic clusters of still-open targets** (isolated targets are their own cluster); a plan orders **all** remaining areas | fixed region partition; MCTS depth 15 | fixed 25×25-cell tiles; ordered sequence of `hlp.horizon` (8) regions; nearest-6 candidates |
| g2 | Alg. 2: run the LL intents, then repeatedly move the earliest-free robot to the next open area of its HL plan; **mean** completion time; unassigned work does not occur | — | sum of agent-steps; HL chains + covering-positions spanning tree for unclaimed blocks |
| Weighting | g2 × speed / mean target distance (Eq. 8); insensitive as long as > 0 | blend 0.6/0.4 adapted by coverage/uncertainty/fragmentation | g2_weight = IG of one fresh step (≈ 756 bits/step) |
| D-UCT | discounted UCT in persistent trees, γ = 0.9999 | — | same mechanism (`d_uct.gamma`, default 0.99); separate step-based intent ageing (`decay_factor`, `threshold_steps`) |
| Dec-MCTS baseline | run **with g2** and random HL plans, so the comparison isolates the hierarchy | — | run **with g2** and no HL plans (our g2 prices unplanned work itself, so random plans are not needed); same async scheduler |
| LLP horizon | 100 m paths (half executed) | MCTS depth 5 | 3 steps, first action executed |

---

## 9. Known limitations

- g2 ignores altitude: an agent flying high covers more per move than g2 assumes, so g2 overestimates the remaining work for high-flying agents.
- Per-region bounding boxes make `reach` optimistic for regions with scattered holes. Greedy cover and the MST approximate the real tour (rare small non-monotonicity, see §4).
- g1 does not update the belief along a rollout (repeat visits count twice).
- With MCTS LLP, UCB values are in bits (hundreds to thousands) against `ucb_c ≈ 1.4`, so the tree search is close to greedy exploitation.
- Coverage knowledge does not relay through intermediaries. With BS, an agent that is out of range when the news is sent never learns about those cells, and may later re-cover them.

---

**Planned:** HLP regions are fixed tiles for now. The paper's dynamic clusters of still-uncovered cells (isolated spots as their own cluster), planned over the whole remaining mission, are the next step to test.

---

## 10. Files

| File | Role |
|---|---|
| `src/hierarchical_dec_mcts.py` | `LLIntent`, `HLIntent`, `IntentBus` (range-limited, deferred delivery), `LowLevelPlanner`, `HighLevelPlanner` (both anytime), `HierarchicalDecMCTSPlanner` (`begin_step` / `run_slice` / `finish_step`, or sync `plan`), `create_hierarchical_planner` |
| `src/d_uct.py` | D-UCT statistics (paper Eq. 4–5) |
| `src/g2_evaluator.py` | `CoverageStepModel` (g2 in steps), multi-root MST |
| `src/planner.py` | builds the planner from config; creates the range-limited `IntentBus`; passes `current_step` and `coverage_mask` |
| `src/simulator.py` | step loop; simulated-async scheduler (`_select_agent_actions_async`); per-agent `coverage_mask`; calls coverage news |
| `src/multi_agent_mapper.py` | belief news and `share_coverage_news` |
| `tests/test_mh_coverage_g2.py` | regression tests for all of the above |
