# Configuration

Experiments use a **master config** plus one **strategy config** per planner.

```
configs/
├── master_config.json            # shared settings + which strategies to run
├── strategies/
│   ├── greedy_ig.json
│   ├── dec_mcts.json
│   ├── mh_dec_mcts_full.json
│   ├── mh_dec_mcts_efficient.json
│   └── sweeps/                   # variants used by sweep.json (see note below)
├── experiments/
│   └── gaussian_60x110_ig_bs_n4.json   # Gaussian r=5 at orthomap size, IG_BS unlimited range, 4 agents, 20 runs
├── baseline_greedy_ig_bs_rinf.json   # standalone reference-compatible greedy configs
└── baseline_greedy_igd_bm_r5.json
```

Run from the project root:

```bash
python src/main.py --config configs/master_config.json
```

A standalone strategy file (no `strategies` key) can also be passed with `--config`. Missing shared values are then filled with defaults from `config_loader.load_config`.

## Master config (`master_config.json`, `_version` 3.0)

| Section | Contents |
|---|---|
| `strategies` | strategy names to run, in order |
| `strategy_configs` | name → file, **relative to the master config's folder** (e.g. `strategies/greedy_ig.json`) |
| `shared` | `field_type`, `cluster_radius`, `field_size` (Gaussian only: [x, y] in m, default [50, 50], 0.125 m cells; [60, 110] = orthomap size → 880 × 480 grid; non-default sizes are tagged in run folder names, e.g. `r5_60x110m`), `start_position` (`corner`: fixed corners; `sample`: random lattice start per agent quadrant, seeded per run), `num_agents`, `n_steps`, `iters` ([first, last) iteration; seed = `seed` + iteration), `correlation_types`, `error_margins`, `mode_labels`, `mode_radius_multiplier` (per-mode range override, e.g. `{"IG_BS": -1}` = unlimited; applied last), `enable_plotting`, `enable_logging`, `debug_logs` |
| `decentralized` | shared comm settings, overridden per strategy; see below |
| `experiment` | logging/output settings; `common_metrics` + each strategy's `strategy_metrics` |
| `limited_testing` | if true, `IG_BS` runs with `radius_multiplier = 16` (effectively unlimited) and `IGd_BM` with `radius_multiplier = 5`, overriding the strategy files |

### `decentralized`

| Key | Unit | Meaning |
|---|---|---|
| `radius_multiplier` | × 3.125 m | communication range, checked per axis (x, y, z); `-1` = unlimited. Takes precedence over `communication_range` |
| `communication_range` | m | used only when no `radius_multiplier` is set |
| `d_uct.gamma` | per MCTS iteration | D-UCT tree discount (paper Eq. 4–5) for Dec-MCTS and MH trees; default 0.99 |
| `d_uct.decay_factor` | – | teammate-intent weight `decay ** (age / threshold_steps)` |
| `d_uct.threshold_steps` | **simulation steps** | intent staleness scale; intents older than 2× are ignored. The old key `threshold_sec` is still accepted and its value is read as steps |
| `async.enabled` | – | simulated asynchronous planning for `dec_mcts` and `mh_*` (default true). False: one synchronous plan per agent per step |
| `async.slices_per_step` | count | planning slices per step; intents are exchanged between slices (default 3) |
| `overlap_penalty_weight` | – | Dec-MCTS teammate-overlap penalty (not used by MH) |
| `communication_delay`, `message_drop_probability` | – | not used by the synchronous simulator |

### Mode labels

`mode_labels` selects what is shared; each label is a separate run. `src/main.py` first looks for `mode_labels` inside a section named after the strategy (`greedy_ig.mode_labels`, `dec_mcts.mode_labels`, `mh_dec_mcts_full.mode_labels`, …), then falls back to `shared.mode_labels`. `hierarchical_dec_mcts.mode_labels` is **not** read, so MH strategies always use the shared list unless you add a `mh_dec_mcts_full` / `mh_dec_mcts_efficient` section.

| Label | News (beliefs + coverage) | Intents (Dec-MCTS / MH) | Position IoU discount (greedy only) |
|---|---|---|---|
| `IG` | – | – | – |
| `IGd` | – | – | yes |
| `IG_BS` / `IG_BM` | broadcast / per-neighbour | yes, in range | – |
| `IGd_BS` / `IGd_BM` | broadcast / per-neighbour | yes, in range | yes |

## Strategy configs

Merge rule: shared → strategy overrides. The `decentralized` and `experiment` sections are merged key by key. Any other top-level key in the strategy file replaces the shared one. For example, an `n_steps` in a strategy file wins over a sweep's `shared.n_steps`.

### `greedy_ig.json`

`radius_multiplier` 5. `greedy_ig.overlap_penalty_weight` is ignored (the planner always uses 0).

### `dec_mcts.json`

```json
"dec_mcts": {"horizon": 5, "iterations": 50, "ucb_c": 1.4, "discount_factor": 0.95,
             "timeout": 5.0, "parallel": 1, "use_g2": true, "g2_weight": null},
"decentralized": {"overlap_penalty_weight": 0.3, "radius_multiplier": 5}
```

- `iterations` is per simulation step, split over the async slices.
- `use_g2` (default true) adds the same coverage term as MH-Dec-MCTS, so Dec-MCTS is the "MCTS without hierarchy" baseline.
- `timeout` is a wall-clock budget in seconds, used only in synchronous mode.
- `parallel` is not used.

### `mh_dec_mcts_full.json` / `mh_dec_mcts_efficient.json`

```json
"hierarchical_dec_mcts": {
  "use_mcts_llp": true,            // full: true, efficient: false
  "use_g2": true,                  // add steps-saved term to the LLP reward
  "g2_weight": null,               // bits per saved step; null = auto (~756)
  "llp": {"horizon": 3, "iterations": 30, "ucb_c": 1.4, "discount_factor": 0.95},
  "hlp": {"horizon": 8, "iterations": 30, "ucb_c": 1.0, "discount_factor": 0.98,
          "tile_size": [25, 25], "replan_interval": 2, "candidate_regions": 6},
  "intent_sharing": {"ll_broadcast_interval": 0.1, "hl_broadcast_interval": 0.5,
                     "max_history": 10}
}
```

| Key | Unit | Notes |
|---|---|---|
| `llp.horizon` | steps | |
| `llp.iterations` | rollouts / MCTS iterations **per step** | split over async slices |
| `hlp.horizon` | regions | |
| `hlp.iterations` | MCTS iterations **per step** | the HLP tree accumulates them over its cycle (sync mode: per replan) |
| `hlp.tile_size` | grid cells | 25 cells = 3.125 m = one lateral move |
| `hlp.replan_interval` | **simulation steps** | length of an HLP search cycle (fresh tree); a new cycle also starts as soon as the target region is fully observed |
| `hlp.candidate_regions` | count | nearest uncovered regions considered per HLP tree node |
| `hlp.discount_factor` | – | not used |
| `intent_sharing.*_broadcast_interval` | – | not used (intents are published every slice) |

`config_loader.flatten_hierarchical_config` converts the nested `llp` / `hlp` blocks into flat keys (`llp_horizon`, `hlp_replan_interval`, `tile_size`, …). Both nested and flat forms are accepted.

What these parameters do: `docs/mh_dec_mcts_multiagent.md`.

## Sweeps

`./sweep.sh` (reads `sweep.json`) deep-merges overrides into `master_config.json` and runs each experiment. Strategy paths in `sweep.json` are relative to `experiments/temp/`, where the merged configs are written.

**Several files in `strategies/sweeps/` have names that do not match their contents** (see `docs/method_comparison_summary.md`). Runs are labelled by `experiment.name`, not by file name.

## Experiment configs (`configs/experiments/`)

Self-contained master configs for specific studies; strategy paths are relative to that folder. Example:

```bash
python src/main.py --config configs/experiments/gaussian_60x110_ig_bs_n4.json
python collect_metrics.py --match r5_60x110m_sample_N4_IG_BS_commRinf --out experiments/summaries/gaussian_60x110_ig_bs_n4
```

`collect_metrics.py` parses every matching run (`txt/run.log`, `txt/timestamps.csv`) and writes `per_run_metrics.csv`, `summary.csv` / `summary.md` (mean ± std per strategy) and the plots `curves.png` and `steps_to_full_coverage.png`.

## Output

Command-line runs write to `experiments/temp/<run_base>/` and are moved to `experiments/runs/<strategy>/run_<timestamp>_<run_base>/` when a mode finishes. `run_base` = `<strategy>_<field>_r<radius>_<start>_N<agents>_<mode>_comm<R…>__<experiment.name>`.
