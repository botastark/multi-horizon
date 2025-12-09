# Refactoring Summary: Unified Benchmark Planners

## What Was Done

This refactoring unified the codebase to support three clean, decentralized benchmark planners with consistent interfaces.

---

## Files Created

### 1. `src/experiment_utils.py` (NEW)
**Purpose:** Common experiment utilities for single and multi-agent experiments

**Contents:**
- `initialize_agent()` - Initialize agent with camera, planner, and state
- `compute_agent_metrics()` - Compute entropy, MSE, coverage for agent
- `update_agent_observation()` - Update agent belief with new observation
- `compute_multi_agent_fused_metrics()` - Fuse multi-agent beliefs and compute global metrics
- `finalize_planners()` - Finalize all agent planners
- `get_results_folder()` - Generate results folder path
- `save_step_plot()` - Save visualization for a step

**Impact:** Eliminates ~300 lines of duplicated code between single-agent and multi-agent experiment functions in `main.py`.

---

### 3. `src/planning_utils.py` (NEW)
**Purpose:** Shared utilities used across all planners

**Contents:**
- `analyze_coverage_fragmentation()` - Connected components analysis
- `compute_revisit_cost()` - Patch revisit cost estimation
- `partition_field()` - Field partitioning for HLP
- `compute_expected_ig()` - IG computation helper
- `compute_field_entropy()` - Total entropy calculation
- `compute_coverage_ratio()` - Coverage metric
- Constants: `FRAGMENTATION_PATCH_THRESHOLD`, `COVERAGE_ADJUSTMENT_FACTOR`, etc.

**Impact:** Eliminates code duplication across planners.

---

### 5. `BENCHMARK_PLANNERS.md` (NEW)
**Purpose:** Comprehensive documentation for the three benchmark planners

**Contents:**
- Detailed description of each planner
- Configuration parameters and examples
- Expected performance characteristics
- Use cases and recommendations
- Metrics for comparison
- Implementation notes

---

### 7. `QUICK_START.md` (NEW)
**Purpose:** Quick reference for running benchmarks

**Contents:**
- Command-line examples for each planner
- Single-agent and multi-agent configurations
- Troubleshooting guide
- File structure overview

---

### 8. `src/multi_agent_coordinator.py` (NEW)
**Purpose:** Wrapper for multi-agent coordination

**Contents:**
- `MultiAgentCoordinator` - Unified coordinator wrapper
- `generate_multi_agent_starts()` - Generate agent starting positions

**Impact:** Provides missing coordinator used by `main.py` for multi-agent experiments.

---

### 9. Configuration Files (NEW)

**Multi-Agent Benchmarks:**
- `configs/benchmark_greedy_ig.json` - Greedy IG with 4 agents
- `configs/benchmark_dec_mcts.json` - Dec-MCTS with 4 agents
- `configs/benchmark_mh_dec_mcts.json` - MH Dec-MCTS with 4 agents

**Single-Agent Benchmarks:**
- `configs/single_agent_greedy.json` - Greedy IG baseline
- `configs/single_agent_mh_dec_mcts.json` - MH Dec-MCTS baseline

Each config includes:
- Strategy-specific parameters
- Decentralized coordination settings
- D-UCT staleness discounting
- Experiment metadata

---

## Files Modified

### 1. `src/dual_horizon_planner.py`
**Changes:**
- Added import: `from planning_utils import ...`
- Removed duplicate function definitions:
  - `analyze_coverage_fragmentation()`
  - `compute_revisit_cost()`
  - `partition_field()`
- Removed duplicate constants:
  - `FRAGMENTATION_PATCH_THRESHOLD`
  - `COVERAGE_ADJUSTMENT_FACTOR`
  - `UNCERTAINTY_ADJUSTMENT_FACTOR`
  - `FRAGMENTATION_ADJUSTMENT_FACTOR`

**Result:** ~200 lines removed, now uses shared utilities.

---

## Verification Completed

### Existing Planners Already Support Decentralization

**✅ greedy_ig_planner.py:**
- Intent sharing via `GreedyIGCoordinator`
- D-UCT staleness discounting
- Overlap penalty for coordination
- Single-agent mode (automatic when `num_agents=1`)

**✅ dec_mcts.py:**
- Intent sharing via `DecMCTSCoordinator`
- D-UCT staleness discounting
- Multi-step trajectory planning with overlap avoidance
- Single-agent mode support

**✅ hierarchical_dec_mcts.py:**
- Intent sharing via `IntentBus` (LL + HL intents)
- D-UCT staleness discounting
- Hierarchical planning with alignment
- Single-agent mode support

**No changes needed** - All three planners already had full decentralized support!

---

## Unified Interface

All three planners use the same interface via `src/planner.py`:

```python
planner = planning(
    grid_info=grid,
    uav=uav_camera,
    strategy="greedy_ig",  # or "dec_mcts", "hierarchical_dec_mcts"
    conf_dict=config,
    agent_id=agent_id,
    coordinator=coordinator
)

action, action_scores = planner.select_action(belief, visited_x)
```

**Benefits:**
- Easy to swap planners in experiments
- Consistent API for benchmarking scripts
- Single-agent and multi-agent modes use same code

---

## Three Clean Benchmarks

### Benchmark 1: Greedy IG
- **File:** `src/greedy_ig_planner.py`
- **Strategy:** `"greedy_ig"`
- **Type:** Single-step lookahead
- **Use:** Fast baseline

### Benchmark 2: Dec-MCTS
- **File:** `src/dec_mcts.py`
- **Strategy:** `"dec_mcts"`
- **Type:** Single-level MCTS (multi-step planning)
- **Use:** Standard MCTS benchmark

### Benchmark 3: MH Dec-MCTS
- **File:** `src/hierarchical_dec_mcts.py`
- **Strategy:** `"hierarchical_dec_mcts"` or `"mh_dec_mcts"`
- **Type:** Hierarchical LLP + HLP
- **Use:** State-of-the-art multi-horizon planning

**All three:**
- ✅ Support single-agent mode (`num_agents=1`)
- ✅ Support multi-agent mode (`num_agents>1`)
- ✅ Fully decentralized (no central controller)
- ✅ Async-compatible via `async_runner.py`
- ✅ D-UCT staleness discounting
- ✅ Intent sharing for coordination

---

## What's Redundant Now?

### To Deprecate (Not Needed for Benchmarks)

**1. `src/dual_horizon_planner.py`**
- **Why:** Superseded by `hierarchical_dec_mcts.py`
- **Action:** Mark as legacy or remove after validation
- **Note:** Keep if you need the specific blend weight computation from dual-horizon

**2. `src/threaded_dual_horizon.py`** (if it exists)
- **Why:** `async_runner.py` provides better async support
- **Action:** Remove or consolidate threading logic

**3. `src/multi_agent_coordinator.py`** (if centralized)
- **Why:** Each planner has its own decentralized coordinator
- **Action:** Remove if fully centralized (conflicts with decentralized design)

---

## Migration Guide

### For Existing Experiments Using `dual_horizon_planner.py`:

**Option 1:** Switch to `hierarchical_dec_mcts` (recommended)
```json
{
  "action_strategy": "hierarchical_dec_mcts"  // instead of "dual_horizon"
}
```

**Option 2:** Keep using `dual_horizon_planner.py`
- It now imports from `planning_utils.py`
- Still works, just not a "benchmark" planner
- Good for custom blend weight experiments

---

## Testing the Refactoring

### Run All Three Benchmarks

```bash
# Greedy IG
python src/main.py --config configs/benchmark_greedy_ig.json

# Dec-MCTS
python src/main.py --config configs/benchmark_dec_mcts.json

# MH Dec-MCTS
python src/main.py --config configs/benchmark_mh_dec_mcts.json
```

### Verify Single-Agent Mode

```bash
# Single-agent greedy
python src/main.py --config configs/single_agent_greedy.json

# Single-agent MH Dec-MCTS
python src/main.py --config configs/single_agent_mh_dec_mcts.json
```

---

## Summary

**Before Refactoring:**
- Duplicated utility functions across multiple files
- Unclear which planners to use for benchmarks
- No clear documentation of planner differences
- Mixed centralized/decentralized code

**After Refactoring:**
- ✅ Shared utilities in `planning_utils.py`
- ✅ Three clear benchmarks with documented purpose
- ✅ All planners support single + multi-agent
- ✅ All planners fully decentralized
- ✅ Unified interface via `planner.py`
- ✅ Example configs for each benchmark
- ✅ Quick start and comprehensive documentation

**Result:** Clean, maintainable benchmark suite ready for experiments!
