# Benchmark Planners for Multi-Agent UAV Coverage

This document describes the three benchmark planners available for multi-agent decentralized UAV coverage experiments.

## Overview

All three planners support:
- **Single-agent mode** (set `num_agents=1` in config)
- **Multi-agent decentralized mode** (set `num_agents>1`)
- **Asynchronous operation** via `async_runner.py`
- **Intent sharing** for coordination
- **D-UCT staleness discounting** for handling communication delays

## The Three Benchmark Planners

### 1. Greedy IG (Information Gain)

**File:** `src/greedy_ig_planner.py`  
**Strategy name:** `"greedy_ig"`

**Description:**  
The simplest baseline planner that uses one-step lookahead to maximize information gain. For each possible action, it computes the expected IG from the resulting camera footprint and selects the action with maximum IG.

**Key Characteristics:**
- ✅ **Fast**: Minimal computation per step
- ✅ **Myopic**: Only considers immediate reward
- ✅ **Multi-agent**: Reduces IG for cells teammates will observe
- ❌ **No planning**: Doesn't consider future trajectories
- ❌ **Can get stuck**: May leave fragmented uncovered regions

**Use Cases:**
- Fast baseline for comparison
- Benchmarking information gain collection speed
- Testing coordination mechanisms without planning overhead

**Configuration Example:**
```json
{
  "action_strategy": "greedy_ig",
  "num_agents": 4,
  "greedy_ig": {
    "intent_discount": 0.5,
    "overlap_penalty_weight": 0.3
  },
  "decentralized": {
    "d_uct": {
      "decay_factor": 0.9,
      "threshold_sec": 2.0
    }
  }
}
```

**Key Parameters:**
- `intent_discount`: Discount factor for cells teammates will observe (0=no discount, 1=full discount)
- `overlap_penalty_weight`: Weight for penalizing overlapping observations with teammates

---

### 2. Dec-MCTS (Decentralized Monte Carlo Tree Search)

**File:** `src/dec_mcts.py`  
**Strategy name:** `"dec_mcts"`

**Description:**  
Single-level MCTS planner that plans multi-step trajectories using tree search and rollouts. Each agent builds an MCTS tree independently, using teammate intents to avoid redundant coverage via overlap penalties.

**Key Characteristics:**
- ✅ **Multi-step planning**: Looks ahead 5-15 steps
- ✅ **Considers future rewards**: Discount factor balances exploration
- ✅ **Intent-aware**: Uses teammate planned trajectories for coordination
- ✅ **D-UCT discounting**: Handles stale intents gracefully
- ⚠️ **Medium compute**: ~50-200 MCTS iterations per decision
- ❌ **No hierarchical structure**: Uniform horizon planning

**Use Cases:**
- Benchmark against greedy (shows value of multi-step planning)
- Benchmark against MH Dec-MCTS (shows value of hierarchical structure)
- Standard decentralized MCTS baseline from the literature

**Configuration Example:**
```json
{
  "action_strategy": "dec_mcts",
  "num_agents": 4,
  "dec_mcts": {
    "horizon": 10,
    "iterations": 100,
    "ucb_c": 1.4,
    "discount_factor": 0.95,
    "timeout": 5.0,
    "parallel": 1
  },
  "decentralized": {
    "overlap_penalty_weight": 0.3,
    "d_uct": {
      "decay_factor": 0.9,
      "threshold_sec": 2.0
    }
  }
}
```

**Key Parameters:**
- `horizon`: MCTS tree depth (number of steps to plan ahead)
- `iterations`: Number of MCTS iterations (more = better quality, slower)
- `ucb_c`: UCB1 exploration constant (higher = more exploration)
- `discount_factor`: Reward discount factor γ (0=myopic, 1=no discount)
- `overlap_penalty_weight`: Penalty for overlapping with teammate footprints
- `d_uct.decay_factor`: D-UCT staleness decay rate
- `d_uct.threshold_sec`: Intent age threshold for full discount

---

### 3. MH Dec-MCTS (Multi-Horizon Decentralized MCTS)

**File:** `src/hierarchical_dec_mcts.py`  
**Strategy name:** `"hierarchical_dec_mcts"` or `"mh_dec_mcts"`

**Description:**  
Two-level hierarchical planner combining:
- **LLP (Low-Level Planner)**: Short-horizon MCTS for detailed motion planning (5-10 steps)
- **HLP (High-Level Planner)**: Long-horizon region allocation planner (3-5 regions)

The key innovation is **reward decomposition**: `g = g1(LL intents) + g2(all intents)`, where LLP focuses on immediate IG maximization while HLP provides strategic guidance toward valuable regions, preventing fragmented coverage.

**Key Characteristics:**
- ✅ **Hierarchical planning**: Separates tactical and strategic reasoning
- ✅ **Best coverage quality**: Avoids leaving isolated uncovered patches
- ✅ **Intent sharing at both levels**: LL and HL intents for coordination
- ✅ **Alignment bonus**: LLP biased toward HLP target regions
- ✅ **Fully decentralized**: No central controller
- ⚠️ **Higher compute**: Runs two planners per agent
- ⚠️ **More complex**: More parameters to tune

**Use Cases:**
- State-of-the-art multi-horizon planning benchmark
- Demonstrates value of hierarchical decomposition
- Best performance on coverage quality metrics

**Configuration Example:**
```json
{
  "action_strategy": "hierarchical_dec_mcts",
  "num_agents": 4,
  "hierarchical_dec_mcts": {
    "llp_horizon": 7,
    "llp_iterations": 50,
    "hlp_horizon": 3,
    "hlp_iterations": 30,
    "tile_size": [50, 50],
    "hlp_replan_interval": 1.0,
    "alignment_bonus_weight": 0.2,
    "region_conflict_penalty": 10.0
  },
  "decentralized": {
    "overlap_penalty_weight": 0.3,
    "d_uct": {
      "decay_factor": 0.9,
      "threshold_sec": 2.0
    }
  }
}
```

**Key Parameters:**
- `llp_horizon`: LLP planning depth (short-horizon steps)
- `llp_iterations`: MCTS iterations for LLP
- `hlp_horizon`: Number of regions HLP plans ahead
- `hlp_iterations`: MCTS iterations for HLP
- `tile_size`: [height, width] of each region in grid cells
- `hlp_replan_interval`: Seconds between HLP replanning
- `alignment_bonus_weight`: How much LLP is biased toward HLP target
- `region_conflict_penalty`: Penalty for multiple agents targeting same region

---

## Running Benchmarks

### Single-Agent Example

```python
# In config.json
{
  "action_strategy": "greedy_ig",  # or "dec_mcts", "hierarchical_dec_mcts"
  "num_agents": 1,
  "max_steps": 200
}
```

### Multi-Agent Example (4 UAVs)

```python
# In config.json
{
  "action_strategy": "hierarchical_dec_mcts",
  "num_agents": 4,
  "max_steps": 200,
  "decentralized": {
    "communication_range": 100.0,  # meters
    "communication_delay": 0.1,    # seconds
    "overlap_penalty_weight": 0.3
  }
}
```

### Using async_runner.py

```bash
python src/async_runner.py --config config.json --num-agents 4 --strategy dec_mcts
```

---

## Metrics for Comparison

Recommended metrics to compare the three planners:

1. **Cumulative Information Gain**: Total IG collected over time
2. **Coverage Progress**: Percentage of field covered vs. time
3. **Final Coverage Quality**: Coverage percentage at episode end
4. **Fragmentation**: Number of isolated uncovered patches
5. **Communication Cost**: Messages sent / bytes exchanged
6. **Compute Time**: Planning time per step (ms)
7. **Overlap Redundancy**: Fraction of cells observed by >1 agent

---

## Expected Performance Characteristics

| Metric                    | Greedy IG | Dec-MCTS | MH Dec-MCTS |
|---------------------------|-----------|----------|-------------|
| **Planning Speed**        | ⚡⚡⚡      | ⚡⚡      | ⚡          |
| **Coverage Quality**      | ⭐⭐      | ⭐⭐⭐    | ⭐⭐⭐⭐     |
| **Coordination Quality**  | ⭐⭐      | ⭐⭐⭐    | ⭐⭐⭐⭐     |
| **Fragmentation Avoidance**| ❌        | ⚠️       | ✅          |
| **Scalability (agents)**  | ⭐⭐⭐⭐   | ⭐⭐⭐    | ⭐⭐⭐      |

---

## Implementation Notes

### Unified Interface

All three planners expose the same interface via `planner.py`:

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

### Decentralized Coordinators

Each planner has its own coordinator for intent sharing:
- `GreedyIGCoordinator` (simple position + footprint sharing)
- `DecMCTSCoordinator` (trajectory intent sharing)
- `IntentBus` (LL + HL intent sharing for hierarchical)

Single-agent mode automatically disables coordinator features.

### Logging

Each planner logs detailed decision information:
- `logs/greedy_ig_*.log`: Action scores, overlap penalties
- `logs/dec_mcts_*.log`: MCTS statistics, trajectory details
- `logs/hierarchical_dec_mcts_*.log`: LLP/HLP decisions, alignment

---

## References

**Paper:** "Multi-Horizon Multi-Agent Planning Using Decentralised Monte Carlo Tree Search"

**Greedy IG**: Section 5.1 (Baseline)  
**Dec-MCTS**: Section 3 (Dec-MCTS framework)  
**MH Dec-MCTS**: Section 4 (Multi-Horizon extension)
