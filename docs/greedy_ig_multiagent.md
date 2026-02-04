# Greedy Information Gain Multi-Agent Baseline

## Overview

The Greedy IG strategy is a **single-step lookahead baseline** that selects actions purely based on immediate information gain maximization. Each agent independently evaluates all possible next positions and chooses the one that maximizes expected belief reduction about the underlying spatial field.

This is the simplest and fastest baseline, serving as a lower bound for comparison with more sophisticated planning methods.

## How It Works

### 1. Action Selection Process

At each time step, for each agent:
1. **Generate candidates**: Enumerate all reachable positions from current location
2. **Evaluate IG**: For each candidate position, compute expected information gain using current belief map
3. **Select maximum**: Choose the position with highest IG value
4. **No lookahead**: Decisions are myopic - no consideration of future steps beyond immediate next position

### 2. Information Gain Calculation

The information gain for a candidate position is computed as:
- **IG = H(current belief) - E[H(belief after observation)]**
- Uses the sensor model to predict expected entropy reduction
- Incorporates camera footprint and sensor noise characteristics

### 3. Multi-Agent Coordination

Greedy IG uses **decentralized coordination** through:
- **Communication range**: Agents share information within a specified radius
- **Position sharing**: Optional - agents can broadcast their current positions
- **News sharing**: Optional - agents can share recent observations (news)
- **No explicit collision avoidance**: Overlap penalty can be added but is set to 0 by default

## Key Parameters

### Strategy-Specific (greedy_ig.json)

```json
{
  "greedy_ig": {
    "overlap_penalty_weight": 0.0
  },
  "decentralized": {
    "radius_multiplier": 5
  }
}
```

- **overlap_penalty_weight**: Penalty for visiting positions near other agents (default: 0.0)
- **radius_multiplier**: Communication range as multiple of grid step size
  - Formula: `communication_range = radius_multiplier × grid_step`
  - Example: `5 × 3.125m = 15.625m`

### Shared Parameters (master_config.json)

- **num_agents**: Number of UAVs in the swarm
- **cluster_radius**: Spatial correlation parameter for Gaussian field
- **mode_labels**: Communication modes (IG, IGd, IG_BS, IG_BM, IGd_BS, IGd_BM)
  - **IG**: Information gain only, no sharing
  - **IGd**: Position sharing enabled
  - **IG_BS/IG_BM**: News sharing with broadcast/per-neighbor modes
  - **IGd_BS/IGd_BM**: Position + news sharing

## Communication Modes

### Limited Testing Mode
When `limited_testing: true` in master config:
- **IG_BS**: Infinite communication range (radius_multiplier = -1)
- **IGd_BM**: Limited communication at 3× cluster_radius

### Standard Mode
Uses default `radius_multiplier: 5` for all modes unless overridden by limited_testing.

## Computational Complexity

- **Time per step**: O(|A| × |C|)
  - |A| = number of agents
  - |C| = number of candidate positions per agent (typically 9-25)
- **No tree search**: No MCTS simulation overhead
- **Extremely fast**: Minimal computational cost compared to tree-based methods

## Use Cases

✅ **Good for:**
- Fast baseline comparison
- Real-time applications requiring immediate response
- Scenarios where myopic decisions are acceptable
- Understanding pure belief-driven behavior

❌ **Limitations:**
- No long-term planning or strategic positioning
- May get stuck in local optima
- Poor performance in scenarios requiring coordination or future positioning
- Cannot anticipate dynamic changes or multi-step strategies

## Comparison with Other Baselines

| Method | Lookahead | Tree Search | Coordination |
|--------|-----------|-------------|--------------||
| **Greedy IG** | 1 step | None | Reactive only |
| Dec-MCTS | Tree depth | Single-level MCTS | Communication-based |
| MH-Dec-MCTS | Multi-horizon | Two-level hierarchical | Hierarchical planning |

## Expected Performance

- **Information Gain**: Moderate - good immediate coverage but suboptimal long-term
- **Coverage**: Tends to spread out initially, may miss strategic areas
- **Messages Sent**: Depends on communication mode and range
- **Scalability**: Excellent - linear with agents and steps

## Configuration Example

```json
{
  "action_strategy": "greedy_ig",
  "greedy_ig": {
    "overlap_penalty_weight": 0.0
  },
  "decentralized": {
    "radius_multiplier": 5
  },
  "shared": {
    "num_agents": 4,
    "cluster_radius": 4,
    "mode_labels": ["IG_BS", "IGd_BM"]
  }
}
```

## References

- Pure greedy information gain is a standard baseline in adaptive sensing and active learning
- Implements single-step lookahead version of the general information-theoretic planning framework
- See `src/greedy_ig_planner.py` for implementation details
