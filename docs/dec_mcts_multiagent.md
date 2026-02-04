# Decentralized Monte Carlo Tree Search (Dec-MCTS) Multi-Agent Baseline

## Overview

Dec-MCTS is a **single-level decentralized planning** baseline that extends standard MCTS to multi-agent scenarios. Each agent runs independent MCTS planning while incorporating information from neighbors through communication. Unlike greedy methods, Dec-MCTS builds a search tree and simulates future trajectories to make better long-term decisions.

This serves as a mid-tier baseline between myopic greedy approaches and sophisticated multi-horizon methods.

## How It Works

### 1. Decentralized MCTS Process

Each agent independently executes:

1. **Selection**: Traverse tree using UCT policy to select promising branches
2. **Expansion**: Add new child nodes for unexplored actions
3. **Simulation**: Rollout from leaf node to estimate future value
4. **Backpropagation**: Update node values along traversed path

The key difference from single-agent MCTS: agents exchange information during planning to coordinate actions.

### 2. Tree Construction

```
Root (current state)
├── Action 1
│   ├── Obs 1 → Belief update
│   └── Obs 2 → Belief update
├── Action 2
│   └── ...
└── Action N
```

- **Tree depth**: Controlled by iteration count and expansion policy
- **Branching**: Action space × observation space
- **Node values**: Cumulative information gain estimates

### 3. Rollout Policy

During simulation phase:
- **Random actions**: Default rollout policy selects actions uniformly
- **Depth-limited**: Rollouts terminate at fixed horizon or entropy threshold
- **Belief propagation**: Each simulated observation updates belief state

### 4. Communication Integration

Agents share information through:
- **D-UCT algorithm**: Decentralized UCT with message passing
- **Decay factor**: Recent information weighted more heavily (default: 0.9)
- **Threshold**: Messages older than threshold are discarded (default: 2.0s)
- **Range-limited**: Only neighbors within communication_range exchange data

## Key Parameters

### Strategy-Specific (dec_mcts.json)

```json
{
  "mcts": {
    "iterations": 20,
    "rollout_depth": 3,
    "exploration_constant": 1.414,
    "discount_factor": 0.95,
    "enable_progressive_widening": false
  },
  "decentralized": {
    "communication_range": 15.625,
    "d_uct": {
      "decay_factor": 0.9,
      "threshold_sec": 2.0
    }
  }
}
```

#### MCTS Parameters
- **iterations**: Number of MCTS iterations per step (default: 20)
  - Higher = better decisions but slower
  - Typical range: 10-50
- **rollout_depth**: Simulation horizon in steps (default: 3)
  - Deeper rollouts = more lookahead but more computation
- **exploration_constant**: UCT exploration parameter (default: √2 ≈ 1.414)
  - Controls exploration vs exploitation tradeoff
- **discount_factor**: Future reward discounting (default: 0.95)
  - γ = 0.95 means rewards 3 steps ahead worth 0.86× immediate reward
- **enable_progressive_widening**: Advanced branching control (default: false)

#### Decentralized Parameters
- **communication_range**: Fixed range in meters (default: 15.625m)
- **decay_factor**: Message age weighting (0-1, default: 0.9)
  - Older messages contribute less to decision making
- **threshold_sec**: Maximum message age in seconds (default: 2.0)
  - Messages older than this are ignored

### Shared Parameters (master_config.json)

- **num_agents**: Number of UAVs in the swarm
- **cluster_radius**: Spatial correlation parameter
- **mode_labels**: Communication modes (same as Greedy IG)
- **n_steps**: Total mission steps

## Communication Modes

### Mode Configuration
Same as Greedy IG:
- **IG**: No sharing, independent planning
- **IGd**: Position sharing enabled
- **IG_BS/IG_BM**: News sharing (broadcast/per-neighbor)
- **IGd_BS/IGd_BM**: Position + news sharing

### Limited Testing Mode
When `limited_testing: true`:
- **IG_BS**: Infinite communication (communication_range = -1)
- **IGd_BM**: Limited communication (communication_range = 3 × cluster_radius)

## Computational Complexity

- **Time per step**: O(|A| × I × D × |C|^D)
  - |A| = number of agents
  - I = MCTS iterations
  - D = rollout depth
  - |C| = candidate actions per node
- **Scaling**: Exponential with depth, linear with iterations
- **Computational cost**: Higher than greedy methods due to tree search and simulation

## Tree Search Characteristics

### UCT Selection Formula
```
UCT(node) = value(node) + c × √(ln(parent_visits) / node_visits)
           └─exploitation─┘   └────────exploration────────┘
```

### Belief Update
At each tree node after observation:
```
belief_new = bayesian_update(belief_old, observation, sensor_model)
```

### Value Estimation
Node value = cumulative IG along trajectory from root:
```
V(node) = Σ γ^t × IG(t)  for t=0 to depth
```

## Use Cases

✅ **Good for:**
- Scenarios requiring multi-step lookahead
- Moderate computational budgets
- Balancing exploration and exploitation
- Comparing single-level vs multi-horizon planning

❌ **Limitations:**
- Single planning horizon (no hierarchical decomposition)
- Computational cost grows exponentially with depth
- May struggle with very long-term strategic positioning
- No explicit high-level/low-level task decomposition

## Comparison with Other Baselines

| Method | Lookahead | Tree Depth | Coordination |
|--------|-----------|------------|--------------||
| Greedy IG | 1 step | None | Reactive |
| **Dec-MCTS** | Multi-step | Single-level | Communication-based |
| MH-Dec-MCTS | Multi-horizon | Two-level hierarchical | Hierarchical planning |

## Expected Performance

- **Information Gain**: Good - better than greedy due to lookahead
- **Coverage**: Strategic positioning through simulation
- **Planning Quality**: Moderate - limited by single-level tree depth
- **Scalability**: Good for small-medium swarms (2-8 agents)

## Tuning Guidelines

### For Better Performance
- ↑ iterations (20 → 50): More accurate value estimates, higher computational cost
- ↑ rollout_depth (3 → 5): Longer lookahead, exponentially higher computational cost
- ↑ exploration_constant (1.4 → 2.0): More exploration, less greedy

### For Faster Execution
- ↓ iterations (20 → 10): Lower computational cost but less optimal
- ↓ rollout_depth (3 → 2): Significantly lower computational cost, shorter lookahead
- Disable progressive_widening: Simpler tree structure, lower overhead

### For Better Coordination
- ↑ communication_range: More information sharing
- ↓ threshold_sec: Use only very recent messages
- Use IGd_BM mode: Position + per-neighbor news

## Configuration Example

```json
{
  "action_strategy": "dec_mcts",
  "mcts": {
    "iterations": 20,
    "rollout_depth": 3,
    "exploration_constant": 1.414,
    "discount_factor": 0.95
  },
  "decentralized": {
    "communication_range": 15.625,
    "d_uct": {
      "decay_factor": 0.9,
      "threshold_sec": 2.0
    }
  },
  "shared": {
    "num_agents": 4,
    "cluster_radius": 4,
    "mode_labels": ["IG_BS", "IGd_BM"]
  }
}
```

## Implementation Details

- **Code**: `src/dec_mcts.py`
- **Helper functions**: `src/helper.py` (entropy, IG calculations)
- **Communication**: `src/multi_agent_coordinator.py`
- **Belief updates**: `src/mapper_LBP.py`, `src/multi_agent_mapper.py`

## References

- Based on UCT (Upper Confidence Bounds for Trees) algorithm
- Decentralized extension using D-UCT message passing
- See Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012)
- Implementation adapted for continuous spatial information gathering
