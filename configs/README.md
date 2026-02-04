# Configuration System Documentation

## Overview

The multi-horizon project uses a **master config + strategy-specific configs** architecture for running experiments. This allows you to:
- Define shared parameters once (field type, agents, simulation settings)
- Run multiple strategies (planners) in sequence
- Keep strategy-specific parameters organized and maintainable

## Directory Structure

```
configs/
├── master_config.json          # Master configuration (v3.0)
├── strategies/                 # Strategy-specific configs
│   ├── greedy_ig.json         # Strategy 1: Greedy IG baseline
│   ├── dec_mcts.json          # Strategy 2: Dec-MCTS
│   ├── mh_dec_mcts_full.json  # Strategy 3: MH-Dec-MCTS (full MCTS)
│   └── mh_dec_mcts_efficient.json  # Strategy 4: MH-Dec-MCTS (efficient)
└── [other configs]            # Additional test and utility configs
```

## Master Config Format

**File**: `configs/master_config.json`

```json
{
  "_version": "3.0",
  
  "strategies": [
    "greedy_ig",
    "dec_mcts",
    "mh_dec_mcts_full",
    "mh_dec_mcts_efficient"
  ],
  
  "strategy_configs": {
    "greedy_ig": "configs/strategies/greedy_ig.json",
    "dec_mcts": "configs/strategies/dec_mcts.json",
    "mh_dec_mcts_full": "configs/strategies/mh_dec_mcts_full.json",
    "mh_dec_mcts_efficient": "configs/strategies/mh_dec_mcts_efficient.json"
  },
  
  "shared": {
    "project_path": "./",
    "field_type": "Gaussian",
    "start_position": "corner",
    "num_agents": 4,
    "n_steps": 100,
    "iters": [0, 20],
    "correlation_types": ["adaptive", "equal", "biased"],
    "error_margins": [null],
    "enable_plotting": false,
    "enable_logging": true
  },
  
  "decentralized": {
    "communication_range": 15.625,
    "communication_delay": 0.05,
    "message_drop_probability": 0.0,
    "d_uct": {
      "decay_factor": 0.9,
      "threshold_sec": 2.0
    }
  },
  
  "experiment": {
    "base_log_dir": "trials",
    "save_trajectory": true,
    "save_belief_maps": false,
    "common_metrics": [
      "cumulative_ig",
      "coverage_ratio",
      "fragmentation_score",
      "messages_sent",
      "compute_time_ms"
    ]
  }
}
```

### Key Sections

| Section | Purpose |
|---------|---------|
| `strategies` | List of strategies to run (in order) |
| `strategy_configs` | Mapping from strategy name to config file path |
| `shared` | Common simulation parameters (field type, agents, steps) |
| `decentralized` | Common multi-agent communication settings |
| `experiment` | Common logging and output settings |

## Strategy Config Format

Each strategy has its own config file in `configs/strategies/`.

### Example: Greedy IG

**File**: `configs/strategies/greedy_ig.json`

```json
{
  "action_strategy": "greedy_ig",
  
  "greedy_ig": {
    "overlap_penalty_weight": 0.0,
    "mode_labels": ["IG", "IGd", "IG_BM", "IG_BS", "IGd_BM", "IGd_BS"]
  },
  
  "decentralized": {
    "radius_multiplier": 5
  },
  
  "experiment": {
    "name": "greedy_ig_benchmark",
    "log_dir_suffix": "greedy_ig",
    "strategy_metrics": []
  }
}
```

### Example: MH-Dec-MCTS (full)

**File**: `configs/strategies/mh_dec_mcts_full.json`

```json
{
  "action_strategy": "mh_dec_mcts_both",
  
  "hierarchical_dec_mcts": {
    "use_mcts_llp": true,
    "mode_labels": ["IG", "IGd", "IG_BM", "IG_BS", "IGd_BM", "IGd_BS"],
    
    "llp": {
      "horizon": 3,
      "iterations": 50,
      "ucb_c": 1.4,
      "discount_factor": 0.95
    },
    
    "hlp": {
      "horizon": 10,
      "iterations": 30,
      "ucb_c": 1.0,
      "discount_factor": 0.98,
      "tile_size": [50, 50],
      "replan_interval": 1.0
    },
    
    "intent_sharing": {
      "ll_broadcast_interval": 0.1,
      "hl_broadcast_interval": 0.5,
      "max_history": 10
    }
  },
  
  "decentralized": {
    "overlap_penalty_weight": 0.3
  },
  
  "experiment": {
    "name": "mh_dec_mcts_full_benchmark",
    "log_dir_suffix": "mh_dec_mcts_full",
    "strategy_metrics": [
      "llp_iterations",
      "llp_planning_time_ms",
      "hlp_iterations",
      "hlp_planning_time_ms",
      "ll_intents_sent",
      "hl_intents_sent"
    ]
  }
}
```

## Config Merging Rules

When a strategy config is loaded, settings are merged as follows:

1. **Shared settings** are applied first (from master config)
2. **Strategy-specific settings** override shared settings
3. **Decentralized** section: strategy can override specific fields
4. **Experiment** section:
   - `common_metrics` + `strategy_metrics` = final metrics list
   - `log_dir` = `base_log_dir` + `log_dir_suffix`

## Running Experiments

### Run all strategies

```bash
python src/main.py --config configs/master_config.json
```

This will run all strategies listed in `master_config.json` in sequence.

### Run single strategy (legacy)

```bash
python src/main.py --config configs/benchmark_greedy_ig.json
```

Old single-file configs are still supported for backward compatibility.

## Four Baseline Strategies

| Strategy | Config File | LLP Method | HLP Method | Description |
|----------|-------------|------------|------------|-------------|
| 1. Greedy IG | `greedy_ig.json` | N/A | N/A | Single-step IG maximization |
| 2. Dec-MCTS | `dec_mcts.json` | N/A (single-level) | N/A | UCB tree search (10 steps) |
| 3. MH-Dec-MCTS (full) | `mh_dec_mcts_full.json` | **UCB tree** | **UCB tree** | Both planners use MCTS |
| 4. MH-Dec-MCTS (efficient) | `mh_dec_mcts_efficient.json` | **Random rollout** | **UCB tree** | LLP optimized for speed |

**Incremental Progression:**
- **1 → 2**: Add multi-step planning
- **2 → 3**: Add hierarchical structure with both planners using MCTS
- **3 → 4**: Optimize LLP with random rollout (hypothesis: HLP guidance makes LLP tree search unnecessary)

## Hierarchical Config Structure

The `hierarchical_dec_mcts` section supports both flat and nested formats:

### Nested Format (preferred for v3.0)

```json
"hierarchical_dec_mcts": {
  "use_mcts_llp": true,
  "llp": {
    "horizon": 3,
    "iterations": 50,
    "ucb_c": 1.4
  },
  "hlp": {
    "horizon": 10,
    "iterations": 30,
    "ucb_c": 1.0
  }
}
```

### Flat Format (backward compatible)

```json
"hierarchical_dec_mcts": {
  "use_mcts_llp": true,
  "llp_horizon": 3,
  "llp_iterations": 50,
  "llp_ucb_c": 1.4,
  "hlp_horizon": 10,
  "hlp_iterations": 30,
  "hlp_ucb_c": 1.0
}
```

The config loader automatically flattens nested configs for backward compatibility.

## Migration Guide

### From Old Configs (v1.0) to New System (v3.0)

1. **Create master config** with shared settings
2. **Extract strategy-specific sections** to separate files
3. **Update `action_strategy`** field in each strategy config
4. **Test** with `python src/main.py --config configs/master_config.json`

### Example Migration

**Old** (`benchmark_dec_mcts.json`):
```json
{
  "field_type": "Gaussian",
  "num_agents": 4,
  "action_strategy": "dec_mcts",
  "dec_mcts": {
    "horizon": 10,
    "iterations": 100
  }
}
```

**New** (split into 2 files):

`master_config.json`:
```json
{
  "shared": {
    "field_type": "Gaussian",
    "num_agents": 4
  }
}
```

`strategies/dec_mcts.json`:
```json
{
  "action_strategy": "dec_mcts",
  "dec_mcts": {
    "horizon": 10,
    "iterations": 100
  }
}
```

## Backward Compatibility

The system maintains **full backward compatibility**:

- ✅ v1.0 configs (legacy flat format) work unchanged
- ✅ v2.0 configs (structured format) work unchanged
- ✅ v3.0 configs (master + strategy) is the new preferred format

**Detection logic**:
- If `_version == "3.0"` → master config mode
- If `_schema_version == "2.0"` → v2.0 structured config
- Otherwise → v1.0 legacy config

## Best Practices

1. **Use master config** for running multiple strategies together
2. **Keep shared settings in master** to avoid duplication
3. **Document strategy changes** in config file `_comment` fields
4. **Version control configs** to track experiment evolution
5. **Use descriptive log_dir_suffix** for easy identification of results

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Strategy config not found" | Check path in `strategy_configs` is correct |
| "No strategies to run" | Ensure `strategies` list is not empty |
| Nested config not recognized | Add `flatten_hierarchical_config()` call |
| Metrics missing | Check both `common_metrics` and `strategy_metrics` |

## See Also

- [Method Comparison Summary](../docs/method_comparison_summary.md)
- [MH-Dec-MCTS Architecture](../docs/mh_dec_mcts_multiagent.md)
- [Baseline Configurations](./strategies/)
