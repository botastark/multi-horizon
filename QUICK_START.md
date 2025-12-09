# Benchmark Quick Start Guide

## Running the Three Benchmark Planners

### 1. Greedy IG (Baseline)

**Single-Agent:**
```bash
python src/main.py --config configs/single_agent_greedy.json
```

**Multi-Agent (4 UAVs):**
```bash
python src/main.py --config configs/benchmark_greedy_ig.json
```

---

### 2. Dec-MCTS (Single-Level MCTS)

**Multi-Agent (4 UAVs):**
```bash
python src/main.py --config configs/benchmark_dec_mcts.json
```

---

### 3. MH Dec-MCTS (Multi-Horizon Hierarchical)

**Single-Agent:**
```bash
python src/main.py --config configs/single_agent_mh_dec_mcts.json
```

**Multi-Agent (4 UAVs):**
```bash
python src/main.py --config configs/benchmark_mh_dec_mcts.json
```

---

## Switching Number of Agents

Edit the config file:
```json
{
  "num_agents": 1,   // Single-agent
  "num_agents": 4,   // 4 agents
  "num_agents": 8    // 8 agents
}
```

All three planners automatically adapt to single or multi-agent mode.

---

## Key Configuration Parameters

### Communication Settings (Multi-Agent Only)

```json
"decentralized": {
  "communication_range": 150.0,      // meters
  "communication_delay": 0.05,       // seconds
  "message_drop_probability": 0.0,   // 0.0 - 1.0
  "overlap_penalty_weight": 0.3      // coordination strength
}
```

### D-UCT Staleness Discounting

```json
"decentralized": {
  "d_uct": {
    "decay_factor": 0.9,        // 0.9 = fast decay, 0.99 = slow decay
    "threshold_sec": 2.0        // intent age threshold
  }
}
```

---

## Comparing the Three Planners

Run all three and compare:

```bash
# Run benchmarks
python src/main.py --config configs/benchmark_greedy_ig.json
python src/main.py --config configs/benchmark_dec_mcts.json
python src/main.py --config configs/benchmark_mh_dec_mcts.json

# Analyze results
python analyze_dual_horizon_logs.py --log-dir logs/
```

Expected trends:
- **Greedy IG**: Fastest, but leaves fragmented coverage
- **Dec-MCTS**: Better coverage than greedy, medium compute
- **MH Dec-MCTS**: Best coverage quality, highest compute cost

---

## File Structure

```
src/
├── greedy_ig_planner.py       # Benchmark 1: Greedy IG
├── dec_mcts.py                # Benchmark 2: Dec-MCTS
├── hierarchical_dec_mcts.py   # Benchmark 3: MH Dec-MCTS
├── planner.py                 # Unified interface
├── planning_utils.py          # Shared utilities (NEW)
└── async_runner.py            # Async execution

configs/
├── benchmark_greedy_ig.json        # Greedy IG multi-agent
├── benchmark_dec_mcts.json         # Dec-MCTS multi-agent
├── benchmark_mh_dec_mcts.json      # MH Dec-MCTS multi-agent
├── single_agent_greedy.json        # Greedy IG single-agent
└── single_agent_mh_dec_mcts.json   # MH Dec-MCTS single-agent

logs/
├── greedy_ig/          # Greedy IG logs
├── dec_mcts/           # Dec-MCTS logs
└── mh_dec_mcts/        # MH Dec-MCTS logs
```

---

## Troubleshooting

### "No module named planning_utils"
Make sure `src/planning_utils.py` exists and src/ is in PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Slow planning with Dec-MCTS
Reduce iterations:
```json
"dec_mcts": {
  "iterations": 50  // instead of 100
}
```

### Memory issues with large grids
Reduce tile size for MH Dec-MCTS:
```json
"hierarchical_dec_mcts": {
  "tile_size": [100, 100]  // larger tiles = fewer regions
}
```

---

## Next Steps

See [`BENCHMARK_PLANNERS.md`](BENCHMARK_PLANNERS.md) for:
- Detailed planner descriptions
- Parameter tuning guides
- Metrics for comparison
- Implementation notes
