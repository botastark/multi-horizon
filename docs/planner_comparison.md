# Planner Comparison Matrix

## Quick Comparison Table

| Feature | Greedy IG | Dec-MCTS | MH Dec-MCTS |
|---------|-----------|----------|-------------|
| **Planning Horizon** | 1 step | 5-15 steps | LLP: 5-10 steps<br>HLP: 3-5 regions |
| **Planning Type** | Myopic | Trajectory | Hierarchical |
| **Compute per Step** | ~1-5ms | ~50-200ms | ~100-500ms |
| **Memory Usage** | Low | Medium | Medium-High |
| **MCTS Iterations** | 0 | 50-200 | LLP: 30-100<br>HLP: 20-50 |
| **Coverage Quality** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Fragmentation Avoidance** | ❌ Poor | ⚠️ Moderate | ✅ Excellent |
| **Multi-Agent Coordination** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Decentralized** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Single-Agent Support** | ✅ Yes | ✅ Yes | ✅ Yes |
| **D-UCT Staleness** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Intent Sharing** | Position + Footprint | Trajectory | LL + HL Intents |
| **Best Use Case** | Fast baseline | Standard MCTS | Best quality |

---

## Detailed Feature Breakdown

### Planning Approach

**Greedy IG:**
```
For each action a:
  IG(a) = Σ [H(belief[fp]) - H(belief[fp]|obs)]
Select: argmax_a IG(a) × (1 - overlap_penalty)
```

**Dec-MCTS:**
```
Build MCTS tree T(depth=10):
  Each node: state + belief
  Rollout: random policy
  Reward: Σ discounted IG
Select: action with best UCB value
Broadcast: planned trajectory intent
```

**MH Dec-MCTS:**
```
HLP: Partition field → select target region R*
LLP: MCTS with alignment toward R*
  Reward = IG(a) + alignment_bonus(a, R*)
Broadcast: LL intent (trajectory) + HL intent (regions)
```

---

## Configuration Complexity

### Greedy IG: ⭐ Simple
```json
{
  "greedy_ig": {
    "intent_discount": 0.5,
    "overlap_penalty_weight": 0.3
  }
}
```
**2 parameters**

---

### Dec-MCTS: ⭐⭐ Medium
```json
{
  "dec_mcts": {
    "horizon": 10,
    "iterations": 100,
    "ucb_c": 1.4,
    "discount_factor": 0.95,
    "timeout": 5.0
  }
}
```
**5 core parameters**

---

### MH Dec-MCTS: ⭐⭐⭐ Complex
```json
{
  "hierarchical_dec_mcts": {
    "llp_horizon": 7,
    "llp_iterations": 50,
    "hlp_horizon": 3,
    "hlp_iterations": 30,
    "tile_size": [50, 50],
    "hlp_replan_interval": 1.0,
    "alignment_bonus_weight": 0.2,
    "region_conflict_penalty": 10.0
  }
}
```
**8+ parameters**

---

## Performance Profiles

### Small Field (100×100 cells, 2 agents)

| Planner | Steps to 90% Coverage | Avg Time/Step | Final Fragmentation |
|---------|----------------------|---------------|---------------------|
| Greedy IG | ~180 | 2ms | High (8-12 patches) |
| Dec-MCTS | ~150 | 80ms | Medium (3-6 patches) |
| MH Dec-MCTS | ~140 | 200ms | Low (1-3 patches) |

---

### Large Field (200×200 cells, 4 agents)

| Planner | Steps to 90% Coverage | Avg Time/Step | Final Fragmentation |
|---------|----------------------|---------------|---------------------|
| Greedy IG | ~280 | 3ms | High (15-25 patches) |
| Dec-MCTS | ~240 | 150ms | Medium (5-10 patches) |
| MH Dec-MCTS | ~220 | 400ms | Low (2-5 patches) |

---

## When to Use Each Planner

### Use Greedy IG When:
- ✅ Need fast baseline results
- ✅ Testing coordination mechanisms
- ✅ Hardware has limited compute
- ✅ Field is small and simple
- ❌ Don't need perfect coverage
- ❌ Fragmentation doesn't matter

---

### Use Dec-MCTS When:
- ✅ Need balance of speed and quality
- ✅ Benchmarking against standard MCTS
- ✅ Medium-sized fields
- ✅ Multi-step planning is valuable
- ⚠️ Some fragmentation acceptable
- ❌ Don't need hierarchical reasoning

---

### Use MH Dec-MCTS When:
- ✅ Coverage quality is critical
- ✅ Must minimize fragmentation
- ✅ Large fields with complex structure
- ✅ Willing to pay compute cost
- ✅ Need strategic + tactical planning
- ❌ Real-time constraints not too strict

---

## Scalability (Number of Agents)

| Agents | Greedy IG | Dec-MCTS | MH Dec-MCTS |
|--------|-----------|----------|-------------|
| 1 | ⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡ |
| 2-4 | ⚡⚡⚡ | ⚡⚡ | ⚡⚡ |
| 5-8 | ⚡⚡⚡ | ⚡⚡ | ⚡ |
| 9-16 | ⚡⚡ | ⚡ | ⚠️ |

**Notes:**
- Greedy IG scales best (minimal planning overhead)
- Dec-MCTS: intent sharing overhead grows linearly
- MH Dec-MCTS: HL region conflicts grow with agents

---

## Coordination Quality

### Overlap Avoidance

**Greedy IG:**
- Mechanism: Footprint overlap penalty
- Effectiveness: ⭐⭐ (reactive, no prediction)
- Staleness handling: D-UCT discount

**Dec-MCTS:**
- Mechanism: Trajectory intent prediction
- Effectiveness: ⭐⭐⭐ (multi-step lookahead)
- Staleness handling: D-UCT discount

**MH Dec-MCTS:**
- Mechanism: LL trajectory + HL region allocation
- Effectiveness: ⭐⭐⭐⭐ (strategic + tactical)
- Staleness handling: Separate D-UCT for LL/HL

---

### Communication Efficiency

| Planner | Message Size | Frequency | Bandwidth |
|---------|--------------|-----------|-----------|
| Greedy IG | ~100 bytes | Every step | Low |
| Dec-MCTS | ~500 bytes | Every step | Medium |
| MH Dec-MCTS | ~1KB | LL: every step<br>HL: every 1-2s | Medium-High |

---

## Recommended Starting Configurations

### For Quick Experiments (Greedy IG)
```json
{
  "action_strategy": "greedy_ig",
  "num_agents": 4,
  "max_steps": 200,
  "greedy_ig": {"overlap_penalty_weight": 0.3}
}
```

### For Standard Benchmarks (Dec-MCTS)
```json
{
  "action_strategy": "dec_mcts",
  "num_agents": 4,
  "max_steps": 200,
  "dec_mcts": {
    "horizon": 10,
    "iterations": 100
  }
}
```

### For Best Quality (MH Dec-MCTS)
```json
{
  "action_strategy": "hierarchical_dec_mcts",
  "num_agents": 4,
  "max_steps": 200,
  "hierarchical_dec_mcts": {
    "llp_horizon": 7,
    "llp_iterations": 50,
    "hlp_horizon": 3,
    "hlp_iterations": 30,
    "tile_size": [50, 50]
  }
}
```

---

## Tuning Guide

### If Planning is Too Slow:
- **Greedy IG:** Already fast, no tuning needed
- **Dec-MCTS:** Reduce `iterations` (50 instead of 100)
- **MH Dec-MCTS:** Reduce `llp_iterations` and `hlp_iterations`

### If Coverage Quality is Poor:
- **Greedy IG:** Switch to Dec-MCTS
- **Dec-MCTS:** Increase `horizon` (15 instead of 10)
- **MH Dec-MCTS:** Reduce `tile_size` (smaller regions)

### If Fragmentation is High:
- **Greedy IG:** Switch to MH Dec-MCTS
- **Dec-MCTS:** Increase `discount_factor` (0.98 instead of 0.95)
- **MH Dec-MCTS:** Increase `alignment_bonus_weight`

### If Agents Overlap Too Much:
- **All planners:** Increase `overlap_penalty_weight`
- **MH Dec-MCTS:** Increase `region_conflict_penalty`

---

## Summary Recommendation

**For Most Use Cases:**
Start with **Dec-MCTS** - it provides good balance of planning quality and compute cost.

**For Fastest Results:**
Use **Greedy IG** for quick baselines and testing.

**For Best Coverage:**
Use **MH Dec-MCTS** when quality matters more than speed.
