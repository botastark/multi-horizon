# Multi-Horizon Dec-MCTS (MH_DEC_MCTS) Multi-Agent Architecture

> **Strategy:** `mh_dec_mcts` / `hierarchical_dec_mcts` — Two-level hierarchical planning

---

## 1. Overview

MH-Dec-MCTS is a **hierarchical** planner with two levels:
- **HLP (High-Level Planner)**: Long-horizon region allocation (slow cycle)
- **LLP (Low-Level Planner)**: Short-horizon motion planning (fast cycle)

**Key Features:**
- Two-level planning with reward decomposition: `g = g1(LL) + g2(HL)`
- Intent sharing at both levels (LL-intent + HL-intent)
- D-UCT discounting for asynchronous operation
- HLP guides LLP toward target regions (soft guidance)

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT (per agent)                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          HierarchicalDecMCTSPlanner                  │   │
│  │                                                      │   │
│  │   ┌───────────────┐      ┌────────────────┐         │   │
│  │   │     HLP       │      │      LLP       │         │   │
│  │   │  (Regions)    │─────▶│   (Actions)    │         │   │
│  │   │               │ guid │               │          │   │
│  │   │  horizon: 3   │ ance │  horizon: 5   │          │   │
│  │   │  regions      │      │  steps        │          │   │
│  │   └───────┬───────┘      └───────┬───────┘          │   │
│  │           │                      │                   │   │
│  │           ▼                      ▼                   │   │
│  │     HL-Intent               LL-Intent                │   │
│  │    (region seq)           (action seq)               │   │
│  └──────────┬─────────────────────┬─────────────────────┘   │
│             │                     │                         │
│             └──────────┬──────────┘                         │
│                        │                                    │
│                        ▼                                    │
│                  ┌───────────┐                              │
│                  │ IntentBus │◀───── Shared across agents   │
│                  └───────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Call Flow Diagram

```
main.py
│
├── run_multi_agent_experiment()
│   │
│   ├── [1] MultiAgentCoordinator()
│   │       └── LBPBeliefFusion()
│   │
│   ├── [2] generate_multi_agent_starts()
│   │
│   ├── [3] FOR agent_id in num_agents:
│   │       └── initialize_agent()
│   │           ├── Camera()
│   │           ├── planning("mh_dec_mcts")
│   │           │   └── create_hierarchical_planner()
│   │           │       ├── IntentBus()        # Shared
│   │           │       ├── LowLevelPlanner()
│   │           │       └── HighLevelPlanner()
│   │           └── OccupancyMap()
│   │
│   └── [4] MAIN LOOP (n_steps):
│           │
│           ├── Phase 1: OBSERVE
│           │   └── process_agent_observations()
│           │
│           ├── Phase 2: FUSE BELIEFS
│           │   └── perform_belief_fusion()
│           │
│           ├── Phase 3: SELECT ACTIONS
│           │   └── select_agent_actions()
│           │       FOR each agent:
│           │           └── HierarchicalDecMCTSPlanner.select_action()
│           │               │
│           │               ├── [A] receive_intents()
│           │               │       └── intent_bus.get_teammate_ll/hl_intents()
│           │               │
│           │               ├── [B] HLP.plan(grid_position)
│           │               │       ├── _get_teammate_target_regions()
│           │               │       ├── _compute_region_score() for each region
│           │               │       └── Return HLIntent (region sequence)
│           │               │
│           │               ├── [C] LLP.update_hl_guidance(hl_intent)
│           │               │
│           │               ├── [D] LLP.plan(current_state)
│           │               │       ├── _compute_teammate_coverage_mask()
│           │               │       ├── MCTS iterations:
│           │               │       │   ├── _simulate_trajectory()
│           │               │       │   │   ├── _compute_ig()
│           │               │       │   │   └── _compute_alignment_bonus()
│           │               │       │   └── Pick best trajectory
│           │               │       └── Return LLIntent (action sequence)
│           │               │
│           │               └── [E] BROADCAST INTENTS
│           │                       ├── intent_bus.broadcast_ll_intent()
│           │                       └── intent_bus.broadcast_hl_intent()
│           │
│           └── Phase 4: UPDATE POSITIONS
│               └── update_agent_positions()
```

---

## 4. Key Components

### 4.1 IntentBus (`hierarchical_dec_mcts.py`)

```python
class IntentBus:
    """Thread-safe communication for intent sharing."""
    
    def __init__(self, num_agents, max_history=10):
        self._ll_intents: Dict[int, LLIntent] = {}
        self._hl_intents: Dict[int, HLIntent] = {}
        self._ll_history: Dict[int, List[LLIntent]] = defaultdict(list)
        self._hl_history: Dict[int, List[HLIntent]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def broadcast_ll_intent(self, intent: LLIntent):
        with self._lock:
            self._ll_intents[intent.agent_id] = intent
            # Keep history for temporal reasoning
            self._ll_history[intent.agent_id].append(intent)
    
    def get_teammate_ll_intents(self, agent_id) -> Dict[int, LLIntent]:
        return {k: v for k, v in self._ll_intents.items() if k != agent_id}
```

### 4.2 Intent Structures

**LL-Intent (Low-Level):**
```python
@dataclass
class LLIntent:
    agent_id: int
    action_sequence: List[str]           # ['front', 'front', 'right', ...]
    state_sequence: List[Tuple]          # [(row, col, alt), ...]
    footprint_sequence: List[Tuple]      # [(imin, imax, jmin, jmax), ...]
    ig_sequence: List[float]             # Expected IG per step
    total_expected_ig: float
    timestamp: float
    horizon: int = 5
    value: float = 0.0
    
    def staleness_discount(self, decay=0.9, threshold=2.0):
        """Fast decay for short-horizon intents."""
        age = time.time() - self.timestamp
        return decay ** (age / threshold)
```

**HL-Intent (High-Level):**
```python
@dataclass
class HLIntent:
    agent_id: int
    region_sequence: List[int]           # [region_3, region_7, region_12]
    eta_sequence: List[float]            # Estimated time to reach
    completion_sequence: List[float]     # Estimated completion time
    score_sequence: List[float]          # Priority scores
    current_target_region: int           # Immediate target
    target_center: Tuple[float, float]   # Center of target region
    timestamp: float
    horizon: int = 3
    
    def staleness_discount(self, decay=0.95, threshold=5.0):
        """Slow decay for long-horizon intents."""
        age = time.time() - self.timestamp
        return decay ** (age / threshold)
```

---

## 5. High-Level Planner (HLP)

### 5.1 Region Partitioning

```python
class HighLevelPlanner:
    def __init__(self, agent_id, num_agents, grid_shape, tile_size=(100, 100)):
        self.regions = self._partition_grid()
    
    def _partition_grid(self) -> Dict[int, Dict]:
        """Divide grid into rectangular regions."""
        regions = {}
        region_id = 0
        
        for i in range(0, H, tile_h):
            for j in range(0, W, tile_w):
                regions[region_id] = {
                    "bounds": ((i, min(i+tile_h, H)), (j, min(j+tile_w, W))),
                    "center": ((i + i_end) / 2, (j + j_end) / 2),
                    "area": (i_end - i) * (j_end - j),
                }
                region_id += 1
        
        return regions
```

### 5.2 Region Scoring

```python
def _compute_region_score(self, region_id, agent_position, teammate_targets):
    """Score = uncertainty - distance_penalty - teammate_conflict."""
    
    # Base score: remaining uncertainty
    remaining_uncertainty = 1.0 - self._region_coverage[region_id]
    
    # Distance penalty
    center = self.regions[region_id]["center"]
    distance = np.linalg.norm(agent_position - center)
    distance_penalty = 0.3 * (distance / max_distance)
    
    # Teammate conflict with D-UCT discount
    conflict_penalty = 0.0
    for teammate_id, (targets, staleness_discount) in teammate_targets.items():
        if region_id in targets:
            conflict_penalty += 0.5 * staleness_discount
    
    return max(0.0, remaining_uncertainty - distance_penalty - conflict_penalty)
```

### 5.3 HLP Planning

```python
def plan(self, current_position) -> HLIntent:
    if not self._should_replan():
        return self.current_intent  # Reuse cached plan
    
    # Get teammate target regions
    teammate_targets = self._get_teammate_target_regions()
    
    # Score all regions
    region_scores = {}
    for region_id in self.regions:
        region_scores[region_id] = self._compute_region_score(
            region_id, current_position, teammate_targets
        )
    
    # Select top regions (greedy)
    sorted_regions = sorted(region_scores.items(), key=lambda x: -x[1])
    region_sequence = [r for r, s in sorted_regions[:self.horizon] if s > 0]
    
    return HLIntent(
        agent_id=self.agent_id,
        region_sequence=region_sequence,
        current_target_region=region_sequence[0] if region_sequence else None,
        target_center=self.regions[region_sequence[0]]["center"],
        ...
    )
```

---

## 6. Low-Level Planner (LLP)

### 6.1 Teammate Coverage Mask

```python
def _compute_teammate_coverage_mask(self) -> np.ndarray:
    """Discount cells covered by teammate LL intents."""
    coverage_discount = np.ones((H, W), dtype=float)
    
    for teammate_id, ll_intent in self._teammate_ll_intents.items():
        if ll_intent.is_stale():
            continue
        
        staleness = ll_intent.staleness_discount()
        
        for step_idx, fp in enumerate(ll_intent.footprint_sequence):
            step_discount = self.intent_discount ** step_idx * staleness
            coverage_discount[fp_slice] *= step_discount
    
    return coverage_discount
```

### 6.2 Alignment Bonus (HLP Guidance)

```python
def _compute_alignment_bonus(self, position, prev_position) -> float:
    """Bonus for moving toward HLP target."""
    if self._hl_guidance is None or self._hl_guidance.target_center is None:
        return 0.0
    
    target = self._hl_guidance.target_center
    
    dist_before = np.linalg.norm(prev_position - target)
    dist_after = np.linalg.norm(position - target)
    
    improvement = dist_before - dist_after
    if improvement > 0:
        return 0.3 * improvement / max_distance
    return 0.0  # No penalty for moving away (soft guidance)
```

### 6.3 LLP Planning (MCTS-lite)

```python
def plan(self, current_state) -> LLIntent:
    coverage_discount = self._compute_teammate_coverage_mask()
    
    best_reward = -inf
    best_actions = []
    
    # Random sampling MCTS (simplified)
    for _ in range(self.num_iterations):
        actions = [random.choice(self.actions) for _ in range(self.horizon)]
        
        reward, states, footprints, igs = self._simulate_trajectory(
            current_state, actions, coverage_discount
        )
        
        if reward > best_reward:
            best_reward = reward
            best_actions = actions
            best_states = states
            best_footprints = footprints
            best_igs = igs
    
    return LLIntent(
        agent_id=self.agent_id,
        action_sequence=best_actions,
        state_sequence=best_states,
        footprint_sequence=best_footprints,
        ig_sequence=best_igs,
        total_expected_ig=sum(best_igs),
        ...
    )
```

### 6.4 Trajectory Simulation

```python
def _simulate_trajectory(self, start_state, actions, coverage_discount):
    """Simulate trajectory and compute total reward (g1)."""
    total_reward = 0.0
    state_sequence = [start_state]
    
    for step_idx, action in enumerate(actions):
        next_state = camera.x_future(action)
        
        # IG with teammate discount
        ig = self._compute_ig(next_state, coverage_discount)
        
        # Alignment with HLP target
        alignment = self._compute_alignment_bonus(next_state, current_pos)
        
        # Discounted reward
        step_reward = (self.discount ** step_idx) * (ig + alignment)
        total_reward += step_reward
        
        state_sequence.append(next_state)
    
    return total_reward, state_sequence, footprints, igs
```

---

## 7. Hierarchical Planner Integration

```python
class HierarchicalDecMCTSPlanner:
    def __init__(self, agent_id, num_agents, camera, grid_info, intent_bus, ...):
        self.llp = LowLevelPlanner(agent_id, camera, grid_info, ...)
        self.hlp = HighLevelPlanner(agent_id, num_agents, grid_info.shape, ...)
        self.intent_bus = intent_bus
    
    def plan(self) -> Tuple[str, Dict]:
        # Step 1: Receive teammate intents
        self.receive_intents()
        
        # Step 2: Run HLP (may reuse cached plan)
        grid_pos = camera.convert_xy_ij(self._current_position)
        hl_intent = self.hlp.plan(grid_pos)
        
        # Step 3: Update LLP with HLP guidance
        self.llp.update_hl_guidance(hl_intent)
        
        # Step 4: Run LLP
        current_state = (x, y, altitude)
        ll_intent = self.llp.plan(current_state)
        
        # Step 5: Broadcast intents
        self.intent_bus.broadcast_ll_intent(ll_intent)
        self.intent_bus.broadcast_hl_intent(hl_intent)
        
        return ll_intent.action_sequence[0], metrics
```

---

## 8. Reward Decomposition

The total reward follows the paper's formulation:

```
g = g1(LL intents) + g2(all intents)

g1 = Σ_t γ^t * [IG(t) + alignment(t) - teammate_overlap(t)]
     └── LLP computes this

g2 = Σ_r [region_uncertainty(r) - distance(r) - teammate_conflict(r)]
     └── HLP computes this
```

---

## 9. Belief Fusion (Same Infrastructure)

MH-Dec-MCTS uses the same belief fusion as other strategies:

```
1. Per-agent: OG update + local LBP (mapper_LBP.py)
2. Coordinator: LBPBeliefFusion with news mode (multi_agent_coordinator.py)
3. Pairwise potentials: equal/biased/adaptive
```

---

## 10. Configuration

```json
{
  "action_strategy": "mh_dec_mcts",
  "num_agents": 2,
  
  "hierarchical_mcts": {
    "llp_horizon": 5,
    "llp_iterations": 100,
    "hlp_horizon": 3,
    "hlp_iterations": 50,
    "tile_size": [100, 100],
    "hlp_replan_interval": 1.0
  },
  
  "decentralized": {
    "communication_range": 150.0,
    "d_uct": {
      "decay_factor": 0.9,
      "threshold_sec": 2.0
    }
  }
}
```

---

## 11. Comparison: Greedy vs Dec-MCTS vs MH-Dec-MCTS

| Aspect | Greedy IG | Dec-MCTS | MH-Dec-MCTS |
|--------|-----------|----------|-------------|
| Levels | 1 | 1 | 2 (HLP + LLP) |
| Lookahead | 1 step | N steps | HLP: regions, LLP: steps |
| Intent | Footprint | Trajectory | LL + HL intents |
| Region allocation | None | None | Yes (HLP) |
| Guidance | None | None | HLP → LLP |
| Computation | Lowest | Medium | Highest |

---

## 12. Debugging Tips

### Check HLP Decisions
```python
# In HierarchicalDecMCTSPlanner.plan():
print(f"Agent {self.agent_id} HLP:")
print(f"  Target region: {hl_intent.current_target_region}")
print(f"  Region scores: {dict(self.hlp._region_coverage)}")
print(f"  Teammate targets: {self.hlp._get_teammate_target_regions()}")
```

### Check LLP Guidance
```python
# In LowLevelPlanner.plan():
if self._hl_guidance:
    print(f"LLP guidance: target_center={self._hl_guidance.target_center}")
print(f"Alignment adjustments: {self._alignment_adjustments}")
```

### Check Intent Bus
```python
# In HierarchicalDecMCTSPlanner.receive_intents():
ll = self.intent_bus.get_teammate_ll_intents(self.agent_id)
hl = self.intent_bus.get_teammate_hl_intents(self.agent_id)
print(f"Agent {self.agent_id} received:")
print(f"  LL intents from: {list(ll.keys())}")
print(f"  HL intents from: {list(hl.keys())}")
for tid, hi in hl.items():
    print(f"    Agent {tid} targeting region {hi.current_target_region}")
```

### Check Region Partitioning
```python
# At initialization:
print(f"HLP partitioned into {len(self.hlp.regions)} regions")
for rid, r in self.hlp.regions.items():
    print(f"  Region {rid}: bounds={r['bounds']}, center={r['center']}")
```

### Check Trajectory Simulation
```python
# In _simulate_trajectory():
for step_idx, action in enumerate(actions):
    print(f"  Step {step_idx}: action={action}, ig={ig:.2f}, align={alignment:.2f}")
```

---

## 13. File References

| File | Purpose |
|------|---------|
| `src/main.py` | Entry point, experiment loop |
| `src/experiment_utils.py` | Agent init, observation, fusion utilities |
| `src/hierarchical_dec_mcts.py` | HierarchicalDecMCTSPlanner, LLP, HLP, IntentBus |
| `src/multi_agent_coordinator.py` | MultiAgentCoordinator, LBPBeliefFusion |
| `src/mapper_LBP.py` | OccupancyMap (per-agent local LBP) |
| `src/planner.py` | Planner factory (dispatches to hierarchical) |
| `src/helper.py` | H(), cH(), adaptive_weights_matrix() |

---

## 14. Visualization Support

The planner exposes metadata for visualization:

```python
# In planner.py or main.py:
region_metadata = planner._hierarchical_planner.current_region_metadata
selected_region = planner._hierarchical_planner.current_selected_region
region_scores = planner._hierarchical_planner.current_region_scores

# Use in plot_terrain():
plot_terrain(
    ...,
    region_metadata=region_metadata,
    selected_region_id=selected_region,
    region_scores=region_scores,
)
```
