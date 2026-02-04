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
│  │   ┌───────────────┐      ┌────────────────┐          │   │
│  │   │     HLP       │      │      LLP       │          │   │
│  │   │  (Regions)    │─────▶│   (Actions)    │          │   │
│  │   │               │ guid │                │          │   │
│  │   │  horizon: 3   │ ance │  horizon: 5    │          │   │
│  │   │  regions      │      │  steps         │          │   │
│  │   └───────┬───────┘      └───────┬────────┘          │   │
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
│           │               │       │   │   └── _compute_ig()
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

**Current Implementation: MCTS Region Search**

```python
def plan(self, current_position) -> HLIntent:
    if not self._should_replan():
        return self.current_intent  # Reuse cached plan
    
    # Get teammate target regions
    teammate_targets = self._get_teammate_target_regions()
    
    # Run MCTS over region sequences
    best_sequence = self._run_mcts_region_search(current_position, teammate_targets)
    
    # Build intent from best sequence
    region_sequence = best_sequence
    eta_sequence = []
    completion_sequence = []
    score_sequence = []
    
    cumulative_time = 0.0
    pos = current_position
    
    for region_id in region_sequence:
        # Compute marginal score
        score = self._compute_region_score(region_id, pos, teammate_targets)
        
        # Estimate completion time
        eta = self._estimate_region_completion_time(region_id, pos)
        cumulative_time += eta
        
        eta_sequence.append(cumulative_time)
        completion_sequence.append(cumulative_time + eta)
        score_sequence.append(score)
        
        pos = self.regions[region_id]["center"]
    
    return HLIntent(
        agent_id=self.agent_id,
        region_sequence=region_sequence,
        current_target_region=region_sequence[0] if region_sequence else None,
        target_center=self.regions[region_sequence[0]]["center"],
        value=sum(score_sequence),
        iterations=self.num_iterations,
        ...
    )
```

**Note:** HLP now uses **full MCTS tree search** over region sequences, making it symmetric with LLP. The MCTS search explores different region orderings using UCB selection, random rollout policies, and marginal g₂ evaluation for complete sequences.

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

### 6.2 LLP Reward Computation (Paper-Correct)

**No explicit alignment bonus** - alignment emerges through g2 conditioning:

```python
def _simulate_trajectory(self, start_state, actions, coverage_discount):
    """
    Simulate trajectory and compute reward: r_LLP = g1 + g2
    
    g1: Immediate IG (discounted sum)
    g2: Mission completion time estimate (includes HL intents)
    
    Alignment emerges because g2 is conditioned on LL trajectory,
    and HLP guides by setting target regions that reduce g2.
    """
    # Compute g1: discounted IG over trajectory
    g1_reward = 0.0
    for step_idx, action in enumerate(actions):
        ig = self._compute_ig(next_pos, next_alt, coverage_discount)
        g1_reward += (self.discount ** step_idx) * ig
    
    # Compute g2: mission completion time
    g2_value = self._compute_g2_for_trajectory(
        state_sequence, footprint_sequence, ig_sequence
    )
    
    # Total reward (no explicit alignment bonus)
    total_reward = g1_reward + g2_value
    return total_reward, state_sequence, footprint_sequence, ig_sequence
```

### 6.3 LLP Planning (Random Rollout MCTS)

```python
def plan(self, current_state) -> LLIntent:
    coverage_discount = self._compute_teammate_coverage_mask()
    
    best_reward = -inf
    best_actions = []
    
    # Random rollout MCTS
    for _ in range(self.num_iterations):
        # Sample random action sequence
        actions = [random.choice(self.actions) for _ in range(self.horizon)]
        
        # Simulate and evaluate using g1 + g2
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
        value=best_reward,  # This is g1 + g2
        ...
    )
```

**Key Properties:**
- Uses random rollout sampling (simplified MCTS without UCB tree)
- Evaluates complete trajectories using g1 + g2 reward
- Coordination through teammate coverage discount mask
- HLP guidance emerges naturally through g2 conditioning

**Why Random Rollout (Not UCB Tree)?**

The LLP uses a simplified MCTS approach called **random rollout sampling** instead of the full UCB tree search used in Dec-MCTS:

| Aspect | Dec-MCTS (UCB Tree) | MH-LLP (Random Rollout) |
|--------|---------------------|-------------------------|
| **Search method** | UCB tree with Selection→Expansion→Simulation→Backpropagation | Random sampling of complete trajectories |
| **Tree structure** | Builds explicit tree, reuses nodes across iterations | No tree structure, each iteration is independent |
| **Action selection** | UCB1 formula balances exploitation vs exploration | Uniform random sampling |
| **Memory** | O(iterations × horizon) nodes stored | O(1) - only stores best trajectory |
| **Compute per iteration** | Tree traversal + UCB calculation | Direct trajectory simulation |
| **Planning horizon** | 10 steps | 3 steps (shorter) |

**Why this design choice?**

1. **Computational efficiency**: LLP runs frequently (every step) and needs to be fast
   - Random sampling: ~50 iterations × 3 steps = 150 simulations
   - UCB tree would need more iterations to build meaningful tree statistics

2. **Short horizon**: With only 3 steps, exhaustive search space is manageable
   - 7 actions³ = 343 possible sequences (small enough for random sampling)
   - UCB tree benefits more from longer horizons where search space is huge

3. **HLP provides guidance**: The g2 reward component already incorporates long-horizon reasoning
   - HLP (with full UCB tree) handles strategic region allocation
   - LLP focuses on short-term tactical motion planning
   
4. **Hierarchical division of labor**:
   - HLP: Complex search (UCB tree over regions, marginal g2 evaluation)
   - LLP: Fast reactive planning (random rollout, immediate IG + g2)

**What "MCTS" means in LLP:**
The LLP is called "MCTS" because it uses Monte Carlo simulation (random rollouts) to estimate trajectory values, which is a core MCTS concept. However, it's a **simplified MCTS variant** that skips the tree-building and UCB selection phases, keeping only the simulation component.

**Implementation Details** ([hierarchical_dec_mcts.py](../src/hierarchical_dec_mcts.py)):

| Function | Line | Purpose |
|----------|------|---------|
| `LowLevelPlanner.plan()` | ~747 | **Main planning loop** - generates random action sequences |
| `_simulate_trajectory()` | ~661 | **Trajectory simulation** - evaluates g1 + g2 for action sequence |
| `_compute_ig()` | ~511 | **IG computation** - applies teammate coverage discount |
| `_compute_teammate_coverage_mask()` | ~480 | **Coordination** - creates discount mask from teammate intents |

**Key code snippet** (line 781 in `plan()`):
```python
for _ in range(self.num_iterations):  # 50 iterations
    # Random action sequence (THIS IS THE RANDOM ROLLOUT)
    actions = [np.random.choice(self.actions) for _ in range(self.horizon)]  # 3 steps
    
    # Simulate and evaluate using g1 + g2
    reward, states, footprints, igs = self._simulate_trajectory(
        current_state, actions, coverage_discount
    )
    
    # Keep best trajectory found
    if reward > best_reward:
        best_reward = reward
        best_actions = actions
        ...
```

**No UCB tree construction** - the code shows:
- No `DecMCTSNode` class instantiation (unlike Dec-MCTS)
- No `best_child()` or `_tree_policy()` calls
- No tree structure stored (`self._tree` exists but marked as `_tree_valid = False`)
- Each iteration samples independently: `np.random.choice(self.actions)`

---

**For comparison: HLP's UCB Tree Implementation** ([hierarchical_dec_mcts.py](../src/hierarchical_dec_mcts.py)):

| Function | Line | Purpose |
|----------|------|---------|
| `HighLevelPlanner._run_mcts_region_search()` | ~1183 | **Main MCTS loop** - builds UCB tree over region sequences |
| `_mcts_iteration()` | ~1222 | **Selection + Expansion + Backprop** - full MCTS iteration |
| `_select_best_region_ucb()` | ~1298 | **UCB selection** - chooses region using UCB1 formula |
| `_rollout_region_sequence()` | ~1328 | **Random completion** - completes partial sequence to horizon |
| `_backpropagate()` | ~1379 | **Update tree** - propagates value back through path |

**Key code snippet** (line 1236 in `_mcts_iteration()`):
```python
# Selection phase: traverse tree using UCB
while len(sequence) < self.horizon:
    node = tree[state]
    node["visits"] += 1
    
    unexplored = [r for r in available_regions if r not in node["children"]]
    
    if unexplored:
        # Expansion: pick random unexplored action
        action = random.choice(unexplored)
        node["children"][action] = {"visits": 0, "value": 0.0}
        # ... rollout and backpropagate
    else:
        # Selection: use UCB1 formula (THIS IS THE UCB TREE)
        action = self._select_best_region_ucb(node, available_regions)
        # ... traverse to child
        
# UCB1 formula (line 1318):
exploitation = child_value / child_visits
exploration = ucb_c * np.sqrt(np.log(parent_visits) / child_visits)
ucb_score = exploitation + exploration
```

**HLP builds explicit tree** - dictionary structure stores:
- Visit counts per state-action pair
- Cumulative values for each node
- Parent-child relationships for tree traversal
    
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

The total reward follows the paper's formulation (Seiler et al., 2024):

```
g = g1(LL intents) + g2(all intents)
```

### Paper-Correct Implementation (Current)

**LLP Reward:**
```
r_LLP = g1 + g2

g1 = Σ_t γ^t * IG(t)
     └── Immediate information gain (discounted)

g2 = remaining_uncovered_area / nominal_coverage_rate
     └── Mission completion time estimate
```

**HLP Reward:**
```
r_HLP = g2(with my HL intent) - g2(with null HL intent)
        └── Marginal contribution only
```

### Key Properties

✅ **Two-component reward**: LLP uses g1 (IG) + g2 (mission completion time)
✅ **Centralized g2**: Single g2() function in `g2_evaluator.py` used by both planners
✅ **Time-like g2**: Lower values = better (faster mission completion)
✅ **Marginal HLP**: HLP evaluates its contribution, not absolute region value

### g2 Implementation

Located in `src/g2_evaluator.py`:

**Two-Phase Evaluation** (Algorithm 2 from paper):

```python
def g2(ll_intents, hl_intents, env_state, agent_id=None):
    """
    Phase 1: Execute LL intents (fixed)
        - Compute LL execution time
        - Get cells covered by LL footprints
        - Compute LL overlap penalty
        - Get agent end positions after LL
    
    Phase 2: Estimate HL completion
        - Remaining area = total uncertain - LL covered
        - Estimate time to complete using HL intents
        - Starting from LL end positions
    
    Return: g2 = LL_time + HL_time + overlap_penalty
    """
```

**Key Properties:**
- ✅ Time-like (units: coverage time)
- ✅ Conditioned on LL intents (Phase 1 → Phase 2)
- ✅ g2 decreases as LL progresses
- ✅ HL naturally avoids LL-covered regions
- ✅ Bottom-up information flow (LL → HL)
- ✅ HLP guidance emerges through g2 conditioning

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

### Benchmark Configuration (`configs/benchmark_mh_dec_mcts.json`)

```json
{
  "action_strategy": "mh_dec_mcts",
  "num_agents": 4,
  "n_steps": 100,
  "iters": [0, 20],
  "correlation_types": ["adaptive", "equal", "biased"],
  
  "hierarchical_dec_mcts": {
    "mode_labels": ["IG","IGd","IG_BM","IG_BS","IGd_BM","IGd_BS"],
    "llp_horizon": 3,
    "llp_iterations": 50,
    "llp_ucb_c": 1.4,
    "llp_discount_factor": 0.95,
    
    "hlp_horizon": 10,
    "hlp_iterations": 30,
    "hlp_ucb_c": 1.0,
    "hlp_discount_factor": 0.98,
    
    "tile_size": [50, 50],
    "hlp_replan_interval": 1.0,
    
    "intent_sharing": {
      "ll_broadcast_interval": 0.1,
      "hl_broadcast_interval": 0.5,
      "max_history": 10
    }
  },
  
  "decentralized": {
    "communication_range": 15.625,
    "overlap_penalty_weight": 0.3,
    "d_uct": {
      "decay_factor": 0.9,
      "threshold_sec": 2.0
    }
  }
}
```

**Key Parameters:**
- **LLP**: `llp_horizon=3` steps, `llp_iterations=50`, random rollout MCTS (Note: `llp_ucb_c=1.4` is a legacy parameter - **not used** since LLP uses random sampling, not UCB tree)
- **HLP**: `hlp_horizon=10` regions, `hlp_iterations=30`, UCB tree MCTS with `hlp_ucb_c=1.0`
- **tile_size**: `[50, 50]` defines region grid (50×50 tiles per region)
- **Communication**: `communication_range=15.625m` (matches other strategies for comparison)
- **Mode Labels**: 6 modes available - `IG` (no sharing), `IGd` (position sharing), `IG_BS`/`IG_BM` (IG + news sharing), `IGd_BS`/`IGd_BM` (IGd + news sharing)
- **Intent sharing**: LL broadcasts every 0.1s, HL every 0.5s

---

## 11. Full MCTS for HLP (Step 7)

**Status:** ✅ Implemented

The HLP now uses **full MCTS tree search** over region sequences, making it symmetric with LLP:

### MCTS Components

| Component | Implementation |
|-----------|----------------|
| **State** | `frozenset` of visited regions |
| **Action** | Next region ID to visit |
| **Planning** | UCB tree search with random rollout |
| **Horizon** | Tree depth = `hlp_horizon` regions |
| **Rollout** | Random region ordering to complete sequence |
| **Reward** | Marginal g₂ (full sequence evaluation) |
| **Selection** | UCB with exploration constant = 1.0 |
| **Expansion** | Add random unexplored region |
| **Backpropagation** | Update visit counts and cumulative values |

### Key Features

✅ **Full UCB tree search**: Unlike LLP which uses random rollout, HLP builds and traverses UCB tree  
✅ **Marginal g₂ evaluation**: Evaluates complete region sequence's contribution  
✅ **Strategic planning**: Handles long-horizon region allocation (10 regions vs LLP's 3 steps)  
✅ **Coordination**: Bottom-up LL→HL flow via g₂ conditioning  

### Algorithm Flow

```python
def _run_mcts_region_search(start_position, teammate_targets):
    tree = {frozenset(): {'visits': 0, 'value': 0.0, 'children': {}}}
    
    for iteration in range(num_iterations):
        # 1. Selection: traverse tree using UCB
        state = frozenset()  # root
        sequence = []
        path = []
        
        while len(sequence) < horizon:
            available_regions = get_unvisited(state)
            
            if has_unexplored_children(state):
                # 2. Expansion: pick random unexplored region
                action = random.choice(unexplored)
                sequence.append(action)
                path.append((state, action))
                
                # 3. Simulation: random rollout to complete sequence
                value = rollout_and_evaluate(sequence, start_position)
                
                # 4. Backpropagation: update tree
                backpropagate(tree, path, value)
                break
            else:
                # Selection: UCB to pick best child
                action = select_best_ucb(state, available_regions)
                sequence.append(action)
                path.append((state, action))
                state = frozenset(sequence)
        
    return best_sequence_found
```

### Benefits Over Greedy

✅ **Better exploration**: Discovers non-obvious region orderings  
✅ **Sequence optimization**: Considers full trajectory, not just individual regions  
✅ **Adaptivity**: Adjusts to teammate conflicts through marginal g₂  
✅ **Consistency**: Both planning levels use same algorithmic approach

---

## 12. Comparison: Incremental Planner Progression

The framework supports **4 baselines** representing incremental planning complexity:

| Aspect | 1. Greedy IG | 2. Dec-MCTS | 3. MH-Dec-MCTS<br>(full) | 4. MH-Dec-MCTS<br>(efficient) |
|--------|-------------|------------|---------------------------|-------------------------------|
| **Levels** | 1 | 1 | 2 (HLP + LLP) | 2 (HLP + LLP) |
| **Lookahead** | 1 step | 10 steps | HLP: 10 regions<br>LLP: 3 steps | HLP: 10 regions<br>LLP: 3 steps |
| **Search method** | Enumerate actions | **UCB tree** | HLP: **UCB tree**<br>LLP: **UCB tree** | HLP: **UCB tree**<br>LLP: **Random rollout** |
| **Tree structure** | None | Yes | HLP: Yes<br>LLP: Yes | HLP: Yes<br>LLP: No |
| **Action selection** | Max IG | UCB1 formula | HLP: UCB1<br>LLP: UCB1 | HLP: UCB1<br>LLP: Random sampling |
| **Intent** | Footprint | Trajectory (10 steps) | LL (3 steps) + HL (region seq) | LL (3 steps) + HL (region seq) |
| **Region allocation** | None | None | Yes (HLP MCTS) | Yes (HLP MCTS) |
| **Guidance** | None | None | HLP → LLP via g2 | HLP → LLP via g2 |
| **Reward** | IG only | IG + overlap penalty | g1 (IG) + g2 (mission time) | g1 (IG) + g2 (mission time) |
| **Iterations** | 1 per action | 100 | HLP: 30<br>LLP: 50 | HLP: 30<br>LLP: 50 |
| **Computation** | Lowest<br>(~7 evals) | Medium<br>(~1000 sims) | Highest<br>(~450 sims, best quality) | High<br>(~450 sims, optimized) |
| **Memory** | O(1) | O(iter × horizon) | HLP: O(iter × regions)<br>LLP: O(iter × horizon) | HLP: O(iter × regions)<br>LLP: O(1) |
| **use_mcts_llp** | N/A | N/A | **True** | **False** |

**Incremental Progression:**
1. **Greedy IG**: Reactive baseline, no lookahead
2. **Dec-MCTS**: Adds multi-step planning with UCB tree search
3. **MH-Dec-MCTS (full)**: Adds hierarchical structure, both HLP and LLP use full MCTS tree search
4. **MH-Dec-MCTS (efficient)**: Optimizes (3) by replacing LLP tree search with random rollout to reduce computational cost

**Key Design Insight (3) → (4):**
- Baseline **(3)** establishes the full hierarchical approach with both planners using MCTS
- Baseline **(4)** tests an optimization hypothesis: Can we replace LLP's expensive tree search with simple random rollout and maintain similar performance?
  - **Rationale**: HLP already provides strategic guidance via g₂, so LLP may not need full tree search
  - **Trade-off**: (4) is computationally cheaper but may find slightly lower-quality short-horizon plans
  - **Result**: Experiments compare whether the performance loss (if any) justifies the computational savings

---

## 13. Debugging Tips

### Check HLP Decisions (MCTS)
```python
# In HierarchicalDecMCTSPlanner.plan():
print(f"Agent {self.agent_id} HLP:")
print(f"  Target region: {hl_intent.current_target_region}")
print(f"  Best sequence: {hl_intent.region_sequence}")
print(f"  Sequence value: {hl_intent.value}")
print(f"  MCTS iterations: {hl_intent.iterations}")
print(f"  Region coverage: {dict(self.hlp._region_coverage)}")
print(f"  Teammate targets: {self.hlp._get_teammate_target_regions()}")
```

### Check LLP Guidance
```python
# In LowLevelPlanner.plan():
if self._hl_guidance:
    print(f"LLP guidance: target_center={self._hl_guidance.target_center}")
print(f"Coverage discount stats: min={coverage_discount.min():.2f}, mean={coverage_discount.mean():.2f}")
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
    print(f"  Step {step_idx}: action={action}, ig={ig:.2f}, g1={g1_reward:.2f}, g2={g2_value:.2f}")
```

---

## 14. File References

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

## 15. Visualization Support

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
