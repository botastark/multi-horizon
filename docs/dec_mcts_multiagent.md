# Dec-MCTS Multi-Agent Architecture

> **Strategy:** `dec_mcts` — Decentralized Monte Carlo Tree Search for multi-step planning

---

## 1. Overview

Dec-MCTS is a single-level MCTS planner with multi-agent coordination:
- **Multi-step lookahead**: Plans trajectories via MCTS tree search
- **Decentralized**: Each agent runs MCTS independently
- **Intent sharing**: Agents share planned trajectories for coordination
- **D-UCT discounting**: Handles asynchronous intent staleness

**Key Difference from Greedy IG:**
- Greedy: Evaluates single actions → picks best immediate IG
- Dec-MCTS: Simulates rollouts → picks action with best cumulative IG

---

## 2. Call Flow Diagram

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
│   │           ├── planning("dec_mcts")      # Creates DecMCTSPlanner
│   │           │   └── DecMCTSPlanner()
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
│           │           └── DecMCTSPlanner.select_action()
│           │               ├── [A] _compute_teammate_mask()
│           │               ├── [B] BUILD MCTS TREE
│           │               │   └── LOOP iterations:
│           │               │       ├── _tree_policy() → Selection + Expansion
│           │               │       ├── node.rollout() → Simulation
│           │               │       └── node.backpropagate() → Update
│           │               ├── [C] _extract_best_action()
│           │               └── [D] _build_intent() → DecMCTSIntent
│           │
│           └── Phase 4: UPDATE POSITIONS
│               └── update_agent_positions()
```

---

## 3. Key Components

### 3.1 DecMCTSPlanner (`dec_mcts.py`)

```python
class DecMCTSPlanner:
    def __init__(self, agent_id, camera, grid_info, config):
        self.config = {
            "horizon": 10,           # MCTS planning depth
            "iterations": 100,       # MCTS iterations per cycle
            "ucb_c": 1.4,            # UCB exploration constant
            "discount_factor": 0.95, # Gamma for future rewards
            "overlap_penalty_weight": 0.3,
            "d_uct_decay": 0.9,      # D-UCT staleness decay
            "d_uct_threshold": 2.0,  # Staleness threshold (seconds)
        }
        
        self._teammate_intents: Dict[int, DecMCTSIntent] = {}
        self.current_intent: DecMCTSIntent = None
```

### 3.2 MCTS Planning Flow

```python
def plan(self) -> DecMCTSIntent:
    # Build initial state
    state = {
        "uav_pos": uav_position((self.position, self.altitude)),
        "belief": self.belief.copy(),
        "remaining_steps": self.config["horizon"],
        "teammate_mask": self._compute_teammate_mask(),  # D-UCT discounted
    }
    
    # Create MCTS root
    root = DecMCTSNode(state, camera=self.camera, ...)
    
    # MCTS iterations
    for _ in range(self.config["iterations"]):
        # Selection + Expansion
        node, path = self._tree_policy(root)
        
        # Simulation (random rollout)
        reward, trajectory = node.rollout(
            max_depth=self.config["horizon"],
            discount_factor=self.config["discount_factor"],
        )
        
        # Backpropagation
        node.backpropagate(reward)
    
    # Extract best action & trajectory
    best_action, action_values = self._extract_best_action(root)
    trajectory = self._extract_trajectory(root)
    
    return self._build_intent(best_action, trajectory)
```

### 3.3 DecMCTSNode

```python
class DecMCTSNode:
    def __init__(self, state, camera, parent, action, ...):
        self.state = copy_state(state)
        self.parent = parent
        self.action_from_parent = action
        self.children: Dict[str, DecMCTSNode] = {}
        self.visit_count = 0
        self.value = 0.0
        self.untried_actions = camera.permitted_actions(state["uav_pos"])
    
    def best_child(self, c_param=1.4) -> DecMCTSNode:
        """UCB1 selection."""
        for action, child in self.children.items():
            exploitation = child.value / child.visit_count
            exploration = c_param * sqrt(2 * log(self.visit_count) / child.visit_count)
            ucb = exploitation + exploration
        return child_with_max_ucb
    
    def expand(self) -> DecMCTSNode:
        """Add child for untried action."""
        action = self.untried_actions.pop()
        new_state = self.apply_action(self.state, action)
        child = DecMCTSNode(new_state, parent=self, action=action, ...)
        self.children[action] = child
        return child
    
    def rollout(self, max_depth, discount_factor) -> Tuple[float, List]:
        """Random simulation to estimate value."""
        state = copy_state(self.state)
        total_reward = 0.0
        discount = 1.0
        
        for t in range(max_depth):
            action = random.choice(permitted_actions)
            state = self.apply_action(state, action)
            
            # Compute IG reward
            ig_reward = self.compute_ig_reward(state, footprint)
            overlap_penalty = self.compute_overlap_penalty(state, footprint)
            
            reward = ig_reward - overlap_penalty
            total_reward += discount * reward
            discount *= discount_factor
            
            # Update belief for next step
            state["belief"] = self.belief_update(state["belief"], ...)
        
        return total_reward, trajectory
```

---

## 4. Intent Structure

```python
@dataclass
class DecMCTSIntent:
    agent_id: int
    action_sequence: List[str]       # Planned actions
    position_sequence: List[Tuple]   # Trajectory positions
    altitude_sequence: List[float]   # Trajectory altitudes
    footprint_sequence: List[Tuple]  # [(imin, imax, jmin, jmax), ...]
    ig_sequence: List[float]         # Expected IG per step
    total_expected_ig: float
    timestamp: float
    
    def staleness_discount(self, decay=0.9, threshold=2.0):
        """D-UCT discount for asynchronous drift."""
        age = time.time() - self.timestamp
        return decay ** (age / threshold)
```

---

## 5. D-UCT Staleness Handling

### 5.1 Teammate Mask Computation

```python
def _compute_teammate_mask(self) -> np.ndarray:
    """Discount teammate footprints based on intent age."""
    mask = np.zeros((H, W), dtype=float)
    
    for teammate_id, intent in self._teammate_intents.items():
        if intent.is_stale(max_age=self.config["d_uct_threshold"] * 2):
            continue
        
        # D-UCT staleness discount (0 to 1)
        discount = intent.staleness_discount(
            decay_factor=self.config["d_uct_decay"],
            threshold_sec=self.config["d_uct_threshold"],
        )
        
        # Add discounted footprints
        for footprint in intent.footprint_sequence:
            mask[imin:imax, jmin:jmax] += discount
    
    return mask
```

### 5.2 Overlap Penalty in Rollout

```python
def compute_overlap_penalty(self, state, imin, imax, jmin, jmax) -> float:
    """Penalize overlap with teammate planned footprints."""
    if "teammate_mask" not in state:
        return 0.0
    
    mask = state["teammate_mask"]
    overlap = np.sum(mask[imin:imax, jmin:jmax])
    footprint_size = (imax - imin) * (jmax - jmin)
    
    overlap_ratio = overlap / footprint_size
    return self.config["overlap_penalty_weight"] * overlap_ratio
```

---

## 6. Belief Fusion (Same as Greedy)

Dec-MCTS uses the same belief fusion mechanism:

```
1. Each agent: OG update + local LBP
2. Coordinator: LBPBeliefFusion with news mode (BS/BM)
3. Pairwise potentials: equal/biased/adaptive
```

**Key Difference:**
- In Dec-MCTS, belief is also updated during rollout simulation (for multi-step planning)

```python
# In DecMCTSNode.rollout():
def belief_update(self, belief, imin, imax, jmin, jmax, Pz0, Pz1, p10, p11):
    """Update belief using expected posterior."""
    expected_post = Pz1 * p11 + Pz0 * p10
    belief[imin:imax, jmin:jmax, 1] = expected_post
    belief[imin:imax, jmin:jmax, 0] = 1 - expected_post
    return belief
```

---

## 7. Information Gain Computation

```python
def compute_ig_reward(self, state, imin, imax, jmin, jmax) -> float:
    """IG = H(prior) - E[H(posterior)]"""
    prior = state["belief"][imin:imax, jmin:jmax, 1]
    s0, s1 = self._get_sensor_params(state["uav_pos"].altitude)
    
    # Expected posterior via sensor model
    Pz0, Pz1, p10, p11 = expected_posterior(prior, s0, s1)
    
    # Entropy reduction
    curr_entropy = H(prior)
    expected_entropy = Pz0 * H(p10) + Pz1 * H(p11)
    
    return np.sum(curr_entropy - expected_entropy)
```

---

## 8. Multi-Agent Coordination

### 8.1 Intent Sharing via Coordinator

```python
class DecMCTSCoordinator:
    def __init__(self, num_agents):
        self._intents: Dict[int, DecMCTSIntent] = {}
        self._lock = threading.Lock()
    
    def share_intent(self, intent: DecMCTSIntent):
        with self._lock:
            self._intents[intent.agent_id] = intent
    
    def get_teammate_intents(self, agent_id) -> Dict[int, DecMCTSIntent]:
        with self._lock:
            return {aid: i for aid, i in self._intents.items() if aid != agent_id}
```

### 8.2 Action Selection with Coordination

```python
def select_agent_actions(agents, ...):
    for agent in agents:
        # Get teammate intents
        teammate_intents = coordinator.get_teammate_intents(agent_id)
        agent["planner"].update_teammate_intents(teammate_intents)
        
        # Run MCTS planning
        action, scores = planner.select_action(belief, positions)
        
        # Share own intent
        coordinator.share_intent(planner.current_intent)
```

---

## 9. Configuration

**Current Benchmark:** `configs/benchmark_dec_mcts.json`

```json
{
  "action_strategy": "dec_mcts",
  "num_agents": 4,
  "n_steps": 100,
  "iters": [0, 5],
  "correlation_types": ["equal", "biased", "adaptive"],
  
  "dec_mcts": {
    "horizon": 10,
    "iterations": 100,
    "ucb_c": 1.4,
    "discount_factor": 0.95
  },
  
  "decentralized": {
    "communication_range": 15.0,
    "overlap_penalty_weight": 0.3,
    "d_uct": {
      "decay_factor": 0.9,
      "threshold_sec": 2.0
    }
  }
}
```

**Key Parameters:**
- **Planning**: `horizon=10` steps, `iterations=100` MCTS rollouts per cycle
- **UCB exploration**: `ucb_c=1.4` balances exploration vs exploitation
- **Reward**: `discount_factor=0.95` for future trajectory value
- **Coordination**: `overlap_penalty_weight=0.3` penalizes teammate trajectory overlap
- **Communication**: `communication_range=15.0m` (direct specification)
- **D-UCT**: `decay_factor=0.9`, `threshold_sec=2.0` for staleness discounting
- **Testing**: 5 iterations across 3 correlation types (equal, biased, adaptive)

---
## 10. Comparison: Greedy vs Dec-MCTS

| Aspect | Greedy IG | Dec-MCTS |
|--------|-----------|----------|
| Lookahead | 1 step | N steps (horizon) |
| Search | Enumerate actions | MCTS tree search |
| Intent | Next footprint only | Full trajectory |
| Computation | O(actions) | O(iterations × horizon) |
| Coordination | Footprint overlap | Trajectory overlap |
| Belief update | None during planning | Simulated during rollout |

---

## 11. Debugging Tips

### Check MCTS Statistics
```python
# After planner.plan():
print(f"MCTS: {planner._stats['total_iterations']} iterations")
print(f"Action values: {planner.get_action_values()}")
print(f"Visit counts: {planner.get_action_visits()}")
```

### Check Teammate Mask
```python
# In _compute_teammate_mask():
print(f"Agent {self.agent_id} teammate mask:")
for tid, intent in self._teammate_intents.items():
    discount = intent.staleness_discount()
    print(f"  Agent {tid}: {len(intent.footprint_sequence)} footprints, discount={discount:.2f}")
```

### Check Rollout Rewards
```python
# In node.rollout():
print(f"Rollout step {t}: action={action}, ig={ig_reward:.2f}, penalty={overlap_penalty:.2f}")
```

### Check Tree Structure
```python
# After MCTS iterations:
def print_tree(node, depth=0):
    indent = "  " * depth
    print(f"{indent}{node.action_from_parent}: visits={node.visit_count}, value={node.value:.2f}")
    for child in node.children.values():
        print_tree(child, depth + 1)
print_tree(root)
```

---

## 12. File References

| File | Purpose |
|------|---------|
| `src/main.py` | Entry point, experiment loop |
| `src/experiment_utils.py` | Agent init, observation, fusion utilities |
| `src/dec_mcts.py` | DecMCTSPlanner, DecMCTSNode, DecMCTSIntent |
| `src/multi_agent_coordinator.py` | MultiAgentCoordinator, LBPBeliefFusion |
| `src/mapper_LBP.py` | OccupancyMap (per-agent local LBP) |
| `src/helper.py` | H(), expected_posterior(), cH() |
| `src/planner.py` | Planner factory (dispatches to dec_mcts) |
