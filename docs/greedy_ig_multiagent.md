# Greedy IG Multi-Agent Architecture

> **Strategy:** `greedy_ig` — Single-step Information Gain maximization baseline

---

## 1. Overview

### Paper's Approach to Multi-Agent Coordination

In the paper's decentralized multi-UAV framework, each agent maintains its own local occupancy-belief map. Agents coordinate through:

1. **Position sharing** — determines communication neighbors (proximity-based)
2. **News belief sharing** — only newly acquired information since last exchange

**Communication Range:**
- Controlled by `radius_multiplier` in config
- Actual range = `radius_multiplier × h_displacement`
- Where `h_displacement = (field_len/2) / n_h_act`
- For 8 agents: `n_h_act=8`, `h_displacement=3.125m`
- Example: `radius_multiplier=5` → `15.625m`, `-1` → unlimited

Key properties:
- **Incremental updates** — agents share "news" not full belief maps
- **Product-of-experts fusion** — shared evidence strengthens common beliefs
- **Avoids double-counting** — news is cleared after fusion

---

## 2. Call Flow Diagram

```- **Avoids double-counting** — news is cleared after fusion

main.py
│
├── run_multi_agent_experiment()
│   │
│   ├── [1] MultiAgentCoordinator()          # Central coordination
│   │       └── MultiAgentMapper()           # Per-agent maps + news fusion
│   │
│   ├── [2] generate_multi_agent_starts()    # Start positions
│   │
│   ├── [3] FOR agent_id in num_agents:
│   │       └── initialize_agent()           # experiment_utils.py
│   │           ├── Camera()                 # Per-agent camera
│   │           ├── planning()               # Creates Planner with GreedyIGPlanner
│   │           │   └── create_greedy_ig_planner()
│   │           └── OccupancyMap()           # Per-agent local belief (OG + LBP)
│   │
│   └── [4] MAIN LOOP (n_steps):
│           │
│           ├── Phase 1: OBSERVE
│           │   └── process_agent_observations()
│           │       FOR each agent:
│           │           ├── map_obj.get_observations()
│           │           ├── occupancy_map.update_belief_OG()
│           │           ├── occupancy_map.propagate_messages()  # Local LBP
│           │           └── Store local_belief_map (before fusion)
│           │
│           ├── Phase 2: FUSE BELIEFS
│           │   └── perform_belief_fusion()
│           │       ├── mapper.update_all_news()      # Update news beliefs
│           │       ├── mapper.fuse_news_with_neighbors()  # Product-of-experts
│           │       └── Feed fused beliefs back to agents
│           │
│           ├── Phase 3: SELECT ACTIONS
│           │   └── select_agent_actions()
│           │       FOR each agent:
│           │           └── planner.select_action(fused_belief)
│           │               └── Planner.greedy_ig_decision()
│           │                   └── GreedyIGPlanner.plan()
│           │                       └── _compute_ig() using FUSED belief
│           │
│           └── Phase 4: UPDATE POSITIONS
│               └── update_agent_positions()
│                   FOR each agent:
│                       ├── camera.x_future(action)
│                       └── coordinator.update_agent_state()
```

---

## 3. Key Components

### 3.1 Agent Initialization (`experiment_utils.py`)

```python
def initialize_agent(...):
    # Each agent gets independent instances:
    camera = Camera(grid_info, 60, seed=seed+agent_id, ...)
    
    agent_planner = planning(                    # Creates Planner wrapper
        grid_info, camera, "greedy_ig",
        agent_id=agent_id,
        coordinator=coordinator                  # Shared coordinator reference
    )
    
    occupancy_map = OM(grid_info.shape,          # Independent local belief
                       conf_dict=conf_dict,
                       correlation_type=corr_type)  # 'equal'|'biased'|'adaptive'
    
    return {
        "agent_id": agent_id,
        "camera": camera,
        "planner": agent_planner,
        "occupancy_map": occupancy_map,
        "belief_map": np.full((..., 2), 0.5),    # Per-agent belief [H,W,2]
        "local_belief_map": None,                # Stored before fusion
        ...
    }
```

### 3.2 GreedyIGPlanner (`greedy_ig_planner.py`)

**Core Planning Logic (Pure Belief-Based IG):**

```python
class GreedyIGPlanner:
    def __init__(self, agent_id, camera, grid_info, conf_dict=None):
        self.actions = ["front", "back", "left", "right", "up", "down", "hover"]
        self.belief = None  # Will receive FUSED belief from coordinator
    
    def plan(self, current_position, current_altitude) -> GreedyIGIntent:
        """Single-step greedy action selection using fused belief."""
        
        # Set camera state for x_future computation
        self.camera.set_position(current_position)
        self.camera.set_altitude(current_altitude)
        
        for action in self.actions:
            # 1. Compute future state
            future_state = self.camera.x_future(action)
            if future_state is None:
                continue
            next_pos, next_alt = future_state
            
            # 2. Compute IG using FUSED belief (no penalties)
            ig = self._compute_ig(next_pos, next_alt)
            
            # 3. Score = pure IG (coordination via belief fusion, not penalties)
            score = ig
        
        # Return best action
        return GreedyIGIntent(action=best_action, ...)
```

**Information Gain Computation:**

```python
def _compute_ig(self, position, altitude) -> float:
    """IG = H(prior) - E[H(posterior)]"""
    
    # Get footprint cells
    [[imin, imax], [jmin, jmax]] = self.camera.get_range(
        position=position, altitude=altitude, index_form=True
    )
    
    # Clip to grid bounds
    H_grid, W_grid = self.belief.shape[:2]
    imin, imax = max(0, min(imin, H_grid)), max(0, min(imax, H_grid))
    jmin, jmax = max(0, min(jmin, W_grid)), max(0, min(jmax, W_grid))
    
    # Extract belief in footprint
    if self.belief.ndim == 3:
        prior = self.belief[imin:imax, jmin:jmax, 1]  # P(occupied)
    else:
        prior = self.belief[imin:imax, jmin:jmax]
    
    # Get sensor parameters for this altitude
    s0, s1 = self._get_sensor_params(altitude)
    
    # Entropy before observation
    prior_entropy = H(prior)  # -p*log2(p) - (1-p)*log2(1-p)
    
    # Expected entropy after observation (using sensor model)
    conditional_entropy = cH(prior, s0, s1)
    
    return float(np.sum(prior_entropy - conditional_entropy))
```

### 3.3 Belief Map Structure

Each agent maintains a 3D belief map:
```python
belief_map = np.full((H, W, 2), 0.5)
# belief_map[:,:,0] = P(free)
# belief_map[:,:,1] = P(occupied)
```

---

## 4. Belief Fusion Flow

### 4.1 Per-Agent Local Update

```
Agent Observation → OG Update → Local LBP → local_belief_map (stored)
```

**In `process_agent_observations()`:**

```python
for agent in agents:
    occupancy_map = agent["occupancy_map"]
    
    # 1. Get observation from environment
    fp_vertices_ij, submap = map_obj.get_observations(uav_pos, sigmas)
    
    # 2. Bayesian update (Occupancy Grid)
    occupancy_map.update_belief_OG(fp_vertices_ij, submap, uav_pos)
    
    # 3. Local LBP propagation (spatial consistency with pairwise factors)
    occupancy_map.propagate_messages(fp_vertices_ij, submap)
    
    # 4. Update agent's belief map
    belief_map[:,:,1] = occupancy_map.get_belief().copy()
    belief_map[:,:,0] = 1 - belief_map[:,:,1]
    agent["belief_map"] = belief_map
    
    # 5. Store local belief BEFORE fusion (for visualization)
    agent["local_belief_map"] = belief_map.copy()
```

### 4.2 Multi-Agent Fusion (via MultiAgentMapper)

```
All Agents' News → Update News → Fuse with Neighbors → Back to Agents
```

**In `perform_belief_fusion()`:**

```python
# Access MultiAgentMapper through coordinator
mapper = coordinator.map

# Phase 2a: Update all agents' news beliefs (synchronous)
mapper.update_all_news(agent_observations)

# Phase 2b: Fuse news with neighbors
for agent_id in range(coordinator.num_agents):
    neighbor_ids = coordinator.get_neighbors_in_range(agent_id)
    if neighbor_ids:
        mapper.fuse_news_with_neighbors(agent_id, neighbor_ids)

# Phase 2c: Feed fused beliefs back to agents
for agent in agents:
    agent_id = agent["agent_id"]
    fused_belief = mapper.get_agent_belief(agent_id)
    if fused_belief is not None:
        agent["belief_map"][:,:,1] = fused_belief
        agent["belief_map"][:,:,0] = 1 - fused_belief
```

### 4.3 News Modes (BS vs BM)

| Mode | Description | Clearing Behavior |
|------|-------------|-------------------|
| `BS` (Belief Single) | Single news buffer per agent, broadcast to all neighbors | Cleared after broadcast |
| `BM` (Belief Multi) | Per-neighbor news buffers, selectively fused | Selectively cleared per neighbor |

**Key Points:**
- Neither BS nor BM introduces penalties
- All coordination emerges from Bayesian fusion of news beliefs
- No heuristic interaction terms

**Fusion Formula (Product-of-Experts):**
```
P(m=1|z_A, z_B) = [P(m=1|z_A) × P(m=1|z_B)] / 
                  [P(m=1|z_A)×P(m=1|z_B) + P(m=0|z_A)×P(m=0|z_B)]
```

This ensures shared evidence strengthens common beliefs while avoiding double-counting (news is cleared after fusion).

---

## 5. Pairwise Potential Types

The **local LBP** (inside OccupancyMap) uses pairwise potentials `ψ(m_i, m_j)`:

| Type | Matrix | Effect |
|------|--------|--------|
| `equal` | `[[0.5, 0.5], [0.5, 0.5]]` | No spatial correlation |
| `biased` | `[[0.7, 0.3], [0.3, 0.7]]` | Neighbors likely same state |
| `adaptive` | Computed from observations | Pearson correlation-based |

**Note:** Pairwise factors are applied ONLY in local LBP (`OccupancyMap.propagate_messages()`), NOT in multi-agent fusion. The fusion uses unary-only Bayesian combination.

**Adaptive Computation (`helper.py`):**

```python
def adaptive_weights_matrix(obs_map):
    samples = collect_sample_set(obs_map)  # 3x3 blocks
    p = pearson_correlation_coeff(samples)  # Correlation
    exp_neg_p = np.exp(-p)
    return [[1/(1+exp_neg_p), exp_neg_p/(1+exp_neg_p)],
            [exp_neg_p/(1+exp_neg_p), 1/(1+exp_neg_p)]]
```

---

## 6. Multi-Agent Coordination (Paper Approach)

Greedy IG agents coordinate through **decentralized belief fusion** only:

1. **Position Sharing**: Determines communication neighbors (proximity-based)
2. **News Belief Sharing**: Agents exchange only newly acquired information
3. **Bayesian Fusion**: Product-of-experts combination of news beliefs

**No Penalties — Pure Belief-Based Planning:**
- Planner uses **fused belief** for IG computation
- Coordination emerges naturally from shared information
- Areas observed by teammates show reduced uncertainty → lower IG → agents explore elsewhere

```python
# In select_agent_actions():
belief_map = agent["belief_map"]  # This is the FUSED belief

# Planner computes IG using fused belief
next_action, info_gain_action = planner.select_action(
    belief_map,  # Fused belief passed to planner
    agent["uav_positions"]
)
```

**Why This Works:**
- When agent B observes an area, its news is fused into agent A's belief
- Agent A's belief for that area becomes more certain (closer to 0 or 1)
- IG = H(prior) - H(posterior) is LOW for certain cells
- Agent A naturally avoids re-observing areas teammates have covered

---

## 7. Configuration

**Current Benchmark:** `configs/benchmark_greedy_ig.json`

```json
{
  "action_strategy": "greedy_ig",
  "num_agents": 4,
  "n_steps": 100,
  "iters": [0, 20],
  "correlation_types": ["adaptive", "equal", "biased"],
  
  "greedy_ig": {
    "overlap_penalty_weight": 0.0,
    "mode_labels": ["IG", "IGd", "IG_BM", "IG_BS", "IGd_BM", "IGd_BS"]
  },
  
  "decentralized": {
    "radius_multiplier": 5
  }
}
```

**Key Parameters:**
- **num_agents**: 4 agents (consistent with Dec-MCTS and MH-Dec-MCTS benchmarks)
- **overlap_penalty_weight**: 0.0 (pure belief-based IG, paper-compliant)
- **mode_labels**: 6 news modes tested:
  - `IG`: No information sharing (baseline)
  - `IGd`: Position sharing only (discounted IoU)
  - `IG_BS`: IG + broadcast single news belief
  - `IG_BM`: IG + per-neighbor private news beliefs
  - `IGd_BS`: Position + broadcast news
  - `IGd_BM`: Position + per-neighbor news
- **radius_multiplier**: 5 → 15.625m range (calculated as `radius_multiplier × h_displacement`)
  - For agents: `h_displacement = (field_len/2) / n_h_act = 25/8 = 3.125m`
  - Set to `-1` for unlimited range
- **correlation_types**: `["adaptive", "equal", "biased"]` for comprehensive LBP testing
- **Testing**: 20 iterations across 3 correlation types

---

## 8. Debugging Tips

### Check Belief Fusion is Working
```python
# In perform_belief_fusion():
for agent in agents:
    local = agent["local_belief_map"][:,:,1].mean()
    fused = mapper.get_agent_belief(agent_id).mean()
    diff = np.sum(agent["local_belief_map"][:,:,1] != fused)
    print(f"Agent {agent_id}: local={local:.4f}, fused={fused:.4f}, changed_cells={diff}")
```

### Verify Planner Uses Fused Belief
```python
# In select_agent_actions():
print(f"Agent {agent_id}: belief passed to planner is FUSED: {agent['belief_map'] is agent['local_belief_map']}")
# Should print False after fusion
```

### Check News Clearing (BS mode)
```python
# After fusion:
for i in range(num_agents):
    news = mapper.news_map_beliefs[i, i, :, :]
    non_prior = np.sum(news != 0.5)
    print(f"Agent {i} news cells != 0.5: {non_prior}")  # Should be 0 after clearing
```

### Check Action Selection
```python
# After planner.select_action():
print(f"Agent {agent_id} action scores (pure IG, no penalties):")
for action, score in sorted(info_gain_action.items(), key=lambda x: -x[1]):
    print(f"  {action}: {score:.4f}")
```

---

## 9. File References

| File | Purpose |
|------|---------|
| `src/main.py` | Entry point, experiment loop |
| `src/experiment_utils.py` | Agent init, observation, fusion utilities |
| `src/greedy_ig_planner.py` | GreedyIGPlanner class (pure IG computation) |
| `src/planner.py` | Planner wrapper (calls GreedyIGPlanner) |
| `src/multi_agent_coordinator.py` | MultiAgentCoordinator (position tracking) |
| `src/multi_agent_mapper.py` | MultiAgentMapper (news fusion, BS/BM modes) |
| `src/mapper_LBP.py` | OccupancyMap (per-agent local OG + LBP) |
| `src/helper.py` | H(), cH(), adaptive_weights_matrix() |
| `src/uav_camera.py` | Camera model, x_future(), get_range() |
| `configs/benchmark_greedy_ig.json` | Configuration |

---

## 10. Paper Compliance Checklist

✅ **No overlap penalties** — `overlap_penalty_weight: 0.0`  
✅ **Pure belief-based IG** — no intent influence on action selection  
✅ **BS or BM fusion** — news beliefs shared and cleared appropriately  
✅ **Fused beliefs used by planner** — IG computed on post-fusion belief map  
✅ **Incremental news** — only new observations shared, not full maps  
✅ **Proximity-based neighbors** — `communication_range` determines fusion partners
