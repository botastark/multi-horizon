# Hierarchical Dec-MCTS Planner Documentation

## Overview

This document describes the hierarchical multi-agent planner with shared beliefs and shared intents, following the Dec-MCTS (Decentralized Monte Carlo Tree Search) framework from "Multi-Horizon Multi-Agent Planning Using Decentralised Monte Carlo Tree Search".

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Multi-Agent System                                │
│                                                                             │
│  ┌─────────────────┐    IntentBus    ┌─────────────────┐                   │
│  │     Agent 0     │◄───────────────►│     Agent 1     │                   │
│  │  ┌───────────┐  │  (LL + HL       │  ┌───────────┐  │                   │
│  │  │    LLP    │  │   intents)      │  │    LLP    │  │                   │
│  │  └─────┬─────┘  │                 │  └─────┬─────┘  │                   │
│  │        │        │                 │        │        │                   │
│  │  ┌─────▼─────┐  │                 │  ┌─────▼─────┐  │                   │
│  │  │    HLP    │  │                 │  │    HLP    │  │                   │
│  │  └───────────┘  │                 │  └───────────┘  │                   │
│  └─────────────────┘                 └─────────────────┘                   │
│           │                                   │                             │
│           └───────────┬───────────────────────┘                             │
│                       ▼                                                     │
│              ┌────────────────┐                                             │
│              │ LBP Belief     │                                             │
│              │ Fusion         │                                             │
│              └────────────────┘                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Intent Data Structures

#### LLIntent (Low-Level Intent)
Short-horizon motion plan containing:
- `action_sequence`: List of planned primitive actions (e.g., ['front', 'front', 'right'])
- `state_sequence`: Predicted (x, y, altitude) positions
- `footprint_sequence`: Camera footprint bounds at each step
- `ig_sequence`: Expected Information Gain at each step
- `total_expected_ig`: Sum of expected IG
- `timestamp`: When intent was generated
- `value`: MCTS value of this plan

#### HLIntent (High-Level Intent)  
Long-horizon region/cluster plan containing:
- `region_sequence`: List of region IDs to visit
- `eta_sequence`: Estimated time to reach each region
- `completion_sequence`: Estimated completion time for each region
- `current_target_region`: Current target region ID
- `target_center`: (row, col) center of current target
- `timestamp`: When intent was generated
- `value`: HLP value of this plan

### 2. IntentBus
Thread-safe communication channel for sharing intents between agents:
- Broadcasts LL and HL intents
- Stores latest intents from each agent
- Maintains history for temporal reasoning
- Statistics tracking

### 3. LowLevelPlanner (LLP)
Short-horizon planner using MCTS with intent-aware rewards:

**Reward Function (g1):**
```
g1 = Σ (γ^t × (IG_t × teammate_discount + alignment_bonus))
```

Where:
- `IG_t`: Information gain at step t
- `teammate_discount`: Reduces IG for cells teammates plan to cover
- `alignment_bonus`: Bonus for moving toward HLP target (soft guidance)
- `γ`: Discount factor

**Key Features:**
- Uses teammates' LL-intents to reduce IG in cells they'll cover
- Respects HLP guidance via alignment bonus
- Fast planning cycle (5-10 step horizon)

### 4. HighLevelPlanner (HLP)
Long-horizon planner for region allocation:

**Region Scoring (g2):**
```
score = remaining_uncertainty - distance_penalty - conflict_penalty
```

Where:
- `remaining_uncertainty`: 1 - coverage of region
- `distance_penalty`: Normalized distance to region
- `conflict_penalty`: Heavy penalty if teammates target same region

**Key Features:**
- Partitions grid into rectangular regions
- Considers teammates' HL-intents to avoid conflicts
- Slower planning cycle with tree persistence

### 5. HierarchicalDecMCTSPlanner
Main orchestrator combining LLP and HLP:

**Planning Loop:**
```python
loop:
    # Step 1: Receive teammate intents
    ll_intents = intent_bus.get_teammate_ll_intents(agent_id)
    hl_intents = intent_bus.get_teammate_hl_intents(agent_id)
    
    # Step 2: Update planners with intents
    llp.update_teammate_intents(ll_intents, hl_intents)
    hlp.update_teammate_intents(ll_intents, hl_intents)
    
    # Step 3: Run HLP (slow cycle, may reuse cached plan)
    hl_intent = hlp.plan(current_position)
    
    # Step 4: Update LLP with HLP guidance
    llp.update_hl_guidance(hl_intent)
    
    # Step 5: Run LLP (fast cycle)
    ll_intent = llp.plan(current_state)
    
    # Step 6: Broadcast intents
    intent_bus.broadcast_ll_intent(ll_intent)
    intent_bus.broadcast_hl_intent(hl_intent)
    
    return ll_intent.action_sequence[0]
end loop
```

## Reward Decomposition

Following the paper's decomposition:
```
g = g1(LL intents) + g2(all intents)
```

Where:
- **g1**: Immediate task quality (IG from LLP)
- **g2**: Long-horizon mission estimate (region allocation from HLP)

## Configuration

In `config.json`:
```json
{
    "action_strategy": "hierarchical_dec_mcts",
    "hierarchical_dec_mcts": {
        "llp_horizon": 5,
        "llp_iterations": 100,
        "hlp_horizon": 3,
        "hlp_iterations": 50,
        "tile_size": [100, 100],
        "hlp_replan_interval": 1.0,
        "intent_discount": 0.8,
        "enable_belief_sharing": true
    }
}
```

## Integration with Belief Fusion

The hierarchical planner integrates with the existing LBP belief fusion:

1. **Belief Updates**: After LBP fusion, fused beliefs are fed back to agents
2. **Planning with Fused Beliefs**: Both LLP and HLP use the fused belief map
3. **Intent-Belief Interaction**: Incoming intents predict where teammates will observe, allowing preemptive IG adjustment

## Usage

### In main.py (automatic via strategy)
```python
# Set in config.json:
# "action_strategy": "hierarchical_dec_mcts"

# The planning class automatically creates and manages hierarchical planners
planner = planning(
    grid_info,
    camera,
    "hierarchical_dec_mcts",
    agent_id=agent_id,
    coordinator=coordinator,
)
action, scores = planner.select_action(belief_map, visited_positions)
```

### Manual Creation
```python
from hierarchical_dec_mcts import IntentBus, create_hierarchical_planner

# Create shared intent bus
intent_bus = IntentBus(num_agents=2)

# Create planners for each agent
planners = []
for agent_id in range(num_agents):
    planner = create_hierarchical_planner(
        agent_id=agent_id,
        num_agents=num_agents,
        camera=cameras[agent_id],
        grid_info=grid_info,
        intent_bus=intent_bus,
        config=config,
    )
    planners.append(planner)
```

## Key Design Principles

1. **Decentralized**: No central controller, agents share intents via broadcast
2. **Asynchronous**: LLP and HLP can run at different rates
3. **Soft HLP Guidance**: HLP suggests, never blocks LLP from following IG
4. **Intent-Aware Rewards**: Both planners adjust rewards based on teammates' plans
5. **Belief Sharing**: Fused beliefs ensure consistent world view across agents

## Files

- `src/hierarchical_dec_mcts.py`: Core implementation
- `src/planner.py`: Integration with planning class
- `src/multi_agent_coordinator.py`: Integration with coordinator
- `config.json`: Configuration options

## Future Improvements

1. **Full D-UCT Implementation**: Currently uses simplified MCTS; could add D-UCT discounting for asynchronous drift handling
2. **Trajectory Prediction**: Use more sophisticated models for teammate trajectory prediction
3. **Communication Constraints**: Add realistic communication delays and packet loss
4. **Adaptive Intent Frequency**: Adjust broadcast frequency based on plan changes
