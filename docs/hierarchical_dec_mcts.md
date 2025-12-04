# Hierarchical Dec-MCTS Planner Documentation

## Overview

This document describes the **fully decentralized** multi-agent planner with shared beliefs and shared intents, following the Dec-MCTS (Decentralized Monte Carlo Tree Search) framework from "Multi-Horizon Multi-Agent Planning Using Decentralised Monte Carlo Tree Search".

### Key Design Principles

**Full Decentralization**: Each agent maintains its own terrain belief while sharing only incremental updates through "news beliefs." There is no central controller.

**News-Based Belief Sharing**: When agents are within communication range, they exchange their news beliefs—not their full maps—so the receiver can fuse this fresh information exactly once through renormalization, avoiding double counting and preventing overconfident or inconsistent posteriors.

**Per-Neighbor News Beliefs**: To maintain alignment in asynchronous, distributed settings, each UAV stores separate news beliefs δ_ij for each neighbor j, ensuring that updates from any agent are never re-fused.

**Position Broadcasting for Coordination**: Agents broadcast their positions to modulate action choices via footprint overlap penalties, promoting non-redundant exploration without relying on any central controller.

## Architecture

### Fully Decentralized Architecture
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Fully Decentralized Multi-Agent System                    │
│                              (No Central Controller)                             │
│                                                                                  │
│  ┌──────────────────────────┐        ┌──────────────────────────┐              │
│  │   DecentralizedAgent 0   │◄──────►│   DecentralizedAgent 1   │              │
│  │  ┌────────────────────┐  │  P2P   │  ┌────────────────────┐  │              │
│  │  │  LocalBeliefMgr B  │  │  Comm  │  │  LocalBeliefMgr B  │  │              │
│  │  │  ┌──────────────┐  │  │        │  │  ┌──────────────┐  │  │              │
│  │  │  │ News δ_01    │  │  │ News   │  │  │ News δ_10    │  │  │              │
│  │  │  │ News δ_02    │  │◄─┼─Beliefs│  │  │ News δ_12    │  │  │              │
│  │  │  │ ...          │  │  │ Only   │  │  │ ...          │  │  │              │
│  │  │  └──────────────┘  │  │        │  │  └──────────────┘  │  │              │
│  │  └────────────────────┘  │        │  └────────────────────┘  │              │
│  │  ┌─────────┐ ┌─────────┐ │Position│  ┌─────────┐ ┌─────────┐ │              │
│  │  │   LLP   │ │   HLP   │ │Broadcast│ │   LLP   │ │   HLP   │ │              │
│  │  └─────────┘ └─────────┘ │        │  └─────────┘ └─────────┘ │              │
│  └──────────────────────────┘        └──────────────────────────┘              │
│           │                                   │                                 │
│           │         ┌───────────────┐         │                                 │
│           └────────►│ Overlap Penalty│◄────────┘                                │
│                     │  Calculation  │                                           │
│                     └───────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Legacy Centralized Architecture (Deprecated)
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

### 0. Decentralized Agent (NEW)

The `DecentralizedAgent` class encapsulates a fully autonomous agent:

**Components:**
- `LocalBeliefManager`: Maintains local CRF-based belief map B
- Per-neighbor news beliefs δ_ij: Tracks new observations for each neighbor
- Position broadcaster: Shares location for overlap avoidance
- Intent tracker: Stores neighbor intents for coordination

**News Belief Fusion Mathematics:**

Using log-odds representation for numerical stability:
```
log_odds(p) = log(p / (1-p))
p = 1 / (1 + exp(-log_odds))
```

**Observation Update (in log-odds):**
```
log_odds_new = log_odds_old + log(P(z|m=1) / P(z|m=0))
```

**News Fusion (prevents double-counting):**
```
log_odds_fused = log_odds_local + log_odds_news
```

This works because:
1. Local belief contains all previously fused information
2. News belief contains ONLY new observations since last communication
3. No overlap = additive fusion in log-odds is correct

**Communication Protocol:**
```python
# When agent i communicates with agent j:
1. Agent i sends news_beliefs[j] to agent j
2. Agent j fuses: belief_log_odds += received_news_log_odds
3. Agent i resets: news_beliefs[j] = 0 (all zeros in log-odds = uniform prior)
```

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

- `src/decentralized_agent.py`: **NEW** Fully decentralized agent implementation
- `src/hierarchical_dec_mcts.py`: Core planning implementation
- `src/planner.py`: Integration with planning class
- `src/multi_agent_coordinator.py`: Integration with coordinator (legacy)
- `config.json`: Configuration options

## Decentralized System Usage (NEW)

### Creating Decentralized Agents
```python
from decentralized_agent import create_decentralized_system

# Create complete decentralized system
system = create_decentralized_system(
    num_agents=3,
    cameras=cameras,  # List of camera objects
    grid_info=grid_info,
    config={
        "communication_range": 50.0,  # -1 for unlimited
        "use_lbp": True,
        "overlap_penalty_weight": 0.5,
        "intent_horizon": 5,
    },
)

# Access components
agents = system["agents"]
network = system["network"]

# Run one communication step
system["step_all"]()

# Get all beliefs
beliefs = system["get_all_beliefs"]()  # {agent_id: belief_array}

# Get statistics
stats = system["get_all_statistics"]()
```

### Manual Agent Control
```python
from decentralized_agent import DecentralizedAgent, DecentralizedCommNetwork

# Create agents
agents = []
for agent_id in range(num_agents):
    agent = DecentralizedAgent(
        agent_id=agent_id,
        num_agents=num_agents,
        camera=cameras[agent_id],
        grid_info=grid_info,
        communication_range=50.0,
    )
    agents.append(agent)

# Create network
network = DecentralizedCommNetwork(agents)

# Simulation loop
for step in range(max_steps):
    for agent in agents:
        # Update position
        agent.update_position(position, altitude)
        
        # Process observation
        agent.observe_and_update(observation, fp_ij, sigma0, sigma1)
        
        # Compute IG with overlap penalty
        ig = agent.compute_information_gain(proposed_pos, altitude)
        
        # Plan action using agent's local belief
        belief = agent.get_belief()
        action = planner.plan(belief)
        
        # Broadcast position and intent
        agent.broadcast_position()
        agent.share_intent()
    
    # Share news beliefs (only new observations)
    for agent in agents:
        agent.share_news_with_neighbors()
    
    # Route messages
    network.step()
```

### News Belief Mechanics

Each agent maintains per-neighbor news beliefs to prevent double-counting:

```python
# Agent 0 observes cell (5,5) with probability 0.8
# Before communication with Agent 1:
agent0.news_beliefs[1]  # Contains log-odds update for cell (5,5)
agent0.news_masks[1]    # True at (5,5), False elsewhere

# During communication:
news_msg = agent0.belief_manager.get_news_for_neighbor(1)
agent1.belief_manager.fuse_received_news(news_msg)  # Fuses once
agent0.belief_manager.reset_news_for_neighbor(1)    # Reset to prevent re-fusion

# Key property: If agent 0 observes more cells before next communication,
# only the NEW observations accumulate in news_beliefs[1]
```

## Configuration

### Decentralized System Config
```json
{
    "decentralized": {
        "communication_range": 50.0,
        "use_lbp": true,
        "lbp_iterations": 1,
        "overlap_penalty_weight": 0.5,
        "intent_horizon": 5,
        "stale_message_threshold": 5.0,
        "message_delay": 0.0,
        "drop_probability": 0.0
    }
}
```

### Legacy Hierarchical Config
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

## Future Improvements

1. **Full D-UCT Implementation**: Currently uses simplified MCTS; could add D-UCT discounting for asynchronous drift handling
2. **Trajectory Prediction**: Use more sophisticated models for teammate trajectory prediction
3. **Communication Constraints**: Add realistic communication delays and packet loss
4. **Adaptive Intent Frequency**: Adjust broadcast frequency based on plan changes
5. **Consensus Mechanisms**: Implement distributed consensus for coordinated region allocation

