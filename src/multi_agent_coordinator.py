"""
Multi-Agent Coordinator - Swarm Orchestration Only

This module handles ONLY swarm-level orchestration:
- Inter-agent communication (CommunicationBus)
- Collision avoidance
- Neighbor discovery (within communication range)
- Agent state tracking

DOES NOT handle:
- Belief mapping (see mapper_LBP.py, multi_agent_mapper.py)
- Region allocation (handled by HLP/MCTS for hierarchical planning)
- Planning (see greedy_ig_planner.py, hierarchical_dec_mcts.py, etc.)

Architecture:
- mapper_LBP.py: Single-agent OccupancyMap (OG + LBP, uses pairwise factors)
- multi_agent_mapper.py: Multi-agent mapping (per-agent maps, news beliefs, fusion)
- multi_agent_coordinator.py: Swarm orchestration ONLY (this file)
- Planners: Planning strategies in separate files
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from multi_agent_mapper import MultiAgentMapper

logger = logging.getLogger(__name__)


# =============================================================================
# Message Types for Inter-Agent Communication
# =============================================================================


class MessageType(Enum):
    """Types of messages that can be exchanged between agents."""

    POSITION_UPDATE = auto()  # Agent position broadcast
    BELIEF_UPDATE = auto()  # Belief map update notification
    OBSERVATION = auto()  # Raw observation data
    INTENT = auto()  # Planned action intent (for coordination)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""

    msg_type: MessageType
    sender_id: int
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    ttl: float = 5.0  # Time-to-live in seconds


@dataclass
class AgentState:
    """Current state of an agent."""

    agent_id: int
    position: Tuple[float, float]
    altitude: float
    last_update: float = field(default_factory=time.time)


# =============================================================================
# Communication Bus - Message Passing Between Agents
# =============================================================================


class CommunicationBus:
    """
    Simulated communication channel for multi-agent coordination.

    In a real deployment, this would be replaced with actual network
    communication (e.g., ROS topics, ZeroMQ, etc.).

    This implementation uses a broadcast model where all agents
    can send/receive messages from all other agents.
    """

    def __init__(self, num_agents: int, max_queue_size: int = 1000):
        """
        Initialize communication bus.

        Args:
            num_agents: Number of agents in the system
            max_queue_size: Maximum messages per agent queue
        """
        self.num_agents = num_agents

        # Per-agent message queues (agent_id -> queue)
        self._queues: Dict[int, queue.Queue] = {
            i: queue.Queue(maxsize=max_queue_size) for i in range(num_agents)
        }

        # Lock for thread-safe operations
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_dropped": 0,
            "messages_received": 0,
        }

    def broadcast(self, message: AgentMessage, exclude_sender: bool = True):
        """
        Broadcast a message to all agents.

        Args:
            message: Message to broadcast
            exclude_sender: If True, don't send to the sender
        """
        with self._lock:
            for agent_id, q in self._queues.items():
                if exclude_sender and agent_id == message.sender_id:
                    continue

                try:
                    q.put_nowait(message)
                    self._stats["messages_sent"] += 1
                except queue.Full:
                    self._stats["messages_dropped"] += 1
                    logger.warning(f"Message queue full for agent {agent_id}")

    def send_to(self, message: AgentMessage, target_id: int) -> bool:
        """
        Send a message to a specific agent.

        Args:
            message: Message to send
            target_id: ID of target agent

        Returns:
            True if sent successfully, False if queue full
        """
        if target_id not in self._queues:
            return False

        with self._lock:
            try:
                self._queues[target_id].put_nowait(message)
                self._stats["messages_sent"] += 1
                return True
            except queue.Full:
                self._stats["messages_dropped"] += 1
                return False

    def receive(self, agent_id: int, timeout: float = 0.01) -> Optional[AgentMessage]:
        """
        Receive a message for a specific agent.

        Args:
            agent_id: ID of receiving agent
            timeout: How long to wait for a message

        Returns:
            Message if available, None otherwise
        """
        if agent_id not in self._queues:
            return None

        try:
            msg = self._queues[agent_id].get(timeout=timeout)
            self._stats["messages_received"] += 1

            # Filter stale messages
            if time.time() - msg.timestamp > msg.ttl:
                return None  # Message expired

            return msg
        except queue.Empty:
            return None

    def receive_all(self, agent_id: int) -> List[AgentMessage]:
        """
        Receive all pending messages for an agent.

        Args:
            agent_id: ID of receiving agent

        Returns:
            List of pending messages (may be empty)
        """
        messages = []
        while True:
            msg = self.receive(agent_id, timeout=0.001)
            if msg is None:
                break
            messages.append(msg)
        return messages

    def get_stats(self) -> Dict[str, int]:
        """Get communication statistics."""
        with self._lock:
            return self._stats.copy()

    def clear_queue(self, agent_id: int) -> int:
        """
        Clear all pending messages for an agent.

        Call this at the start of each step to prevent message buildup.

        Args:
            agent_id: ID of agent whose queue to clear

        Returns:
            Number of messages cleared
        """
        if agent_id not in self._queues:
            return 0

        cleared = 0
        with self._lock:
            while not self._queues[agent_id].empty():
                try:
                    self._queues[agent_id].get_nowait()
                    cleared += 1
                except queue.Empty:
                    break
        return cleared

    def clear_all_queues(self) -> int:
        """
        Clear all message queues for all agents.

        Returns:
            Total number of messages cleared
        """
        total_cleared = 0
        for agent_id in self._queues:
            total_cleared += self.clear_queue(agent_id)
        return total_cleared


# =============================================================================
# Multi-Agent Coordinator - Swarm Orchestration
# =============================================================================


class MultiAgentCoordinator:
    """
    Coordinator for multi-agent UAV swarm orchestration.

    Handles ONLY:
    - Communication infrastructure
    - Collision avoidance
    - Neighbor discovery (communication range)
    - Agent state tracking

    Does NOT handle:
    - Belief mapping (use MultiAgentMapper directly)
    - Region allocation (handled by HLP for hierarchical planning)
    - Planning (handled by planner classes)
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        config: Dict[str, Any],
        conf_dict: Dict[str, Any],
        correlation_type: str = "equal",
    ):
        """
        Initialize multi-agent coordinator.

        Args:
            grid_shape: (H, W) shape of the belief map
            config: Configuration dict with multi_agent settings
            conf_dict: Confidence dictionary - sensor model parameters
            correlation_type: Pairwise correlation type ('equal', 'biased', 'adaptive')
        """
        self.grid_shape = grid_shape
        self.config = config
        self.conf_dict = conf_dict
        self.num_agents = self.config.get("num_agents", 1)

        # Extract configuration
        ma_config = config.get("multi_agent", {})
        # Use explicit correlation_type parameter, fallback to config, then default
        self.correlation_type = correlation_type or ma_config.get(
            "correlation_type", ""
        )
        if self.correlation_type == "":
            print(f"Warning: correlation_type not specified, defaulting to 'equal'")
            breakpoint()

        self.news_mode = ma_config.get("news_mode", "BM")
        self.lbp_iterations = ma_config.get("lbp_iterations", 1)

        self.enable_coordination = ma_config.get("enable_coordination", True)
        self.communication_range = ma_config.get(
            "communication_range", -1
        )  # -1 = unlimited
        self.collision_distance = ma_config.get("collision_avoidance_distance", 5.0)

        # Initialize communication bus
        self.comm_bus = CommunicationBus(self.num_agents)

        # Initialize multi-agent mapper (handles all belief mapping)
        self.map = MultiAgentMapper(
            self.grid_shape,
            self.num_agents,
            self.conf_dict,
            correlation_type=self.correlation_type,
            news_mode=self.news_mode,
            lbp_iterations=self.lbp_iterations,
        )

        # Agent states tracking
        self._agent_states: Dict[int, AgentState] = {}
        self._states_lock = threading.Lock()

        # Statistics
        self._stats = {
            "collision_avoidances": 0,
            "position_updates": 0,
        }

        logger.info(
            f"MultiAgentCoordinator initialized: {self.num_agents} agents, "
            f"coordination={'enabled' if self.enable_coordination else 'disabled'}, "
            f"correlation_type={self.correlation_type}, "
            f"news_mode={self.news_mode}, "
            f"comm_range={self.communication_range}"
        )

    def update_agent_state(
        self,
        agent_id: int,
        position: Tuple[float, float],
        altitude: float,
    ):
        """
        Update an agent's state (called by each agent).

        Args:
            agent_id: ID of agent
            position: Current (row, col) position
            altitude: Current altitude
        """
        with self._states_lock:
            self._agent_states[agent_id] = AgentState(
                agent_id=agent_id,
                position=position,
                altitude=altitude,
                last_update=time.time(),
            )

        self._stats["position_updates"] += 1

        # Broadcast position update to other agents
        if self.enable_coordination:
            msg = AgentMessage(
                msg_type=MessageType.POSITION_UPDATE,
                sender_id=agent_id,
                data={
                    "position": position,
                    "altitude": altitude,
                },
            )
            self.comm_bus.broadcast(msg)

    def get_neighbors_in_range(self, agent_id: int) -> List[int]:
        """
        Get IDs of agents within communication range.

        Args:
            agent_id: ID of querying agent

        Returns:
            List of neighbor agent IDs
        """
        if self.communication_range < 0:
            # Unlimited range - all other agents are neighbors
            return [i for i in range(self.num_agents) if i != agent_id]

        neighbors = []
        agent_state = self._agent_states.get(agent_id)
        if agent_state is None:
            return neighbors

        with self._states_lock:
            for other_id, state in self._agent_states.items():
                if other_id == agent_id:
                    continue

                distance = np.sqrt(
                    (agent_state.position[0] - state.position[0]) ** 2
                    + (agent_state.position[1] - state.position[1]) ** 2
                )

                if distance <= self.communication_range:
                    neighbors.append(other_id)

        return neighbors

    def get_collision_penalty(
        self,
        agent_id: int,
        proposed_position: Tuple[float, float],
    ) -> float:
        """
        Compute collision avoidance penalty for a proposed position.

        Returns a penalty (0-1) based on proximity to other agents.
        Higher penalty = closer to other agents = less desirable.

        Args:
            agent_id: ID of agent considering the move
            proposed_position: (row, col) position being considered

        Returns:
            Penalty value (0 = no collision risk, 1 = imminent collision)
        """
        if self.collision_distance <= 0:
            return 0.0  # Collision avoidance disabled

        penalty = 0.0

        with self._states_lock:
            for other_id, state in self._agent_states.items():
                if other_id == agent_id:
                    continue

                # Check if state is recent (within 5 seconds)
                if time.time() - state.last_update > 5.0:
                    continue

                # Compute distance to other agent
                distance = np.sqrt(
                    (proposed_position[0] - state.position[0]) ** 2
                    + (proposed_position[1] - state.position[1]) ** 2
                )

                if distance < self.collision_distance:
                    # Within collision zone - compute penalty
                    # Penalty increases as distance decreases
                    relative_proximity = 1.0 - (distance / self.collision_distance)
                    penalty = max(penalty, relative_proximity)

        if penalty > 0:
            self._stats["collision_avoidances"] += 1

        return penalty

    def get_other_agent_positions(
        self,
        agent_id: int,
        max_age: float = 5.0,
    ) -> List[Tuple[int, Tuple[float, float], float]]:
        """
        Get positions of other agents.

        Args:
            agent_id: ID of requesting agent
            max_age: Maximum age of position data in seconds

        Returns:
            List of (agent_id, position, altitude) tuples
        """
        result = []
        current_time = time.time()

        with self._states_lock:
            for other_id, state in self._agent_states.items():
                if other_id == agent_id:
                    continue
                if current_time - state.last_update <= max_age:
                    result.append((other_id, state.position, state.altitude))

        return result

    def process_messages(self, agent_id: int) -> List[AgentMessage]:
        """
        Process pending messages for an agent.

        Updates internal state based on received messages.

        Args:
            agent_id: ID of agent processing messages

        Returns:
            List of processed messages
        """
        messages = self.comm_bus.receive_all(agent_id)

        for msg in messages:
            if msg.msg_type == MessageType.POSITION_UPDATE:
                # Update tracked position of other agent
                with self._states_lock:
                    if msg.sender_id not in self._agent_states:
                        self._agent_states[msg.sender_id] = AgentState(
                            agent_id=msg.sender_id,
                            position=msg.data.get("position", (0, 0)),
                            altitude=msg.data.get("altitude", 0),
                        )
                    else:
                        state = self._agent_states[msg.sender_id]
                        state.position = msg.data.get("position", state.position)
                        state.altitude = msg.data.get("altitude", state.altitude)
                        state.last_update = time.time()

        return messages

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        stats = self._stats.copy()
        stats["comm_bus"] = self.comm_bus.get_stats()
        stats["num_agents"] = self.num_agents
        stats["agents_tracked"] = len(self._agent_states)
        return stats

    # =========================================================================
    # Mapper Access Methods (delegated to MultiAgentMapper)
    # =========================================================================

    def get_agent_belief(self, agent_id: int) -> Optional[np.ndarray]:
        """Get belief map for a specific agent."""
        return self.map.get_agent_belief(agent_id)

    def get_fused_belief(self) -> Optional[np.ndarray]:
        """Get the fused global belief map."""
        return self.map.get_fused_belief()


# =============================================================================
# Utility: Generate Start Positions for Multiple Agents
# =============================================================================


def generate_multi_agent_starts(
    num_agents: int,
    grid_info: Any,
    start_position: str = "corner",
    min_distance: float = 10.0,
    seed: int = None,
) -> List[Tuple[float, float]]:
    """
    Generate starting positions for multiple agents.

    Spreads agents across the field to minimize initial overlap.

    Args:
        num_agents: Number of agents
        grid_info: Grid configuration object
        start_position: "corner", "edge", or "spread"
        min_distance: Minimum distance between start positions
        seed: Random seed for reproducible random positions

    Returns:
        List of (x, y) start positions
    """
    # Set random seed if provided
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    positions = []

    if start_position == "corner":
        # Use corners, then edges, then interior
        corners = [
            (-grid_info.x / 2, -grid_info.y / 2),  # bottom-left
            (grid_info.x / 2, -grid_info.y / 2),  # bottom-right
            (-grid_info.x / 2, grid_info.y / 2),  # top-left
            (grid_info.x / 2, grid_info.y / 2),  # top-right
        ]

        # Shuffle corners to randomize starting positions
        corners = list(corners)  # Make a copy
        rng.shuffle(corners)

        for i in range(min(num_agents, len(corners))):
            positions.append(corners[i])

        # If more agents than corners, add edge positions
        if num_agents > len(corners):
            edges = [
                (0, -grid_info.y / 2),  # bottom-center
                (0, grid_info.y / 2),  # top-center
                (-grid_info.x / 2, 0),  # left-center
                (grid_info.x / 2, 0),  # right-center
            ]
            rng.shuffle(edges)
            for i in range(min(num_agents - len(corners), len(edges))):
                positions.append(edges[i])

    elif start_position == "edge":
        # Distribute along edges
        total_perimeter = 2 * (grid_info.x + grid_info.y)
        spacing = total_perimeter / num_agents

        for i in range(num_agents):
            distance = i * spacing

            if distance < grid_info.x:
                # Bottom edge
                positions.append((distance - grid_info.x / 2, -grid_info.y / 2))
            elif distance < grid_info.x + grid_info.y:
                # Right edge
                d = distance - grid_info.x
                positions.append((grid_info.x / 2, d - grid_info.y / 2))
            elif distance < 2 * grid_info.x + grid_info.y:
                # Top edge
                d = distance - grid_info.x - grid_info.y
                positions.append((grid_info.x / 2 - d, grid_info.y / 2))
            else:
                # Left edge
                d = distance - 2 * grid_info.x - grid_info.y
                positions.append((-grid_info.x / 2, grid_info.y / 2 - d))

    elif start_position == "spread":
        # Grid-based spread across field
        n_cols = int(np.ceil(np.sqrt(num_agents)))
        n_rows = int(np.ceil(num_agents / n_cols))

        x_step = grid_info.x / (n_cols + 1)
        y_step = grid_info.y / (n_rows + 1)

        for i in range(num_agents):
            row = i // n_cols
            col = i % n_cols
            x = -grid_info.x / 2 + (col + 1) * x_step
            y = -grid_info.y / 2 + (row + 1) * y_step
            positions.append((x, y))

    # Fill remaining positions randomly if needed
    while len(positions) < num_agents:
        x = rng.uniform(-grid_info.x / 2, grid_info.x / 2)
        y = rng.uniform(-grid_info.y / 2, grid_info.y / 2)

        # Check minimum distance from existing positions
        valid = True
        for pos in positions:
            if np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2) < min_distance:
                valid = False
                break

        if valid:
            positions.append((x, y))

    return positions[:num_agents]
