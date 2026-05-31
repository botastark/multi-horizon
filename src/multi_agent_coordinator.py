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
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from itertools import product

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
        news_mode: Optional[str] = None,
        mode: Optional[str] = None,
        grid_info=None,
        debug_logs: bool = False,
    ):
        """
        Initialize multi-agent coordinator.

        Args:
            grid_shape: (H, W) shape of the belief map
            config: Configuration dict with multi_agent settings
            conf_dict: Confidence dictionary - sensor model parameters
            correlation_type: Pairwise correlation type ('equal', 'biased', 'adaptive')
            news_mode: News sharing mode ('BS' or 'BM')
            mode: Full mode label (e.g., 'IG', 'IGd', 'IG_BS', 'IGd_BM')
            grid_info: Grid information (for calculating h_displacement)
        """
        self.grid_shape = grid_shape
        self.config = config
        self.conf_dict = conf_dict
        self.num_agents = self.config.get("num_agents", 1)
        self.grid_info = grid_info

        # Store full mode label for planner access
        self.mode = mode if mode is not None else "IG"

        # Extract configuration
        ma_config = config.get("multi_agent", {})
        dec_config = config.get("decentralized", {})
        # Use explicit correlation_type parameter, fallback to config, then default
        self.correlation_type = correlation_type or ma_config.get(
            "correlation_type", ""
        )
        if self.correlation_type == "":
            if debug_logs:
                print(f"Warning: correlation_type not specified, defaulting to 'equal'")
            self.correlation_type = "equal"

        # Prefer provided `news_mode` override, otherwise `multi_agent.news_mode`, fall back to `decentralized.news_mode`, default to BM
        if news_mode is not None:
            self.news_mode = news_mode
        else:
            self.news_mode = ma_config.get(
                "news_mode", dec_config.get("news_mode", "BM")
            )
        self.lbp_iterations = ma_config.get("lbp_iterations", 1)

        self.enable_coordination = ma_config.get("enable_coordination", True)

        # Communication range calculation (matching reference paper)
        # Can be specified as:
        # 1. radius_multiplier: multiplied by h_displacement (field_len/2/n_h_act), -1 = unlimited
        # 2. communication_range: direct value in meters (-1 = unlimited)
        radius_multiplier = ma_config.get(
            "radius_multiplier",
            dec_config.get("radius_multiplier", None),
        )
        self._pa_componentwise_comm_distances = None

        if radius_multiplier is not None:
            if radius_multiplier == -1:
                # Unlimited range
                self.communication_range = -1
                if debug_logs:
                    print(f"Communication range: unlimited (radius_multiplier=-1)")
            else:
                # PA's adhoc environment uses n_h_act=8 for the 50x50m field,
                # independent of agent count, and proximity is componentwise in
                # x/y/z rather than Euclidean in x/y.
                n_h_act = ma_config.get("n_h_act", dec_config.get("n_h_act", 8))
                h_displacement = (grid_info.x / 2) / n_h_act
                fov_deg = ma_config.get("fov", dec_config.get("fov", 60.0))
                v_displacement = h_displacement / math.tan(math.radians(fov_deg) * 0.5)
                self.communication_range = radius_multiplier * h_displacement
                self._pa_componentwise_comm_distances = radius_multiplier * np.array(
                    [h_displacement, h_displacement, v_displacement],
                    dtype=float,
                )
                if debug_logs:
                    print(
                        f"Communication range calculation: radius_multiplier={radius_multiplier}, "
                        f"n_h_act={n_h_act}, field_len={grid_info.x}, "
                        f"h_displacement={h_displacement:.3f}, "
                        f"v_displacement={v_displacement:.3f}, "
                        f"comm_range={self.communication_range:.3f}m"
                    )
        else:
            # Fallback to direct communication_range specification
            self.communication_range = ma_config.get(
                "communication_range",
                dec_config.get("communication_range", -1),
            )  # -1 = unlimited
            if debug_logs:
                print(
                    f"Communication range: {self.communication_range}m (direct specification)"
                )

        self.collision_distance = ma_config.get("collision_avoidance_distance", 0.0)

        # Initialize communication bus
        self.comm_bus = CommunicationBus(self.num_agents)

        # Initialize multi-agent mapper (handles all belief mapping)
        # Use LBP news inference by default (matches paper's LBP_single/LBP_multi)
        news_inference = ma_config.get("news_inference_type", "LBP")
        fusion_eps = ma_config.get("fusion_eps", ma_config.get("eps", 1e-20))
        self.map = MultiAgentMapper(
            self.grid_shape,
            self.num_agents,
            self.conf_dict,
            correlation_type=self.correlation_type,
            news_mode=self.news_mode,
            lbp_iterations=self.lbp_iterations,
            news_inference_type=news_inference,
            eps=fusion_eps,
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

    def reset_start_position(
        self,
        grid_info: Any,
        start_position: str = "corner",
        seed: Optional[int] = None,
        min_distance: float = 10.0,
        altitude: Optional[float] = None,
        camera_hrange: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[float, float, float]]:
        """
        Reset and assign start positions for all agents.

        Args:
            grid_info: Object with `x` and `y` attributes (field extents).
            start_position: "corner" | "sample".
            seed: RNG seed for reproducible assignment.
            min_distance: Minimum distance between start positions (unused for corner).
            altitude: Starting altitude for all agents (optional).
            camera_hrange: Optional tuple `(min_h, max_h)` from a camera; if provided
                the minimum altitude `min_h` is used as the start altitude.

        Returns:
            List of assigned (x, y, altitude) tuples for each agent id.
        """

        # Determine starting altitude. Prefer camera_hrange minimum if provided,
        # otherwise use explicit altitude argument. If neither provided, raise.
        if camera_hrange is not None and len(camera_hrange) > 0:
            try:
                altitude = float(camera_hrange[0])
            except Exception:
                pass

        if altitude is None:
            raise ValueError(
                "altitude or camera_hrange must be provided for start positions"
            )

        # RNG for deterministic placement when seed provided
        rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

        positions: List[Tuple[float, float, float]] = []

        if start_position == "corner":
            corners = [
                (-grid_info.x / 2, -grid_info.y / 2),
                (grid_info.x / 2, -grid_info.y / 2),
                (-grid_info.x / 2, grid_info.y / 2),
                (grid_info.x / 2, grid_info.y / 2),
            ]

            # Assign corner positions in fixed order
            for i in range(min(self.num_agents, len(corners))):
                x, y = corners[i]
                positions.append((x, y, altitude))

            # If more agents than corners, add edge positions deterministically
            if self.num_agents > len(corners):
                edges = [
                    (0, -grid_info.y / 2),
                    (0, grid_info.y / 2),
                    (-grid_info.x / 2, 0),
                    (grid_info.x / 2, 0),
                ]
                idx = 0
                while len(positions) < self.num_agents:
                    ex, ey = edges[idx % len(edges)]
                    positions.append((ex, ey, altitude))
                    idx += 1
        elif start_position == "sample":
            # Reference simulator logic (regions)

            # Determine n_h_act based on num_agents to match reference simulator
            # Reference uses "adhoc" (n_h_act=8) for 8 agents, "normal" (n_h_act=5) otherwise
            n_h_act = 8 if self.num_agents == 8 else 5

            # Calculate h_displacement
            h_displacement = (grid_info.x / 2) / n_h_act

            # Region splitted field limits per agent
            n_agents_to_n_regions = {
                1: [1, 1],
                2: [2, 1],
                4: [2, 2],
                6: [3, 2],
                8: [4, 2],
                10: [5, 2],
            }

            if self.num_agents in n_agents_to_n_regions:
                n_regions = n_agents_to_n_regions[self.num_agents]

                min_space_x = -grid_info.x / 2
                max_space_x = grid_info.x / 2
                min_space_y = -grid_info.y / 2
                max_space_y = grid_info.y / 2

                num_points_x = 2 * n_h_act + 1
                x_positions = np.linspace(min_space_x, max_space_x, num_points_x)

                # Assume square grid for regions as per reference (using h_displacement for Y steps)
                num_points_y = int((grid_info.y / h_displacement)) + 1
                y_positions = np.linspace(min_space_y, max_space_y, num_points_y)

                x_intervals = len(x_positions) - 1
                y_intervals = len(y_positions) - 1

                region_x_limits = []
                for i in range(n_regions[0]):
                    idx_start = int(i * x_intervals / n_regions[0])
                    idx_end = int((i + 1) * x_intervals / n_regions[0])
                    region_x_limits.append(
                        [x_positions[idx_start], x_positions[idx_end]]
                    )

                region_y_limits = []
                for i in range(n_regions[1]):
                    idx_start = int(i * y_intervals / n_regions[1])
                    idx_end = int((i + 1) * y_intervals / n_regions[1])
                    region_y_limits.append(
                        [y_positions[idx_start], y_positions[idx_end]]
                    )

                regions_limits = list(product(region_x_limits, region_y_limits))

                # Assign agents
                for i in range(self.num_agents):
                    if i < len(regions_limits):
                        rl = regions_limits[i]
                        # Randomly sample within the region (discrete grid points)
                        # Filter x_positions/y_positions to find those within this region's limits
                        valid_x = x_positions[
                            (x_positions >= rl[0][0]) & (x_positions <= rl[0][1])
                        ]
                        valid_y = y_positions[
                            (y_positions >= rl[1][0]) & (y_positions <= rl[1][1])
                        ]

                        x = rng.choice(valid_x)
                        y = rng.choice(valid_y)
                        positions.append((x, y, altitude))
                    else:
                        positions.append((0.0, 0.0, altitude))
            else:
                raise NotImplementedError(
                    "start_position='sample' supports agent counts: "
                    f"{sorted(n_agents_to_n_regions.keys())}"
                )
        else:
            raise NotImplementedError("start_position must be 'corner' or 'sample'")

        # Update internal agent tracking and broadcast positions
        now = time.time()
        with self._states_lock:
            for agent_id in range(self.num_agents):
                x, y, z = positions[agent_id]
                self._agent_states[agent_id] = AgentState(
                    agent_id=agent_id, position=(x, y), altitude=z, last_update=now
                )

        # Broadcast position updates
        for agent_id in range(self.num_agents):
            pos = positions[agent_id]
            msg = AgentMessage(
                msg_type=MessageType.POSITION_UPDATE,
                sender_id=agent_id,
                data={"position": (pos[0], pos[1]), "altitude": pos[2]},
            )
            self.comm_bus.broadcast(msg, exclude_sender=False)

        return positions

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
            print(
                f"[DEBUG] Agent {agent_id} state not found in _agent_states (keys: {list(self._agent_states.keys())})"
            )
            return neighbors

        with self._states_lock:
            for other_id, state in self._agent_states.items():
                if other_id == agent_id:
                    continue

                if self._pa_componentwise_comm_distances is not None:
                    delta = np.array(
                        [
                            agent_state.position[0] - state.position[0],
                            agent_state.position[1] - state.position[1],
                            agent_state.altitude - state.altitude,
                        ],
                        dtype=float,
                    )
                    if np.all(
                        np.abs(delta) <= self._pa_componentwise_comm_distances
                    ):
                        neighbors.append(other_id)
                    continue

                # Position is already stored in METERS (from reset_start_position)
                # So we can calculate distance directly without cell_size conversion
                distance_meters = np.sqrt(
                    (agent_state.position[0] - state.position[0]) ** 2
                    + (agent_state.position[1] - state.position[1]) ** 2
                )

                if distance_meters <= self.communication_range:
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
