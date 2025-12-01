# multi_agent_coordinator.py
"""
Decentralized Multi-Agent Coordination Module with LBP Belief Fusion

This module provides coordination mechanisms for multiple UAV agents:
- Belief Fusion: LBP-based probabilistic belief fusion (inspired by Precision-Agriculture-Dev)
- Region Allocation: Decentralized auction/voting for target region assignment
- Collision Avoidance: Soft penalties for proximity to other agents
- Communication: Simulated message passing between agents

Key Design Principles:
1. Decentralized: No central controller, agents coordinate via message passing
2. Asynchronous: Agents operate independently with periodic coordination
3. Scalable: O(N) communication per agent (broadcast to all)
4. Robust: Graceful degradation if communication fails

Belief Fusion Methods:
- OG (Occupancy Grid): Simple Bayesian fusion of observations
- LBP (Loopy Belief Propagation): Spatial consistency via message passing
- News-based: Agents share only "news" (new observations) to avoid double-counting

Reference: LBP_cts_vectorized from Precision-Agriculture-Dev/simulator.py
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import time
import copy
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Message Types for Inter-Agent Communication
# =============================================================================


class MessageType(Enum):
    """Types of messages exchanged between agents."""

    # Coordination messages
    BELIEF_UPDATE = "belief_update"  # Share belief map updates
    NEWS_BELIEF = "news_belief"  # Share "news" observations (LBP style)
    REGION_CLAIM = "region_claim"  # Announce intent to cover a region
    REGION_RELEASE = "region_release"  # Release claim on a region
    POSITION_UPDATE = "position_update"  # Share current position

    # Observation messages (for LBP fusion)
    OBSERVATION = "observation"  # Share raw observation data
    LIKELIHOOD = "likelihood"  # Share likelihood values for fusion

    # Auction messages (for region allocation)
    AUCTION_BID = "auction_bid"  # Bid for a region
    AUCTION_RESULT = "auction_result"  # Announce auction winner

    # System messages
    SYNC_REQUEST = "sync_request"  # Request full state sync
    HEARTBEAT = "heartbeat"  # Alive signal


@dataclass
class AgentMessage:
    """
    Message passed between agents.

    Attributes:
        msg_type: Type of message
        sender_id: ID of sending agent
        timestamp: When the message was created
        data: Message-specific payload
        ttl: Time-to-live in seconds (for stale message filtering)
    """

    msg_type: MessageType
    sender_id: int
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    ttl: float = 5.0  # Message expires after 5 seconds


@dataclass
class AgentState:
    """
    State information for a single agent.

    Attributes:
        agent_id: Unique agent identifier
        position: Current (row, col) position in grid indices
        altitude: Current altitude
        target_region: Currently assigned target region
        coverage_progress: Agent's local coverage fraction
        last_update: Timestamp of last state update
    """

    agent_id: int
    position: Tuple[float, float]
    altitude: float
    target_region: Optional[int] = None
    coverage_progress: float = 0.0
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
# Belief Fusion - LBP-based Combining Observations from Multiple Agents
# =============================================================================


class LBPBeliefFusion:
    """
    LBP-based belief fusion for multi-agent systems.

    Architecture follows the paper's design:
    - NEWS UPDATE: Synchronous - all agents update their news beliefs first
    - NEWS FUSION: Synchronous - then all agents fuse with neighbors
    - LBP/CRF INFERENCE: Asynchronous/Decentralized - each agent runs LBP locally

    Two news modes available:
    - BS (Belief Single): Each agent maintains ONE news belief (diagonal), shared
      with all neighbors, then reset.
    - BM (Belief Multi): Each agent maintains SEPARATE news beliefs per neighbor
      (off-diagonal), allowing independent accumulation.

    Fusion formula (Occupancy Grid style):
        P(m=1|z_A, z_B) = [P(m=1|z_A) × P(m=1|z_B)] /
                          [P(m=1|z_A)×P(m=1|z_B) + P(m=0|z_A)×P(m=0|z_B)]

    Reference: simulator.py from Precision-Agriculture-Dev

    Design Decisions (confirmed):
    - Epsilon (1e-20) added for numerical stability: YES
    - Belief clipping to [0.001, 0.999]: YES
    - Neighbors determined by communication_range: YES
    - LBP runs asynchronously per agent (decentralized): YES
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        num_agents: int,
        use_lbp: bool = True,
        lbp_iterations: int = 1,
        pairwise_potential: Optional[np.ndarray] = None,
        news_mode: str = "BM",  # "BS" (single) or "BM" (multi)
    ):
        """
        Initialize LBP belief fusion.

        Args:
            grid_shape: (H, W) shape of belief map
            num_agents: Number of agents in the system
            use_lbp: Whether to use LBP for spatial consistency
            lbp_iterations: Number of LBP iterations
            pairwise_potential: 2x2 matrix for pairwise correlations
            news_mode: "BS" for single news belief per agent,
                      "BM" for per-neighbor news beliefs
        """
        self.grid_shape = grid_shape
        self.num_agents = num_agents
        self.use_lbp = use_lbp
        self.lbp_iterations = lbp_iterations
        self.news_mode = news_mode  # BS or BM

        # Pairwise potential for LBP (spatial correlation)
        if pairwise_potential is None:
            # Default: slight spatial correlation (neighboring cells likely similar)
            self.pairwise_potential = np.array([[0.6, 0.4], [0.4, 0.6]], dtype=float)
        else:
            self.pairwise_potential = pairwise_potential

        # Per-agent belief maps: map_beliefs[H, W, agent_id]
        self.map_beliefs = np.full((*grid_shape, num_agents), 0.5, dtype=float)

        # News belief maps depend on mode:
        # BS mode: news_map_beliefs[agent_id, agent_id, H, W] - only diagonal used
        # BM mode: news_map_beliefs[sender_id, receiver_id, H, W] - full matrix used
        self.news_map_beliefs = np.full(
            (num_agents, num_agents, *grid_shape), 0.5, dtype=float
        )

        # Messages for LBP: msgs[direction, H, W]
        # Directions: 0=up, 1=right, 2=down, 3=left, 4=local_evidence
        self.msgs = np.ones((5, *grid_shape), dtype=float) * 0.5
        self.msgs_buffer = np.ones_like(self.msgs) * 0.5

        # Direction slicing data for LBP message passing
        self._init_direction_slicing()

        # Lock for thread-safe operations
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "og_updates": 0,
            "lbp_propagations": 0,
            "news_fusions": 0,
            "news_mode": news_mode,
        }

        logger.info(f"LBPBeliefFusion initialized: mode={news_mode}, use_lbp={use_lbp}")

    def _init_direction_slicing(self):
        """Initialize slicing rules for message passing in each direction."""
        I, J = 0, 1
        n_cell_h, n_cell_w = self.grid_shape

        self.direction_to_slicing_data = {
            "up": {
                "product_slice": lambda fp_ij: (
                    (1, 2, 3, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(
                        1 if fp_ij["ul"][I] == 0 else 0, fp_ij["bl"][I] - fp_ij["ul"][I]
                    ),
                    slice(0, fp_ij["ur"][J] - fp_ij["ul"][J]),
                ),
                "write_slice": lambda fp_ij: (
                    2,
                    slice(
                        max(0, fp_ij["ul"][I] - 1), min(n_cell_h, fp_ij["bl"][I] - 1)
                    ),
                    slice(max(0, fp_ij["ul"][J]), min(n_cell_w, fp_ij["br"][J])),
                ),
            },
            "right": {
                "product_slice": lambda fp_ij: (
                    (0, 2, 3, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(0, fp_ij["bl"][I] - fp_ij["ul"][I]),
                    slice(
                        0,
                        (
                            fp_ij["ur"][J] - fp_ij["ul"][J] - 1
                            if fp_ij["ur"][J] == n_cell_w
                            else fp_ij["ur"][J] - fp_ij["ul"][J]
                        ),
                    ),
                ),
                "write_slice": lambda fp_ij: (
                    3,
                    slice(max(0, fp_ij["ul"][I]), min(n_cell_h, fp_ij["bl"][I])),
                    slice(
                        max(0, fp_ij["ul"][J] + 1), min(n_cell_w, fp_ij["br"][J] + 1)
                    ),
                ),
            },
            "down": {
                "product_slice": lambda fp_ij: (
                    (0, 1, 3, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(
                        0,
                        (
                            fp_ij["bl"][I] - fp_ij["ul"][I] - 1
                            if fp_ij["bl"][I] == n_cell_h
                            else fp_ij["bl"][I] - fp_ij["ul"][I]
                        ),
                    ),
                    slice(0, fp_ij["ur"][J] - fp_ij["ul"][J]),
                ),
                "write_slice": lambda fp_ij: (
                    0,
                    slice(
                        max(0, fp_ij["ul"][I] + 1), min(n_cell_h, fp_ij["bl"][I] + 1)
                    ),
                    slice(max(0, fp_ij["ul"][J]), min(n_cell_w, fp_ij["br"][J])),
                ),
            },
            "left": {
                "product_slice": lambda fp_ij: (
                    (0, 1, 2, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(0, fp_ij["bl"][I] - fp_ij["ul"][I]),
                    slice(
                        1 if fp_ij["ul"][J] == 0 else 0, fp_ij["ur"][J] - fp_ij["ul"][J]
                    ),
                ),
                "write_slice": lambda fp_ij: (
                    1,
                    slice(max(0, fp_ij["ul"][I]), min(n_cell_h, fp_ij["bl"][I])),
                    slice(
                        max(0, fp_ij["ul"][J] - 1), min(n_cell_w, fp_ij["br"][J] - 1)
                    ),
                ),
            },
        }

    def update_belief_OG(
        self,
        agent_id: int,
        fp_ij: Dict[str, Tuple[int, int]],
        observation: np.ndarray,
        sigma0: float,
        sigma1: float,
    ):
        """
        Update agent's belief using Occupancy Grid (Bayesian) approach.

        Args:
            agent_id: ID of observing agent
            fp_ij: Footprint indices dict with keys 'ul', 'bl', 'ur', 'br'
            observation: Binary observation (0=free, 1=occupied)
            sigma0: P(observe 1 | true state is 0) - false positive rate
            sigma1: P(observe 0 | true state is 1) - false negative rate
        """
        I, J = 0, 1

        with self._lock:
            # Compute likelihoods
            likelihood_m_zero = np.where(observation == 0, 1 - sigma0, sigma0)
            likelihood_m_one = np.where(observation == 0, sigma1, 1 - sigma1)

            # Extract prior
            prior = self.map_beliefs[
                fp_ij["ul"][I] : fp_ij["bl"][I],
                fp_ij["ul"][J] : fp_ij["ur"][J],
                agent_id,
            ]

            # Bayesian update
            posterior_m_zero = likelihood_m_zero * (1.0 - prior)
            posterior_m_one = likelihood_m_one * prior

            # Normalize
            epsilon = 1e-20
            denominator = posterior_m_zero + posterior_m_one + epsilon
            posterior_m_one_norm = posterior_m_one / denominator

            # Clip to valid range
            posterior_m_one_norm = np.clip(posterior_m_one_norm, 0.001, 0.999)

            # Update belief
            self.map_beliefs[
                fp_ij["ul"][I] : fp_ij["bl"][I],
                fp_ij["ul"][J] : fp_ij["ur"][J],
                agent_id,
            ] = posterior_m_one_norm

            self._stats["og_updates"] += 1

    def update_belief_LBP(
        self,
        agent_id: int,
        fp_ij: Dict[str, Tuple[int, int]],
        observation: np.ndarray,
        sigma0: float,
        sigma1: float,
    ):
        """
        Update agent's belief using LBP for spatial consistency (ASYNCHRONOUS).

        This runs LOCALLY on each agent in a DECENTRALIZED manner.
        Each agent runs its own LBP inference independently.

        First does OG update, then propagates messages for spatial smoothing.

        Args:
            agent_id: ID of observing agent
            fp_ij: Footprint indices dict
            observation: Binary observation
            sigma0, sigma1: Sensor model parameters
        """
        # First do OG update
        self.update_belief_OG(agent_id, fp_ij, observation, sigma0, sigma1)

        if not self.use_lbp:
            return

        I, J = 0, 1

        with self._lock:
            # Reset messages
            self.msgs[:] = 0.5
            self.msgs_buffer[:] = 0.5

            # Inject current beliefs as local evidence
            self.msgs[4, :, :] = self.map_beliefs[:, :, agent_id]

            # Run LBP iterations
            for _ in range(self.lbp_iterations):
                for direction, data in self.direction_to_slicing_data.items():
                    product_slice = data["product_slice"](fp_ij)
                    read_slice = data["read_slice"](fp_ij)
                    write_slice = data["write_slice"](fp_ij)

                    # Element-wise product of incoming messages
                    mul_0 = np.prod(1 - self.msgs[product_slice], axis=0)
                    mul_1 = np.prod(self.msgs[product_slice], axis=0)

                    # Message computation with pairwise potential
                    psi = self.pairwise_potential
                    msg_0 = psi[0, 0] * mul_0 + psi[0, 1] * mul_1
                    msg_1 = psi[1, 0] * mul_0 + psi[1, 1] * mul_1

                    # Normalize
                    norm_msg_1 = msg_1 / (msg_0 + msg_1 + 1e-20)

                    # Buffer the message
                    self.msgs_buffer[write_slice] = norm_msg_1[read_slice]

                # Update messages (except local evidence)
                self.msgs[:4, :, :] = self.msgs_buffer[:4, :, :]

            # Compute final beliefs
            product_slice = self.direction_to_slicing_data["up"]["product_slice"](fp_ij)
            bel_0 = np.prod(
                1 - self.msgs[:, product_slice[1], product_slice[2]], axis=0
            )
            bel_1 = np.prod(self.msgs[:, product_slice[1], product_slice[2]], axis=0)

            new_beliefs = bel_1 / (bel_0 + bel_1 + 1e-20)
            new_beliefs = np.clip(new_beliefs, 0.001, 0.999)

            self.map_beliefs[product_slice[1], product_slice[2], agent_id] = new_beliefs

            self._stats["lbp_propagations"] += 1

    def update_news_and_fuse(
        self,
        agent_id: int,
        fp_ij: Dict[str, Tuple[int, int]],
        observation: np.ndarray,
        sigma0: float,
        sigma1: float,
        neighbor_ids: List[int],
    ):
        """
        Update news beliefs and fuse with neighbors.

        IMPORTANT: In the paper, observation update and fusion are SEPARATE phases:
        1. ALL agents update their news beliefs first
        2. THEN all agents fuse with neighbors

        This method combines both for a single agent. For true paper compliance,
        the coordinator should call update_news_belief() for ALL agents first,
        then call fuse_news_with_neighbors() for ALL agents.

        Two modes (from the paper):
        - BS (Belief Single): Agent has ONE news belief (diagonal), shared with all
          neighbors, then reset. Paper: _update_news_belief_OG_and_fuse_single
        - BM (Belief Multi): Agent has SEPARATE news beliefs per neighbor (off-diagonal).
          Paper: _update_news_belief_OG_and_fuse_multi

        ASSUMPTIONS vs Paper:
        1. We call update+fuse per agent (paper does all updates, then all fusions)
        2. We clip beliefs to [0.001, 0.999] (paper doesn't clip)
        3. We add epsilon to denominator (paper may get div-by-zero)
        4. This is called AFTER update_belief_OG (agent's own belief already updated)

        Args:
            agent_id: ID of observing agent
            fp_ij: Footprint indices
            observation: Binary observation
            sigma0, sigma1: Sensor model parameters
            neighbor_ids: IDs of neighboring agents to share with
        """
        I, J = 0, 1

        with self._lock:
            # Compute likelihoods (same as paper)
            likelihood_m_zero = np.where(observation == 0, 1 - sigma0, sigma0)
            likelihood_m_one = np.where(observation == 0, sigma1, 1 - sigma1)

            epsilon = 1e-20  # ASSUMPTION: paper doesn't use epsilon

            if self.news_mode == "BS":
                # ===== BS MODE: Single news belief per agent =====
                # Paper: _update_news_belief_OG_and_fuse_single
                # Uses news_map_beliefs[agent_id, agent_id, :, :] (diagonal only)

                # Update agent's single news belief (diagonal element)
                # Only update the footprint region, not the whole map
                prior_news = self.news_map_beliefs[
                    agent_id,
                    agent_id,
                    fp_ij["ul"][I] : fp_ij["bl"][I],
                    fp_ij["ul"][J] : fp_ij["ur"][J],
                ]

                posterior_m_zero = likelihood_m_zero * (1.0 - prior_news)
                posterior_m_one = likelihood_m_one * prior_news

                # Paper: posterior_m_one / (posterior_m_zero + posterior_m_one)
                posterior_m_one_norm = posterior_m_one / (
                    posterior_m_zero + posterior_m_one + epsilon
                )
                # ASSUMPTION: clipping (paper doesn't clip)
                posterior_m_one_norm = np.clip(posterior_m_one_norm, 0.001, 0.999)

                self.news_map_beliefs[
                    agent_id,
                    agent_id,
                    fp_ij["ul"][I] : fp_ij["bl"][I],
                    fp_ij["ul"][J] : fp_ij["ur"][J],
                ] = posterior_m_one_norm

                # Fuse this single news with ALL neighbors
                # Paper: mul = news_map_beliefs[agent_id, agent_id, :, :] * map_beliefs[:,:,neighbor_id]
                #        map_beliefs[:, :, neighbor_id] = mul / (mul + (1-news)*(1-neighbor_belief))
                for neighbor_id in neighbor_ids:
                    news = self.news_map_beliefs[agent_id, agent_id, :, :]
                    neighbor_belief = self.map_beliefs[:, :, neighbor_id]

                    # Bayesian fusion (matches paper formula)
                    mul = news * neighbor_belief
                    denominator = mul + (1.0 - news) * (1.0 - neighbor_belief)
                    # ASSUMPTION: add epsilon to avoid div-by-zero
                    fused_belief = mul / (denominator + epsilon)
                    # ASSUMPTION: clipping
                    fused_belief = np.clip(fused_belief, 0.001, 0.999)

                    # Update NEIGHBOR's belief (matches paper)
                    self.map_beliefs[:, :, neighbor_id] = fused_belief

                # Reset news after sharing (if we had neighbors) - matches paper
                if len(neighbor_ids) > 0:
                    self.news_map_beliefs[agent_id, agent_id, :, :] = 0.5

            else:
                # ===== BM MODE: Per-neighbor news beliefs =====
                # Paper: _update_news_belief_OG_and_fuse_multi
                # Uses news_map_beliefs[agent_id, other_id, :, :] (off-diagonal)

                # Update news beliefs for EACH other agent separately
                # Paper: iterates over all news_map_belief_id where agent_id != news_map_belief_id
                for news_target_id in range(self.num_agents):
                    if news_target_id == agent_id:
                        continue  # Skip self (diagonal)

                    prior_news = self.news_map_beliefs[
                        agent_id,
                        news_target_id,
                        fp_ij["ul"][I] : fp_ij["bl"][I],
                        fp_ij["ul"][J] : fp_ij["ur"][J],
                    ]

                    posterior_m_zero = likelihood_m_zero * (1.0 - prior_news)
                    posterior_m_one = likelihood_m_one * prior_news

                    posterior_m_one_norm = posterior_m_one / (
                        posterior_m_zero + posterior_m_one + epsilon
                    )
                    posterior_m_one_norm = np.clip(posterior_m_one_norm, 0.001, 0.999)

                    self.news_map_beliefs[
                        agent_id,
                        news_target_id,
                        fp_ij["ul"][I] : fp_ij["bl"][I],
                        fp_ij["ul"][J] : fp_ij["ur"][J],
                    ] = posterior_m_one_norm

                # Fuse with each neighbor using their specific news belief
                # Paper: mul = news_map_beliefs[agent_id, neighbor_id, :, :] * map_beliefs[:,:,neighbor_id]
                for neighbor_id in neighbor_ids:
                    news = self.news_map_beliefs[agent_id, neighbor_id, :, :]
                    neighbor_belief = self.map_beliefs[:, :, neighbor_id]

                    mul = news * neighbor_belief
                    denominator = mul + (1.0 - news) * (1.0 - neighbor_belief)
                    fused_belief = mul / (denominator + epsilon)
                    fused_belief = np.clip(fused_belief, 0.001, 0.999)

                    self.map_beliefs[:, :, neighbor_id] = fused_belief

                    # Reset only the news belief for THIS neighbor after sharing
                    # (matches paper behavior)
                    self.news_map_beliefs[agent_id, neighbor_id, :, :] = 0.5

            self._stats["news_fusions"] += 1

    def update_news_belief_only(
        self,
        agent_id: int,
        fp_ij: Dict[str, Tuple[int, int]],
        observation: np.ndarray,
        sigma0: float,
        sigma1: float,
    ):
        """
        Update ONLY the news belief without fusing.

        For paper-compliant operation: call this for ALL agents first,
        then call fuse_news_with_neighbors() for ALL agents.

        Args:
            agent_id: ID of observing agent
            fp_ij: Footprint indices
            observation: Binary observation
            sigma0, sigma1: Sensor model parameters
        """
        I, J = 0, 1

        with self._lock:
            likelihood_m_zero = np.where(observation == 0, 1 - sigma0, sigma0)
            likelihood_m_one = np.where(observation == 0, sigma1, 1 - sigma1)
            epsilon = 1e-20

            if self.news_mode == "BS":
                # Update diagonal only
                prior_news = self.news_map_beliefs[
                    agent_id,
                    agent_id,
                    fp_ij["ul"][I] : fp_ij["bl"][I],
                    fp_ij["ul"][J] : fp_ij["ur"][J],
                ]

                posterior_m_zero = likelihood_m_zero * (1.0 - prior_news)
                posterior_m_one = likelihood_m_one * prior_news
                posterior_m_one_norm = posterior_m_one / (
                    posterior_m_zero + posterior_m_one + epsilon
                )
                posterior_m_one_norm = np.clip(posterior_m_one_norm, 0.001, 0.999)

                self.news_map_beliefs[
                    agent_id,
                    agent_id,
                    fp_ij["ul"][I] : fp_ij["bl"][I],
                    fp_ij["ul"][J] : fp_ij["ur"][J],
                ] = posterior_m_one_norm
            else:
                # Update all off-diagonal entries for this agent
                for news_target_id in range(self.num_agents):
                    if news_target_id == agent_id:
                        continue

                    prior_news = self.news_map_beliefs[
                        agent_id,
                        news_target_id,
                        fp_ij["ul"][I] : fp_ij["bl"][I],
                        fp_ij["ul"][J] : fp_ij["ur"][J],
                    ]

                    posterior_m_zero = likelihood_m_zero * (1.0 - prior_news)
                    posterior_m_one = likelihood_m_one * prior_news
                    posterior_m_one_norm = posterior_m_one / (
                        posterior_m_zero + posterior_m_one + epsilon
                    )
                    posterior_m_one_norm = np.clip(posterior_m_one_norm, 0.001, 0.999)

                    self.news_map_beliefs[
                        agent_id,
                        news_target_id,
                        fp_ij["ul"][I] : fp_ij["bl"][I],
                        fp_ij["ul"][J] : fp_ij["ur"][J],
                    ] = posterior_m_one_norm

    def fuse_news_with_neighbors(
        self,
        agent_id: int,
        neighbor_ids: List[int],
    ):
        """
        Fuse accumulated news with neighbors (SEPARATE from update).

        For paper-compliant operation: call update_news_belief_only() for ALL
        agents first, then call this for ALL agents.

        Args:
            agent_id: ID of agent doing the fusion
            neighbor_ids: IDs of neighboring agents to share with
        """
        epsilon = 1e-20

        with self._lock:
            if self.news_mode == "BS":
                for neighbor_id in neighbor_ids:
                    news = self.news_map_beliefs[agent_id, agent_id, :, :]
                    neighbor_belief = self.map_beliefs[:, :, neighbor_id]

                    mul = news * neighbor_belief
                    denominator = mul + (1.0 - news) * (1.0 - neighbor_belief)
                    fused_belief = mul / (denominator + epsilon)
                    fused_belief = np.clip(fused_belief, 0.001, 0.999)

                    self.map_beliefs[:, :, neighbor_id] = fused_belief

                if len(neighbor_ids) > 0:
                    self.news_map_beliefs[agent_id, agent_id, :, :] = 0.5
            else:
                for neighbor_id in neighbor_ids:
                    news = self.news_map_beliefs[agent_id, neighbor_id, :, :]
                    neighbor_belief = self.map_beliefs[:, :, neighbor_id]

                    mul = news * neighbor_belief
                    denominator = mul + (1.0 - news) * (1.0 - neighbor_belief)
                    fused_belief = mul / (denominator + epsilon)
                    fused_belief = np.clip(fused_belief, 0.001, 0.999)

                    self.map_beliefs[:, :, neighbor_id] = fused_belief
                    self.news_map_beliefs[agent_id, neighbor_id, :, :] = 0.5

    def get_agent_belief(self, agent_id: int) -> np.ndarray:
        """Get belief map for a specific agent."""
        with self._lock:
            return self.map_beliefs[:, :, agent_id].copy()

    def get_fused_belief(self) -> np.ndarray:
        """
        Get consensus belief by fusing all agents' beliefs.

        Uses product-of-experts fusion:
        P(m|z_all) ∝ ∏_i P(m|z_i)
        """
        with self._lock:
            # Product of all beliefs (occupied probability)
            prod_occupied = np.prod(self.map_beliefs, axis=2)
            prod_free = np.prod(1.0 - self.map_beliefs, axis=2)

            # Normalize
            epsilon = 1e-20
            fused = prod_occupied / (prod_occupied + prod_free + epsilon)
            return np.clip(fused, 0.001, 0.999)

    def get_agent_coverage(self, agent_id: int, threshold: float = 0.3) -> float:
        """
        Compute coverage for an agent (fraction of cells with low entropy).

        Args:
            agent_id: Agent ID
            threshold: Entropy threshold for "covered"

        Returns:
            Coverage fraction (0-1)
        """
        with self._lock:
            beliefs = self.map_beliefs[:, :, agent_id]
            entropy = -beliefs * np.log(beliefs + 1e-10) - (1 - beliefs) * np.log(
                1 - beliefs + 1e-10
            )
            covered = entropy < threshold
            return float(np.mean(covered))

    def reset(self):
        """Reset all beliefs to prior."""
        with self._lock:
            self.map_beliefs[:] = 0.5
            self.news_map_beliefs[:] = 0.5
            self.msgs[:] = 0.5
            self.msgs_buffer[:] = 0.5
            self._stats = {k: 0 for k in self._stats}

    def get_stats(self) -> Dict[str, int]:
        """Get fusion statistics."""
        with self._lock:
            return self._stats.copy()


class BeliefFusion:
    """
    Handles fusion of belief maps from multiple agents.

    Uses a weighted average approach where more recent observations
    have higher weight, and agents can have different confidence levels.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        fusion_weight: float = 0.7,
        decay_rate: float = 0.95,
    ):
        """
        Initialize belief fusion.

        Args:
            grid_shape: (H, W) shape of belief map
            fusion_weight: Weight for incoming beliefs vs current (0-1)
            decay_rate: How much to discount older observations
        """
        self.grid_shape = grid_shape
        self.fusion_weight = fusion_weight
        self.decay_rate = decay_rate

        # Fused global belief map
        self._global_belief = np.full((*grid_shape, 2), 0.5)

        # Track last update time per cell
        self._last_update = np.zeros(grid_shape)

        # Per-agent contribution tracking
        self._agent_contributions: Dict[int, np.ndarray] = {}

        # Lock for thread-safe fusion
        self._lock = threading.Lock()

    def fuse_belief(
        self,
        agent_id: int,
        local_belief: np.ndarray,
        observed_mask: Optional[np.ndarray] = None,
    ):
        """
        Fuse an agent's local belief into the global belief.

        Only updates cells that were actually observed by the agent.

        Args:
            agent_id: ID of contributing agent
            local_belief: Agent's local belief map (H, W, 2)
            observed_mask: Boolean mask of cells observed (H, W)
        """
        with self._lock:
            current_time = time.time()

            if observed_mask is None:
                # Assume agent observed everything (fallback)
                observed_mask = np.ones(self.grid_shape, dtype=bool)

            # Compute time-based decay for existing beliefs
            time_delta = current_time - self._last_update
            decay = np.power(self.decay_rate, time_delta)

            # Update only observed cells
            for i in range(self.grid_shape[0]):
                for j in range(self.grid_shape[1]):
                    if observed_mask[i, j]:
                        # Weighted fusion: new = w * incoming + (1-w) * existing * decay
                        self._global_belief[i, j] = (
                            self.fusion_weight * local_belief[i, j]
                            + (1 - self.fusion_weight)
                            * self._global_belief[i, j]
                            * decay[i, j]
                        )
                        self._last_update[i, j] = current_time

            # Normalize beliefs
            self._global_belief = np.clip(self._global_belief, 0.001, 0.999)

            # Track agent's contribution
            self._agent_contributions[agent_id] = observed_mask.copy()

    def get_global_belief(self) -> np.ndarray:
        """Get the fused global belief map."""
        with self._lock:
            return self._global_belief.copy()

    def get_coverage_per_agent(self) -> Dict[int, float]:
        """Get coverage contribution from each agent."""
        with self._lock:
            result = {}
            for agent_id, mask in self._agent_contributions.items():
                result[agent_id] = float(np.mean(mask))
            return result


# =============================================================================
# Region Allocator - Decentralized Region Assignment
# =============================================================================


class RegionAllocator:
    """
    Decentralized region allocation using auction-based mechanism.

    Agents bid for regions based on their proximity and the region's value.
    Conflicts are resolved using a simple priority rule (lower distance wins).
    """

    def __init__(
        self,
        num_agents: int,
        num_regions: int,
        allocation_strategy: str = "auction",
    ):
        """
        Initialize region allocator.

        Args:
            num_agents: Number of agents
            num_regions: Number of regions in the field
            allocation_strategy: "auction", "greedy", or "round_robin"
        """
        self.num_agents = num_agents
        self.num_regions = num_regions
        self.allocation_strategy = allocation_strategy

        # Region assignments: region_id -> agent_id (or None)
        self._assignments: Dict[int, Optional[int]] = {
            r: None for r in range(num_regions)
        }

        # Agent's current region: agent_id -> region_id
        self._agent_regions: Dict[int, Optional[int]] = {
            a: None for a in range(num_agents)
        }

        # Pending bids: region_id -> list of (agent_id, bid_value)
        self._pending_bids: Dict[int, List[Tuple[int, float]]] = {}

        # Lock for thread-safe operations
        self._lock = threading.Lock()

    def request_region(
        self,
        agent_id: int,
        region_scores: Dict[int, float],
        agent_position: Tuple[float, float],
        region_centers: Dict[int, Tuple[float, float]],
    ) -> Optional[int]:
        """
        Request a region for an agent based on scores and availability.

        Args:
            agent_id: ID of requesting agent
            region_scores: HLP scores for each region
            agent_position: Current agent position (row, col)
            region_centers: Center positions of all regions

        Returns:
            Assigned region ID or None if no region available
        """
        with self._lock:
            if self.allocation_strategy == "greedy":
                return self._greedy_allocation(
                    agent_id, region_scores, agent_position, region_centers
                )
            elif self.allocation_strategy == "round_robin":
                return self._round_robin_allocation(agent_id)
            else:  # auction
                return self._auction_allocation(
                    agent_id, region_scores, agent_position, region_centers
                )

    def _greedy_allocation(
        self,
        agent_id: int,
        region_scores: Dict[int, float],
        agent_position: Tuple[float, float],
        region_centers: Dict[int, Tuple[float, float]],
    ) -> Optional[int]:
        """Greedy allocation: agent gets highest-score unassigned region."""
        # Sort regions by score
        sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)

        # Find first unassigned region (or assigned to self)
        for region_id, score in sorted_regions:
            current_owner = self._assignments.get(region_id)
            if current_owner is None or current_owner == agent_id:
                # Assign to this agent
                self._release_agent_region(agent_id)
                self._assignments[region_id] = agent_id
                self._agent_regions[agent_id] = region_id
                return region_id

        # No unassigned regions - keep current if any
        return self._agent_regions.get(agent_id)

    def _auction_allocation(
        self,
        agent_id: int,
        region_scores: Dict[int, float],
        agent_position: Tuple[float, float],
        region_centers: Dict[int, Tuple[float, float]],
    ) -> Optional[int]:
        """
        Auction-based allocation with distance-weighted bids.

        Bid = score - distance_penalty
        Lower distance = higher bid (proximity bonus)
        """
        # Compute bids for each region
        bids = {}
        for region_id, score in region_scores.items():
            center = region_centers.get(region_id, (0, 0))
            distance = np.sqrt(
                (agent_position[0] - center[0]) ** 2
                + (agent_position[1] - center[1]) ** 2
            )
            # Bid includes proximity bonus (closer = better)
            max_distance = 1000.0  # Normalization factor
            proximity_bonus = 1.0 - (distance / max_distance)
            bids[region_id] = score + 0.3 * proximity_bonus

        # Sort by bid value
        sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)

        # Try to claim best available region
        for region_id, bid in sorted_bids:
            current_owner = self._assignments.get(region_id)

            if current_owner is None:
                # Unassigned - claim it
                self._release_agent_region(agent_id)
                self._assignments[region_id] = agent_id
                self._agent_regions[agent_id] = region_id
                return region_id
            elif current_owner == agent_id:
                # Already ours
                return region_id
            else:
                # Contested - check if we have a better bid
                # Simple rule: agent with lower ID wins ties
                # In practice, would use actual bid comparison via messages
                if agent_id < current_owner:
                    # We have priority - take over
                    self._agent_regions[current_owner] = None
                    self._release_agent_region(agent_id)
                    self._assignments[region_id] = agent_id
                    self._agent_regions[agent_id] = region_id
                    return region_id
                # else: keep looking

        return self._agent_regions.get(agent_id)

    def _round_robin_allocation(self, agent_id: int) -> Optional[int]:
        """Simple round-robin allocation based on agent ID."""
        # Divide regions evenly among agents
        regions_per_agent = self.num_regions // self.num_agents
        start_region = agent_id * regions_per_agent

        # Return the first region in agent's "slice"
        if start_region < self.num_regions:
            return start_region
        return None

    def _release_agent_region(self, agent_id: int):
        """Release any region currently assigned to an agent."""
        current_region = self._agent_regions.get(agent_id)
        if current_region is not None:
            self._assignments[current_region] = None
            self._agent_regions[agent_id] = None

    def release_region(self, agent_id: int, region_id: int):
        """Explicitly release a region assignment."""
        with self._lock:
            if self._assignments.get(region_id) == agent_id:
                self._assignments[region_id] = None
            if self._agent_regions.get(agent_id) == region_id:
                self._agent_regions[agent_id] = None

    def get_assignments(self) -> Dict[int, Optional[int]]:
        """Get current region assignments."""
        with self._lock:
            return self._assignments.copy()

    def get_agent_region(self, agent_id: int) -> Optional[int]:
        """Get the region assigned to a specific agent."""
        with self._lock:
            return self._agent_regions.get(agent_id)


# =============================================================================
# Multi-Agent Coordinator - Main Coordination Interface
# =============================================================================


class MultiAgentCoordinator:
    """
    Main coordinator for multi-agent UAV operations.

    Integrates LBP belief fusion, region allocation, and collision avoidance
    into a single interface that each agent can use for coordination.
    """

    def __init__(
        self,
        num_agents: int,
        grid_shape: Tuple[int, int],
        config: Dict[str, Any],
    ):
        """
        Initialize multi-agent coordinator.

        Args:
            num_agents: Number of agents in the system
            grid_shape: (H, W) shape of the belief map
            config: Configuration dict with multi_agent settings
        """
        self.num_agents = num_agents
        self.grid_shape = grid_shape
        self.config = config

        # Extract configuration
        ma_config = config.get("multi_agent", {})
        self.enable_coordination = ma_config.get("enable_coordination", True)
        self.belief_fusion_enabled = ma_config.get("belief_fusion", True)
        self.allocation_strategy = ma_config.get("region_allocation", "auction")
        self.communication_range = ma_config.get(
            "communication_range", -1
        )  # -1 = unlimited
        self.collision_distance = ma_config.get("collision_avoidance_distance", 5.0)
        self.fusion_weight = ma_config.get("belief_fusion_weight", 0.7)
        self.coordination_frequency = ma_config.get("coordination_frequency", 5)

        # LBP-specific settings
        self.use_lbp = ma_config.get("use_lbp", True)
        self.lbp_iterations = ma_config.get("lbp_iterations", 1)
        self.fusion_method = ma_config.get(
            "fusion_method", "lbp"
        )  # "lbp" or "weighted"
        self.news_mode = ma_config.get(
            "news_mode", "BM"
        )  # "BS" (single) or "BM" (multi)

        # Initialize components
        self.comm_bus = CommunicationBus(num_agents)

        # Initialize belief fusion based on method
        if self.belief_fusion_enabled:
            if self.fusion_method == "lbp":
                self.lbp_fusion = LBPBeliefFusion(
                    grid_shape=grid_shape,
                    num_agents=num_agents,
                    use_lbp=self.use_lbp,
                    lbp_iterations=self.lbp_iterations,
                    news_mode=self.news_mode,
                )
                self.belief_fusion = None  # Not using weighted fusion
            else:
                self.lbp_fusion = None
                self.belief_fusion = BeliefFusion(
                    grid_shape=grid_shape,
                    fusion_weight=self.fusion_weight,
                )
        else:
            self.lbp_fusion = None
            self.belief_fusion = None

        # Region allocator (will be initialized when we know num_regions)
        self.region_allocator: Optional[RegionAllocator] = None
        self._num_regions = 0

        # Agent states
        self._agent_states: Dict[int, AgentState] = {}
        self._states_lock = threading.Lock()

        # Coordination step counter (per agent)
        self._step_counters: Dict[int, int] = {i: 0 for i in range(num_agents)}

        # Statistics
        self._stats = {
            "belief_fusions": 0,
            "region_allocations": 0,
            "collision_avoidances": 0,
        }

        logger.info(
            f"MultiAgentCoordinator initialized: {num_agents} agents, "
            f"coordination={'enabled' if self.enable_coordination else 'disabled'}, "
            f"belief_fusion={self.fusion_method if self.belief_fusion_enabled else 'disabled'}, "
            f"news_mode={self.news_mode}, "
            f"allocation={self.allocation_strategy}"
        )

    def initialize_region_allocator(self, num_regions: int):
        """
        Initialize region allocator once we know the number of regions.

        Args:
            num_regions: Number of regions in the field partition
        """
        self._num_regions = num_regions
        self.region_allocator = RegionAllocator(
            num_agents=self.num_agents,
            num_regions=num_regions,
            allocation_strategy=self.allocation_strategy,
        )
        logger.info(f"Region allocator initialized: {num_regions} regions")

    def update_agent_state(
        self,
        agent_id: int,
        position: Tuple[float, float],
        altitude: float,
        target_region: Optional[int] = None,
        coverage: float = 0.0,
    ):
        """
        Update an agent's state (called by each agent).

        Args:
            agent_id: ID of agent
            position: Current (row, col) position
            altitude: Current altitude
            target_region: Currently assigned target region
            coverage: Agent's coverage progress
        """
        with self._states_lock:
            self._agent_states[agent_id] = AgentState(
                agent_id=agent_id,
                position=position,
                altitude=altitude,
                target_region=target_region,
                coverage_progress=coverage,
                last_update=time.time(),
            )

        # Broadcast position update to other agents
        if self.enable_coordination:
            msg = AgentMessage(
                msg_type=MessageType.POSITION_UPDATE,
                sender_id=agent_id,
                data={
                    "position": position,
                    "altitude": altitude,
                    "target_region": target_region,
                },
            )
            self.comm_bus.broadcast(msg)

    def share_belief(
        self,
        agent_id: int,
        local_belief: np.ndarray,
        observed_mask: Optional[np.ndarray] = None,
    ):
        """
        Share an agent's belief map for fusion.

        Args:
            agent_id: ID of contributing agent
            local_belief: Agent's local belief map
            observed_mask: Mask of cells observed by this agent
        """
        if not self.belief_fusion_enabled:
            return

        if self.belief_fusion is not None:
            self.belief_fusion.fuse_belief(agent_id, local_belief, observed_mask)

        self._stats["belief_fusions"] += 1

        # Optionally broadcast belief update message
        if self.enable_coordination:
            msg = AgentMessage(
                msg_type=MessageType.BELIEF_UPDATE,
                sender_id=agent_id,
                data={
                    "observed_cells": (
                        int(np.sum(observed_mask)) if observed_mask is not None else 0
                    )
                },
            )
            self.comm_bus.broadcast(msg)

    def share_observation(
        self,
        agent_id: int,
        fp_ij: Dict[str, Tuple[int, int]],
        observation: np.ndarray,
        sigma0: float,
        sigma1: float,
    ):
        """
        Share an observation for LBP-style fusion (COMBINED update+fuse).

        NOTE: For paper-compliant synchronous operation, use:
        1. update_agent_news() for ALL agents first
        2. fuse_agent_news() for ALL agents second

        This combined method is for convenience when order doesn't matter.

        Args:
            agent_id: ID of observing agent
            fp_ij: Footprint indices dict with keys 'ul', 'bl', 'ur', 'br'
            observation: Binary observation array
            sigma0: False positive rate
            sigma1: False negative rate
        """
        if not self.belief_fusion_enabled or self.lbp_fusion is None:
            return

        # Get neighbor IDs within communication range
        neighbor_ids = self._get_neighbors_in_range(agent_id)

        # Update belief with LBP (asynchronous/decentralized)
        if self.use_lbp:
            self.lbp_fusion.update_belief_LBP(
                agent_id, fp_ij, observation, sigma0, sigma1
            )
        else:
            self.lbp_fusion.update_belief_OG(
                agent_id, fp_ij, observation, sigma0, sigma1
            )

        # Fuse news with neighbors
        if neighbor_ids:
            self.lbp_fusion.update_news_and_fuse(
                agent_id, fp_ij, observation, sigma0, sigma1, neighbor_ids
            )

        self._stats["belief_fusions"] += 1

        # Broadcast observation message
        if self.enable_coordination:
            msg = AgentMessage(
                msg_type=MessageType.OBSERVATION,
                sender_id=agent_id,
                data={
                    "fp_ij": {k: list(v) for k, v in fp_ij.items()},
                    "sigma0": sigma0,
                    "sigma1": sigma1,
                },
            )
            self.comm_bus.broadcast(msg)

    def update_agent_news(
        self,
        agent_id: int,
        fp_ij: Dict[str, Tuple[int, int]],
        observation: np.ndarray,
        sigma0: float,
        sigma1: float,
    ):
        """
        PHASE 1 (SYNCHRONOUS): Update agent's news belief only.

        Call this for ALL agents first, before calling fuse_agent_news().
        This follows the paper's synchronous update-then-fuse pattern.

        Also runs LBP locally (asynchronous/decentralized inference).

        Args:
            agent_id: ID of observing agent
            fp_ij: Footprint indices
            observation: Binary observation
            sigma0, sigma1: Sensor model parameters
        """
        if not self.belief_fusion_enabled or self.lbp_fusion is None:
            return

        # LBP inference runs asynchronously (decentralized)
        if self.use_lbp:
            self.lbp_fusion.update_belief_LBP(
                agent_id, fp_ij, observation, sigma0, sigma1
            )
        else:
            self.lbp_fusion.update_belief_OG(
                agent_id, fp_ij, observation, sigma0, sigma1
            )

        # Update news belief only (no fusion yet)
        self.lbp_fusion.update_news_belief_only(
            agent_id, fp_ij, observation, sigma0, sigma1
        )

    def fuse_agent_news(self, agent_id: int):
        """
        PHASE 2 (SYNCHRONOUS): Fuse agent's news with neighbors.

        Call this for ALL agents AFTER update_agent_news() has been
        called for ALL agents. This follows the paper's pattern.

        Args:
            agent_id: ID of agent doing the fusion
        """
        if not self.belief_fusion_enabled or self.lbp_fusion is None:
            return

        # Get neighbors within communication range
        neighbor_ids = self._get_neighbors_in_range(agent_id)

        if neighbor_ids:
            self.lbp_fusion.fuse_news_with_neighbors(agent_id, neighbor_ids)
            self._stats["belief_fusions"] += 1

    def update_all_news(self, agent_observations: Dict[int, Dict]):
        """
        PHASE 1 (BATCH SYNCHRONOUS): Update news beliefs for ALL agents.

        This is the batch version that updates all agents' news beliefs
        before any fusion occurs. Follows the paper's synchronous pattern.

        Args:
            agent_observations: Dict mapping agent_id to observation dict with:
                - 'fp_ij': Footprint indices
                - 'submap': Binary observation array
                - 'sigmas': Tuple (sigma0, sigma1) or None (uses defaults)
        """
        if not self.belief_fusion_enabled or self.lbp_fusion is None:
            return

        for agent_id, obs in agent_observations.items():
            sigmas = obs.get("sigmas")
            # Use default sigmas if not provided
            if sigmas is None:
                sigma0, sigma1 = 0.1, 0.1  # Default false positive/negative rates
            else:
                sigma0, sigma1 = sigmas

            fp_ij = obs.get("fp_ij")
            submap = obs.get("submap")
            if fp_ij is None or submap is None:
                continue

            # LBP inference runs asynchronously (decentralized per agent)
            if self.use_lbp:
                self.lbp_fusion.update_belief_LBP(
                    agent_id, fp_ij, submap, sigma0, sigma1
                )
            else:
                self.lbp_fusion.update_belief_OG(
                    agent_id, fp_ij, submap, sigma0, sigma1
                )

            # Update news belief only (no fusion yet)
            self.lbp_fusion.update_news_belief_only(
                agent_id, fp_ij, submap, sigma0, sigma1
            )

    def fuse_all_news(self):
        """
        PHASE 2 (BATCH SYNCHRONOUS): Fuse news with neighbors for ALL agents.

        Call this AFTER update_all_news() has completed.
        Follows the paper's synchronous update-then-fuse pattern.
        """
        if not self.belief_fusion_enabled or self.lbp_fusion is None:
            return

        for agent_id in range(self.num_agents):
            neighbor_ids = self._get_neighbors_in_range(agent_id)

            if neighbor_ids:
                self.lbp_fusion.fuse_news_with_neighbors(agent_id, neighbor_ids)
                self._stats["belief_fusions"] += 1

    def coordinate_synchronous_fusion(self, agent_observations: Dict[int, Dict]):
        """
        Convenience method to run complete synchronous fusion.

        Combines update_all_news() followed by fuse_all_news().

        Args:
            agent_observations: Dict mapping agent_id to observation dict
        """
        self.update_all_news(agent_observations)
        self.fuse_all_news()

    def _get_neighbors_in_range(self, agent_id: int) -> List[int]:
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

    def get_agent_belief(self, agent_id: int) -> Optional[np.ndarray]:
        """
        Get belief map for a specific agent.

        Args:
            agent_id: Agent ID

        Returns:
            Belief map or None if not available
        """
        if self.lbp_fusion is not None:
            return self.lbp_fusion.get_agent_belief(agent_id)
        return None

    def get_fused_belief(self) -> Optional[np.ndarray]:
        """Get the fused global belief map."""
        if self.lbp_fusion is not None:
            return self.lbp_fusion.get_fused_belief()
        if self.belief_fusion is not None:
            return self.belief_fusion.get_global_belief()
        return None

    def request_region(
        self,
        agent_id: int,
        region_scores: Dict[int, float],
        agent_position: Tuple[float, float],
        region_centers: Dict[int, Tuple[float, float]],
    ) -> Optional[int]:
        """
        Request a region assignment for an agent.

        Args:
            agent_id: ID of requesting agent
            region_scores: HLP scores for each region
            agent_position: Current agent position
            region_centers: Center positions of all regions

        Returns:
            Assigned region ID or None
        """
        # Initialize region allocator dynamically if needed
        if self.region_allocator is None and region_scores:
            num_regions = len(region_scores)
            self.initialize_region_allocator(num_regions)
            logger.info(f"Auto-initialized region allocator with {num_regions} regions")

        if self.region_allocator is None:
            # Still not initialized (no regions) - just return best-scoring region
            if region_scores:
                return max(region_scores, key=region_scores.get)
            return None

        assigned = self.region_allocator.request_region(
            agent_id=agent_id,
            region_scores=region_scores,
            agent_position=agent_position,
            region_centers=region_centers,
        )
        self._stats["region_allocations"] += 1

        # Broadcast region claim
        if self.enable_coordination and assigned is not None:
            msg = AgentMessage(
                msg_type=MessageType.REGION_CLAIM,
                sender_id=agent_id,
                data={"region_id": assigned},
            )
            self.comm_bus.broadcast(msg)

        return assigned

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
                            target_region=msg.data.get("target_region"),
                        )
                    else:
                        state = self._agent_states[msg.sender_id]
                        state.position = msg.data.get("position", state.position)
                        state.altitude = msg.data.get("altitude", state.altitude)
                        state.target_region = msg.data.get(
                            "target_region", state.target_region
                        )
                        state.last_update = time.time()

            elif msg.msg_type == MessageType.REGION_CLAIM:
                # Note: region allocator handles this internally
                # This is just for logging/visualization
                logger.debug(
                    f"Agent {msg.sender_id} claimed region {msg.data.get('region_id')}"
                )

        return messages

    def should_coordinate(self, agent_id: int) -> bool:
        """
        Check if coordination should run this step.

        Based on coordination_frequency setting.

        Args:
            agent_id: ID of agent

        Returns:
            True if coordination should run
        """
        self._step_counters[agent_id] = self._step_counters.get(agent_id, 0) + 1
        return self._step_counters[agent_id] % self.coordination_frequency == 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        stats = self._stats.copy()
        stats["comm_bus"] = self.comm_bus.get_stats()

        if self.lbp_fusion is not None:
            stats["lbp_fusion"] = self.lbp_fusion.get_stats()
            # Per-agent coverage
            stats["coverage_per_agent"] = {
                i: self.lbp_fusion.get_agent_coverage(i) for i in range(self.num_agents)
            }
        elif self.belief_fusion is not None:
            stats["coverage_per_agent"] = self.belief_fusion.get_coverage_per_agent()

        if self.region_allocator is not None:
            stats["region_assignments"] = self.region_allocator.get_assignments()

        return stats

    def get_assigned_regions(self) -> Dict[int, Optional[int]]:
        """Get current region assignments for all agents."""
        if self.region_allocator is None:
            return {}
        return self.region_allocator.get_assignments()


# =============================================================================
# Utility: Generate Start Positions for Multiple Agents
# =============================================================================


def generate_multi_agent_starts(
    num_agents: int,
    grid_info: Any,
    start_position: str = "corner",
    min_distance: float = 10.0,
) -> List[Tuple[float, float]]:
    """
    Generate starting positions for multiple agents.

    Spreads agents across the field to minimize initial overlap.

    Args:
        num_agents: Number of agents
        grid_info: Grid configuration object
        start_position: "corner", "edge", or "spread"
        min_distance: Minimum distance between start positions

    Returns:
        List of (x, y) start positions
    """
    positions = []

    if start_position == "corner":
        # Use corners, then edges, then interior
        corners = [
            (-grid_info.x / 2, -grid_info.y / 2),  # bottom-left
            (grid_info.x / 2, -grid_info.y / 2),  # bottom-right
            (-grid_info.x / 2, grid_info.y / 2),  # top-left
            (grid_info.x / 2, grid_info.y / 2),  # top-right
        ]
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
        x = np.random.uniform(-grid_info.x / 2, grid_info.x / 2)
        y = np.random.uniform(-grid_info.y / 2, grid_info.y / 2)

        # Check minimum distance from existing positions
        valid = True
        for pos in positions:
            if np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2) < min_distance:
                valid = False
                break

        if valid:
            positions.append((x, y))

    return positions[:num_agents]


# =============================================================================
# Integration: Hierarchical Dec-MCTS Planner Factory
# =============================================================================


def create_hierarchical_planners(
    num_agents: int,
    cameras: List[Any],
    grid_info: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, List[Any]]:
    """
    Factory function to create hierarchical Dec-MCTS planners for all agents.

    Creates:
    - A shared IntentBus for all agents
    - A HierarchicalDecMCTSPlanner for each agent

    Args:
        num_agents: Number of agents
        cameras: List of camera objects (one per agent)
        grid_info: Grid information
        config: Configuration dict with planner parameters

    Returns:
        (intent_bus, list_of_planners)
    """
    from hierarchical_dec_mcts import IntentBus, create_hierarchical_planner

    config = config or {}

    # Create shared intent bus
    intent_bus = IntentBus(num_agents=num_agents)

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

    logger.info(
        f"Created {num_agents} hierarchical Dec-MCTS planners with shared IntentBus"
    )

    return intent_bus, planners
