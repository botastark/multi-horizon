"""
Fully Decentralized Agent Module for Multi-UAV Active Sensing

This module implements fully decentralized agents where:
1. Each agent maintains its own local CRF-based terrain belief map B
2. Each agent maintains separate "news beliefs" δ_ij for each neighbor j
3. Agents exchange ONLY news beliefs (not full maps) when within communication range
4. Fusion uses renormalization to avoid double-counting observations
5. Agents broadcast positions to enable footprint overlap penalties

Key Design Principles:
- No central controller: all coordination via peer-to-peer message passing
- News beliefs ensure observations are fused exactly once
- Per-neighbor news beliefs (δ_ij) prevent re-fusion of stale information
- Position sharing enables non-redundant exploration without centralized planning

Reference: "Multi-Horizon Multi-Agent Planning Using Decentralised Monte Carlo Tree Search"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import threading
import queue
import time
import copy
import logging
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures for Decentralized Communication
# =============================================================================


class DecentralizedMessageType(Enum):
    """Types of messages exchanged between decentralized agents."""

    NEWS_BELIEF = "news_belief"  # Share news belief δ for fusion
    POSITION_BROADCAST = "position"  # Share current position + footprint
    INTENT = "intent"  # Share planned trajectory (backward compat)
    LLP_INTENT = "llp_intent"  # Share LLP short-horizon intent
    HLP_INTENT = "hlp_intent"  # Share HLP long-horizon intent
    HEARTBEAT = "heartbeat"  # Alive signal


@dataclass
class NewsBeliefMessage:
    """
    News belief message for sharing incremental belief updates.

    Contains only the NEW information collected since last communication,
    not the full belief map. This prevents double-counting.
    """

    sender_id: int
    receiver_id: int  # Specific receiver (for per-neighbor news)
    # News belief: log-odds of new observations (delta from uniform prior)
    # Using log-odds allows additive fusion: log_odds_fused = log_odds_A + log_odds_B
    news_log_odds: np.ndarray
    # Mask indicating which cells have news (sparse update)
    news_mask: np.ndarray
    # Timestamp and step for ordering
    timestamp: float = field(default_factory=time.time)
    step: int = 0
    # Last step this receiver acknowledged (for incremental updates)
    last_ack_step: int = 0


@dataclass
class PositionBroadcast:
    """
    Position broadcast for footprint overlap penalties.

    Agents share their current position and planned footprints so others
    can avoid redundant coverage.
    """

    sender_id: int
    position: Tuple[float, float]  # (x, y) world coordinates
    altitude: float
    # Current camera footprint bounds (row_min, row_max, col_min, col_max)
    footprint_bounds: Tuple[int, int, int, int]
    # Planned footprints for next N steps (for trajectory prediction)
    planned_footprints: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Planned positions for next N steps
    planned_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLPIntent:
    """
    Low-Level Planner (LLP) intent message for sharing short-horizon actions.

    Contains immediate action sequence and expected information gains.
    Used for local coordination and collision avoidance.
    """

    sender_id: int
    # Short-horizon action sequence (next few steps)
    action_sequence: List[str] = field(default_factory=list)
    # Predicted state sequence [(x, y, altitude), ...]
    state_sequence: List[Tuple[float, float, float]] = field(default_factory=list)
    # Predicted footprint sequence
    footprint_sequence: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Expected information gain at each step
    ig_sequence: List[float] = field(default_factory=list)
    # Planning horizon
    horizon: int = 5
    timestamp: float = field(default_factory=time.time)
    total_value: float = 0.0

    def is_stale(self, threshold_sec: float = 2.0) -> bool:
        """Check if intent is stale for D-UCT discounting."""
        return time.time() - self.timestamp > threshold_sec

    def staleness_discount(
        self, decay_factor: float = 0.9, threshold_sec: float = 2.0
    ) -> float:
        """
        Compute D-UCT discount factor based on intent age.

        Returns a value in (0, 1] where:
        - 1.0 = fresh intent (no discount)
        - <1.0 = stale intent (should reduce influence)
        """
        age = time.time() - self.timestamp
        if age <= threshold_sec:
            return 1.0
        # Exponential decay for stale intents
        staleness = (age - threshold_sec) / threshold_sec
        return decay_factor**staleness


@dataclass
class HLPIntent:
    """
    High-Level Planner (HLP) intent message for sharing long-horizon goals.

    Contains target region selection and long-term trajectory plans.
    Used for coverage coordination and avoiding region conflicts.
    """

    sender_id: int
    # Target region ID (from region decomposition)
    target_region: Optional[int] = None
    # Target region center coordinates
    target_center: Optional[Tuple[float, float]] = None
    # High-level waypoint sequence
    waypoint_sequence: List[Tuple[float, float]] = field(default_factory=list)
    # Expected coverage contribution per waypoint
    coverage_sequence: List[float] = field(default_factory=list)
    # Planning horizon (longer than LLP)
    horizon: int = 20
    # Priority score for region allocation
    priority: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def is_stale(self, threshold_sec: float = 5.0) -> bool:
        """Check if intent is stale (HLP has longer validity than LLP)."""
        return time.time() - self.timestamp > threshold_sec

    def staleness_discount(
        self, decay_factor: float = 0.95, threshold_sec: float = 5.0
    ) -> float:
        """
        Compute D-UCT discount factor based on intent age.

        HLP intents decay more slowly than LLP intents.
        """
        age = time.time() - self.timestamp
        if age <= threshold_sec:
            return 1.0
        staleness = (age - threshold_sec) / threshold_sec
        return decay_factor**staleness


@dataclass
class DecentralizedIntent:
    """
    Combined intent message for sharing planned actions (backward compatibility).

    More detailed than PositionBroadcast, includes full trajectory info.
    """

    sender_id: int
    # Planned action sequence
    action_sequence: List[str] = field(default_factory=list)
    # Predicted state sequence [(x, y, altitude), ...]
    state_sequence: List[Tuple[float, float, float]] = field(default_factory=list)
    # Predicted footprint sequence
    footprint_sequence: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Expected information gain at each step
    ig_sequence: List[float] = field(default_factory=list)
    # Target region (from HLP)
    target_region: Optional[int] = None
    target_center: Optional[Tuple[float, float]] = None
    timestamp: float = field(default_factory=time.time)
    value: float = 0.0


# =============================================================================
# Local Belief Manager - CRF-based Belief with News Tracking
# =============================================================================


class LocalBeliefManager:
    """
    Manages local CRF-based belief map with news tracking for decentralized fusion.

    Each agent maintains:
    1. Local belief B: Full probabilistic map incorporating all fused information
    2. News beliefs δ_ij for each neighbor j: Incremental updates since last share

    Fusion follows renormalization to prevent double-counting:
        P(m|z_A, z_B) = [P(m|z_A) × P(m|z_B) / P(m)]

    Using log-odds representation for numerical stability:
        log_odds(fused) = log_odds(local) + log_odds(news) - log_odds(prior)
    """

    def __init__(
        self,
        agent_id: int,
        grid_shape: Tuple[int, int],
        neighbor_ids: List[int],
        use_lbp: bool = True,
        lbp_iterations: int = 1,
        prior: float = 0.5,
    ):
        """
        Initialize local belief manager.

        Args:
            agent_id: This agent's ID
            grid_shape: (H, W) shape of belief map
            neighbor_ids: IDs of potential neighbors
            use_lbp: Whether to use LBP for spatial consistency
            lbp_iterations: Number of LBP iterations
            prior: Prior probability (default 0.5 = maximum entropy)
        """
        self.agent_id = agent_id
        self.grid_shape = grid_shape
        self.neighbor_ids = list(neighbor_ids)
        self.use_lbp = use_lbp
        self.lbp_iterations = lbp_iterations
        self.prior = prior

        # Prior log-odds (for 0.5: log_odds = 0)
        self.prior_log_odds = self._prob_to_log_odds(prior)

        # Local belief map B (probability representation)
        self.belief = np.full(grid_shape, prior, dtype=np.float64)

        # Local belief in log-odds form (for fusion)
        self.belief_log_odds = np.full(
            grid_shape, self.prior_log_odds, dtype=np.float64
        )

        # Per-neighbor news beliefs δ_ij: what we've observed since last sharing with j
        # news_beliefs[neighbor_id] = log-odds array
        self.news_beliefs: Dict[int, np.ndarray] = {
            j: np.zeros(grid_shape, dtype=np.float64) for j in neighbor_ids
        }

        # Track which cells have been updated in news for each neighbor
        self.news_masks: Dict[int, np.ndarray] = {
            j: np.zeros(grid_shape, dtype=bool) for j in neighbor_ids
        }

        # Last communication step with each neighbor
        self.last_comm_step: Dict[int, int] = {j: 0 for j in neighbor_ids}

        # Current step counter
        self.current_step: int = 0

        # LBP messages (for spatial consistency)
        if use_lbp:
            # Messages: 4 directions (up, right, down, left) + local evidence
            self.msgs = np.ones((5, *grid_shape), dtype=np.float64) * 0.5
            self.msgs_buffer = np.ones_like(self.msgs) * 0.5
            # Pairwise potential (spatial correlation)
            self.pairwise_potential = np.array(
                [[0.6, 0.4], [0.4, 0.6]], dtype=np.float64
            )

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "observations": 0,
            "lbp_runs": 0,
            "news_fusions": 0,
            "news_resets": 0,
        }

        logger.debug(
            f"Agent {agent_id}: LocalBeliefManager initialized with neighbors {neighbor_ids}"
        )

    @staticmethod
    def _prob_to_log_odds(p: np.ndarray) -> np.ndarray:
        """Convert probability to log-odds."""
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.log(p / (1 - p))

    @staticmethod
    def _log_odds_to_prob(log_odds: np.ndarray) -> np.ndarray:
        """Convert log-odds to probability."""
        # Clip to prevent overflow
        log_odds = np.clip(log_odds, -20, 20)
        return 1 / (1 + np.exp(-log_odds))

    def update_observation(
        self,
        fp_ij: Dict[str, Tuple[int, int]],
        observation: np.ndarray,
        sigma0: float,
        sigma1: float,
    ) -> None:
        """
        Update local belief with a new observation.

        Also updates news beliefs for all neighbors (they need this info).

        Args:
            fp_ij: Footprint indices dict with keys 'ul', 'bl', 'ur', 'br'
            observation: Binary observation (0=free, 1=occupied)
            sigma0: P(observe 1 | true state is 0) - false positive rate
            sigma1: P(observe 0 | true state is 1) - false negative rate
        """
        I, J = 0, 1

        with self._lock:
            # Extract region bounds
            imin, imax = fp_ij["ul"][I], fp_ij["bl"][I]
            jmin, jmax = fp_ij["ul"][J], fp_ij["ur"][J]

            # Compute observation likelihood in log-odds form
            # P(z|m=0) = (1-sigma0) if z=0 else sigma0
            # P(z|m=1) = sigma1 if z=0 else (1-sigma1)
            likelihood_m0 = np.where(observation == 0, 1 - sigma0, sigma0)
            likelihood_m1 = np.where(observation == 0, sigma1, 1 - sigma1)

            # Log-odds of observation: log(P(z|m=1) / P(z|m=0))
            epsilon = 1e-20
            observation_log_odds = np.log(
                (likelihood_m1 + epsilon) / (likelihood_m0 + epsilon)
            )

            # Update local belief log-odds (Bayesian update in log-odds = addition)
            self.belief_log_odds[imin:imax, jmin:jmax] += observation_log_odds

            # Clip for numerical stability
            self.belief_log_odds = np.clip(self.belief_log_odds, -20, 20)

            # Convert to probability for LBP and external use
            self.belief = self._log_odds_to_prob(self.belief_log_odds)

            # Update news beliefs for ALL neighbors (they need this observation)
            for neighbor_id in self.neighbor_ids:
                self.news_beliefs[neighbor_id][
                    imin:imax, jmin:jmax
                ] += observation_log_odds
                self.news_masks[neighbor_id][imin:imax, jmin:jmax] = True

            self.current_step += 1
            self._stats["observations"] += 1

            # Run LBP for spatial consistency if enabled
            if self.use_lbp:
                self._run_lbp(fp_ij)

    def _run_lbp(self, fp_ij: Dict[str, Tuple[int, int]]) -> None:
        """Run LBP for spatial consistency (local inference)."""
        I, J = 0, 1
        imin, imax = fp_ij["ul"][I], fp_ij["bl"][I]
        jmin, jmax = fp_ij["ul"][J], fp_ij["ur"][J]

        # Inject current beliefs as local evidence
        self.msgs[4, :, :] = self.belief

        for _ in range(self.lbp_iterations):
            # Simplified LBP: pass messages in 4 directions
            psi = self.pairwise_potential

            # For each direction, compute message
            # Message from i to j: m_ij(x_j) = sum_x_i [psi(x_i, x_j) * product of incoming messages to i except from j * local evidence at i]

            # Here we use a simplified version that updates the footprint region
            mul_0 = np.prod(1 - self.msgs[:4, imin:imax, jmin:jmax], axis=0)
            mul_1 = np.prod(self.msgs[:4, imin:imax, jmin:jmax], axis=0)

            # Include local evidence
            mul_0 *= 1 - self.msgs[4, imin:imax, jmin:jmax]
            mul_1 *= self.msgs[4, imin:imax, jmin:jmax]

            # Apply pairwise potential
            msg_0 = psi[0, 0] * mul_0 + psi[0, 1] * mul_1
            msg_1 = psi[1, 0] * mul_0 + psi[1, 1] * mul_1

            # Normalize
            norm = msg_0 + msg_1 + 1e-20
            new_belief = msg_1 / norm

            # Update beliefs in the footprint region
            self.belief[imin:imax, jmin:jmax] = np.clip(new_belief, 0.001, 0.999)

        # Sync log-odds with updated belief
        self.belief_log_odds = self._prob_to_log_odds(self.belief)
        self._stats["lbp_runs"] += 1

    def get_news_for_neighbor(self, neighbor_id: int) -> Optional[NewsBeliefMessage]:
        """
        Get news belief message to send to a specific neighbor.

        Returns None if no news to share.

        Args:
            neighbor_id: ID of neighbor to send news to

        Returns:
            NewsBeliefMessage or None
        """
        with self._lock:
            if neighbor_id not in self.news_beliefs:
                return None

            news_mask = self.news_masks[neighbor_id]

            # Only send if there's actually news
            if not np.any(news_mask):
                return None

            return NewsBeliefMessage(
                sender_id=self.agent_id,
                receiver_id=neighbor_id,
                news_log_odds=self.news_beliefs[neighbor_id].copy(),
                news_mask=news_mask.copy(),
                step=self.current_step,
                last_ack_step=self.last_comm_step[neighbor_id],
            )

    def reset_news_for_neighbor(self, neighbor_id: int, ack_step: int) -> None:
        """
        Reset news belief for a neighbor after successful sharing.

        Args:
            neighbor_id: ID of neighbor
            ack_step: Step number acknowledged by neighbor
        """
        with self._lock:
            if neighbor_id in self.news_beliefs:
                self.news_beliefs[neighbor_id][:] = 0.0
                self.news_masks[neighbor_id][:] = False
                self.last_comm_step[neighbor_id] = ack_step
                self._stats["news_resets"] += 1

    def fuse_received_news(self, news_msg: NewsBeliefMessage) -> None:
        """
        Fuse received news belief from another agent.

        Uses renormalization to prevent double-counting:
        log_odds(fused) = log_odds(local) + log_odds(news)

        This works because news contains only NEW information not yet in local belief.

        Args:
            news_msg: NewsBeliefMessage from another agent
        """
        with self._lock:
            sender_id = news_msg.sender_id

            # Only fuse cells that have news
            mask = news_msg.news_mask

            if not np.any(mask):
                return

            # Additive fusion in log-odds space
            # This is correct because:
            # - Local belief contains all previously fused information
            # - News contains only observations since last communication
            # - No overlap = no double counting
            self.belief_log_odds[mask] += news_msg.news_log_odds[mask]

            # Clip for stability
            self.belief_log_odds = np.clip(self.belief_log_odds, -20, 20)

            # Update probability representation
            self.belief = self._log_odds_to_prob(self.belief_log_odds)

            self._stats["news_fusions"] += 1

            logger.debug(
                f"Agent {self.agent_id}: Fused news from agent {sender_id}, "
                f"{np.sum(mask)} cells updated"
            )

    def get_belief(self) -> np.ndarray:
        """Get current belief map (probability form)."""
        with self._lock:
            return self.belief.copy()

    def get_entropy_map(self) -> np.ndarray:
        """Get entropy map for information gain calculation."""
        with self._lock:
            p = np.clip(self.belief, 1e-10, 1 - 1e-10)
            return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    def add_neighbor(self, neighbor_id: int) -> None:
        """Add a new neighbor to track news for."""
        with self._lock:
            if neighbor_id not in self.news_beliefs:
                self.neighbor_ids.append(neighbor_id)
                self.news_beliefs[neighbor_id] = np.zeros(
                    self.grid_shape, dtype=np.float64
                )
                self.news_masks[neighbor_id] = np.zeros(self.grid_shape, dtype=bool)
                self.last_comm_step[neighbor_id] = 0

    def remove_neighbor(self, neighbor_id: int) -> None:
        """Remove a neighbor (e.g., out of communication range)."""
        with self._lock:
            if neighbor_id in self.news_beliefs:
                self.neighbor_ids.remove(neighbor_id)
                del self.news_beliefs[neighbor_id]
                del self.news_masks[neighbor_id]
                del self.last_comm_step[neighbor_id]

    def get_statistics(self) -> Dict[str, Any]:
        """Get belief manager statistics."""
        with self._lock:
            return {
                **self._stats,
                "current_step": self.current_step,
                "num_neighbors": len(self.neighbor_ids),
                "mean_belief": float(np.mean(self.belief)),
                "belief_entropy": float(np.mean(self.get_entropy_map())),
            }


# =============================================================================
# Decentralized Agent - Full Agent Implementation
# =============================================================================


class DecentralizedAgent:
    """
    Fully decentralized UAV agent for multi-agent active sensing.

    Each agent:
    1. Maintains its own local CRF-based terrain belief map B
    2. Maintains separate news beliefs δ_ij for each neighbor j
    3. Exchanges only news beliefs when within communication range (configurable)
    4. Broadcasts position for footprint overlap penalties (configurable)
    5. Shares LLP intent for short-horizon coordination (configurable)
    6. Shares HLP intent for long-horizon coverage planning (configurable)

    No central controller is required - all coordination is peer-to-peer.

    Configuration options (in config dict or 'decentralized' section):
        enable_belief_fusion: bool - Enable news belief sharing and fusion (default: True)
        enable_llp_intent_sharing: bool - Enable LLP intent sharing (default: True)
        enable_hlp_intent_sharing: bool - Enable HLP intent sharing (default: True)
        enable_position_sharing: bool - Enable position broadcast (default: True)
    """

    def __init__(
        self,
        agent_id: int,
        num_agents: int,
        camera: Any,
        grid_info: Any,
        communication_range: float = -1,  # -1 = unlimited
        use_lbp: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize decentralized agent.

        Args:
            agent_id: Unique agent ID
            num_agents: Total number of agents in system
            camera: UAV camera model
            grid_info: Grid configuration
            communication_range: Max distance for communication (-1 = unlimited)
            use_lbp: Whether to use LBP for spatial consistency
            config: Additional configuration options including:
                - enable_belief_fusion: Enable news belief sharing (default: True)
                - enable_llp_intent_sharing: Enable LLP intent sharing (default: True)
                - enable_hlp_intent_sharing: Enable HLP intent sharing (default: True)
                - enable_position_sharing: Enable position broadcast (default: True)
        """
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.camera = camera
        self.grid_info = grid_info
        self.communication_range = communication_range
        self.config = config or {}

        # Get decentralized-specific config (nested or flat)
        dec_config = self.config.get("decentralized", self.config)

        # Feature enable flags
        self.enable_belief_fusion = dec_config.get("enable_belief_fusion", True)
        self.enable_llp_intent_sharing = dec_config.get(
            "enable_llp_intent_sharing", True
        )
        self.enable_hlp_intent_sharing = dec_config.get(
            "enable_hlp_intent_sharing", True
        )
        self.enable_position_sharing = dec_config.get("enable_position_sharing", True)

        # Current state
        self.position: Tuple[float, float] = (0.0, 0.0)
        self.altitude: float = 0.0
        self.current_footprint: Optional[Tuple[int, int, int, int]] = None

        # All other agent IDs (potential neighbors)
        neighbor_ids = [i for i in range(num_agents) if i != agent_id]

        # Initialize local belief manager with per-neighbor news tracking
        self.belief_manager = LocalBeliefManager(
            agent_id=agent_id,
            grid_shape=grid_info.shape,
            neighbor_ids=neighbor_ids,
            use_lbp=use_lbp,
            lbp_iterations=self.config.get("lbp_iterations", 1),
        )

        # Received position broadcasts from other agents
        self.neighbor_positions: Dict[int, PositionBroadcast] = {}

        # Received intents from other agents (separated by type)
        self.neighbor_intents: Dict[int, DecentralizedIntent] = {}  # backward compat
        self.neighbor_llp_intents: Dict[int, LLPIntent] = {}
        self.neighbor_hlp_intents: Dict[int, HLPIntent] = {}

        # Outgoing message queue (for async communication)
        self.outbox: queue.Queue = queue.Queue(
            maxsize=dec_config.get("message_queue_size", 100)
        )

        # Inbox for received messages
        self.inbox: queue.Queue = queue.Queue(
            maxsize=dec_config.get("message_queue_size", 100)
        )

        # Current intents (our planned actions)
        self.current_intent: Optional[DecentralizedIntent] = None  # backward compat
        self.current_llp_intent: Optional[LLPIntent] = None
        self.current_hlp_intent: Optional[HLPIntent] = None

        # Thread safety
        self._lock = threading.RLock()

        # Configuration
        self.overlap_penalty_weight = dec_config.get("overlap_penalty_weight", 0.5)
        self.intent_horizon = dec_config.get("intent_horizon", 5)
        self.stale_message_threshold = dec_config.get("stale_message_threshold", 5.0)

        # Statistics
        self._stats = {
            "observations": 0,
            "news_sent": 0,
            "news_received": 0,
            "positions_broadcast": 0,
            "intents_shared": 0,
            "llp_intents_shared": 0,
            "hlp_intents_shared": 0,
        }

        logger.info(
            f"DecentralizedAgent {agent_id} initialized: "
            f"belief_fusion={self.enable_belief_fusion}, "
            f"llp_intent={self.enable_llp_intent_sharing}, "
            f"hlp_intent={self.enable_hlp_intent_sharing}, "
            f"position_sharing={self.enable_position_sharing}"
        )

    def update_position(
        self,
        position: Tuple[float, float],
        altitude: float,
    ) -> None:
        """
        Update agent's current position.

        Args:
            position: (x, y) world coordinates
            altitude: Current altitude
        """
        with self._lock:
            self.position = position
            self.altitude = altitude

            # Update footprint
            try:
                [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                    position=position,
                    altitude=altitude,
                    index_form=True,
                )
                self.current_footprint = (imin, imax, jmin, jmax)
            except Exception:
                self.current_footprint = None

    def observe_and_update(
        self,
        observation: np.ndarray,
        fp_ij: Dict[str, Tuple[int, int]],
        sigma0: float = 0.1,
        sigma1: float = 0.1,
    ) -> None:
        """
        Process a new observation and update local belief.

        Also prepares news beliefs for sharing with neighbors.

        Args:
            observation: Binary observation array
            fp_ij: Footprint indices
            sigma0: False positive rate
            sigma1: False negative rate
        """
        with self._lock:
            self.belief_manager.update_observation(fp_ij, observation, sigma0, sigma1)
            self._stats["observations"] += 1

    def get_neighbors_in_range(self) -> List[int]:
        """
        Get IDs of agents within communication range.

        Returns:
            List of neighbor agent IDs currently in range
        """
        if self.communication_range < 0:
            # Unlimited range
            return [i for i in range(self.num_agents) if i != self.agent_id]

        neighbors = []
        current_time = time.time()

        with self._lock:
            for neighbor_id, pos_msg in self.neighbor_positions.items():
                # Skip stale positions
                if current_time - pos_msg.timestamp > self.stale_message_threshold:
                    continue

                # Check distance
                dist = np.sqrt(
                    (self.position[0] - pos_msg.position[0]) ** 2
                    + (self.position[1] - pos_msg.position[1]) ** 2
                )

                if dist <= self.communication_range:
                    neighbors.append(neighbor_id)

        return neighbors

    def create_position_broadcast(self) -> PositionBroadcast:
        """
        Create a position broadcast message.

        Includes current position and planned trajectory for overlap avoidance.

        Returns:
            PositionBroadcast message
        """
        with self._lock:
            planned_footprints = []
            planned_positions = []

            # Include planned trajectory if we have an intent
            if self.current_intent is not None:
                planned_footprints = self.current_intent.footprint_sequence[
                    : self.intent_horizon
                ]
                planned_positions = self.current_intent.state_sequence[
                    : self.intent_horizon
                ]

            return PositionBroadcast(
                sender_id=self.agent_id,
                position=self.position,
                altitude=self.altitude,
                footprint_bounds=self.current_footprint or (0, 0, 0, 0),
                planned_footprints=planned_footprints,
                planned_positions=planned_positions,
            )

    def broadcast_position(self) -> None:
        """Broadcast current position to all agents (if enabled)."""
        if not self.enable_position_sharing:
            return

        msg = self.create_position_broadcast()

        try:
            self.outbox.put_nowait(("position", msg))
            self._stats["positions_broadcast"] += 1
        except queue.Full:
            logger.warning(
                f"Agent {self.agent_id}: Position broadcast dropped (queue full)"
            )

    def share_news_with_neighbors(self) -> int:
        """
        Share news beliefs with all neighbors in communication range (if enabled).

        For each neighbor, sends only the news (new observations since last share).

        Returns:
            Number of news messages sent
        """
        if not self.enable_belief_fusion:
            return 0

        neighbors = self.get_neighbors_in_range()
        sent_count = 0

        with self._lock:
            for neighbor_id in neighbors:
                news_msg = self.belief_manager.get_news_for_neighbor(neighbor_id)

                if news_msg is not None:
                    try:
                        self.outbox.put_nowait(("news", news_msg))
                        sent_count += 1
                        self._stats["news_sent"] += 1
                    except queue.Full:
                        logger.warning(
                            f"Agent {self.agent_id}: News to {neighbor_id} dropped"
                        )

        return sent_count

    def receive_position(self, pos_msg: PositionBroadcast) -> None:
        """
        Process received position broadcast from another agent.

        Args:
            pos_msg: Position broadcast from another agent
        """
        with self._lock:
            self.neighbor_positions[pos_msg.sender_id] = pos_msg

    def receive_news(self, news_msg: NewsBeliefMessage) -> None:
        """
        Process received news belief from another agent (if belief fusion enabled).

        Fuses the news into local belief and acknowledges receipt.

        Args:
            news_msg: News belief message
        """
        # Only process if we're the intended receiver
        if news_msg.receiver_id != self.agent_id:
            return

        # Skip fusion if disabled
        if not self.enable_belief_fusion:
            return

        with self._lock:
            # Fuse received news into local belief
            self.belief_manager.fuse_received_news(news_msg)
            self._stats["news_received"] += 1

            # The sender will reset their news for us after we acknowledge
            # In this implementation, we assume reliable delivery

    def receive_intent(self, intent_msg: DecentralizedIntent) -> None:
        """
        Process received intent from another agent (backward compatibility).

        Args:
            intent_msg: Intent message
        """
        with self._lock:
            self.neighbor_intents[intent_msg.sender_id] = intent_msg
            self._stats["intents_shared"] += 1

    def receive_llp_intent(self, intent_msg: LLPIntent) -> None:
        """
        Process received LLP (short-horizon) intent from another agent.

        Args:
            intent_msg: LLP intent message
        """
        if not self.enable_llp_intent_sharing:
            return

        with self._lock:
            self.neighbor_llp_intents[intent_msg.sender_id] = intent_msg
            self._stats["llp_intents_shared"] += 1

    def receive_hlp_intent(self, intent_msg: HLPIntent) -> None:
        """
        Process received HLP (long-horizon) intent from another agent.

        Args:
            intent_msg: HLP intent message
        """
        if not self.enable_hlp_intent_sharing:
            return

        with self._lock:
            self.neighbor_hlp_intents[intent_msg.sender_id] = intent_msg
            self._stats["hlp_intents_shared"] += 1

    def process_inbox(self) -> int:
        """
        Process all messages in inbox.

        Returns:
            Number of messages processed
        """
        processed = 0

        while True:
            try:
                msg_type, msg = self.inbox.get_nowait()

                if msg_type == "position":
                    self.receive_position(msg)
                elif msg_type == "news":
                    self.receive_news(msg)
                elif msg_type == "intent":
                    self.receive_intent(msg)
                elif msg_type == "llp_intent":
                    self.receive_llp_intent(msg)
                elif msg_type == "hlp_intent":
                    self.receive_hlp_intent(msg)

                processed += 1
            except queue.Empty:
                break

        return processed

    def compute_footprint_overlap_penalty(
        self,
        proposed_position: Tuple[float, float],
        proposed_altitude: float,
    ) -> float:
        """
        Compute penalty for overlapping with other agents' footprints.

        This encourages non-redundant exploration by penalizing positions
        that would observe cells other agents are planning to observe.

        Args:
            proposed_position: (x, y) proposed position
            proposed_altitude: Proposed altitude

        Returns:
            Overlap penalty (0 = no overlap, 1 = complete overlap)
        """
        # Get proposed footprint
        try:
            [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                position=proposed_position,
                altitude=proposed_altitude,
                index_form=True,
            )
        except Exception:
            return 0.0

        H, W = self.grid_info.shape
        imin = max(0, min(imin, H))
        imax = max(0, min(imax, H))
        jmin = max(0, min(jmin, W))
        jmax = max(0, min(jmax, W))

        if imax <= imin or jmax <= jmin:
            return 0.0

        our_area = (imax - imin) * (jmax - jmin)
        if our_area == 0:
            return 0.0

        total_overlap = 0.0
        current_time = time.time()

        with self._lock:
            for neighbor_id, pos_msg in self.neighbor_positions.items():
                # Skip stale positions
                if current_time - pos_msg.timestamp > self.stale_message_threshold:
                    continue

                # Check current footprint overlap
                n_fp = pos_msg.footprint_bounds
                overlap = self._compute_rect_overlap((imin, imax, jmin, jmax), n_fp)
                total_overlap += overlap

                # Check planned footprint overlaps (with time discount)
                for step_idx, planned_fp in enumerate(pos_msg.planned_footprints):
                    discount = 0.8**step_idx  # Discount future overlaps
                    overlap = self._compute_rect_overlap(
                        (imin, imax, jmin, jmax), planned_fp
                    )
                    total_overlap += overlap * discount

        # Normalize by our footprint area
        penalty = min(1.0, total_overlap / our_area)

        return penalty * self.overlap_penalty_weight

    @staticmethod
    def _compute_rect_overlap(
        rect1: Tuple[int, int, int, int],
        rect2: Tuple[int, int, int, int],
    ) -> float:
        """Compute overlap area between two rectangles."""
        imin1, imax1, jmin1, jmax1 = rect1
        imin2, imax2, jmin2, jmax2 = rect2

        # Intersection
        imin = max(imin1, imin2)
        imax = min(imax1, imax2)
        jmin = max(jmin1, jmin2)
        jmax = min(jmax1, jmax2)

        if imin < imax and jmin < jmax:
            return float((imax - imin) * (jmax - jmin))
        return 0.0

    def compute_information_gain(
        self,
        position: Tuple[float, float],
        altitude: float,
    ) -> float:
        """
        Compute expected information gain at a position.

        IG = sum of entropy in footprint, adjusted by overlap penalty.

        Args:
            position: (x, y) position
            altitude: Altitude

        Returns:
            Information gain value
        """
        try:
            [[imin, imax], [jmin, jmax]] = self.camera.get_range(
                position=position,
                altitude=altitude,
                index_form=True,
            )
        except Exception:
            return 0.0

        H, W = self.grid_info.shape
        imin = max(0, min(imin, H))
        imax = max(0, min(imax, H))
        jmin = max(0, min(jmin, W))
        jmax = max(0, min(jmax, W))

        if imax <= imin or jmax <= jmin:
            return 0.0

        # Get entropy in footprint
        entropy_map = self.belief_manager.get_entropy_map()
        entropy_sum = np.sum(entropy_map[imin:imax, jmin:jmax])

        # Apply overlap penalty
        overlap_penalty = self.compute_footprint_overlap_penalty(position, altitude)
        adjusted_ig = entropy_sum * (1 - overlap_penalty)

        return float(adjusted_ig)

    def update_intent(self, intent: DecentralizedIntent) -> None:
        """
        Update agent's current intent (backward compatibility).

        Args:
            intent: New intent from planner
        """
        with self._lock:
            self.current_intent = intent

    def update_llp_intent(self, intent: LLPIntent) -> None:
        """
        Update agent's current LLP (short-horizon) intent.

        Args:
            intent: New LLP intent from planner
        """
        with self._lock:
            self.current_llp_intent = intent

    def update_hlp_intent(self, intent: HLPIntent) -> None:
        """
        Update agent's current HLP (long-horizon) intent.

        Args:
            intent: New HLP intent from planner
        """
        with self._lock:
            self.current_hlp_intent = intent

    def share_intent(self) -> None:
        """Share current intent with neighbors (backward compatibility)."""
        if self.current_intent is None:
            return

        try:
            self.outbox.put_nowait(("intent", self.current_intent))
        except queue.Full:
            logger.warning(f"Agent {self.agent_id}: Intent share dropped")

    def share_llp_intent(self) -> None:
        """Share current LLP intent with neighbors (if enabled)."""
        if not self.enable_llp_intent_sharing:
            return

        if self.current_llp_intent is None:
            return

        try:
            self.outbox.put_nowait(("llp_intent", self.current_llp_intent))
            self._stats["llp_intents_shared"] += 1
        except queue.Full:
            logger.warning(f"Agent {self.agent_id}: LLP intent share dropped")

    def share_hlp_intent(self) -> None:
        """Share current HLP intent with neighbors (if enabled)."""
        if not self.enable_hlp_intent_sharing:
            return

        if self.current_hlp_intent is None:
            return

        try:
            self.outbox.put_nowait(("hlp_intent", self.current_hlp_intent))
            self._stats["hlp_intents_shared"] += 1
        except queue.Full:
            logger.warning(f"Agent {self.agent_id}: HLP intent share dropped")

    def share_all_intents(self) -> None:
        """Share both LLP and HLP intents (convenience method)."""
        self.share_llp_intent()
        self.share_hlp_intent()

    def get_belief(self) -> np.ndarray:
        """Get current belief map."""
        return self.belief_manager.get_belief()

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "position": self.position,
            "altitude": self.altitude,
            "config": {
                "enable_belief_fusion": self.enable_belief_fusion,
                "enable_llp_intent_sharing": self.enable_llp_intent_sharing,
                "enable_hlp_intent_sharing": self.enable_hlp_intent_sharing,
                "enable_position_sharing": self.enable_position_sharing,
            },
            **self._stats,
            "belief_manager": self.belief_manager.get_statistics(),
            "neighbors_in_range": len(self.get_neighbors_in_range()),
            "neighbor_llp_intents": len(self.neighbor_llp_intents),
            "neighbor_hlp_intents": len(self.neighbor_hlp_intents),
        }

    def get_neighbor_llp_intents(self) -> Dict[int, LLPIntent]:
        """Get all received LLP intents from neighbors."""
        with self._lock:
            return dict(self.neighbor_llp_intents)

    def get_neighbor_hlp_intents(self) -> Dict[int, HLPIntent]:
        """Get all received HLP intents from neighbors."""
        with self._lock:
            return dict(self.neighbor_hlp_intents)


# =============================================================================
# Communication Network - Simulates Peer-to-Peer Communication
# =============================================================================


class DecentralizedCommNetwork:
    """
    Simulated peer-to-peer communication network for decentralized agents.

    In a real deployment, this would be replaced by actual network protocols.
    This simulation:
    - Handles message routing between agents
    - Simulates communication range limits
    - Provides broadcast and unicast messaging
    """

    def __init__(
        self,
        agents: List[DecentralizedAgent],
        message_delay: float = 0.0,
        drop_probability: float = 0.0,
    ):
        """
        Initialize communication network.

        Args:
            agents: List of decentralized agents
            message_delay: Simulated message delay (seconds)
            drop_probability: Probability of dropping a message (0-1)
        """
        self.agents = {a.agent_id: a for a in agents}
        self.message_delay = message_delay
        self.drop_probability = drop_probability

        # Statistics
        self._stats = {
            "messages_routed": 0,
            "messages_dropped": 0,
            "position_broadcasts": 0,
            "news_messages": 0,
            "intent_messages": 0,
            "llp_intent_messages": 0,
            "hlp_intent_messages": 0,
        }

    def route_messages(self) -> int:
        """
        Route all pending messages between agents.

        Processes each agent's outbox and delivers to appropriate inboxes.

        Returns:
            Total messages routed
        """
        routed = 0

        for agent_id, agent in self.agents.items():
            while True:
                try:
                    msg_type, msg = agent.outbox.get_nowait()

                    # Simulate message drop
                    if np.random.random() < self.drop_probability:
                        self._stats["messages_dropped"] += 1
                        continue

                    # Route based on message type
                    if msg_type == "position":
                        # Broadcast to all other agents
                        for other_id, other_agent in self.agents.items():
                            if other_id != agent_id:
                                self._deliver(other_agent, msg_type, msg)
                        self._stats["position_broadcasts"] += 1

                    elif msg_type == "news":
                        # Unicast to specific receiver
                        receiver_id = msg.receiver_id
                        if receiver_id in self.agents:
                            self._deliver(self.agents[receiver_id], msg_type, msg)
                            # Acknowledge and reset sender's news for this receiver
                            agent.belief_manager.reset_news_for_neighbor(
                                receiver_id, msg.step
                            )
                        self._stats["news_messages"] += 1

                    elif msg_type == "intent":
                        # Broadcast to all other agents (backward compat)
                        for other_id, other_agent in self.agents.items():
                            if other_id != agent_id:
                                self._deliver(other_agent, msg_type, msg)
                        self._stats["intent_messages"] += 1

                    elif msg_type == "llp_intent":
                        # Broadcast LLP intent to all other agents
                        for other_id, other_agent in self.agents.items():
                            if other_id != agent_id:
                                self._deliver(other_agent, msg_type, msg)
                        self._stats["llp_intent_messages"] += 1

                    elif msg_type == "hlp_intent":
                        # Broadcast HLP intent to all other agents
                        for other_id, other_agent in self.agents.items():
                            if other_id != agent_id:
                                self._deliver(other_agent, msg_type, msg)
                        self._stats["hlp_intent_messages"] += 1

                    routed += 1
                    self._stats["messages_routed"] += 1

                except queue.Empty:
                    break

        return routed

    def _deliver(
        self,
        agent: DecentralizedAgent,
        msg_type: str,
        msg: Any,
    ) -> bool:
        """Deliver a message to an agent's inbox."""
        try:
            agent.inbox.put_nowait((msg_type, msg))
            return True
        except queue.Full:
            self._stats["messages_dropped"] += 1
            return False

    def step(self) -> Dict[str, int]:
        """
        Execute one communication step.

        1. Route all outgoing messages
        2. Process all incoming messages for each agent

        Returns:
            Statistics for this step
        """
        # Route messages
        routed = self.route_messages()

        # Process inboxes
        processed = 0
        for agent in self.agents.values():
            processed += agent.process_inbox()

        return {
            "messages_routed": routed,
            "messages_processed": processed,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        return dict(self._stats)


# =============================================================================
# Factory Functions
# =============================================================================


def create_decentralized_agents(
    num_agents: int,
    cameras: List[Any],
    grid_info: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[DecentralizedAgent], DecentralizedCommNetwork]:
    """
    Factory function to create decentralized agents and their communication network.

    Args:
        num_agents: Number of agents to create
        cameras: List of camera objects (one per agent)
        grid_info: Grid configuration
        config: Configuration options including:
            - enable_belief_fusion: Enable news belief sharing (default: True)
            - enable_llp_intent_sharing: Enable LLP intent sharing (default: True)
            - enable_hlp_intent_sharing: Enable HLP intent sharing (default: True)
            - enable_position_sharing: Enable position broadcast (default: True)

    Returns:
        (list of agents, communication network)
    """
    config = config or {}

    # Get decentralized-specific config (nested or flat)
    dec_config = config.get("decentralized", config)

    # Extract config
    communication_range = config.get("communication_range", -1)
    use_lbp = config.get("use_lbp", True)
    message_delay = config.get("message_delay", 0.0)
    drop_probability = config.get("drop_probability", 0.0)

    # Create agents
    agents = []
    for agent_id in range(num_agents):
        agent = DecentralizedAgent(
            agent_id=agent_id,
            num_agents=num_agents,
            camera=cameras[agent_id],
            grid_info=grid_info,
            communication_range=communication_range,
            use_lbp=use_lbp,
            config=config,
        )
        agents.append(agent)

    # Create communication network
    network = DecentralizedCommNetwork(
        agents=agents,
        message_delay=message_delay,
        drop_probability=drop_probability,
    )

    # Log configuration
    enable_belief_fusion = dec_config.get("enable_belief_fusion", True)
    enable_llp = dec_config.get("enable_llp_intent_sharing", True)
    enable_hlp = dec_config.get("enable_hlp_intent_sharing", True)
    enable_pos = dec_config.get("enable_position_sharing", True)

    logger.info(
        f"Created {num_agents} decentralized agents: "
        f"comm_range={communication_range}, use_lbp={use_lbp}, "
        f"belief_fusion={enable_belief_fusion}, llp_intent={enable_llp}, "
        f"hlp_intent={enable_hlp}, position_sharing={enable_pos}"
    )

    return agents, network


def create_decentralized_system(
    num_agents: int,
    cameras: List[Any],
    grid_info: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a complete decentralized multi-agent system.

    Args:
        num_agents: Number of agents
        cameras: List of cameras
        grid_info: Grid configuration
        config: System configuration including:
            - enable_belief_fusion: Enable news belief sharing (default: True)
            - enable_llp_intent_sharing: Enable LLP intent sharing (default: True)
            - enable_hlp_intent_sharing: Enable HLP intent sharing (default: True)
            - enable_position_sharing: Enable position broadcast (default: True)

    Returns:
        Dict with 'agents', 'network', and helper functions
    """
    agents, network = create_decentralized_agents(
        num_agents=num_agents,
        cameras=cameras,
        grid_info=grid_info,
        config=config,
    )

    def step_all():
        """Execute one step for all agents (respects enable flags)."""
        # 1. Broadcast positions (if enabled per agent)
        for agent in agents:
            agent.broadcast_position()

        # 2. Route position messages
        network.route_messages()

        # 3. Process incoming messages
        for agent in agents:
            agent.process_inbox()

        # 4. Share news beliefs (if enabled per agent)
        for agent in agents:
            agent.share_news_with_neighbors()

        # 5. Route news messages
        network.route_messages()

        # 6. Process news
        for agent in agents:
            agent.process_inbox()

        # 7. Share LLP and HLP intents (if enabled per agent)
        for agent in agents:
            agent.share_llp_intent()
            agent.share_hlp_intent()

        # 8. Route intents
        network.route_messages()

        # 9. Process intents
        for agent in agents:
            agent.process_inbox()

    def get_all_beliefs():
        """Get belief maps from all agents."""
        return {a.agent_id: a.get_belief() for a in agents}

    def get_all_statistics():
        """Get statistics from all components."""
        return {
            "agents": {a.agent_id: a.get_statistics() for a in agents},
            "network": network.get_statistics(),
        }

    def get_config_summary():
        """Get configuration summary for all agents."""
        return {
            a.agent_id: {
                "enable_belief_fusion": a.enable_belief_fusion,
                "enable_llp_intent_sharing": a.enable_llp_intent_sharing,
                "enable_hlp_intent_sharing": a.enable_hlp_intent_sharing,
                "enable_position_sharing": a.enable_position_sharing,
            }
            for a in agents
        }

    return {
        "agents": agents,
        "network": network,
        "step_all": step_all,
        "get_all_beliefs": get_all_beliefs,
        "get_all_statistics": get_all_statistics,
        "get_config_summary": get_config_summary,
    }
