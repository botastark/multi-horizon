"""
Asynchronous Multi-Agent Runner for True Dec-MCTS Execution

This module implements truly asynchronous multi-agent execution where:
1. Each agent runs in its own thread with independent planning cycles
2. Agents communicate via message queues with configurable delays
3. D-UCT discounting handles stale intents from asynchronous drift
4. Planning and communication happen at independent rates

This matches the Dec-MCTS paper's asynchronous model where agents:
- Plan continuously at their own rate
- Broadcast intents when ready (not at sync points)
- Receive teammate intents with realistic delays
- Must handle uncertainty about teammates' current plans

Reference: "Multi-Horizon Multi-Agent Planning Using Decentralised Monte Carlo Tree Search"
"""

import threading
import queue
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy

logger = logging.getLogger(__name__)


# =============================================================================
# Async Message Types and Data Structures
# =============================================================================


class AsyncMessageType(Enum):
    """Types of async messages between agents."""

    LL_INTENT = "ll_intent"
    HL_INTENT = "hl_intent"
    POSITION = "position"
    BELIEF_NEWS = "belief_news"
    OBSERVATION = "observation"
    FUSED_BELIEF = "fused_belief"  # Belief fusion result from coordinator
    STOP = "stop"


@dataclass
class AsyncMessage:
    """Message with timestamp for async communication."""

    msg_type: AsyncMessageType
    sender_id: int
    payload: Any
    send_time: float = field(default_factory=time.time)
    # Scheduled delivery time (send_time + delay)
    delivery_time: float = 0.0


@dataclass
class AgentState:
    """Thread-safe agent state snapshot."""

    agent_id: int
    position: Tuple[float, float]
    altitude: float
    step: int
    planning_cycles: int
    last_action: str
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# Async Communication Network
# =============================================================================


class AsyncCommNetwork:
    """
    Asynchronous communication network with realistic delays.

    Features:
    - Per-agent inbox queues
    - Configurable communication delay (simulates network latency)
    - Message timestamping for D-UCT staleness tracking
    - Thread-safe message routing
    """

    def __init__(
        self,
        num_agents: int,
        comm_delay_ms: float = 50.0,
        drop_probability: float = 0.0,
        queue_size: int = 100,
    ):
        """
        Initialize async communication network.

        Args:
            num_agents: Number of agents
            comm_delay_ms: Communication delay in milliseconds
            drop_probability: Probability of dropping a message (0-1)
            queue_size: Max messages per inbox
        """
        self.num_agents = num_agents
        self.comm_delay_sec = comm_delay_ms / 1000.0
        self.drop_probability = drop_probability

        # Per-agent inboxes
        self.inboxes: Dict[int, queue.Queue] = {
            i: queue.Queue(maxsize=queue_size) for i in range(num_agents)
        }

        # Pending messages (waiting for delivery time)
        self._pending: List[Tuple[int, AsyncMessage]] = []
        self._pending_lock = threading.Lock()

        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_dropped": 0,
        }
        self._stats_lock = threading.Lock()

        # Running flag
        self._running = False
        self._delivery_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the message delivery thread."""
        self._running = True
        self._delivery_thread = threading.Thread(
            target=self._delivery_loop, daemon=True, name="AsyncCommNetwork-Delivery"
        )
        self._delivery_thread.start()
        logger.info(
            f"AsyncCommNetwork started with {self.comm_delay_sec*1000:.0f}ms delay"
        )

    def stop(self):
        """Stop the message delivery thread."""
        self._running = False
        if self._delivery_thread:
            self._delivery_thread.join(timeout=1.0)
        logger.info("AsyncCommNetwork stopped")

    def _delivery_loop(self):
        """Background thread that delivers messages after delay."""
        while self._running:
            current_time = time.time()
            to_deliver = []

            with self._pending_lock:
                # Find messages ready for delivery
                remaining = []
                for receiver_id, msg in self._pending:
                    if current_time >= msg.delivery_time:
                        to_deliver.append((receiver_id, msg))
                    else:
                        remaining.append((receiver_id, msg))
                self._pending = remaining

            # Deliver ready messages
            for receiver_id, msg in to_deliver:
                self._deliver_to_inbox(receiver_id, msg)

            # Small sleep to avoid busy loop
            time.sleep(0.001)  # 1ms resolution

    def _deliver_to_inbox(self, receiver_id: int, msg: AsyncMessage):
        """Deliver a message to agent's inbox."""
        if receiver_id not in self.inboxes:
            return

        try:
            self.inboxes[receiver_id].put_nowait(msg)
            with self._stats_lock:
                self._stats["messages_delivered"] += 1
        except queue.Full:
            # Drop oldest message if queue full
            try:
                self.inboxes[receiver_id].get_nowait()
                self.inboxes[receiver_id].put_nowait(msg)
            except queue.Empty:
                pass

    def send(
        self,
        sender_id: int,
        receiver_id: int,
        msg_type: AsyncMessageType,
        payload: Any,
    ):
        """
        Send a message with delay.

        Args:
            sender_id: Sending agent ID
            receiver_id: Receiving agent ID
            msg_type: Message type
            payload: Message payload
        """
        # Check for random drop
        if np.random.random() < self.drop_probability:
            with self._stats_lock:
                self._stats["messages_dropped"] += 1
            return

        current_time = time.time()
        msg = AsyncMessage(
            msg_type=msg_type,
            sender_id=sender_id,
            payload=payload,
            send_time=current_time,
            delivery_time=current_time + self.comm_delay_sec,
        )

        with self._pending_lock:
            self._pending.append((receiver_id, msg))

        with self._stats_lock:
            self._stats["messages_sent"] += 1

    def broadcast(
        self,
        sender_id: int,
        msg_type: AsyncMessageType,
        payload: Any,
    ):
        """Broadcast message to all other agents."""
        for receiver_id in range(self.num_agents):
            if receiver_id != sender_id:
                self.send(sender_id, receiver_id, msg_type, payload)

    def receive(self, agent_id: int, timeout: float = 0.0) -> Optional[AsyncMessage]:
        """
        Receive a message from inbox.

        Args:
            agent_id: Agent ID
            timeout: Max time to wait (0 = non-blocking)

        Returns:
            AsyncMessage or None
        """
        if agent_id not in self.inboxes:
            return None

        try:
            if timeout > 0:
                return self.inboxes[agent_id].get(timeout=timeout)
            else:
                return self.inboxes[agent_id].get_nowait()
        except queue.Empty:
            return None

    def receive_all(self, agent_id: int) -> List[AsyncMessage]:
        """Receive all pending messages from inbox."""
        messages = []
        while True:
            msg = self.receive(agent_id)
            if msg is None:
                break
            messages.append(msg)
        return messages

    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        with self._stats_lock:
            return dict(self._stats)


# =============================================================================
# Async Agent Thread
# =============================================================================


class AsyncAgentThread(threading.Thread):
    """
    Agent running in its own thread with independent planning cycle.

    Each agent:
    1. Observes environment at fixed rate
    2. Runs LLP planning continuously
    3. Runs HLP planning at slower rate
    4. Broadcasts intents when ready
    5. Receives teammate intents asynchronously
    """

    def __init__(
        self,
        agent_id: int,
        agent_config: Dict[str, Any],
        comm_network: AsyncCommNetwork,
        environment: Any,  # Shared environment reference
        planning_rate_hz: float = 10.0,
        hlp_rate_hz: float = 2.0,
        observation_callback: Optional[Callable] = None,
    ):
        """
        Initialize async agent thread.

        Args:
            agent_id: Agent ID
            agent_config: Agent configuration dict
            comm_network: Async communication network
            environment: Shared environment for observations
            planning_rate_hz: LLP planning rate (cycles/second)
            hlp_rate_hz: HLP planning rate (cycles/second)
            observation_callback: Callback for getting observations
        """
        super().__init__(name=f"Agent-{agent_id}", daemon=True)

        self.agent_id = agent_id
        self.config = agent_config
        self.comm_network = comm_network
        self.environment = environment

        # Planning rates
        self.llp_period = 1.0 / planning_rate_hz
        self.hlp_period = 1.0 / hlp_rate_hz

        # Callbacks
        self.observation_callback = observation_callback

        # Agent components (set by initialize())
        self.camera = None
        self.planner = None
        self.belief_map = None
        self.occupancy_map = None

        # Current state
        self.position = (0.0, 0.0)
        self.altitude = 0.0
        self.step = 0
        self.planning_cycles = 0

        # Thread control
        self._running = False
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused initially

        # State lock
        self._state_lock = threading.RLock()

        # Received intents from teammates
        self._teammate_ll_intents: Dict[int, Any] = {}
        self._teammate_hl_intents: Dict[int, Any] = {}
        self._intents_lock = threading.Lock()

        # Action queue (for external control)
        self._action_queue: queue.Queue = queue.Queue()

        # Statistics
        self._stats = {
            "observations": 0,
            "llp_cycles": 0,
            "hlp_cycles": 0,
            "intents_broadcast": 0,
            "intents_received": 0,
            "actions_executed": 0,
        }

        # Logging
        self._log_buffer: List[Dict] = []
        self._log_lock = threading.Lock()

        # Coordinator for collision avoidance and belief fusion (set externally)
        self.coordinator = None

        # Teammate positions for collision avoidance (updated via POSITION messages)
        self._teammate_positions: Dict[int, Tuple[Tuple[float, float], float]] = {}
        self._positions_lock = threading.Lock()

    def initialize(
        self,
        camera: Any,
        planner: Any,
        belief_map: np.ndarray,
        occupancy_map: Any,
        initial_position: Tuple[float, float],
        initial_altitude: float,
    ):
        """
        Initialize agent components.

        Args:
            camera: UAV camera model
            planner: Planning instance (with hierarchical planner)
            belief_map: Initial belief map
            occupancy_map: Occupancy grid map
            initial_position: Starting (x, y) position
            initial_altitude: Starting altitude
        """
        with self._state_lock:
            self.camera = camera
            self.planner = planner
            self.belief_map = belief_map.copy()
            self.occupancy_map = occupancy_map
            self.position = initial_position
            self.altitude = initial_altitude

            # Set camera state
            self.camera.set_position(initial_position)
            self.camera.set_altitude(initial_altitude)

    def set_coordinator(self, coordinator: Any):
        """
        Set the multi-agent coordinator for collision avoidance and belief fusion.

        Args:
            coordinator: MultiAgentCoordinator instance
        """
        self.coordinator = coordinator

    def run(self):
        """Main agent loop - runs in separate thread."""
        logger.info(f"Agent {self.agent_id} thread started")
        self._running = True

        last_llp_time = time.time()
        last_hlp_time = time.time()

        while self._running:
            # Check for pause
            self._pause_event.wait()

            if not self._running:
                break

            current_time = time.time()

            # Process incoming messages
            self._process_messages()

            # Run LLP at configured rate
            if current_time - last_llp_time >= self.llp_period:
                self._run_llp_cycle()
                last_llp_time = current_time

            # Run HLP at slower rate
            if current_time - last_hlp_time >= self.hlp_period:
                self._run_hlp_cycle()
                last_hlp_time = current_time

            # Small sleep to avoid busy loop
            time.sleep(0.001)

        logger.info(f"Agent {self.agent_id} thread stopped")

    def stop(self):
        """Stop the agent thread."""
        self._running = False
        self._pause_event.set()  # Unpause if paused

    def pause(self):
        """Pause the agent."""
        self._paused = True
        self._pause_event.clear()

    def resume(self):
        """Resume the agent."""
        self._paused = False
        self._pause_event.set()

    def _process_messages(self):
        """Process all pending messages from teammates."""
        messages = self.comm_network.receive_all(self.agent_id)

        for msg in messages:
            if msg.msg_type == AsyncMessageType.LL_INTENT:
                with self._intents_lock:
                    self._teammate_ll_intents[msg.sender_id] = msg.payload
                self._stats["intents_received"] += 1

            elif msg.msg_type == AsyncMessageType.HL_INTENT:
                with self._intents_lock:
                    self._teammate_hl_intents[msg.sender_id] = msg.payload
                self._stats["intents_received"] += 1

            elif msg.msg_type == AsyncMessageType.POSITION:
                # Update teammate position (for collision avoidance)
                with self._positions_lock:
                    self._teammate_positions[msg.sender_id] = (
                        msg.payload.get("position", (0.0, 0.0)),
                        msg.payload.get("altitude", 0.0),
                    )

            elif msg.msg_type == AsyncMessageType.BELIEF_NEWS:
                # Handle belief news for fusion (via coordinator)
                if self.coordinator is not None and hasattr(
                    self.coordinator, "receive_news"
                ):
                    self.coordinator.receive_news(
                        sender_id=msg.sender_id,
                        news_belief=msg.payload.get("news_belief"),
                        footprint=msg.payload.get("footprint"),
                    )

            elif msg.msg_type == AsyncMessageType.FUSED_BELIEF:
                # Update local belief with fused version
                fused = msg.payload.get("fused_belief")
                if fused is not None:
                    with self._state_lock:
                        if self.belief_map is not None:
                            self.belief_map[:, :, 1] = fused
                            self.belief_map[:, :, 0] = 1 - fused

            elif msg.msg_type == AsyncMessageType.STOP:
                self.stop()

    def _run_llp_cycle(self):
        """
        Run one LLP planning cycle.

        Supports multiple planner types:
        - mh_dec_mcts_efficient/mh_dec_mcts_full: Hierarchical LLP/HLP
        - dec_mcts: Single-level MCTS
        - greedy_ig: Greedy information gain
        """
        if self.planner is None:
            return

        with self._state_lock:
            strategy = self.planner.strategy

            # Multi-Horizon Dec-MCTS (hierarchical)
            if hasattr(self.planner, "_hierarchical_planner"):
                hier_planner = self.planner._hierarchical_planner

                # Update belief (use fused belief if available via coordinator)
                if self.coordinator is not None:
                    fused_belief = self.coordinator.get_agent_belief(self.agent_id)
                    if fused_belief is not None:
                        self.belief_map[:, :, 1] = fused_belief
                        self.belief_map[:, :, 0] = 1 - fused_belief
                        self.planner.M = self.belief_map.copy()
                        # Update LLP belief as well
                        hier_planner.llp.belief = self.planner.M.copy()

                # Update teammate intents (with D-UCT staleness)
                with self._intents_lock:
                    ll_intents = dict(self._teammate_ll_intents)
                    hl_intents = dict(self._teammate_hl_intents)

                # These intents may be stale - D-UCT discounting applies
                hier_planner.llp.update_teammate_intents(ll_intents, hl_intents)

                # Run LLP planning
                current_state = (self.position[0], self.position[1], self.altitude)
                ll_intent = hier_planner.llp.plan(current_state)

                # Broadcast LL intent
                self.comm_network.broadcast(
                    self.agent_id,
                    AsyncMessageType.LL_INTENT,
                    ll_intent,
                )
                self._stats["intents_broadcast"] += 1

                # Get best action (collision avoidance handled within LLP via coordinator)
                best_action = hier_planner.llp.get_best_action()

            # Dec-MCTS (single-level)
            elif hasattr(self.planner, "_dec_mcts_planner"):
                dec_planner = self.planner._dec_mcts_planner

                # Update teammate intents
                with self._intents_lock:
                    teammate_intents = dict(self._teammate_ll_intents)
                dec_planner.update_teammate_intents(teammate_intents)

                # Update belief (use fused belief if available via coordinator)
                if self.coordinator is not None:
                    fused_belief = self.coordinator.get_agent_belief(self.agent_id)
                    if fused_belief is not None:
                        self.belief_map[:, :, 1] = fused_belief
                        self.belief_map[:, :, 0] = 1 - fused_belief
                        self.planner.M = self.belief_map.copy()

                # Update state
                dec_planner.update_state(
                    position=self.position,
                    altitude=self.altitude,
                    belief=self.planner.M.copy(),
                )

                # Run planning
                intent = dec_planner.plan()

                # Apply collision avoidance if coordinator available
                best_action = (
                    intent.action_sequence[0] if intent.action_sequence else "hover"
                )
                if self.coordinator is not None and hasattr(
                    self.coordinator, "get_collision_penalty"
                ):
                    # Dec-MCTS provides action values in the intent
                    if hasattr(intent, "action_values") and intent.action_values:
                        action_scores = intent.action_values
                        best_score = float("-inf")

                        # Get max score for normalization
                        valid_scores = [abs(s) for s in action_scores.values()]
                        max_score = max(valid_scores) if valid_scores else 1.0
                        max_score = max(max_score, 1.0)

                        for action, score in action_scores.items():
                            proposed_state = self.camera.x_future(action)
                            if proposed_state is None:
                                continue

                            proposed_pos, _ = proposed_state
                            proposed_row, proposed_col = self.camera.convert_xy_ij(
                                proposed_pos[0],
                                proposed_pos[1],
                                self.camera.grid.center,
                            )

                            collision_penalty = self.coordinator.get_collision_penalty(
                                self.agent_id, (proposed_row, proposed_col)
                            )

                            collision_weight = self.config.get(
                                "collision_penalty_weight", 1.0
                            )
                            adjusted_score = (
                                score - collision_penalty * max_score * collision_weight
                            )

                            if adjusted_score > best_score:
                                best_score = adjusted_score
                                best_action = action

                # Broadcast intent
                self.comm_network.broadcast(
                    self.agent_id,
                    AsyncMessageType.LL_INTENT,
                    intent,
                )
                self._stats["intents_broadcast"] += 1

            # Greedy IG
            elif hasattr(self.planner, "_greedy_ig_planner"):
                greedy_planner = self.planner._greedy_ig_planner

                # Update teammate intents
                with self._intents_lock:
                    teammate_intents = dict(self._teammate_ll_intents)
                greedy_planner.update_teammate_intents(teammate_intents)

                # Update belief (use fused belief if available via coordinator)
                if self.coordinator is not None:
                    fused_belief = self.coordinator.get_agent_belief(self.agent_id)
                    if fused_belief is not None:
                        self.belief_map[:, :, 1] = fused_belief
                        self.belief_map[:, :, 0] = 1 - fused_belief
                        self.planner.M = self.belief_map.copy()

                greedy_planner.update_belief(self.planner.M.copy())

                # Run planning
                intent = greedy_planner.plan(
                    current_position=self.position,
                    current_altitude=self.altitude,
                )

                # Apply collision avoidance if coordinator available
                best_action = intent.action
                if self.coordinator is not None and hasattr(
                    self.coordinator, "get_collision_penalty"
                ):
                    action_scores = greedy_planner.get_action_scores()
                    best_score = float("-inf")

                    # Get max IG for normalization
                    valid_scores = [
                        s for s in action_scores.values() if s > float("-inf")
                    ]
                    max_ig = max(valid_scores) if valid_scores else 1.0
                    max_ig = max(max_ig, 1.0)  # Ensure minimum scale

                    for action, ig_score in action_scores.items():
                        if ig_score <= float("-inf"):
                            continue

                        # Get proposed position for this action
                        proposed_state = self.camera.x_future(action)
                        if proposed_state is None:
                            continue

                        proposed_pos, proposed_alt = proposed_state
                        proposed_row, proposed_col = self.camera.convert_xy_ij(
                            proposed_pos[0], proposed_pos[1], self.camera.grid.center
                        )

                        # Get collision penalty from coordinator (0-1 range)
                        collision_penalty = self.coordinator.get_collision_penalty(
                            self.agent_id, (proposed_row, proposed_col)
                        )

                        # Adjusted score = IG - collision_penalty * max_ig * weight
                        # This ensures collision penalty has meaningful impact regardless of IG magnitude
                        collision_weight = self.config.get(
                            "collision_penalty_weight", 1.0
                        )
                        adjusted_score = (
                            ig_score - collision_penalty * max_ig * collision_weight
                        )

                        if adjusted_score > best_score:
                            best_score = adjusted_score
                            best_action = action

                    # Log collision avoidance decision if penalty was applied
                    if best_action != intent.action:
                        logger.debug(
                            f"Agent {self.agent_id}: collision avoidance changed action {intent.action} -> {best_action}"
                        )

                # Broadcast intent
                self.comm_network.broadcast(
                    self.agent_id,
                    AsyncMessageType.LL_INTENT,
                    intent,
                )
                self._stats["intents_broadcast"] += 1

            else:
                # Fallback: use basic planning
                best_action, _ = self.planner.select_action(
                    self.planner.M, visited_x=[]
                )

            # Execute action
            self._execute_action(best_action)

        self._stats["llp_cycles"] += 1
        self.planning_cycles += 1

    def _run_hlp_cycle(self):
        """
        Run one HLP planning cycle (slower rate).

        Only applies to MH planners.
        For non-hierarchical planners, this is a no-op.
        """
        if self.planner is None:
            return

        with self._state_lock:
            # Only run HLP for hierarchical planners
            if hasattr(self.planner, "_hierarchical_planner"):
                hier_planner = self.planner._hierarchical_planner

                # Get teammate intents
                with self._intents_lock:
                    ll_intents = dict(self._teammate_ll_intents)
                    hl_intents = dict(self._teammate_hl_intents)

                hier_planner.hlp.update_teammate_intents(ll_intents, hl_intents)

                # Convert position to grid coordinates
                grid_pos = self.camera.convert_xy_ij(
                    self.position[0],
                    self.position[1],
                    self.camera.grid.center,
                )

                # Run HLP planning
                hl_intent = hier_planner.hlp.plan((grid_pos[0], grid_pos[1]))

                # Broadcast HL intent
                self.comm_network.broadcast(
                    self.agent_id,
                    AsyncMessageType.HL_INTENT,
                    hl_intent,
                )
                self._stats["intents_broadcast"] += 1

                # Update LLP with new HLP guidance
                if hl_intent.target_center is not None:
                    hl_intent_for_llp = copy.copy(hl_intent)
                    grid_center = hl_intent.target_center
                    world_x, world_y = self.camera.ij_to_xy(
                        grid_center[0], grid_center[1]
                    )
                    hl_intent_for_llp.target_center = (world_x, world_y)
                    hier_planner.llp.update_hl_guidance(hl_intent_for_llp)

                self._stats["hlp_cycles"] += 1

    def _execute_action(self, action: str):
        """Execute an action and update state."""
        with self._state_lock:
            # Get next state from camera model
            future_state = self.camera.x_future(action)
            if future_state is None:
                return

            # Update position
            self.position = future_state[0]
            self.altitude = future_state[1]

            # Update camera
            self.camera.set_position(self.position)
            self.camera.set_altitude(self.altitude)

            self.step += 1
            self._stats["actions_executed"] += 1

            # Log action
            self._log_action(action)

            # Update coordinator with new position (for collision avoidance)
            if self.coordinator is not None:
                current_row, current_col = self.camera.convert_xy_ij(
                    self.position[0], self.position[1], self.camera.grid.center
                )
                self.coordinator.update_agent_state(
                    agent_id=self.agent_id,
                    position=(current_row, current_col),
                    altitude=self.altitude,
                )

            # Broadcast position
            self.comm_network.broadcast(
                self.agent_id,
                AsyncMessageType.POSITION,
                {
                    "position": self.position,
                    "altitude": self.altitude,
                    "step": self.step,
                },
            )

    def _log_action(self, action: str):
        """Log action to buffer."""
        with self._log_lock:
            self._log_buffer.append(
                {
                    "timestamp": time.time(),
                    "step": self.step,
                    "action": action,
                    "position": self.position,
                    "altitude": self.altitude,
                    "planning_cycles": self.planning_cycles,
                }
            )

    def get_state(self) -> AgentState:
        """Get current agent state (thread-safe)."""
        with self._state_lock:
            return AgentState(
                agent_id=self.agent_id,
                position=self.position,
                altitude=self.altitude,
                step=self.step,
                planning_cycles=self.planning_cycles,
                last_action=(
                    self._log_buffer[-1]["action"] if self._log_buffer else "none"
                ),
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return dict(self._stats)

    def get_log(self) -> List[Dict]:
        """Get action log."""
        with self._log_lock:
            return list(self._log_buffer)


# =============================================================================
# Async Multi-Agent Experiment Runner
# =============================================================================


class AsyncMultiAgentRunner:
    """
    Coordinates multiple async agent threads for experiment execution.

    Features:
    - Spawns and manages agent threads
    - Provides synchronized start/stop
    - Collects metrics across all agents
    - Supports step-based checkpoints for logging/visualization
    """

    def __init__(
        self,
        num_agents: int,
        planning_rate_hz: float = 10.0,
        hlp_rate_hz: float = 2.0,
        comm_delay_ms: float = 50.0,
        drop_probability: float = 0.0,
    ):
        """
        Initialize async runner.

        Args:
            num_agents: Number of agents
            planning_rate_hz: LLP planning rate
            hlp_rate_hz: HLP planning rate
            comm_delay_ms: Communication delay in milliseconds
            drop_probability: Message drop probability
        """
        self.num_agents = num_agents
        self.planning_rate_hz = planning_rate_hz
        self.hlp_rate_hz = hlp_rate_hz

        # Create communication network
        self.comm_network = AsyncCommNetwork(
            num_agents=num_agents,
            comm_delay_ms=comm_delay_ms,
            drop_probability=drop_probability,
        )

        # Agent threads
        self.agents: Dict[int, AsyncAgentThread] = {}

        # Running state
        self._running = False
        self._start_time: float = 0.0

        # Checkpoint callback
        self._checkpoint_callback: Optional[Callable] = None
        self._checkpoint_interval: float = 1.0  # seconds
        self._checkpoint_thread: Optional[threading.Thread] = None

        # Shared coordinator for collision avoidance and belief fusion
        self.coordinator = None

    def set_coordinator(self, coordinator: Any):
        """
        Set the multi-agent coordinator for all agents.

        Args:
            coordinator: MultiAgentCoordinator instance
        """
        self.coordinator = coordinator
        # Propagate to existing agents
        for agent in self.agents.values():
            agent.set_coordinator(coordinator)

    def add_agent(
        self,
        agent_id: int,
        camera: Any,
        planner: Any,
        belief_map: np.ndarray,
        occupancy_map: Any,
        initial_position: Tuple[float, float],
        initial_altitude: float,
        config: Optional[Dict] = None,
        coordinator: Optional[Any] = None,
    ):
        """
        Add an agent to the runner.

        Args:
            agent_id: Agent ID
            camera: UAV camera
            planner: Planning instance
            belief_map: Initial belief
            occupancy_map: OG map
            initial_position: Start position
            initial_altitude: Start altitude
            config: Agent config
            coordinator: MultiAgentCoordinator for collision avoidance/belief fusion
        """
        agent = AsyncAgentThread(
            agent_id=agent_id,
            agent_config=config or {},
            comm_network=self.comm_network,
            environment=None,  # Set later if needed
            planning_rate_hz=self.planning_rate_hz,
            hlp_rate_hz=self.hlp_rate_hz,
        )

        agent.initialize(
            camera=camera,
            planner=planner,
            belief_map=belief_map,
            occupancy_map=occupancy_map,
            initial_position=initial_position,
            initial_altitude=initial_altitude,
        )

        # Set coordinator for collision avoidance and belief fusion
        effective_coordinator = (
            coordinator if coordinator is not None else self.coordinator
        )
        if effective_coordinator is not None:
            agent.set_coordinator(effective_coordinator)
            # Register initial position with coordinator
            initial_row, initial_col = camera.convert_xy_ij(
                initial_position[0], initial_position[1], camera.grid.center
            )
            effective_coordinator.update_agent_state(
                agent_id=agent_id,
                position=(initial_row, initial_col),
                altitude=initial_altitude,
            )
            logger.info(
                f"Agent {agent_id} initial position registered: ({initial_row:.1f}, {initial_col:.1f})"
            )

        self.agents[agent_id] = agent
        logger.info(f"Added agent {agent_id} to async runner")

    def set_checkpoint_callback(
        self,
        callback: Callable[[Dict[int, AgentState]], None],
        interval: float = 1.0,
    ):
        """
        Set callback for periodic checkpoints.

        Args:
            callback: Function called with dict of agent states
            interval: Checkpoint interval in seconds
        """
        self._checkpoint_callback = callback
        self._checkpoint_interval = interval

    def start(self):
        """Start all agent threads and communication network."""
        if self._running:
            return

        logger.info(f"Starting async multi-agent runner with {len(self.agents)} agents")

        # Start communication network
        self.comm_network.start()

        # Start all agents
        self._running = True
        self._start_time = time.time()

        for agent in self.agents.values():
            agent.start()

        # Start checkpoint thread if callback set
        if self._checkpoint_callback:
            self._checkpoint_thread = threading.Thread(
                target=self._checkpoint_loop, daemon=True, name="Checkpoint-Thread"
            )
            self._checkpoint_thread.start()

        logger.info("Async runner started")

    def stop(self):
        """Stop all agent threads."""
        if not self._running:
            return

        logger.info("Stopping async runner...")
        self._running = False

        # Stop all agents
        for agent in self.agents.values():
            agent.stop()

        # Wait for agents to finish
        for agent in self.agents.values():
            agent.join(timeout=2.0)

        # Stop communication network
        self.comm_network.stop()

        logger.info("Async runner stopped")

    def _checkpoint_loop(self):
        """Periodic checkpoint callback."""
        while self._running:
            time.sleep(self._checkpoint_interval)

            if not self._running:
                break

            # Collect all agent states
            states = {
                agent_id: agent.get_state() for agent_id, agent in self.agents.items()
            }

            # Call checkpoint callback
            if self._checkpoint_callback:
                try:
                    self._checkpoint_callback(states)
                except Exception as e:
                    logger.error(f"Checkpoint callback error: {e}")

    def run_for_duration(self, duration_sec: float):
        """
        Run experiment for specified duration.

        Args:
            duration_sec: Duration in seconds
        """
        self.start()

        try:
            time.sleep(duration_sec)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def run_until_steps(self, target_steps: int, timeout_sec: float = 300.0):
        """
        Run until all agents complete target steps.

        Args:
            target_steps: Target step count per agent
            timeout_sec: Maximum runtime
        """
        self.start()

        start_time = time.time()

        try:
            while self._running:
                # Check if all agents reached target
                all_done = all(
                    agent.step >= target_steps for agent in self.agents.values()
                )

                if all_done:
                    logger.info(f"All agents completed {target_steps} steps")
                    break

                # Check timeout
                if time.time() - start_time > timeout_sec:
                    logger.warning(f"Timeout after {timeout_sec}s")
                    break

                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def get_all_states(self) -> Dict[int, AgentState]:
        """Get current states of all agents."""
        return {agent_id: agent.get_state() for agent_id, agent in self.agents.items()}

    def get_all_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics from all agents."""
        return {
            agent_id: agent.get_statistics() for agent_id, agent in self.agents.items()
        }

    def get_all_logs(self) -> Dict[int, List[Dict]]:
        """Get action logs from all agents."""
        return {agent_id: agent.get_log() for agent_id, agent in self.agents.items()}

    def get_network_statistics(self) -> Dict[str, Any]:
        """Get communication network statistics."""
        return self.comm_network.get_statistics()


# =============================================================================
# Utility Functions
# =============================================================================


def create_async_runner_from_config(config: Dict[str, Any]) -> AsyncMultiAgentRunner:
    """
    Create AsyncMultiAgentRunner from config dict.

    Config keys:
        - async.planning_rate_hz: LLP planning rate (default: 10.0)
        - async.hlp_rate_hz: HLP planning rate (default: 2.0)
        - async.comm_delay_ms: Communication delay (default: 50.0)
        - async.drop_probability: Message drop rate (default: 0.0)
    """
    async_config = config.get("async", {})
    num_agents = config.get("multi_agent", {}).get("num_agents", 1)

    return AsyncMultiAgentRunner(
        num_agents=num_agents,
        planning_rate_hz=async_config.get("planning_rate_hz", 10.0),
        hlp_rate_hz=async_config.get("hlp_rate_hz", 2.0),
        comm_delay_ms=async_config.get("comm_delay_ms", 50.0),
        drop_probability=async_config.get("drop_probability", 0.0),
    )
