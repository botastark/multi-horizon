# threaded_dual_horizon.py
"""
Threaded Dual-Horizon MCTS Planner with Intent Bus

This module extends the dual-horizon planning approach with asynchronous execution:
- LLP (Low-Level Planner): Runs in its own thread, focuses on immediate IG exploitation
- HLP (High-Level Planner): Runs in its own thread, focuses on region selection
- Intent Bus: Thread-safe communication channel between LLP and HLP

The key advantage is that HLP can run expensive region analysis while LLP continues
to make real-time decisions, improving responsiveness without sacrificing quality.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import time
import copy
import logging
from datetime import datetime

from helper import uav_position, H, expected_posterior
from mcts import MCTSPlanner
from dual_horizon_planner import (
    DualHorizonPlanner,
    analyze_coverage_fragmentation,
    setup_dual_horizon_logger,
    log_planning_step,
    UNCERTAINTY_ADJUSTMENT_FACTOR,
    FRAGMENTATION_ADJUSTMENT_FACTOR,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Intent Types and Data Classes
# =============================================================================


class IntentType(Enum):
    """Types of intents exchanged between LLP and HLP."""

    # HLP -> LLP intents
    TARGET_REGION = "target_region"  # HLP selected a new target region
    FRAGMENTATION_ALERT = "fragmentation_alert"  # HLP detected fragmentation

    # LLP -> HLP intents
    REGION_REACHED = "region_reached"  # LLP reached target region

    # Bidirectional
    SHUTDOWN = "shutdown"  # Shutdown signal


@dataclass
class Intent:
    """
    Message passed between LLP and HLP via the Intent Bus.

    Attributes:
        intent_type: Type of intent
        source: 'LLP' or 'HLP'
        timestamp: When the intent was created
        data: Intent-specific payload
        priority: Higher priority intents are processed first
    """

    intent_type: IntentType
    source: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more urgent

    def __lt__(self, other):
        """Priority queue ordering (higher priority first)."""
        return self.priority > other.priority


@dataclass
class HLPGuidance:
    """
    Guidance from HLP to LLP containing target region information.

    Attributes:
        target_region_id: Selected region ID
        target_center: (row, col) center of target region in grid indices
        region_scores: Scores for all regions
        isolation_scores: Isolation scores for fragmented regions
        blend_weight_suggestion: Suggested w_long weight
        valid_until: Timestamp after which guidance should be refreshed
    """

    target_region_id: int
    target_center: Tuple[float, float]
    region_scores: Dict[int, float]
    isolation_scores: Dict[int, float]
    blend_weight_suggestion: float
    valid_until: float
    region_metadata: Dict[int, Dict] = field(default_factory=dict)


@dataclass
class LLPFeedback:
    """
    Feedback from LLP to HLP about current state and progress.

    Attributes:
        current_position: Current UAV position (row, col) in grid indices
        current_altitude: Current altitude
        last_action: Last action taken
        coverage_progress: Current coverage fraction
        reached_target: Whether UAV has reached target region
        ig_scores: Recent IG scores for actions
    """

    current_position: Tuple[float, float]
    current_altitude: float
    last_action: Optional[str]
    coverage_progress: float
    reached_target: bool
    ig_scores: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Intent Bus - Thread-Safe Communication Channel
# =============================================================================


class IntentBus:
    """
    Thread-safe communication channel between LLP and HLP.

    Uses priority queues for each direction to ensure important
    messages are processed first.
    """

    def __init__(self, max_queue_size: int = 100):
        """
        Initialize the intent bus.

        Args:
            max_queue_size: Maximum number of pending intents per queue
        """
        # Separate queues for each direction
        self._llp_to_hlp: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=max_queue_size
        )
        self._hlp_to_llp: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=max_queue_size
        )

        # Latest guidance/feedback for quick access (non-blocking)
        self._latest_hlp_guidance: Optional[HLPGuidance] = None
        self._latest_llp_feedback: Optional[LLPFeedback] = None
        self._guidance_lock = threading.Lock()
        self._feedback_lock = threading.Lock()

        # Shutdown flag
        self._shutdown = threading.Event()

        # Statistics
        self._stats = {
            "llp_to_hlp_count": 0,
            "hlp_to_llp_count": 0,
            "guidance_updates": 0,
            "feedback_updates": 0,
        }
        self._stats_lock = threading.Lock()

    def send_to_hlp(self, intent: Intent) -> bool:
        """
        Send intent from LLP to HLP.

        Args:
            intent: Intent to send

        Returns:
            True if sent successfully, False if queue is full
        """
        try:
            self._llp_to_hlp.put_nowait(intent)
            with self._stats_lock:
                self._stats["llp_to_hlp_count"] += 1
            return True
        except queue.Full:
            logger.warning("LLP->HLP queue full, dropping intent")
            return False

    def send_to_llp(self, intent: Intent) -> bool:
        """
        Send intent from HLP to LLP.

        Args:
            intent: Intent to send

        Returns:
            True if sent successfully, False if queue is full
        """
        try:
            self._hlp_to_llp.put_nowait(intent)
            with self._stats_lock:
                self._stats["hlp_to_llp_count"] += 1
            return True
        except queue.Full:
            logger.warning("HLP->LLP queue full, dropping intent")
            return False

    def receive_from_llp(self, timeout: float = 0.1) -> Optional[Intent]:
        """
        Receive intent from LLP (called by HLP).

        Args:
            timeout: How long to wait for an intent

        Returns:
            Intent if available, None otherwise
        """
        try:
            return self._llp_to_hlp.get(timeout=timeout)
        except queue.Empty:
            return None

    def receive_from_hlp(self, timeout: float = 0.01) -> Optional[Intent]:
        """
        Receive intent from HLP (called by LLP).

        Non-blocking with very short timeout since LLP needs to be responsive.

        Args:
            timeout: How long to wait for an intent

        Returns:
            Intent if available, None otherwise
        """
        try:
            return self._hlp_to_llp.get(timeout=timeout)
        except queue.Empty:
            return None

    def update_guidance(self, guidance: HLPGuidance):
        """
        Update the latest HLP guidance (thread-safe).

        This is the primary way HLP communicates with LLP.
        """
        with self._guidance_lock:
            self._latest_hlp_guidance = guidance
            with self._stats_lock:
                self._stats["guidance_updates"] += 1

    def get_guidance(self) -> Optional[HLPGuidance]:
        """
        Get the latest HLP guidance (thread-safe, non-blocking).
        """
        with self._guidance_lock:
            return self._latest_hlp_guidance

    def update_feedback(self, feedback: LLPFeedback):
        """
        Update the latest LLP feedback (thread-safe).

        This is the primary way LLP communicates with HLP.
        """
        with self._feedback_lock:
            self._latest_llp_feedback = feedback
            with self._stats_lock:
                self._stats["feedback_updates"] += 1

    def get_feedback(self) -> Optional[LLPFeedback]:
        """
        Get the latest LLP feedback (thread-safe, non-blocking).
        """
        with self._feedback_lock:
            return self._latest_llp_feedback

    def shutdown(self):
        """Signal shutdown to all threads."""
        self._shutdown.set()
        # Send shutdown intents to unblock waiting threads
        shutdown_intent = Intent(
            intent_type=IntentType.SHUTDOWN, source="bus", priority=999
        )
        try:
            self._llp_to_hlp.put_nowait(shutdown_intent)
            self._hlp_to_llp.put_nowait(shutdown_intent)
        except queue.Full:
            pass

    def is_shutdown(self) -> bool:
        """Check if shutdown has been signaled."""
        return self._shutdown.is_set()

    def get_stats(self) -> Dict[str, int]:
        """Get bus statistics."""
        with self._stats_lock:
            return self._stats.copy()


# =============================================================================
# LLP Worker Thread
# =============================================================================


class LLPWorker(threading.Thread):
    """
    Low-Level Planner worker thread.

    Runs MCTS for immediate action selection while receiving
    guidance from HLP via the intent bus.
    """

    def __init__(
        self,
        intent_bus: IntentBus,
        camera: Any,
        conf_dict: Dict,
        mcts_params: Dict,
        initial_state: Dict[str, Any],
    ):
        """
        Initialize LLP worker.

        Args:
            intent_bus: Communication channel with HLP
            camera: UAV camera object
            conf_dict: Sensor confusion matrix
            mcts_params: MCTS configuration
            initial_state: Initial planning state
        """
        super().__init__(name="LLP-Worker", daemon=True)

        self.intent_bus = intent_bus
        self.camera = camera
        self.conf_dict = conf_dict
        self.mcts_params = mcts_params

        # State (protected by lock)
        self._state = copy.deepcopy(initial_state)
        self._state_lock = threading.Lock()

        # Latest action request and result
        self._action_request = threading.Event()
        self._action_result: Optional[Tuple[str, Dict]] = None
        self._result_ready = threading.Event()

        # Configuration
        self.short_horizon_depth = mcts_params.get("horizon_weights", {}).get(
            "short_horizon_depth", 5
        )
        self.num_iterations = mcts_params.get("num_iterations", 100)
        self.timeout = mcts_params.get("timeout", 2000)
        self.ucb1_c = mcts_params.get("ucb1_c", 0.95)
        self.discount_factor = mcts_params.get("discount_factor", 0.99)
        self.w_ig = mcts_params.get("horizon_weights", {}).get("w_ig", 0.8)

        # Blend weights (updated from HLP guidance)
        self._w_short = 0.6
        self._w_long = 0.4
        self._blend_lock = threading.Lock()

        # Statistics
        self.step_count = 0
        self.total_planning_time = 0.0

    def run(self):
        """Main LLP loop - wait for action requests and compute actions."""
        logger.info("LLP Worker started")

        while not self.intent_bus.is_shutdown():
            # Wait for action request
            if self._action_request.wait(timeout=0.1):
                self._action_request.clear()

                # Process any pending HLP intents
                self._process_hlp_intents()

                # Compute action
                start_time = time.time()
                action, metrics = self._compute_action()
                planning_time = time.time() - start_time

                self.step_count += 1
                self.total_planning_time += planning_time

                # Store result and signal completion
                self._action_result = (action, metrics)
                self._result_ready.set()

                # Send feedback to HLP
                self._send_feedback(action, metrics)

        logger.info("LLP Worker shutting down")

    def request_action(
        self, state: Dict[str, Any], timeout: float = 5.0
    ) -> Tuple[str, Dict]:
        """
        Request an action from LLP (called from main thread).

        Args:
            state: Current planning state
            timeout: Maximum time to wait

        Returns:
            Tuple of (action, metrics)
        """
        # Update state
        with self._state_lock:
            self._state = copy.deepcopy(state)

        # Clear previous result and request new action
        self._result_ready.clear()
        self._action_request.set()

        # Wait for result
        if self._result_ready.wait(timeout=timeout):
            return self._action_result
        else:
            logger.error("LLP action request timed out")
            return "hover", {"error": "timeout"}

    def _process_hlp_intents(self):
        """Process any pending intents from HLP."""
        while True:
            intent = self.intent_bus.receive_from_hlp(timeout=0.001)
            if intent is None:
                break

            if intent.intent_type == IntentType.SHUTDOWN:
                break
            elif intent.intent_type == IntentType.TARGET_REGION:
                # HLP selected a new target region
                logger.debug(
                    f"LLP received target region: {intent.data.get('region_id')}"
                )
            elif intent.intent_type == IntentType.FRAGMENTATION_ALERT:
                # HLP detected fragmentation - may need to adjust behavior
                logger.debug(f"LLP received fragmentation alert")

    def _compute_action(self) -> Tuple[str, Dict]:
        """
        Compute next action using MCTS with HLP guidance.

        Returns:
            Tuple of (action, metrics)
        """
        with self._state_lock:
            state = copy.deepcopy(self._state)

        # Run MCTS for short-horizon planning
        planner = MCTSPlanner(
            initial_state=state,
            uav_camera=self.camera,
            conf_dict=self.conf_dict,
            discount_factor=self.discount_factor,
            max_depth=self.short_horizon_depth,
            parallel=1,  # Single-threaded within LLP
            ucb1_c=self.ucb1_c,
            plan_cfg={"use_coverage_reward": False},
        )

        action, action_scores = planner.search(
            num_iterations=self.num_iterations,
            timeout=self.timeout,
            return_action_scores=True,
        )

        # Get HLP guidance for blending
        guidance = self.intent_bus.get_guidance()

        if guidance is not None and time.time() < guidance.valid_until:
            # Apply HLP guidance - compute alignment and blend
            blended_action, blended_scores = self._apply_guidance(
                state, action_scores, guidance
            )

            metrics = {
                "strategy": "blended",
                "ig_action": action,
                "blended_action": blended_action,
                "ig_scores": action_scores,
                "blended_scores": blended_scores,
                "target_region": guidance.target_region_id,
                "guidance_age": time.time()
                - (guidance.valid_until - 5.0),  # Assuming 5s validity
            }
            return blended_action, metrics
        else:
            # No valid guidance - use pure IG
            metrics = {
                "strategy": "ig_only",
                "action_scores": action_scores,
                "no_guidance": True,
            }
            return action, metrics

    def _apply_guidance(
        self, state: Dict[str, Any], ig_scores: Dict[str, float], guidance: HLPGuidance
    ) -> Tuple[str, Dict[str, float]]:
        """
        Apply HLP guidance to blend with IG scores.

        Args:
            state: Current state
            ig_scores: IG scores from MCTS
            guidance: HLP guidance

        Returns:
            Tuple of (best_action, blended_scores)
        """
        uav_pos = state["uav_pos"]
        target_center = guidance.target_center

        # Convert current position to grid indices
        current_row, current_col = self.camera.convert_xy_ij(
            uav_pos.position[0], uav_pos.position[1], self.camera.grid.center
        )
        current_pos = np.array([current_row, current_col])
        target_pos = np.array(target_center)

        # Compute alignment scores
        H_dim, W_dim = state["belief"].shape[:2]
        max_distance = np.sqrt(H_dim**2 + W_dim**2)

        alignment_scores = {}
        for action, ig_score in ig_scores.items():
            if action in ["up", "down"]:
                alignment_scores[action] = 0.0
                continue

            next_state = self.camera.x_future(action, x=uav_pos)
            if next_state is None:
                alignment_scores[action] = 0.0
                continue

            next_uav_pos = uav_position(next_state)
            next_row, next_col = self.camera.convert_xy_ij(
                next_uav_pos.position[0],
                next_uav_pos.position[1],
                self.camera.grid.center,
            )
            next_pos = np.array([next_row, next_col])

            dist_before = np.linalg.norm(current_pos - target_pos)
            dist_after = np.linalg.norm(next_pos - target_pos)

            # SOFT guidance: alignment is a BONUS only, never a penalty
            # Range [0, 1]: 1 = moving toward target, 0 = moving away or neutral
            raw_alignment = (dist_before - dist_after) / max_distance
            alignment = max(0.0, raw_alignment)  # Clamp negative to 0

            alignment_scores[action] = alignment

        # Get blend weights
        with self._blend_lock:
            w_short = self._w_short
            w_long = self._w_long

        # Check if HLP guidance is meaningful
        # If all alignment scores are ~0, HLP has no clear guidance - let LLP maximize IG
        max_alignment = max(alignment_scores.values()) if alignment_scores else 0.0
        hlp_has_guidance = max_alignment > 0.01  # Threshold for "meaningful" guidance

        # Blend scores
        blended_scores = {}
        for action in ig_scores:
            ig_val = ig_scores.get(action, 0.0)
            align_val = alignment_scores.get(action, 0.0)

            if action in ["up", "down"]:
                # Altitude changes: pure IG (HLP doesn't guide altitude)
                blended_scores[action] = w_short * ig_val
            elif not hlp_has_guidance:
                # No clear HLP guidance: pure IG for all actions
                blended_scores[action] = w_short * ig_val
            else:
                # Normal blending with HLP guidance
                blended_scores[action] = w_short * ig_val + w_long * align_val

        best_action = max(blended_scores, key=blended_scores.get)
        return best_action, blended_scores

    def _send_feedback(self, action: str, metrics: Dict):
        """Send feedback to HLP about the action taken."""
        with self._state_lock:
            state = self._state

        uav_pos = state["uav_pos"]
        current_row, current_col = self.camera.convert_xy_ij(
            uav_pos.position[0], uav_pos.position[1], self.camera.grid.center
        )

        covered_mask = state.get("covered_mask")
        coverage = np.mean(covered_mask) if covered_mask is not None else 0.0

        # Check if we reached target region
        guidance = self.intent_bus.get_guidance()
        reached_target = False
        if guidance is not None:
            target_center = guidance.target_center
            dist_to_target = np.sqrt(
                (current_row - target_center[0]) ** 2
                + (current_col - target_center[1]) ** 2
            )
            reached_target = dist_to_target < 20  # Within 20 grid cells

        feedback = LLPFeedback(
            current_position=(current_row, current_col),
            current_altitude=uav_pos.altitude,
            last_action=action,
            coverage_progress=coverage,
            reached_target=reached_target,
            ig_scores=metrics.get("action_scores", metrics.get("ig_scores", {})),
        )

        self.intent_bus.update_feedback(feedback)

        # Send intent for significant events
        if reached_target:
            intent = Intent(
                intent_type=IntentType.REGION_REACHED,
                source="LLP",
                data={"region_id": guidance.target_region_id if guidance else -1},
                priority=5,
            )
            self.intent_bus.send_to_hlp(intent)

    def update_blend_weights(self, w_short: float, w_long: float):
        """Update blend weights (thread-safe)."""
        with self._blend_lock:
            self._w_short = w_short
            self._w_long = w_long


# =============================================================================
# HLP Worker Thread
# =============================================================================


class HLPWorker(threading.Thread):
    """
    High-Level Planner worker thread.

    Runs region analysis and selection asynchronously,
    sending guidance to LLP via the intent bus.
    """

    def __init__(
        self,
        intent_bus: IntentBus,
        camera: Any,
        mcts_params: Dict,
        initial_state: Dict[str, Any],
    ):
        """
        Initialize HLP worker.

        Args:
            intent_bus: Communication channel with LLP
            camera: UAV camera object
            mcts_params: MCTS configuration
            initial_state: Initial planning state
        """
        super().__init__(name="HLP-Worker", daemon=True)

        self.intent_bus = intent_bus
        self.camera = camera
        self.mcts_params = mcts_params

        # State (updated via intent bus)
        self._state = copy.deepcopy(initial_state)
        self._state_lock = threading.Lock()

        # Configuration
        horizon_weights = mcts_params.get("horizon_weights", {})
        self.tile_size = horizon_weights.get("tile_size", [40, 40])
        self.w_coverage = horizon_weights.get("w_coverage", 0.8)
        self.w_fragmentation = horizon_weights.get("w_fragmentation", 0.5)
        self.w_ig = horizon_weights.get("w_ig", 0.8)
        self.w_distance = horizon_weights.get("w_distance", 0.5)

        # Event-driven replanning thresholds (inspired by multi-horizon MCTS)
        # HLP replans when ANY of these conditions are met:
        #   1. Coverage delta exceeds threshold (significant progress)
        #   2. UAV moved far from last replan position (spatial trigger)
        #   3. Target region reached
        #   4. Maximum steps since last replan (fallback)
        self.coverage_delta_threshold = horizon_weights.get(
            "coverage_delta_threshold", 0.05
        )  # 5% coverage change
        self.position_delta_threshold = horizon_weights.get(
            "position_delta_threshold", 30.0
        )  # grid cells
        self.max_steps_between_replans = horizon_weights.get(
            "hlp_replan_frequency", 10
        )  # fallback

        # Hysteresis: current target gets small bonus to avoid ping-pong switching
        self.current_target_bonus = horizon_weights.get("current_target_bonus", 0.15)
        self._current_target_region = None  # Track which region we're targeting

        # Guidance validity duration (used for valid_until timestamp)
        self.guidance_validity = 10.0  # seconds - guidance valid until next replan

        # State tracking for event-driven replanning
        self._last_replan_coverage = 0.0
        self._last_replan_position = None
        self._steps_since_replan = 0

        # Statistics
        self.replan_count = 0
        self.total_planning_time = 0.0

    def run(self):
        """Main HLP loop - continuously analyze and send guidance."""
        logger.info("HLP Worker started")

        while not self.intent_bus.is_shutdown():
            # Process LLP feedback
            self._process_llp_intents()

            # Event-driven replanning check
            should_replan = self._check_replan_triggers()

            if should_replan:
                start_time = time.time()
                self._compute_and_send_guidance()
                planning_time = time.time() - start_time

                self.replan_count += 1
                self.total_planning_time += planning_time
                self._steps_since_replan = 0
            else:
                # Sleep briefly to avoid busy-waiting
                time.sleep(0.05)

        logger.info("HLP Worker shutting down")

    def _check_replan_triggers(self) -> bool:
        """
        Check event-driven replanning conditions.

        Inspired by multi-horizon MCTS: replan when significant state changes occur,
        not on fixed time intervals.

        Returns:
            True if any replan trigger condition is met
        """
        feedback = self.intent_bus.get_feedback()
        if feedback is None:
            return self.replan_count == 0  # Initial replan if no feedback yet

        # Trigger 1: Target region reached
        if feedback.reached_target:
            logger.debug("HLP replan trigger: target region reached")
            return True

        # Trigger 2: Significant coverage progress
        coverage_delta = abs(feedback.coverage_progress - self._last_replan_coverage)
        if coverage_delta >= self.coverage_delta_threshold:
            logger.debug(
                f"HLP replan trigger: coverage delta {coverage_delta:.3f} >= {self.coverage_delta_threshold}"
            )
            return True

        # Trigger 3: UAV moved far from last replan position
        if self._last_replan_position is not None:
            current_pos = np.array(feedback.current_position)
            last_pos = np.array(self._last_replan_position)
            position_delta = np.linalg.norm(current_pos - last_pos)
            if position_delta >= self.position_delta_threshold:
                logger.debug(
                    f"HLP replan trigger: position delta {position_delta:.1f} >= {self.position_delta_threshold}"
                )
                return True

        # Trigger 4: Fallback - max steps since last replan
        self._steps_since_replan += 1
        if self._steps_since_replan >= self.max_steps_between_replans:
            logger.debug(
                f"HLP replan trigger: max steps {self._steps_since_replan} reached"
            )
            return True

        return False

    def update_state(self, state: Dict[str, Any]):
        """Update the state (thread-safe, called from main thread)."""
        with self._state_lock:
            self._state = copy.deepcopy(state)

    def _process_llp_intents(self):
        """Process any pending intents from LLP."""
        while True:
            intent = self.intent_bus.receive_from_llp(timeout=0.01)
            if intent is None:
                break

            if intent.intent_type == IntentType.SHUTDOWN:
                break
            elif intent.intent_type == IntentType.REGION_REACHED:
                logger.debug(f"HLP: LLP reached region {intent.data.get('region_id')}")

    def _compute_and_send_guidance(self):
        """Compute region selection and send guidance to LLP."""
        with self._state_lock:
            state = copy.deepcopy(self._state)

        belief = state["belief"]
        covered_mask = state.get("covered_mask")
        uav_pos = state["uav_pos"]

        if covered_mask is None:
            H_dim, W_dim = belief.shape[:2]
            covered_mask = np.zeros((H_dim, W_dim), dtype=bool)

        H_dim, W_dim = belief.shape[:2]

        # Partition field into regions
        region_map, region_metadata = self._partition_field(belief, covered_mask)

        # Get current position in grid indices
        current_row, current_col = self.camera.convert_xy_ij(
            uav_pos.position[0], uav_pos.position[1], self.camera.grid.center
        )

        # Identify isolated patches
        isolation_scores = self._identify_isolated_patches(
            region_map, covered_mask, region_metadata
        )

        # UNIFIED SCORING: Score = Entropy × (1 + β(1-Coverage)) - γ × Distance
        #
        # This elegantly balances coverage and confidence throughout the flight:
        # - Entropy is ALWAYS the primary driver (high-uncertainty regions are valuable)
        # - Uncovered regions get a BONUS (exploration incentive)
        # - As coverage approaches 1, bonus fades, leaving pure entropy-driven selection
        # - High-entropy covered regions remain attractive (boundaries need refinement)
        #
        # β = w_coverage * 2.0 (exploration bonus weight)
        # γ = w_distance (distance penalty weight)
        exploration_bonus_weight = self.w_coverage * 2.0

        region_scores = {}
        for region_id, metadata in region_metadata.items():
            # Primary metric: average entropy (uncertainty to reduce)
            avg_entropy = metadata.get("entropy", 0.0)

            # Coverage ratio for this region
            coverage = metadata.get("coverage", 0.0)

            center_row, center_col = metadata["center"]
            distance = np.sqrt(
                (current_row - center_row) ** 2 + (current_col - center_col) ** 2
            )
            max_distance = np.sqrt(H_dim**2 + W_dim**2)
            normalized_distance = distance / max_distance

            # UNIFIED FORMULA:
            # Score = Entropy × (1 + β(1-Coverage)) + isolation - γ × Distance
            #
            # - Entropy is ALWAYS the primary driver (even fully covered regions with
            #   high entropy remain attractive for refinement)
            # - Uncovered regions get a BONUS via (1 + β(1-coverage))
            # - As coverage → 1, bonus fades to 1.0 (not zero!)
            # - Isolation bonus prioritizes fragmented patches
            #
            exploration_bonus = 1.0 + exploration_bonus_weight * (1.0 - coverage)
            isolation_bonus = (
                isolation_scores.get(region_id, 0.0) * self.w_fragmentation
            )

            score = (
                avg_entropy * exploration_bonus
                + isolation_bonus
                - self.w_distance * normalized_distance
            )

            # Hysteresis: current target gets bonus to avoid ping-pong switching
            # Only switch when another region is meaningfully better
            if region_id == self._current_target_region:
                score += self.current_target_bonus

            region_scores[region_id] = score

        # Select best region
        if region_scores:
            selected_region_id = max(region_scores, key=region_scores.get)
            selected_metadata = region_metadata[selected_region_id]

            # Update current target tracking
            if selected_region_id != self._current_target_region:
                logger.info(
                    f"HLP: Switching target {self._current_target_region} -> {selected_region_id}"
                )
                self._current_target_region = selected_region_id
        else:
            selected_region_id = 0
            selected_metadata = region_metadata.get(
                0, {"center": (H_dim / 2, W_dim / 2)}
            )

        # Compute suggested blend weight
        coverage_progress = np.mean(covered_mask) if covered_mask is not None else 0.0
        fragmentation_info = analyze_coverage_fragmentation(covered_mask)

        # Suggest higher w_long when fragmentation is high or coverage is low
        base_w_long = 0.4
        frag_boost = (
            FRAGMENTATION_ADJUSTMENT_FACTOR * fragmentation_info["fragmentation_score"]
        )
        coverage_boost = 0.15 * (1.0 - coverage_progress)
        suggested_w_long = min(0.7, base_w_long + frag_boost + coverage_boost)

        # Create guidance
        guidance = HLPGuidance(
            target_region_id=selected_region_id,
            target_center=selected_metadata["center"],
            region_scores=region_scores,
            isolation_scores=isolation_scores,
            blend_weight_suggestion=suggested_w_long,
            valid_until=time.time() + self.guidance_validity,
            region_metadata=region_metadata,
        )

        # Send guidance
        self.intent_bus.update_guidance(guidance)

        # Update state tracking for event-driven replanning
        self._last_replan_coverage = coverage_progress
        current_row, current_col = self.camera.convert_xy_ij(
            uav_pos.position[0], uav_pos.position[1], self.camera.grid.center
        )
        self._last_replan_position = (current_row, current_col)

        # Send intent for significant events
        if fragmentation_info["fragmentation_score"] > 0.3:
            intent = Intent(
                intent_type=IntentType.FRAGMENTATION_ALERT,
                source="HLP",
                data={
                    "score": fragmentation_info["fragmentation_score"],
                    "num_patches": fragmentation_info["num_patches"],
                },
                priority=3,
            )
            self.intent_bus.send_to_llp(intent)

        logger.debug(
            f"HLP: Selected region {selected_region_id}, "
            f"center={selected_metadata['center']}, "
            f"score={region_scores.get(selected_region_id, 0):.3f}"
        )

    def _partition_field(
        self, belief: np.ndarray, covered_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[int, Dict]]:
        """Partition field into regions."""
        H_dim, W_dim = belief.shape[:2]
        tile_h, tile_w = self.tile_size

        n_rows = int(np.ceil(H_dim / tile_h))
        n_cols = int(np.ceil(W_dim / tile_w))

        region_map = np.zeros((H_dim, W_dim), dtype=int)
        region_metadata = {}

        region_id = 0
        for i in range(n_rows):
            for j in range(n_cols):
                row_start = i * tile_h
                row_end = min((i + 1) * tile_h, H_dim)
                col_start = j * tile_w
                col_end = min((j + 1) * tile_w, W_dim)

                region_map[row_start:row_end, col_start:col_end] = region_id

                region_belief = belief[row_start:row_end, col_start:col_end]
                region_covered = covered_mask[row_start:row_end, col_start:col_end]

                entropy = float(np.mean(H(region_belief[:, :, 1])))
                coverage = float(np.mean(region_covered))

                region_metadata[region_id] = {
                    "bounds": ((row_start, row_end), (col_start, col_end)),
                    "center": ((row_start + row_end) / 2, (col_start + col_end) / 2),
                    "entropy": entropy,
                    "coverage": coverage,
                }

                region_id += 1

        return region_map, region_metadata

    def _identify_isolated_patches(
        self,
        region_map: np.ndarray,
        covered_mask: np.ndarray,
        region_metadata: Dict[int, Dict],
    ) -> Dict[int, float]:
        """Identify which regions contain isolated patches."""
        fragmentation_info = analyze_coverage_fragmentation(covered_mask)
        num_patches = fragmentation_info["num_patches"]

        if num_patches <= 1:
            return {}

        from scipy import ndimage

        uncovered_mask = ~covered_mask
        structure = ndimage.generate_binary_structure(2, 2)
        labeled_array, _ = ndimage.label(uncovered_mask, structure=structure)

        patch_sizes = fragmentation_info["patch_sizes"]
        total_uncovered = fragmentation_info["total_uncovered"]

        region_isolation_scores = {}

        for patch_id in range(1, num_patches + 1):
            patch_mask = labeled_array == patch_id
            patch_size = patch_sizes[patch_id - 1]

            patch_isolation = 1.0 - (patch_size / total_uncovered)

            regions_in_patch = {}
            for region_id in region_metadata.keys():
                region_mask = region_map == region_id
                overlap = np.logical_and(patch_mask, region_mask)
                overlap_size = np.sum(overlap)

                if overlap_size > 0:
                    regions_in_patch[region_id] = overlap_size

            if regions_in_patch:
                best_region = max(regions_in_patch, key=regions_in_patch.get)
                current_score = region_isolation_scores.get(best_region, 0.0)
                region_isolation_scores[best_region] = max(
                    current_score, patch_isolation
                )

        return region_isolation_scores


# =============================================================================
# Threaded Dual-Horizon Planner
# =============================================================================


class ThreadedDualHorizonPlanner:
    """
    Main interface for threaded dual-horizon planning.

    Manages LLP and HLP worker threads and coordinates
    action selection via the intent bus.
    """

    def __init__(
        self,
        uav_camera: Any,
        conf_dict: Optional[Dict] = None,
        mcts_params: Optional[Dict] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the threaded planner.

        Args:
            uav_camera: UAV camera object
            conf_dict: Sensor confusion matrix
            mcts_params: MCTS configuration
            initial_state: Initial planning state
        """
        self.camera = uav_camera
        self.conf_dict = conf_dict or None
        self.mcts_params = mcts_params or {}

        # Create intent bus
        self.intent_bus = IntentBus()

        # Set up detailed file logging (same format as non-threaded)
        experiment_name = mcts_params.get("experiment_name") if mcts_params else None
        # Ensure logs go to subdirectory
        import os

        log_dir = "logs/dual_horizon"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = setup_dual_horizon_logger(
            log_dir=log_dir,
            experiment_name=(
                f"threaded_{experiment_name}" if experiment_name else "threaded_default"
            ),
        )
        self._detailed_logger = logging.getLogger("dual_horizon_detailed")

        # Create initial state if not provided
        if initial_state is None:
            initial_state = {"uav_pos": None, "belief": None, "covered_mask": None}

        # Create worker threads (not started yet)
        self._llp_worker: Optional[LLPWorker] = None
        self._hlp_worker: Optional[HLPWorker] = None
        self._initial_state = initial_state
        self._started = False

        # For visualization
        self.current_region_metadata = {}
        self.current_selected_region = 0
        self.current_region_scores = {}

        # Statistics
        self.step_count = 0

    def start(self, state: Dict[str, Any] = None):
        """
        Start the worker threads.

        Args:
            state: Initial state (optional, uses constructor state if not provided)
        """
        if self._started:
            logger.warning("Threaded planner already started")
            return

        if state is not None:
            self._initial_state = state

        # Create and start workers
        self._llp_worker = LLPWorker(
            intent_bus=self.intent_bus,
            camera=self.camera,
            conf_dict=self.conf_dict,
            mcts_params=self.mcts_params,
            initial_state=self._initial_state,
        )

        self._hlp_worker = HLPWorker(
            intent_bus=self.intent_bus,
            camera=self.camera,
            mcts_params=self.mcts_params,
            initial_state=self._initial_state,
        )

        self._llp_worker.start()
        self._hlp_worker.start()
        self._started = True

        logger.info("Threaded Dual-Horizon Planner started")

    def stop(self):
        """Stop the worker threads."""
        if not self._started:
            return

        self.intent_bus.shutdown()

        # Wait for threads to finish
        if self._llp_worker is not None:
            self._llp_worker.join(timeout=2.0)
        if self._hlp_worker is not None:
            self._hlp_worker.join(timeout=2.0)

        self._started = False
        logger.info("Threaded Dual-Horizon Planner stopped")

    def select_action(
        self, state: Dict[str, Any], strategy: str = "dual"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select next action using threaded planners.

        Args:
            state: Current planning state
            strategy: Planning strategy (only 'dual' supported in threaded mode)

        Returns:
            Tuple of (action, metrics)
        """
        if not self._started:
            self.start(state)

        # Update HLP with latest state
        self._hlp_worker.update_state(state)

        # Request action from LLP
        action, metrics = self._llp_worker.request_action(state)

        # Update visualization data from latest guidance
        guidance = self.intent_bus.get_guidance()
        if guidance is not None:
            self.current_region_metadata = guidance.region_metadata
            self.current_selected_region = guidance.target_region_id
            self.current_region_scores = guidance.region_scores

        self.step_count += 1

        # Add bus stats to metrics
        metrics["bus_stats"] = self.intent_bus.get_stats()
        metrics["llp_step_count"] = (
            self._llp_worker.step_count if self._llp_worker else 0
        )
        metrics["hlp_replan_count"] = (
            self._hlp_worker.replan_count if self._hlp_worker else 0
        )

        # Detailed logging
        self._log_step(state, action, metrics, guidance)

        return action, metrics

    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics."""
        stats = {
            "step_count": self.step_count,
            "bus_stats": self.intent_bus.get_stats(),
        }

        if self._llp_worker is not None:
            stats["llp"] = {
                "step_count": self._llp_worker.step_count,
                "avg_planning_time": (
                    self._llp_worker.total_planning_time
                    / max(self._llp_worker.step_count, 1)
                ),
            }

        if self._hlp_worker is not None:
            stats["hlp"] = {
                "replan_count": self._hlp_worker.replan_count,
                "avg_planning_time": (
                    self._hlp_worker.total_planning_time
                    / max(self._hlp_worker.replan_count, 1)
                ),
            }

        return stats

    def _log_step(
        self,
        state: Dict[str, Any],
        action: str,
        metrics: Dict[str, Any],
        guidance: Optional[HLPGuidance],
    ):
        """Log detailed planning step information."""
        log = self._detailed_logger

        log.info(f"\n{'='*60}")
        log.info(f"STEP {self.step_count}")
        log.info(f"{'='*60}")

        # UAV Position
        uav_pos = state.get("uav_pos")
        if uav_pos is not None:
            log.info(
                f"UAV Position: ({uav_pos.position[0]:.2f}, {uav_pos.position[1]:.2f}, {uav_pos.altitude:.2f})"
            )

        # Coverage
        covered_mask = state.get("covered_mask")
        if covered_mask is not None:
            coverage = float(np.mean(covered_mask))
            log.info(f"Coverage: {coverage:.1%}")

        # HLP Guidance
        if guidance is not None:
            log.info(f"\n[HLP] Target Region: {guidance.target_region_id}")
            log.info(f"[HLP] Target Center: {guidance.target_center}")
            log.info(f"[HLP] Region Scores (top 5):")
            sorted_regions = sorted(
                guidance.region_scores.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for rid, score in sorted_regions:
                meta = guidance.region_metadata.get(rid, {})
                cov = meta.get("coverage", 0)
                log.info(f"      Region {rid}: {score:.4f} (coverage: {cov:.1%})")

        # LLP Action Scores
        log.info(f"\n[LLP] Action Scores:")
        blended_scores = metrics.get("blended_scores", metrics.get("action_scores", {}))
        for act, score in sorted(
            blended_scores.items(), key=lambda x: x[1], reverse=True
        ):
            marker = " <--" if act == action else ""
            log.info(f"      {act:8s}: {score:.4f}{marker}")

        # Selected Action
        log.info(f"\n[SELECTED] {action}")

        # Bus Stats
        bus_stats = metrics.get("bus_stats", {})
        log.info(
            f"\n[BUS] Guidance updates: {bus_stats.get('guidance_updates', 0)}, "
            f"HLP replans: {metrics.get('hlp_replan_count', 0)}"
        )

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
