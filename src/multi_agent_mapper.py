import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from mapper_LBP import OccupancyMap


class MultiAgentMapper:
    """
    Multi-agent mapper that wraps multiple OccupancyMap instances
    and handles ONLY:
      - per-agent mapping (via OccupancyMap: OG + LBP)
      - news belief computation (BS / BM)
      - unary-only Bayesian fusion of news between agents

    Design:
    - Each agent i has its own OccupancyMap (local OG+LBP, no sharing).
    - News beliefs are stored as news[sender, receiver, i, j] (BM mode),
      or only on the diagonal news[i, i, i, j] (BS mode).
    - Fusion uses product-of-experts / Bayesian re-normalization, as in the paper.
    - Pairwise correlations are handled ONLY inside OccupancyMap.propagate_messages().
      This class never runs LBP on its own.

    Modes:
    - news_mode = "BS":
        Each agent maintains ONE news map, shared to all neighbors.
        Shape: news_map_beliefs[num_agents, num_agents, H, W], but only (i,i) written.
    - news_mode = "BM":
        Each agent maintains SEPARATE news maps per neighbor.
        Shape: news_map_beliefs[num_agents, num_agents, H, W], off-diagonal used.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        num_agents: int,
        conf_dict: Optional[Dict] = None,
        correlation_type: Optional[str] = None,
        news_mode: str = "BM",  # "BS" or "BM"
        lbp_iterations: int = 5,
    ):
        """
        Args:
            grid_size: (rows, cols) of map.
            num_agents: number of agents.
            conf_dict: sensor model lookup (altitude -> (sigma0, sigma1)).
            correlation_type: 'equal' | 'biased' | 'adaptive' used by OccupancyMap.
            news_mode: "BS" (single news per agent) or "BM" (per-neighbor news).
            lbp_iterations: number of LBP iterations in local mapping.
        """
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.news_mode = news_mode
        self.lbp_iterations = lbp_iterations

        # One local OccupancyMap per agent (per-agent OG + LBP)
        self.maps: List[OccupancyMap] = [
            OccupancyMap(
                grid_size, conf_dict=conf_dict, correlation_type=correlation_type
            )
            for _ in range(num_agents)
        ]

        # News beliefs: [sender, receiver, H, W]
        # We keep full [num_agents, num_agents, H, W] for simplicity.
        # BS mode uses only diagonal [i, i, :, :].
        H, W = grid_size
        self.news_map_beliefs = np.full(
            (num_agents, num_agents, H, W), 0.5, dtype=float
        )

        # Small epsilon for numerical stability
        self.eps = 1e-20

        # Statistics
        self._fusion_count = 0

    # -------------------------------------------------------------------------
    # BASIC ACCESSORS
    # -------------------------------------------------------------------------

    def get_agent_belief(self, agent_id: int) -> np.ndarray:
        """Return current belief map for a specific agent."""
        return self.maps[agent_id].get_belief().copy()

    def get_all_beliefs(self) -> np.ndarray:
        """Return all agents' beliefs stacked as [H, W, num_agents]."""
        H, W = self.grid_size
        beliefs = np.zeros((H, W, self.num_agents), dtype=float)
        for i in range(self.num_agents):
            beliefs[:, :, i] = self.maps[i].get_belief()
        return beliefs

    def get_global_fused_belief(self) -> np.ndarray:
        """
        Product-of-experts fusion across all agents:

            P(m=1 | all) ∝ ∏_i P_i(m=1)
        """
        beliefs = self.get_all_beliefs()
        prod_occ = np.prod(beliefs, axis=2)
        prod_free = np.prod(1.0 - beliefs, axis=2)
        fused = prod_occ / (prod_occ + prod_free + self.eps)
        return np.clip(fused, 0.001, 0.999)

    def reset(self):
        """Reset all agent maps and news beliefs."""
        for m in self.maps:
            m.reset()
        self.news_map_beliefs[:] = 0.5

    # -------------------------------------------------------------------------
    # LOCAL MAPPING (per-agent, single-step OG + LBP)
    # -------------------------------------------------------------------------

    def local_mapping_update(
        self,
        agent_id: int,
        fp_vertices_ij: Dict[str, np.ndarray],
        z: np.ndarray,
        uav_pos: Any,
    ):
        """
        Perform a single local mapping step for one agent:

            OG Bayesian update  -> OccupancyMap.update_belief_OG
            LBP spatial smoothing -> OccupancyMap.propagate_messages

        Args:
            agent_id: which agent to update.
            fp_vertices_ij: bounding box indices (as produced by OccupancyMap.get_indices).
            z: binary observation patch.
            uav_pos: UAV position object (must have altitude attribute).
        """
        omap = self.maps[agent_id]

        # 1) OG update (sets sigma0, sigma1 internally)
        omap.update_belief_OG(fp_vertices_ij, z, uav_pos)

        # 2) LBP propagation for spatial consistency
        if self.lbp_iterations > 0:
            omap.propagate_messages(
                fp_vertices_ij,
                z,
                max_iterations=self.lbp_iterations,
                correlation_type=omap.correlation_type,
            )

    # -------------------------------------------------------------------------
    # NEWS BELIEF UPDATE (no fusion yet)
    # -------------------------------------------------------------------------

    def _update_news_for_target_patch(
        self,
        sender_id: int,
        target_id: int,
        fp_vertices_ij: Dict[str, np.ndarray],
        z: np.ndarray,
        sigma0: float,
        sigma1: float,
    ):
        """
        Low-level helper: OG update of news_map_beliefs[sender, target] on a patch.
        This mirrors the OG update logic, but operates on news beliefs instead
        of on the main map.

        Args:
            sender_id: agent that made the observation.
            target_id: agent for whom this news is intended (can be sender_id).
        """
        I, J = 0, 1

        likelihood_m_zero = np.where(z == 0, 1 - sigma0, sigma0)
        likelihood_m_one = np.where(z == 0, sigma1, 1 - sigma1)

        prior_news = self.news_map_beliefs[
            sender_id,
            target_id,
            fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
            fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
        ]

        post_m_zero = likelihood_m_zero * (1.0 - prior_news)
        post_m_one = likelihood_m_one * prior_news

        denom = post_m_zero + post_m_one + self.eps
        post_m_one_norm = post_m_one / denom
        post_m_one_norm = np.clip(post_m_one_norm, 0.001, 0.999)

        self.news_map_beliefs[
            sender_id,
            target_id,
            fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
            fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
        ] = post_m_one_norm

    def update_news_belief(
        self,
        agent_id: int,
        fp_vertices_ij: Dict[str, np.ndarray],
        z: np.ndarray,
    ):
        """
        Update ONLY the news beliefs for a given agent,
        after its local mapping (OG+LBP) has been done.

        Uses sigma0, sigma1 from the agent's OccupancyMap
        (set during the last update_belief_OG call).

        BS mode:
            - Update news_map_beliefs[agent_id, agent_id, ...]
        BM mode:
            - Update news_map_beliefs[agent_id, target_id, ...] for all target_id != agent_id
        """
        omap = self.maps[agent_id]
        sigma0, sigma1 = omap.sigma0, omap.sigma1

        if sigma0 is None or sigma1 is None:
            # Mapping not updated yet; nothing to do
            return

        if self.news_mode == "BS":
            # Single news per agent (diagonal only)
            self._update_news_for_target_patch(
                sender_id=agent_id,
                target_id=agent_id,
                fp_vertices_ij=fp_vertices_ij,
                z=z,
                sigma0=sigma0,
                sigma1=sigma1,
            )
        else:
            # BM mode: separate news per neighbor
            for target_id in range(self.num_agents):
                if target_id == agent_id:
                    continue
                self._update_news_for_target_patch(
                    sender_id=agent_id,
                    target_id=target_id,
                    fp_vertices_ij=fp_vertices_ij,
                    z=z,
                    sigma0=sigma0,
                    sigma1=sigma1,
                )

    # -------------------------------------------------------------------------
    # FUSION (unary-only Bayesian combination)
    # -------------------------------------------------------------------------

    def _fuse_two_beliefs(self, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
        """
        Product-of-experts fusion of two binary occupancy beliefs:

            P(m=1 | z1,z2) =
                [b1 * b2] / [b1*b2 + (1-b1)*(1-b2)]

        Both inputs are probabilities in [0,1].
        """
        mul = b1 * b2
        denom = mul + (1.0 - b1) * (1.0 - b2) + self.eps
        fused = mul / denom
        return np.clip(fused, 0.001, 0.999)

    def fuse_news_from_sender(
        self,
        sender_id: int,
        neighbor_ids: List[int],
    ):
        """
        Fuse news from a given sender into each neighbor's local map.
        This does NOT overwrite sender's map, only neighbors'.

        BS mode:
            - news = news_map_beliefs[sender, sender, :, :]
        BM mode:
            - news for neighbor j = news_map_beliefs[sender, j, :, :]

        After fusion, the used news entries are reset to 0.5 (neutral).
        """
        if not neighbor_ids:
            return

        for neighbor_id in neighbor_ids:
            # Skip self fusion; if desired, caller can include agent_id in neighbor_ids
            if neighbor_id == sender_id:
                continue

            if self.news_mode == "BS":
                news = self.news_map_beliefs[sender_id, sender_id, :, :]
            else:
                news = self.news_map_beliefs[sender_id, neighbor_id, :, :]

            neighbor_belief = self.maps[neighbor_id].map_beliefs

            fused = self._fuse_two_beliefs(neighbor_belief, news)
            self.maps[neighbor_id].map_beliefs = fused

            # Track fusion count
            self._fusion_count += 1

            # Reset used news to neutral
            if self.news_mode == "BS":
                # Reset sender's single news if it was used at least once
                self.news_map_beliefs[sender_id, sender_id, :, :] = 0.5
            else:
                self.news_map_beliefs[sender_id, neighbor_id, :, :] = 0.5

    def fuse_news_with_neighbors(self, agent_id: int, neighbor_ids: List[int]):
        """
        Fuse news from agent_id into its neighbors.
        Alias for fuse_news_from_sender for compatibility.

        Args:
            agent_id: ID of agent whose news to fuse
            neighbor_ids: List of neighbor agent IDs
        """
        self.fuse_news_from_sender(agent_id, neighbor_ids)

    # -------------------------------------------------------------------------
    # HIGH-LEVEL SYNCHRONOUS PIPELINES
    # -------------------------------------------------------------------------

    def update_all_news(self, agent_observations: Dict[int, Dict[str, Any]]):
        """
        PHASE 1 (BATCH SYNCHRONOUS): Update news beliefs for ALL agents.

        This is the batch version that updates all agents' news beliefs
        before any fusion occurs. Follows the paper's synchronous pattern.

        Note: Local mapping (OG+LBP) should be done by each agent's
        OccupancyMap BEFORE calling this method.

        Args:
            agent_observations: Dict mapping agent_id to observation dict with:
                - 'fp_ij' or 'fp_vertices_ij': Footprint indices
                - 'submap' or 'z': Binary observation array
                - 'sigmas': Tuple (sigma0, sigma1) or None (uses defaults)
        """
        for agent_id, obs in agent_observations.items():
            # Handle different key names for compatibility
            fp_ij = obs.get("fp_ij")
            if fp_ij is None:
                fp_ij = obs.get("fp_vertices_ij")
            
            z = obs.get("submap")
            if z is None:
                z = obs.get("z")
            
            sigmas = obs.get("sigmas")

            if fp_ij is None or z is None:
                continue

            # Set sigmas on the agent's map if provided
            if sigmas is not None:
                self.maps[agent_id].sigma0 = sigmas[0]
                self.maps[agent_id].sigma1 = sigmas[1]

            # Update news belief only (no fusion yet)
            self.update_news_belief(agent_id, fp_ij, z)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the mapper state."""
        return {
            "num_agents": self.num_agents,
            "news_mode": self.news_mode,
            "lbp_iterations": self.lbp_iterations,
            "news_fusions": self._fusion_count,
        }

    def get_fused_belief(self) -> np.ndarray:
        """Alias for get_global_fused_belief for compatibility."""
        return self.get_global_fused_belief()

    def step_all_agents_synchronous(
        self,
        agent_observations: Dict[int, Dict[str, Any]],
        neighbor_map: Dict[int, List[int]],
    ):
        """
        Run one full mapping + sharing step for all agents in a synchronous way.

        Pipeline:
            1) Local mapping (OG + LBP) for ALL agents
            2) Update news beliefs for ALL agents (no fusion yet)
            3) Fuse news from ALL senders into ALL neighbors

        Args:
            agent_observations: dict[agent_id] -> {
                'fp_vertices_ij': dict with 'ul','bl','ur','br' (np.array pairs),
                'z': np.ndarray of binary observations,
                'uav_pos': object with altitude attribute
            }
            neighbor_map: dict[agent_id] -> list of neighbor agent IDs
                          (e.g., from communication range or complete graph)
        """
        # 1) Local mapping
        for agent_id, obs in agent_observations.items():
            fp_ij = obs["fp_vertices_ij"]
            z = obs["z"]
            uav_pos = obs["uav_pos"]
            self.local_mapping_update(agent_id, fp_ij, z, uav_pos)

        # 2) Update news beliefs (no fusion)
        for agent_id, obs in agent_observations.items():
            fp_ij = obs["fp_vertices_ij"]
            z = obs["z"]
            self.update_news_belief(agent_id, fp_ij, z)

        # 3) Fuse news from each sender into its neighbors
        for sender_id in range(self.num_agents):
            neighbors = neighbor_map.get(sender_id, [])
            if neighbors:
                self.fuse_news_from_sender(sender_id, neighbors)
