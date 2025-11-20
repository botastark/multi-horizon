# high_planner.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import numpy as np

# Reuse your helpers
from helper import H, expected_posterior, uav_position  # to match your usage pattern

# Reuse your UAV/camera interfaces the same way mcts.py does:
# - uav_position()
# - Camera.get_range() signature


# --------------------------
# Clusterization utilities
# --------------------------
def build_clusters(belief: np.ndarray, tile_h: int = 20, tile_w: int = 20):
    """
    belief: (H, W, 2), probs in [:,:,1]
    Returns:
        cid_of_cell: (H, W) int cluster-id per cell
        clusters: {cid: {'cells': list[(i,j)], 'entropy': float, 'center_xy': (x,y)}}
    """
    Hh, Ww = belief.shape[:2]
    tile_h = max(1, Hh // 10)
    tile_w = max(1, Ww // 10)
    cid_of_cell = -np.ones((Hh, Ww), dtype=np.int32)

    clusters: Dict[int, dict] = {}
    cid = 0
    for i0 in range(0, Hh, tile_h):
        for j0 in range(0, Ww, tile_w):
            i1 = min(Hh, i0 + tile_h)
            j1 = min(Ww, j0 + tile_w)
            cells = [(i, j) for i in range(i0, i1) for j in range(j0, j1)]
            if not cells:
                continue
            # entropy mass in this tile
            probs = belief[i0:i1, j0:j1, 1]
            ent_mass = float(np.sum(H(probs)))
            clusters[cid] = {
                "cells": cells,
                "entropy": ent_mass,
                "center_ij": ((i0 + i1 - 1) / 2.0, (j0 + j1 - 1) / 2.0),
            }
            for i, j in cells:
                cid_of_cell[i, j] = cid
            cid += 1
    print(f"Built {cid} clusters: each ~{tile_h}x{tile_w} cells")
    return cid_of_cell, clusters


# --------------------------
# Intent extraction from your LL plan
# --------------------------
@dataclass
class LLIntent:
    end_xy: Tuple[float, float]
    duration: float
    footprints: List[Tuple[np.ndarray, float]]  # list of (mask_ij, dt)
    ig_per_sec: float


def make_ll_intent(
    uav, belief: np.ndarray, action_seq: List[str], dt_per_step: float = 1.0
) -> LLIntent:
    """
    UAV/Camera to simulate the 5-step exploitation plan and compute:
        - which cells are observed at each step
        - expected IG from each footprint
    """

    x = uav.get_x()
    total_ig = 0.0
    footprints = []
    M = belief.copy()

    for a in action_seq:
        x = uav_position(uav.x_future(a, x=x))
        [[imin, imax], [jmin, jmax]] = uav.get_range(
            position=x.position, altitude=x.altitude, index_form=True
        )
        obs = M[imin:imax, jmin:jmax, 1]
        # compute expected posterior pieces (same as in mcts.py rollout)
        Pz0, Pz1, p_m1_z0, p_m1_z1 = expected_posterior(obs)
        # IG for this footprint (same as your compute_reward logic)
        curr_H = H(obs)
        exp_H = Pz0 * H(p_m1_z0) + Pz1 * H(p_m1_z1)
        ig = float(np.sum(curr_H - exp_H))
        total_ig += ig
        # store mask for later HP application
        mask = np.zeros(M.shape[:2], dtype=bool)
        mask[imin:imax, jmin:jmax] = True
        footprints.append((mask, dt_per_step))
        # expected map update (so later footprints use updated M like your rollout)
        exp_post = Pz1 * p_m1_z1 + Pz0 * p_m1_z0
        M[imin:imax, jmin:jmax, 1] = exp_post
        M[imin:imax, jmin:jmax, 0] = 1.0 - exp_post

    duration = len(action_seq) * dt_per_step
    ig_per_sec = total_ig / max(duration, 1e-6)
    return LLIntent(
        end_xy=tuple(x.position),
        duration=duration,
        footprints=footprints,
        ig_per_sec=ig_per_sec,
    )


# --------------------------
# High-level evaluator g2
# --------------------------
@dataclass
class HPWeights:
    w_dH: float = 1.0
    w_time: float = 0.1
    w_over: float = 10.0


@dataclass
class RobotHL:
    xy: Tuple[float, float]
    speed: float
    budget: float
    busy_until: float = 0.0

    from dataclasses import dataclass


from typing import List, Tuple, Dict, Optional


@dataclass
class HLGoal:
    cid: int
    center_xy: Tuple[float, float]
    eta_hint: float  # nominal arrival/serve time hint (seconds)


@dataclass
class HLIntent:
    """High-level intent (HP -> LL): ordered cluster goals."""

    goals: List[HLGoal]  # priority queue; first is current target
    gamma: float = 0.999  # for D-UCT consistency if needed


def euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eval_g2_single_candidate(
    robots: List[RobotHL],
    clusters: Dict[int, dict],
    cid_of_cell: np.ndarray,
    ll_intent: Optional[LLIntent],
    candidate_cid: Optional[int],
    weights: HPWeights,
) -> float:
    # shallow copies
    robots = [RobotHL(**vars(r)) for r in robots]
    clusters = {cid: {**c} for cid, c in clusters.items()}
    over_any = False

    # 1) Apply LL intent footprints (reduce cluster entropies)
    if ll_intent is not None:
        # consume time on earliest robot
        r = min(robots, key=lambda rr: rr.busy_until)
        r.busy_until += ll_intent.duration
        r.budget = max(0.0, r.budget - ll_intent.duration)
        r.xy = ll_intent.end_xy

        for mask, dt in ll_intent.footprints:
            # expected IG mass from this footprint is already counted in ll_intent.ig_per_sec,
            # but we only need to *distribute* its effect over clusters for dH accounting.
            # Simple distribution: proportionally to per-cluster mask pixel count.
            idxs = np.where(mask)
            if idxs[0].size == 0:
                continue
            # Count pixels per cluster
            cids, counts = np.unique(cid_of_cell[idxs], return_counts=True)
            # Ignore -1 (out of bounds)
            pix_total = counts.sum()
            if pix_total == 0:
                continue
            ig = ll_intent.ig_per_sec * dt  # IG for this step
            for k, cnt in zip(cids, counts):
                if k < 0:
                    continue
                share = ig * (cnt / pix_total)
                clusters[k]["entropy"] = max(0.0, clusters[k]["entropy"] - share)

    # 2) If candidate cluster provided, assign it to earliest-free robot
    H_before = sum(c["entropy"] for c in clusters.values())
    if candidate_cid is not None:
        r = min(robots, key=lambda rr: rr.busy_until)
        c = clusters[candidate_cid]
        travel_t = euclid(r.xy, ij_to_xy(c["center_ij"])) / max(r.speed, 1e-6)
        cover_t = c["entropy"] / max(ll_intent.ig_per_sec if ll_intent else 1.0, 1e-6)
        used = travel_t + cover_t
        if used > r.budget:
            # partial
            time_for_cover = max(0.0, r.budget - travel_t)
            partial_drop = time_for_cover * (ll_intent.ig_per_sec if ll_intent else 1.0)
            c["entropy"] = max(0.0, c["entropy"] - partial_drop)
            used = travel_t + max(0.0, time_for_cover)
            over_any = True
        else:
            c["entropy"] = 0.0
        r.busy_until += used
        r.budget = max(0.0, r.budget - used)
        r.xy = ij_to_xy(c["center_ij"])

    # 3) Greedy tail (optional for MVP): skip for first runs to keep code minimal
    finish_mean = sum(r.busy_until for r in robots) / len(robots)

    H_after = sum(c["entropy"] for c in clusters.values())
    dH_norm = (H_before - H_after) / max(H_before, 1e-6)
    reward = (
        (weights.w_dH * dH_norm)
        - (weights.w_time * finish_mean)
        - (weights.w_over if over_any else 0.0)
    )
    return reward


# Simple ij->xy using grid indices; replace with your real mapping if needed
def ij_to_xy(center_ij: Tuple[float, float]) -> Tuple[float, float]:
    return (center_ij[1], center_ij[0])  # (x, y) ~ (j, i)


# --------------------------
# Tiny MCTS for HP
# --------------------------
class HPNode:
    def __init__(
        self, robots, clusters, cid_of_cell, ll_intent, parent=None, action=None
    ):
        self.robots = [RobotHL(**vars(r)) for r in robots]
        self.clusters = {cid: {**c} for cid, c in clusters.items()}
        self.cid_of_cell = cid_of_cell
        self.ll_intent = ll_intent
        self.parent = parent
        self.action = action
        self.children: Dict[int, HPNode] = {}
        self.untried: List[int] = [
            cid for cid, c in self.clusters.items() if c["entropy"] > 1e-6
        ]
        self.visits = 0
        self.value = 0.0

    def is_terminal(self):
        return all(c["entropy"] <= 1e-6 for c in self.clusters.values()) or all(
            r.budget <= 1e-6 for r in self.robots
        )

    def expand(self):
        cid = self.untried.pop()
        child = HPNode(
            self.robots,
            self.clusters,
            self.cid_of_cell,
            self.ll_intent,
            parent=self,
            action=cid,
        )
        self.children[cid] = child
        return child

    def best_child(self, c=1.0, weights=HPWeights()):
        import math

        best, best_ucb = None, -1e18
        for cid, ch in self.children.items():
            if ch.visits == 0:
                ucb = float("inf")
            else:
                exploit = ch.value / ch.visits
                explore = c * math.sqrt(math.log(max(1, self.visits)) / ch.visits)
                ucb = exploit + explore
            if ucb > best_ucb:
                best_ucb, best = ucb, ch
        return best


class HighPlanner:
    def __init__(
        self,
        weights: HPWeights = HPWeights(),
        ucb_c: float = 1.0,
        iterations: int = 400,
    ):
        self.W = weights
        self.ucb_c = ucb_c
        self.iterations = iterations

    def plan(
        self,
        robots: List[RobotHL],
        clusters: Dict[int, dict],
        cid_of_cell: np.ndarray,
        ll_intent: Optional[LLIntent],
    ) -> Optional[int]:
        root = HPNode(robots, clusters, cid_of_cell, ll_intent)
        if not root.untried:
            return None
        for _ in range(self.iterations):
            node = root
            # select
            while not node.is_terminal() and not node.untried and node.children:
                node = node.best_child(c=self.ucb_c, weights=self.W)
            # expand
            if (not node.is_terminal()) and node.untried:
                node = node.expand()
            # rollout = single-step eval of candidate (cheap)
            cand = node.action if node.action is not None else None
            r = eval_g2_single_candidate(
                node.robots,
                node.clusters,
                node.cid_of_cell,
                node.ll_intent,
                cand,
                self.W,
            )
            # backup
            while node is not None:
                node.visits += 1
                node.value += r
                node = node.parent
        # pick best by visits/avg
        if not root.children:
            return None
        best = max(root.children.values(), key=lambda ch: ch.visits)
        return best.action
