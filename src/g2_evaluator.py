"""
g2 evaluator: remaining coverage effort, measured in agent-steps.

MH-Dec-MCTS splits the objective as g = g1 + g2 (Seiler et al., "Multi-Horizon
Multi-Agent Planning Using Decentralised Monte Carlo Tree Search"):

  g1  short-horizon information gain of the LL plan (computed in the LLP)
  g2  long-horizon mission estimate (computed here)

Mission goal for g2: every cell of the field must be OBSERVED AT LEAST ONCE
(by any agent, at any altitude). g2 estimates how many agent-steps the team
still needs to reach that state, so LOWER IS BETTER. Planners use
"steps saved" (baseline g2 minus candidate g2) as their reward term.

Motion model (everything priced at minimum altitude; altitude changes ignored)
---------------------------------------------------------------------------
* One lateral move = `step_cells` grid cells (25 = 3.125 m); moves are 4-connected.
* The field is split into BLOCKS of step_cells x step_cells cells. A block is
  "open" if any of its cells is still unobserved.
* Agents stand on LATTICE POSITIONS (block corners). A footprint at minimum
  altitude centred on lattice point (a, b) observes the kh x kh blocks on each
  side of it, kh = round(fp_half / step_cells) (= 1: a 2x2 block window).

Remaining work for a set of open blocks = travel of a tour that visits enough
lattice positions to see every open block:
  1. pick covering positions greedily (most open blocks first);
  2. connect them with a minimum spanning tree in true step distance
     (L1 / step_cells), rooted at the agents' positions (multi-root MST).
Because covered ground can still be crossed at its true distance, covering the
middle of an uncovered strip does not increase the estimate, while leaving an
isolated spot behind adds the detour needed to come back for it.

Team estimate g2 (sum of agent-steps, not makespan):
  1. Claimed work: each agent walks its HL region sequence from its start:
     per still-open region, `reach` (moves until its footprint can see the
     region's open bounding box) + `inner` (tour over the region's own covering
     positions, 0 for a one-block region). The agent then stands where it
     arrived. A region claimed by two agents is paid twice.
  2. Unclaimed work: all open blocks of regions nobody claimed, via the
     covering-positions MST rooted at every agent's end position.

Known approximations: altitude is ignored (a high-flying agent covers more per
move than assumed); greedy cover and MST approximate the optimal tour; per-region
bounding boxes make `reach` optimistic for regions with scattered holes.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Footprint = Tuple[int, int, int, int]  # (imin, imax, jmin, jmax), max exclusive
GridPos = Tuple[float, float]  # (i, j) in grid cells


def camera_step_geometry(camera) -> Tuple[float, float]:
    """
    Movement/footprint geometry in grid cells.

    Returns:
        (step_cells, fp_half): lateral move length in cells, and half-width in
        cells of the footprint at minimum altitude (same rounding as
        Camera.get_range).
    """
    length = camera.grid.length
    step_cells = camera.xy_step / length
    half_m = round(camera.h_range[0] * math.tan(np.deg2rad(camera.fov) / 2) / length) * length
    return float(step_cells), float(half_m / length)


def auto_g2_weight(camera, conf_dict) -> float:
    """
    Default value of one saved step, in bits: the expected information of one
    fresh lateral move at minimum altitude, i.e.
    (new cells per move) x (IG of observing a p = 0.5 cell at minimum altitude).
    ~756 bits/step for the default 50 x 50 m Gaussian setup.
    """
    from helper import cH, get_sensor_params

    step_cells, fp_half = camera_step_geometry(camera)
    s0, s1 = get_sensor_params(camera.h_range[0], conf_dict)
    ig_cell = float(1.0 - cH(np.array([0.5]), s0, s1)[0])
    return max(ig_cell, 0.0) * 2.0 * fp_half * step_cells


def regular_tile_layout(
    grid_shape: Tuple[int, int], tile_size: Tuple[int, int]
) -> Tuple[int, int]:
    """Number of tile rows/cols for a regular partition (last tile may be partial)."""
    H, W = grid_shape
    th, tw = tile_size
    return -(-H // th), -(-W // tw)


def _block_any(mask: np.ndarray, bs: int) -> np.ndarray:
    """(nbi, nbj) bool: does each bs x bs block contain a True cell."""
    H, W = mask.shape
    nbi, nbj = -(-H // bs), -(-W // bs)
    padded = np.zeros((nbi * bs, nbj * bs), dtype=bool)
    padded[:H, :W] = mask
    return padded.reshape(nbi, bs, nbj, bs).any(axis=(1, 3))


class CoverageStepModel:
    """
    Coverage state of one agent's knowledge plus step-cost helpers.

    Region ids follow the HLP partition: id = tile_row * n_tile_cols + tile_col,
    i.e. row-major over tiles of `tile_size` starting at (0, 0).
    """

    def __init__(
        self,
        uncovered: np.ndarray,
        tile_size: Tuple[int, int],
        step_cells: float,
        fp_half: float,
    ):
        """
        Args:
            uncovered: (H, W) bool, True where no agent is known to have observed
                the cell yet.
            tile_size: (rows, cols) of each HLP region tile, in cells.
            step_cells: lateral move length in cells (xy_step / grid length).
            fp_half: half-width in cells of the footprint at minimum altitude.
        """
        self.uncovered = np.asarray(uncovered, dtype=bool)
        self.tile_size = (int(tile_size[0]), int(tile_size[1]))
        self.step_cells = float(step_cells)
        self.fp_half = float(fp_half)
        self.half = max(0.0, self.fp_half - 0.5)  # cell at the footprint edge needs a move
        self.bs = max(1, int(round(self.step_cells)))  # block size in cells
        self.kh = max(1, int(round(self.fp_half / self.step_cells)))  # blocks per half footprint
        self._region_inner: Dict[int, float] = {}
        self._compute_stats()

    # ------------------------------------------------------------------ stats
    def _compute_stats(self) -> None:
        H, W = self.uncovered.shape
        th, tw = self.tile_size
        nr, nc = regular_tile_layout((H, W), (th, tw))
        self.n_tile_rows, self.n_tile_cols = nr, nc

        padded = np.zeros((nr * th, nc * tw), dtype=bool)
        padded[:H, :W] = self.uncovered
        tiles = padded.reshape(nr, th, nc, tw)  # [ri, di, rj, dj]

        self.counts = tiles.sum(axis=(1, 3)).astype(np.int64).ravel()

        rows_any = tiles.any(axis=3)  # (nr, th, nc)
        cols_any = tiles.any(axis=1)  # (nr, nc, tw)
        row_lo = rows_any.argmax(axis=1)
        row_hi = th - 1 - rows_any[:, ::-1, :].argmax(axis=1)
        col_lo = cols_any.argmax(axis=2)
        col_hi = tw - 1 - cols_any[:, :, ::-1].argmax(axis=2)
        base_i = (np.arange(nr) * th)[:, None]
        base_j = (np.arange(nc) * tw)[None, :]
        # Inclusive bounding box of uncovered cells (meaningless when count == 0).
        self.i_lo = (base_i + row_lo).ravel().astype(float)
        self.i_hi = (base_i + row_hi).ravel().astype(float)
        self.j_lo = (base_j + col_lo).ravel().astype(float)
        self.j_hi = (base_j + col_hi).ravel().astype(float)

        self.is_open = self.counts > 0
        self.open_ids = np.flatnonzero(self.is_open)

        # Blocks (one move wide) and the region each block belongs to.
        self.blocks = _block_any(self.uncovered, self.bs)  # (nbi, nbj)
        nbi, nbj = self.blocks.shape
        bi = np.arange(nbi) * self.bs
        bj = np.arange(nbj) * self.bs
        self.block_region = (np.minimum(bi // th, nr - 1)[:, None] * nc
                             + np.minimum(bj // tw, nc - 1)[None, :])

    @property
    def num_regions(self) -> int:
        return self.counts.size

    def total_uncovered(self) -> int:
        return int(self.counts.sum())

    def with_footprints(
        self, footprints: Iterable[Optional[Footprint]]
    ) -> "CoverageStepModel":
        """Return a new model in which the given footprints are marked covered."""
        unc = None
        H, W = self.uncovered.shape
        for fp in footprints:
            if fp is None:
                continue
            imin, imax, jmin, jmax = fp
            imin, imax = max(0, int(imin)), min(H, int(imax))
            jmin, jmax = max(0, int(jmin)), min(W, int(jmax))
            if imax <= imin or jmax <= jmin:
                continue
            if unc is None:
                if not self.uncovered[imin:imax, jmin:jmax].any():
                    continue
                unc = self.uncovered.copy()
            unc[imin:imax, jmin:jmax] = False
        if unc is None:
            return self
        return CoverageStepModel(unc, self.tile_size, self.step_cells, self.fp_half)

    # ------------------------------------------------------- covering tours
    def _cover_points(self, open_blocks: np.ndarray) -> np.ndarray:
        """
        Greedy set of lattice positions (in block units, shape (n, 2)) whose
        minimum-altitude footprints see every open block.
        """
        remaining = open_blocks.copy()
        if not remaining.any():
            return np.zeros((0, 2), dtype=np.int64)
        nbi, nbj = remaining.shape
        k = self.kh
        points = []
        while remaining.any():
            # count[a, b] = open blocks in rows [a-k, a+k) x cols [b-k, b+k)
            sat = np.zeros((nbi + 1, nbj + 1), dtype=np.int64)
            sat[1:, 1:] = remaining.cumsum(0).cumsum(1)
            a = np.arange(nbi + 1)
            b = np.arange(nbj + 1)
            a0, a1 = np.clip(a - k, 0, nbi), np.clip(a + k, 0, nbi)
            b0, b1 = np.clip(b - k, 0, nbj), np.clip(b + k, 0, nbj)
            count = (sat[a1][:, b1] - sat[a0][:, b1] - sat[a1][:, b0] + sat[a0][:, b0])
            pa, pb = np.unravel_index(int(count.argmax()), count.shape)
            points.append((pa, pb))
            remaining[max(0, pa - k):pa + k, max(0, pb - k):pb + k] = False
        return np.asarray(points, dtype=np.int64)

    def _tour_steps(self, points: np.ndarray, roots: List[GridPos]) -> float:
        """Multi-root MST over covering positions, in moves."""
        n = len(points)
        if n == 0:
            return 0.0
        pos = points.astype(float) * self.bs  # grid coordinates
        if roots:
            r = np.asarray(roots, dtype=float)
            root = np.min(np.abs(pos[:, None, :] - r[None, :, :]).sum(-1), axis=1)
            root = np.ceil(root / self.step_cells - 1e-9)
        else:
            root = np.zeros(n)
        if n == 1:
            return float(root[0])
        links = np.abs(points[:, None, :] - points[None, :, :]).sum(-1).astype(float)
        links *= self.bs / self.step_cells
        return _multi_root_mst(root, links)

    # ------------------------------------------------------------ step costs
    def reach_steps_all(self, pos: GridPos) -> np.ndarray:
        """Moves from `pos` until a footprint can see each region's open bounding box."""
        pi, pj = float(pos[0]), float(pos[1])
        di = np.maximum(0.0, np.maximum(self.i_lo - pi, pi - self.i_hi))
        dj = np.maximum(0.0, np.maximum(self.j_lo - pj, pj - self.j_hi))
        return (np.ceil(np.maximum(0.0, di - self.half) / self.step_cells)
                + np.ceil(np.maximum(0.0, dj - self.half) / self.step_cells))

    def reach_steps(self, pos: GridPos, r: int) -> float:
        pi, pj = float(pos[0]), float(pos[1])
        di = max(0.0, self.i_lo[r] - pi, pi - self.i_hi[r])
        dj = max(0.0, self.j_lo[r] - pj, pj - self.j_hi[r])
        return float(
            np.ceil(max(0.0, di - self.half) / self.step_cells)
            + np.ceil(max(0.0, dj - self.half) / self.step_cells)
        )

    def arrival_position(self, pos: GridPos, r: int) -> GridPos:
        """Where an agent coming from `pos` first sees region r (clamped to its view zone)."""
        lo_i, hi_i = self.i_lo[r] - self.half, self.i_hi[r] + self.half
        lo_j, hi_j = self.j_lo[r] - self.half, self.j_hi[r] + self.half
        return (float(min(max(pos[0], lo_i), hi_i)), float(min(max(pos[1], lo_j), hi_j)))

    def region_inner_steps(self, r: int) -> float:
        """Moves to see all of region r's open blocks once there (0 for one block)."""
        if r not in self._region_inner:
            mask = self.blocks & (self.block_region == r)
            pts = self._cover_points(mask)
            if len(pts) <= 1:
                self._region_inner[r] = 0.0
            else:
                links = np.abs(pts[:, None, :] - pts[None, :, :]).sum(-1).astype(float)
                links *= self.bs / self.step_cells
                self._region_inner[r] = _multi_root_mst(np.full(len(pts), np.inf), links, free_first=True)
        return self._region_inner[r]

    def region_cost(self, pos: Optional[GridPos], r: int) -> float:
        """Moves to finish region r starting at pos (0 if r is already covered)."""
        if self.counts[r] == 0:
            return 0.0
        reach = 0.0 if pos is None else self.reach_steps(pos, r)
        return reach + self.region_inner_steps(r)

    def exit_position(self, r: int) -> GridPos:
        """Centre of region r's open bounding box."""
        return (0.5 * (self.i_lo[r] + self.i_hi[r]), 0.5 * (self.j_lo[r] + self.j_hi[r]))

    def sequence_cost(
        self, start: Optional[GridPos], sequence: Sequence[int]
    ) -> Tuple[float, List[float], List[float], Optional[GridPos]]:
        """
        Walk a region sequence (regions already covered or repeated are free).

        Returns:
            (total moves, cumulative moves until each region is reached,
             cumulative moves until each region is finished, end position)
        """
        pos = start
        total = 0.0
        etas: List[float] = []
        finished: List[float] = []
        seen = set()
        for r in sequence:
            if 0 <= r < self.num_regions and self.counts[r] > 0 and r not in seen:
                seen.add(r)
                reach = 0.0 if pos is None else self.reach_steps(pos, r)
                etas.append(total + reach)
                total += reach + self.region_inner_steps(r)
                pos = self.arrival_position(pos, r) if pos is not None else self.exit_position(r)
            else:
                etas.append(total)
            finished.append(total)
        return total, etas, finished, pos

    def leftover_steps(self, claimed: np.ndarray, ends: List[GridPos]) -> float:
        """Moves for all open blocks of regions nobody claimed (covering-positions MST)."""
        open_blocks = self.blocks & ~claimed[self.block_region]
        return self._tour_steps(self._cover_points(open_blocks), ends)

    def team_steps(
        self, agents: Dict[int, Tuple[Optional[GridPos], Sequence[int]]]
    ) -> float:
        """
        g2: estimated agent-steps until every cell is observed at least once.

        Args:
            agents: {agent_id: (start position in grid cells or None, HL region sequence)}
        """
        total = 0.0
        claimed = np.zeros(self.num_regions, dtype=bool)
        ends: List[GridPos] = []
        for start, sequence in agents.values():
            cost, _, _, end = self.sequence_cost(start, sequence)
            total += cost
            for r in sequence:
                if 0 <= r < self.num_regions:
                    claimed[r] = True
            if end is not None:
                ends.append(end)
        return total + self.leftover_steps(claimed, ends)


def _multi_root_mst(root: np.ndarray, links: np.ndarray, free_first: bool = False) -> float:
    """
    Minimum spanning forest cost (Prim) where every node may alternatively be
    attached to a virtual root (the agents) at cost root[k]. With free_first,
    the first node joins at no cost (plain MST).
    """
    best = np.asarray(root, dtype=np.float64).copy()
    n = best.size
    if free_first:
        best[0] = 0.0
    in_tree = np.zeros(n, dtype=bool)
    total = 0.0
    for _ in range(n):
        k = int(np.where(in_tree, np.inf, best).argmin())
        total += best[k]
        in_tree[k] = True
        np.minimum(best, links[k], out=best)
    return float(total)


def uncovered_from_belief(belief: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Fallback when no explicit coverage mask is available: treat cells whose
    belief is still exactly at the 0.5 prior as unobserved.

    Only reliable without LBP smoothing (LBP also moves unobserved neighbours).
    The simulator always passes an explicit observed-at-least-once mask.
    """
    p = belief[:, :, 1] if belief.ndim == 3 else belief
    return np.abs(p - 0.5) <= tol
