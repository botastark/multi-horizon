"""
Regression tests for the step-based coverage objective (g2), the HLP/LLP fixes,
step-based intent ageing, range-limited intent sharing, coverage news and the
Dec-MCTS backpropagation fix.

Each test encodes one behaviour that was previously wrong:
  - g2 is a cost in agent-steps; the LLP must prefer trajectories that SAVE steps
  - the HLP must prefer nearby unexplored regions over already-observed ones,
    and value closing an isolated leftover spot
  - a teammate's NEXT planned footprint is suppressed most, not least
  - intent ages are measured in simulation steps (no wall clock)
  - intents reach only agents in communication range, and none in IG/IGd modes
  - "observed at least once" knowledge travels exactly like BS/BM news
  - Dec-MCTS node values include the immediate reward of every step on the path

Usage: python tests/test_mh_coverage_g2.py   (or: pytest tests/test_mh_coverage_g2.py)
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from uav_camera import Camera
from g2_evaluator import CoverageStepModel
from hierarchical_dec_mcts import (
    HLIntent,
    IntentBus,
    LLIntent,
    camera_step_geometry,
    create_hierarchical_planner,
)
from dec_mcts import DecMCTSCoordinator, DecMCTSIntent, DecMCTSNode
from multi_agent_mapper import MultiAgentMapper

GRID = SimpleNamespace(x=50, y=50, shape=(400, 400), length=0.125, center=True)
TILE = (25, 25)


def make_camera():
    cam = Camera(GRID, 60, camera_altitude=None, f_overlap=None, s_overlap=None,
                 seed=0, a=1.0, b=0.015)
    cam.set_position((0.0, 0.0))
    cam.set_altitude(cam.h_range[0])
    return cam


CAM = make_camera()
CONF = CAM.theoretical_conf_dict()
STEP_CELLS, FP_HALF = camera_step_geometry(CAM)


def model(uncovered):
    return CoverageStepModel(uncovered, TILE, STEP_CELLS, FP_HALF)


def make_planner(agent_id=0, n=1, bus=None, use_g2=True, **cfg):
    config = {"llp_horizon": 3, "llp_iterations": 40, "hlp_horizon": 3,
              "hlp_iterations": 40, "tile_size": list(TILE), "use_g2": use_g2}
    config.update(cfg)
    return create_hierarchical_planner(
        agent_id=agent_id, num_agents=n, camera=CAM, grid_info=GRID,
        intent_bus=bus or IntentBus(n), config=config, conf_dict=CONF, seed=1,
    )


# --------------------------------------------------------------------- g2 model
def test_geometry():
    assert STEP_CELLS == 25.0 and FP_HALF == 25.0, (STEP_CELLS, FP_HALF)


def test_g2_full_field_matches_lawnmower():
    # 400x400 cells, footprint 50 cells wide, 25-cell moves: 8 strips x 16 moves
    steps = model(np.ones((400, 400), bool)).team_steps({0: ((0.0, 0.0), [])})
    assert steps == 128.0, steps


def test_g2_isolated_spot_costs_a_detour():
    unc = np.ones((400, 400), bool)
    unc[:, :200] = False  # left half done
    base = model(unc).team_steps({0: ((0.0, 200.0), [])})
    unc_hole = unc.copy()
    unc_hole[100:110, 50:60] = True  # a small missed spot far inside the done half
    with_hole = model(unc_hole).team_steps({0: ((0.0, 200.0), [])})
    # the spot adds only ~0.08 steps of sweep but several steps of detour
    assert with_hole - base > 2.0, (base, with_hole)


def test_g2_covering_part_of_a_strip_never_costs_more():
    # Regression: an earlier estimator priced thin strips at half cost and then
    # "discovered" travel when their middle got covered, so covering cells raised
    # g2 and the LLP preferred hovering.
    strip = np.zeros((400, 400), bool)
    strip[0:25, :] = True  # one-tile-wide uncovered strip along the top edge
    start = {0: ((0.0, 0.0), [])}
    full = model(strip).team_steps(start)
    middle = strip.copy()
    middle[:, 150:250] = False
    end = strip.copy()
    end[:, 300:] = False
    assert model(middle).team_steps(start) <= full
    assert model(end).team_steps(start) < full


def test_g2_covering_rarely_and_only_slightly_increases_cost():
    rng = np.random.default_rng(0)
    increases = []
    for _ in range(150):
        unc = np.ones((400, 400), bool)
        for _ in range(rng.integers(5, 40)):
            h = int(rng.choice([25, 50, 75, 100, 150]))
            c = rng.integers(0, 401, 2)
            unc[max(0, c[0] - h):c[0] + h, max(0, c[1] - h):c[1] + h] = False
        pos = tuple(rng.integers(0, 401, 2).astype(float))
        m = model(unc)
        h = int(rng.choice([25, 50, 100]))
        c = rng.integers(0, 401, 2)
        after = m.with_footprints([(c[0] - h, c[0] + h, c[1] - h, c[1] + h)])
        increases.append(after.team_steps({0: (pos, [])}) - m.team_steps({0: (pos, [])}))
    increases = np.array(increases)
    # greedy cover + MST are not exactly monotone; keep violations rare and small
    assert (increases > 0).mean() <= 0.05 and increases.max() <= 3, increases.max()
    assert increases.mean() < 0


def test_g2_duplicate_claims_are_paid_twice():
    m = model(np.ones((400, 400), bool))
    single = m.team_steps({0: ((0.0, 0.0), [17]), 1: ((0.0, 0.0), [])})
    double = m.team_steps({0: ((0.0, 0.0), [17]), 1: ((0.0, 0.0), [17])})
    assert double > single


# ------------------------------------------------------------------------- HLP
def test_hlp_prefers_unexplored_over_observed_region():
    p = make_planner()
    covered = np.zeros((400, 400), bool)
    covered[150:250, 150:225] = True  # around the agent, left part observed
    p.update_state((0.0, 0.0), CAM.h_range[0], np.full((400, 400, 2), 0.5),
                   coverage_mask=covered, step=0)
    p.plan()
    seq = p.hlp.current_intent.region_sequence
    m = model(~covered)
    assert seq, "HLP returned no regions"
    assert all(m.counts[r] > 0 for r in seq), f"HLP chose observed regions: {seq}"


def test_hlp_closes_nearby_leftover_spot_first():
    # Everything observed except a tiny spot one tile away and a big open block far away.
    covered = np.ones((400, 400), bool)
    covered[175:180, 230:235] = False  # tiny spot next to the agent (grid ~ (200, 200))
    covered[:, 330:] = False           # unexplored block on the far right
    p = make_planner()
    p.update_state((0.0, 0.0), CAM.h_range[0], np.full((400, 400, 2), 0.5),
                   coverage_mask=covered, step=0)
    p.plan()
    seq = p.hlp.current_intent.region_sequence
    spot_region = (175 // 25) * 16 + (230 // 25)
    assert seq and seq[0] == spot_region, f"expected to finish spot {spot_region} first, got {seq}"


def test_hlp_plans_from_end_of_own_ll_plan():
    # Paper Alg. 2 (bottom-up): the HL plan continues where the agent's own LL
    # plan ends, and cells that LL plan will observe count as covered.
    covered = np.ones((400, 400), bool)
    covered[175:180, 230:235] = False  # spot A, next to the agent at grid (200, 200)
    covered[300:305, 300:305] = False  # spot B, further away
    p = make_planner()
    own_ll = LLIntent(
        agent_id=0,
        step=0,
        footprint_sequence=[(150, 250, 200, 260)],      # covers spot A
        state_sequence=[(0.0, 0.0, CAM.h_range[0]), (12.5, -12.5, CAM.h_range[0])],  # ends at grid (300, 300)
    )
    p.llp.current_intent = own_ll
    p.update_state((0.0, 0.0), CAM.h_range[0], np.full((400, 400, 2), 0.5),
                   coverage_mask=covered, step=1)
    p.plan()
    spot_b = (300 // 25) * 16 + (300 // 25)
    assert p.hlp.last_plan_start == (300.0, 300.0), p.hlp.last_plan_start
    assert p.hlp.current_intent.region_sequence[0] == spot_b, p.hlp.current_intent.region_sequence


def test_hlp_replans_by_steps_and_when_target_done():
    p = make_planner(hlp_replan_interval=3)
    belief = np.full((400, 400, 2), 0.5)
    cov = np.zeros((400, 400), bool)
    replans = []
    for step in range(6):
        p.update_state((0.0, 0.0), CAM.h_range[0], belief, coverage_mask=cov, step=step)
        # isolate the interval rule: without this, the own LL plan (which would
        # cover the target) also triggers a replan - tested separately above
        p.llp.current_intent = None
        p.plan()
        replans.append(p.hlp.last_plan_replanned)
    assert replans == [True, False, False, True, False, False], replans
    # Mark the current target region observed: replan immediately
    target = p.hlp.current_intent.current_target_region
    (i0, i1), (j0, j1) = p.hlp.regions[target]["bounds"]
    cov = cov.copy()
    cov[i0:i1, j0:j1] = True
    p.update_state((0.0, 0.0), CAM.h_range[0], belief, coverage_mask=cov, step=6)
    p.plan()
    assert p.hlp.last_plan_replanned


# ------------------------------------------------------------------------- LLP
def test_llp_g2_rewards_moving_toward_hl_target():
    p = make_planner()
    covered = np.ones((400, 400), bool)
    covered[:, 300:] = False  # only the right side (larger x) is unexplored
    p.update_state((0.0, 0.0), CAM.h_range[0], np.full((400, 400, 2), 0.5),
                   coverage_mask=covered, step=0)
    llp = p.llp
    llp.set_step(0)
    llp.update_hl_guidance(HLIntent(agent_id=0, region_sequence=[12 * 16 + 12]))
    llp._prepare_g2_context((0.0, 0.0, CAM.h_range[0]))
    disc = llp._compute_teammate_coverage_mask()
    toward, *_ = llp._simulate_trajectory((0.0, 0.0, CAM.h_range[0]), ["right"] * 3, disc)
    away, *_ = llp._simulate_trajectory((0.0, 0.0, CAM.h_range[0]), ["left"] * 3, disc)
    # belief is uniform, so g1 is equal for both; only g2 can separate them
    assert toward > away, (toward, away)


def test_teammate_next_footprint_suppressed_most():
    p = make_planner(n=2)
    llp = p.llp
    llp.update_belief(np.full((400, 400, 2), 0.5))
    fps = [(0, 50, 0, 50), (100, 150, 100, 150), (200, 250, 200, 250), (300, 350, 300, 350)]
    llp.update_teammate_intents({1: LLIntent(agent_id=1, footprint_sequence=fps, step=5)}, {})
    llp.set_step(5)  # fresh intent (age 0)
    mask = llp._compute_teammate_coverage_mask()
    w = [float(mask[f[0], f[2]]) for f in fps]
    assert w[0] == 0.0 and w[0] < w[1] < w[2] < w[3] < 1.0, w
    llp.set_step(6)  # one step later: first footprint already executed -> ignored
    mask = llp._compute_teammate_coverage_mask()
    assert float(mask[0, 0]) == 1.0 and float(mask[100, 100]) < 0.1


# -------------------------------------------------------------- intent ageing
def test_intent_age_is_in_steps():
    it = LLIntent(agent_id=1, step=10)
    assert not it.is_stale(current_step=14, max_age_steps=4)
    assert it.is_stale(current_step=15, max_age_steps=4)
    assert it.staleness_discount(10, 0.9, 2.0) == 1.0
    assert abs(it.staleness_discount(12, 0.9, 2.0) - 0.9) < 1e-12
    d = DecMCTSIntent(agent_id=1, step=3)
    assert d.is_stale(10, 4) and not d.is_stale(7, 4)


# --------------------------------------------------------- range-limited intents
def test_intent_bus_respects_range_and_mode():
    in_range = {0: [1], 1: [0], 2: []}
    bus = IntentBus(3, neighbor_fn=lambda a: in_range[a])
    bus.broadcast_ll_intent(LLIntent(agent_id=0, step=0))
    assert set(bus.get_teammate_ll_intents(1)) == {0}
    assert bus.get_teammate_ll_intents(2) == {}
    # agent 1 moves out of range: it keeps the old intent, which then ages
    in_range.update({0: [], 1: []})
    bus.broadcast_ll_intent(LLIntent(agent_id=0, step=5))
    assert bus.get_teammate_ll_intents(1)[0].step == 0

    off = IntentBus(2, enabled=False)
    off.broadcast_hl_intent(HLIntent(agent_id=0))
    assert off.get_teammate_hl_intents(1) == {}

    dc = DecMCTSCoordinator(3, neighbor_fn=lambda a: {0: [2], 1: [], 2: [0]}[a])
    dc.share_intent(DecMCTSIntent(agent_id=0))
    assert set(dc.get_teammate_intents(2)) == {0} and dc.get_teammate_intents(1) == {}


# ---------------------------------------------------------------- coverage news
def _mapper(mode):
    return MultiAgentMapper((40, 40), 3, CONF, news_mode=mode)


def test_coverage_news_bs_and_bm():
    fp = {0: (0, 10, 0, 10), 1: (20, 30, 20, 30), 2: (30, 40, 30, 40)}
    # BS: broadcast to current neighbours, then cleared
    m = _mapper("BS")
    masks = {a: np.zeros((40, 40), bool) for a in range(3)}
    m.share_coverage_news(fp, {0: [1], 1: [0], 2: []}, masks)
    assert masks[1][:10, :10].all() and not masks[2].any()
    m.share_coverage_news({0: (10, 12, 10, 12)}, {0: [2]}, masks)
    assert masks[2][10:12, 10:12].all() and not masks[2][:10, :10].any()
    # BM: per-receiver backlog delivered when the receiver comes into range
    m = _mapper("BM")
    masks = {a: np.zeros((40, 40), bool) for a in range(3)}
    m.share_coverage_news(fp, {0: [1], 1: [0], 2: []}, masks)
    assert masks[1][:10, :10].all() and not masks[2].any()
    m.share_coverage_news({0: (10, 12, 10, 12)}, {0: [2]}, masks)
    assert masks[2][:10, :10].all() and masks[2][10:12, 10:12].all()


# ------------------------------------------------------------ Dec-MCTS backprop
def test_dec_mcts_backprop_includes_path_rewards():
    state = {"uav_pos": SimpleNamespace(position=(0.0, 0.0), altitude=CAM.h_range[0]),
             "belief": np.full((400, 400, 2), 0.5), "remaining_steps": 3}
    root = DecMCTSNode(state, camera=CAM, conf_dict=CONF, config={}, seed=0)
    c1 = DecMCTSNode(state, camera=CAM, parent=root, action="front", conf_dict=CONF, config={})
    c2 = DecMCTSNode(state, camera=CAM, parent=c1, action="front", conf_dict=CONF, config={})
    c1.immediate_reward, c2.immediate_reward = 10.0, 5.0
    c2.backpropagate_return(rollout_reward=2.0, gamma=0.5)
    assert c2.value == 5.0 + 0.5 * 2.0
    assert c1.value == 10.0 + 0.5 * c2.value  # the first step's own reward is included
    assert root.value == c1.value


# ------------------------------------------------------- simulated asynchrony
def test_d_uct_stats_discount_old_episodes():
    from d_uct import DUCTStats

    st = DUCTStats()
    st.add(1, 0.5, 10.0)
    st.add(3, 0.5, 2.0)  # first episode decayed by 0.5**2 before adding
    assert abs(st.count - 1.25) < 1e-12 and abs(st.value - 4.5) < 1e-12
    assert abs(st.count_at(4, 0.5) - 0.625) < 1e-12
    assert abs(st.mean() - 4.5 / 1.25) < 1e-12


def test_deferred_bus_delivers_only_on_flush():
    bus = IntentBus(2)
    bus.deferred = True
    bus.broadcast_ll_intent(LLIntent(agent_id=0, step=0))
    assert bus.get_teammate_ll_intents(1) == {}
    bus.flush()
    assert set(bus.get_teammate_ll_intents(1)) == {0}
    dc = DecMCTSCoordinator(2)
    dc.deferred = True
    dc.share_intent(DecMCTSIntent(agent_id=1))
    assert dc.get_teammate_intents(0) == {}
    dc.flush()
    assert set(dc.get_teammate_intents(0)) == {1}


def _run_async(order, steps=3, n_slices=3):
    """Two MH agents planning concurrently in slices; returns their actions per step."""
    cams = [make_camera(), make_camera()]
    cams[1].set_position((12.5, 12.5))
    bus = IntentBus(2)
    planners = []
    for aid, cam in enumerate(cams):
        planners.append(create_hierarchical_planner(
            agent_id=aid, num_agents=2, camera=cam, grid_info=GRID, intent_bus=bus,
            config={"llp_horizon": 3, "llp_iterations": 12, "hlp_horizon": 3,
                    "hlp_iterations": 9, "tile_size": list(TILE), "use_g2": True,
                    "use_mcts_llp": True},
            conf_dict=CONF, seed=5 + aid))
    cov = [np.zeros((400, 400), bool), np.zeros((400, 400), bool)]
    actions = []
    for step in range(steps):
        for aid, p in enumerate(planners):
            pos = cams[aid].get_x()
            p.update_state(pos.position, pos.altitude, np.full((400, 400, 2), 0.5),
                           coverage_mask=cov[aid], step=step)
            p.begin_step()
        bus.deferred = True
        for k in range(n_slices):
            for aid in order:
                planners[aid].run_slice(4, 3)
            bus.flush()
        bus.deferred = False
        step_actions = []
        for aid, p in enumerate(planners):
            a, _ = p.finish_step()
            step_actions.append(a)
            nxt = cams[aid].x_future(a)
            cams[aid].set_position(nxt[0]); cams[aid].set_altitude(nxt[1])
            [[i0, i1], [j0, j1]] = cams[aid].get_range(index_form=True)
            cov[aid][i0:i1, j0:j1] = True
        actions.append(step_actions)
    return actions, planners


def test_async_is_deterministic_and_order_independent():
    a1, planners = _run_async(order=[0, 1])
    a2, _ = _run_async(order=[0, 1])
    a3, _ = _run_async(order=[1, 0])
    assert a1 == a2, (a1, a2)
    assert a1 == a3, ("agent order changed the outcome", a1, a3)
    # intents were exchanged at slice boundaries
    assert set(planners[0].intent_bus.get_teammate_ll_intents(0)) == {1}


def test_dec_mcts_g2_prefers_unexplored_side():
    from dec_mcts import DecMCTSPlanner

    covered = np.ones((400, 400), bool)
    covered[:, 300:] = False  # only the right side (larger x) is unobserved
    planner = DecMCTSPlanner(agent_id=0, camera=make_camera(), grid_info=GRID, conf_dict=CONF,
                             config={"horizon": 3, "iterations": 60, "use_g2": True,
                                     "timeout": 1e9}, seed=0)
    planner.update_state((0.0, 0.0), CAM.h_range[0], np.full((400, 400, 2), 0.5),
                         step=0, coverage_mask=covered)
    planner.plan()
    values = planner.get_action_values()
    assert values["right"] > values["left"], values


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
