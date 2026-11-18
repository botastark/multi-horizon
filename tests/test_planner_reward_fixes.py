"""
Regression tests for the planner reward and search fixes.

Each test reproduces one reported defect directly, without running a full experiment:
diminishing returns along a plan, teammate discounting, Dec-MCTS backup and tie-breaking,
UCB normalisation, MCTS LLP node identity, efficient LLP root coverage, and g2 / HLP region choice.

Run with: pytest tests/test_planner_reward_fixes.py
"""

import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pytest

from dec_mcts import DecMCTSNode, DecMCTSPlanner
from g2_evaluator import g2
from helper import uav_position
from hierarchical_dec_mcts import HighLevelPlanner, HLIntent, LLIntent, LowLevelPlanner
from uav_camera import Camera

GRID = SimpleNamespace(x=50, y=50, shape=(400, 400), length=0.125, center=True)
SEED = 1000


def setup_camera():
    camera = Camera(GRID, 60, camera_altitude=None, f_overlap=None, s_overlap=None, seed=SEED, a=1.0, b=0.015)
    return camera, camera.theoretical_conf_dict()


def uniform_belief(p=0.5):
    return np.stack([np.full(GRID.shape, 1 - p), np.full(GRID.shape, p)], axis=2)


def random_belief():
    occupied = np.random.default_rng(SEED).uniform(0.02, 0.98, GRID.shape)
    return np.stack([1 - occupied, occupied], axis=2)


def make_llp(camera, conf_dict, belief, **kwargs):
    llp = LowLevelPlanner(0, camera, GRID, conf_dict=conf_dict, seed=SEED, **kwargs)
    llp.update_belief(belief)
    return llp


def start_state(camera, level=2, position=(0.0, 0.0)):
    return (position[0], position[1], camera.h_range[0] + level * camera.h_step)


def footprint_at(camera, position, altitude):
    [[imin, imax], [jmin, jmax]] = camera.get_range(position=position, altitude=altitude, index_form=True)
    return imin, imax, jmin, jmax


# --------------------------------------------------------------------------------------
# Planning ahead sees diminishing returns
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("belief", [uniform_belief(), random_belief()], ids=["uniform", "random"])
def test_dec_mcts_repeat_observation_is_worth_less(belief):
    camera, conf_dict = setup_camera()
    state = {"uav_pos": uav_position(((0.0, 0.0), start_state(camera)[2])), "belief": belief.copy()}
    node = DecMCTSNode(state, camera, conf_dict=conf_dict, seed=0)
    first = node.observe(state)[1]
    second = node.observe(state)[1]
    assert 0.0 < second < 0.9 * first


def test_dec_mcts_simulated_observation_keeps_probabilities_valid():
    camera, conf_dict = setup_camera()
    state = {"uav_pos": uav_position(((0.0, 0.0), start_state(camera)[2])), "belief": random_belief()}
    node = DecMCTSNode(state, camera, conf_dict=conf_dict, seed=0)
    for _ in range(5):
        node.observe(state)
    assert np.all(state["belief"] >= 0.0) and np.all(state["belief"] <= 1.0)
    assert np.allclose(state["belief"].sum(axis=2), 1.0)


def test_llp_repeat_observation_is_worth_less():
    camera, conf_dict = setup_camera()
    llp = make_llp(camera, conf_dict, uniform_belief(), horizon=3)
    _, _, _, igs = llp._simulate_trajectory(start_state(camera), ["hover", "hover", "hover"], [])
    assert igs[0] > igs[1] > igs[2] > 0.0


def test_llp_first_step_uses_the_real_belief():
    camera, conf_dict = setup_camera()
    llp = make_llp(camera, conf_dict, random_belief(), horizon=2)
    state = start_state(camera)
    _, _, _, igs = llp._simulate_trajectory(state, ["hover", "hover"], [])
    direct, _ = llp._compute_ig(state[:2], state[2], [])
    assert igs[0] == pytest.approx(direct)


# --------------------------------------------------------------------------------------
# Teammate discount
# --------------------------------------------------------------------------------------


def teammate_setup():
    camera, conf_dict = setup_camera()
    state = start_state(camera)
    near = footprint_at(camera, (camera.xy_step, 0.0), state[2])
    far = footprint_at(camera, (20.0, 20.0), camera.h_range[0])
    return camera, conf_dict, state, near, far


def test_teammate_next_footprint_is_discounted_most():
    camera, conf_dict, state, near, far = teammate_setup()
    llp = make_llp(camera, conf_dict, uniform_belief())
    no_teammate, _ = llp._compute_ig(state[:2], state[2], [])

    def ig_for(sequence):
        llp.update_teammate_intents({1: LLIntent(agent_id=1, footprint_sequence=sequence)}, {})
        return llp._compute_ig(state[:2], state[2], llp._compute_teammate_coverage_mask())[0]

    next_step, later_step = ig_for([near, far]), ig_for([far, near])
    assert next_step < later_step < no_teammate


def test_plain_ig_mode_ignores_teammates():
    camera, conf_dict, state, near, _ = teammate_setup()
    llp = make_llp(camera, conf_dict, uniform_belief(), discount_teammates=False)
    llp.update_teammate_intents({1: LLIntent(agent_id=1, footprint_sequence=[near])}, {})
    assert llp._compute_teammate_coverage_mask() == []


def test_teammate_out_of_communication_range_is_ignored():
    camera, conf_dict, state, near, _ = teammate_setup()
    limit = camera.xy_step / 2
    llp = make_llp(camera, conf_dict, uniform_belief(), comm_radius=(limit, limit, limit))
    planned = (camera.xy_step, 0.0, state[2])
    llp.update_teammate_intents(
        {1: LLIntent(agent_id=1, footprint_sequence=[near], state_sequence=[planned, planned])}, {}
    )
    assert llp._compute_teammate_coverage_mask(state) == []


def test_dec_mcts_overlap_penalty_is_on_the_scale_of_ig():
    camera, conf_dict, state, near, _ = teammate_setup()
    plans = [[near]]
    node_state = {
        "uav_pos": uav_position((state[:2], state[2])),
        "belief": uniform_belief(),
        "teammate_mask": plans,
    }
    node = DecMCTSNode(node_state, camera, conf_dict=conf_dict, seed=0)
    bounds = footprint_at(camera, state[:2], state[2])
    ig = node.compute_ig_reward(node_state, *bounds)
    assert node.compute_overlap_penalty(node_state, *bounds) > 0.1 * ig


# --------------------------------------------------------------------------------------
# Dec-MCTS backup and ties
# --------------------------------------------------------------------------------------


def make_dec_planner(camera, conf_dict, belief, **config):
    planner = DecMCTSPlanner(0, camera, GRID, conf_dict=conf_dict, seed=SEED, config={"timeout": 60.0, **config})
    state = start_state(camera)
    planner.update_state(state[:2], state[2], belief)
    return planner


def test_dec_mcts_first_action_ig_counts_on_every_visit():
    camera, conf_dict = setup_camera()
    planner = make_dec_planner(camera, conf_dict, uniform_belief(), horizon=2, iterations=60, discount_factor=1.0)
    root_children = {}
    original = planner._extract_best_action

    def capture(root):
        root_children.update(root.children)
        return original(root)

    planner._extract_best_action = capture
    planner.plan()
    revisited = [child for child in root_children.values() if child.visit_count > 1]
    assert revisited
    for child in revisited:
        assert child.value / child.visit_count > child.immediate_reward


def test_dec_mcts_ties_follow_greedy_order():
    camera, conf_dict = setup_camera()
    planner = make_dec_planner(camera, conf_dict, uniform_belief(1e-15), horizon=1, iterations=7)
    assert planner.plan().action_sequence[0] == "left"


def test_llp_ties_follow_greedy_order():
    camera, conf_dict = setup_camera()
    for use_mcts in (False, True):
        llp = make_llp(camera, conf_dict, uniform_belief(1e-15), horizon=1, num_iterations=30, use_mcts_llp=use_mcts)
        assert llp.plan(start_state(camera)).action_sequence[0] == "left"


# --------------------------------------------------------------------------------------
# Search mechanics
# --------------------------------------------------------------------------------------


def test_dec_mcts_ucb_is_scaled_to_the_reward_range():
    camera, conf_dict = setup_camera()
    state = {"uav_pos": uav_position(((0.0, 0.0), start_state(camera)[2])), "belief": uniform_belief()}
    parent = DecMCTSNode(state, camera, conf_dict=conf_dict, seed=0)
    parent.visit_count = 51
    well_known = DecMCTSNode(state, camera, parent=parent, conf_dict=conf_dict, seed=0)
    well_known.visit_count, well_known.value = 50, 50 * 4000.0
    barely_tried = DecMCTSNode(state, camera, parent=parent, conf_dict=conf_dict, seed=0)
    barely_tried.visit_count, barely_tried.value = 1, 3900.0
    parent.children = {"a": well_known, "b": barely_tried}
    assert parent.best_child(c_param=1.4, value_bounds=(3900.0, 4000.0)) is barely_tried


def test_mcts_llp_root_values_only_use_trajectories_starting_with_that_action():
    camera, conf_dict = setup_camera()
    llp = make_llp(camera, conf_dict, random_belief(), horizon=3, num_iterations=80, use_mcts_llp=True)
    trajectories = []
    original = llp._simulate_trajectory

    def recording(start, actions, plans):
        result = original(start, actions, plans)
        trajectories.append((list(actions), result[0]))
        return result

    llp._simulate_trajectory = recording
    llp.plan(start_state(camera))
    for action, value in llp._mcts_action_values.items():
        rewards = [reward for actions, reward in trajectories if actions[0] == action]
        if rewards:
            assert value == pytest.approx(np.mean(rewards))


@pytest.mark.parametrize("seed", range(1000, 1020))
def test_efficient_llp_scores_every_root_action(seed):
    camera, conf_dict = setup_camera()
    llp = LowLevelPlanner(0, camera, GRID, conf_dict=conf_dict, seed=seed, horizon=2, num_iterations=7)
    llp.update_belief(uniform_belief())
    state = start_state(camera)
    llp.plan(state)
    scored = {a for a, v in llp._mcts_action_values.items() if np.isfinite(v)}
    assert scored == set(llp._admissible_actions(state[:2], state[2]))


# --------------------------------------------------------------------------------------
# g2 and region choice
# --------------------------------------------------------------------------------------

TILE = (25, 25)


def make_hlp(num_agents=1, seed=SEED):
    return HighLevelPlanner(
        0, num_agents, GRID.shape, tile_size=TILE, horizon=3, num_iterations=60,
        seed=seed, grid_info=GRID,
    )


def quadrant_mapped_belief():
    belief = uniform_belief()
    belief[:200, :200, 1] = 1e-4
    belief[:200, :200, 0] = 1 - 1e-4
    return belief


def stay_intent(agent_id, x, y, altitude=20.0):
    return LLIntent(agent_id=agent_id, state_sequence=[(x, y, altitude)])


def env_for(hlp, belief):
    return {"belief": belief, "grid_info": GRID, "regions": hlp.regions}


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_hlp_does_not_pick_mapped_regions(seed):
    hlp = make_hlp(seed=seed)
    hlp.update_belief(quadrant_mapped_belief())
    hlp.update_own_ll_intent(stay_intent(0, 20.0, -20.0))
    intent = hlp.plan((360.0, 360.0))
    assert intent.region_sequence
    for region_id in intent.region_sequence:
        row, col = hlp.regions[region_id]["center"]
        assert not (row < 200 and col < 200), f"region {region_id} at ({row}, {col}) is already mapped"


def test_g2_counts_open_regions_outside_every_intent():
    hlp = make_hlp()
    all_uncertain = uniform_belief()
    one_open = np.array(quadrant_mapped_belief())
    ll = {0: stay_intent(0, 20.0, -20.0)}
    plan = {0: HLIntent(agent_id=0, region_sequence=[255])}
    assert g2(ll, plan, env_for(hlp, all_uncertain)) > g2(ll, plan, env_for(hlp, one_open))


def test_g2_does_not_reward_planning_fewer_regions():
    hlp = make_hlp()
    belief = uniform_belief()
    ll = {0: stay_intent(0, 20.0, -20.0)}
    with_plan = {0: HLIntent(agent_id=0, region_sequence=[255])}
    without_plan = {0: HLIntent(agent_id=0, region_sequence=[])}
    assert g2(ll, with_plan, env_for(hlp, belief)) == pytest.approx(g2(ll, without_plan, env_for(hlp, belief)), rel=0.05)


def test_g2_charges_travel_from_the_agents_own_position():
    hlp = make_hlp()
    belief = np.array(quadrant_mapped_belief())
    plan = {0: HLIntent(agent_id=0, region_sequence=[255])}
    near_the_open_area = g2({0: stay_intent(0, 20.0, -20.0)}, plan, env_for(hlp, belief))
    far_from_it = g2({0: stay_intent(0, -20.0, 20.0)}, plan, env_for(hlp, belief))
    assert far_from_it > near_the_open_area


def test_g2_is_the_mean_over_agents_not_the_max():
    hlp = make_hlp()
    belief = uniform_belief()
    belief[:, :, 1] = 1e-4
    belief[:, :, 0] = 1 - 1e-4
    slow = LLIntent(agent_id=0, action_sequence=["hover"] * 10, state_sequence=[(0.0, 0.0, 20.0)])
    fast = LLIntent(agent_id=1, action_sequence=["hover"] * 2, state_sequence=[(0.0, 0.0, 20.0)])
    assert g2({0: slow, 1: fast}, {}, env_for(hlp, belief)) == pytest.approx(0.1 * 6.0)


def test_llp_g2_lowers_the_reward():
    camera, conf_dict = setup_camera()
    without = make_llp(camera, conf_dict, uniform_belief(), horizon=2)
    with_g2 = make_llp(camera, conf_dict, uniform_belief(), horizon=2, use_g2=True)
    with_g2.update_regions(make_hlp().regions)
    state = start_state(camera)
    reward_without = without._simulate_trajectory(state, ["left", "left"], [])[0]
    reward_with = with_g2._simulate_trajectory(state, ["left", "left"], [])[0]
    assert reward_with < reward_without


# --------------------------------------------------------------------------------------
# Repeatable runs and decision order
# --------------------------------------------------------------------------------------


def actions_of_run(case_name, strategy, tmp_path, label):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_planners_smoke import build_simulator, TUNE_SEEDS

    folder = tmp_path / label
    folder.mkdir()
    records = build_simulator(case_name, strategy, TUNE_SEEDS[0], str(folder), n_steps=6).run_measured()
    return [tuple(agent["action"] for agent in record["agents"]) for record in records]


@pytest.mark.parametrize("strategy", ["dec_mcts", "mh_dec_mcts_full", "mh_dec_mcts_efficient"])
def test_same_seed_repeats_the_same_run(strategy, tmp_path):
    assert actions_of_run("case1", strategy, tmp_path, "a") == actions_of_run("case1", strategy, tmp_path, "b")


def test_hlp_replans_on_planning_cycles_not_seconds():
    hlp = HighLevelPlanner(0, 1, GRID.shape, tile_size=(100, 100), horizon=1, num_iterations=2, replan_interval=3, seed=SEED)
    hlp.update_belief(uniform_belief())
    replans = []
    for _ in range(7):
        before = hlp._last_replan_call
        hlp.plan((200.0, 200.0))
        replans.append(hlp._last_replan_call != before)
    assert replans == [True, False, False, True, False, False, True]


def test_every_root_action_is_scored_with_no_budget():
    camera, conf_dict = setup_camera()
    llp = make_llp(camera, conf_dict, uniform_belief(), horizon=1, num_iterations=0)
    state = start_state(camera)
    llp.plan(state)
    scored = {a for a, v in llp._mcts_action_values.items() if np.isfinite(v)}
    assert scored == set(llp._admissible_actions(state[:2], state[2]))
