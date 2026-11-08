"""
Horizon-1 equivalence: at horizon 1 every MCTS planner must act exactly like greedy.

With a one-step horizon there is no future to plan for, so the best action is the one with the
highest one-step score. A planner that disagrees with greedy therefore has a real mismatch in its
footprint, sensor model, discount, reward accounting or action set.

Lockstep shadowing: the real pipeline runs greedy. At every step, before greedy's action is applied,
each planner under test is asked what it would do from the same belief and position. Everyone stays
on greedy's trajectory, so one early mistake cannot hide or cause later ones.

Cases and references come from test_planners_smoke.CASES:
    n1      N=1: plain IG, the only reference that needs no neighbours
    case1   N=4, R=5, BM: greedy IGd
    case2   N=4, R=inf, BS: greedy IG

Planner settings are the shipped strategy JSONs except horizon 1, the iteration budget, and g2. The
g2 term is off by default, because with it on the LLP's reward is not one-step IG by construction.
SMOKE_USE_G2=1 turns it back on to measure its effect.

Iteration budgets:
    0   no search at all
    7   one iteration per primitive action, the least that can see every action
    30  the LLP budget in the shipped MH strategy configs

Ties: in lockstep a planner passes a step if its action scores within TIE_TOLERANCE of greedy's
best, so these tests isolate scoring from tie-breaking.

Closed loop: the real Simulator run with each planner must repeat greedy's run action for action.
Tie-breaking counts here, because one different tied pick sends the run down a different trajectory.
Only case2 runs this way: greedy IGd decides in a random agent order that the MCTS loop does not
share, while lockstep aligns that order for the IGd comparison.

Direct checks test IG, the action set, ties and the teammate discount without any search, RNG or
wall-clock dependence, so a failure there names the cause.

Run with:
    conda activate active_sensing
    python -m pytest tests/test_horizon1_equivalence.py -q    # assertions
    python tests/test_horizon1_equivalence.py                 # summary table

Environment overrides: SMOKE_STEPS (default 100), SMOKE_SEED (default the first tune seed, so the
report seeds stay untouched), SMOKE_USE_G2=1.
"""

import os
import sys
import tempfile
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_planners_smoke import CASES, GRID, REFERENCE, STRATEGY_CONFIGS, TUNE_SEEDS, build_config, build_simulator

from dec_mcts import DecMCTSIntent, DecMCTSNode, DecMCTSPlanner
from experiment_utils import initialize_agent
from greedy_ig_planner import GreedyIGPlanner
from helper import expected_posterior, uav_position
from hierarchical_dec_mcts import LLIntent, LowLevelPlanner
from mcts import MCTSNode
from uav_camera import Camera

SEED = int(os.environ.get("SMOKE_SEED", TUNE_SEEDS[0]))
N_STEPS = int(os.environ.get("SMOKE_STEPS", 100))  # n_steps of the baseline configs
USE_G2 = os.environ.get("SMOKE_USE_G2") == "1"
HORIZON = 1
BUDGETS = (0, 7, 30)
TIE_TOLERANCE = 1e-6  # relative to the best greedy score at that step
VALUE_TOLERANCE = 1e-3  # relative error allowed between planner and greedy values

# The legacy single-agent planner has no strategy JSON and no way to account for teammates.
LEGACY = "mcts"
PLANNERS = (*STRATEGY_CONFIGS, LEGACY)


def horizon_config(case_name, strategy, budget):
    """The shipped strategy settings with only horizon, budget and g2 changed."""
    if strategy == LEGACY:
        return {}
    config = build_config(case_name, strategy, SEED, N_STEPS)
    if strategy == "dec_mcts":
        config["dec_mcts"].update(horizon=HORIZON, iterations=budget, timeout=60.0)
    else:
        config["hierarchical_dec_mcts"].update(llp_horizon=HORIZON, llp_iterations=budget, use_g2=USE_G2)
    return config


def legacy_mcts_params(budget):
    return {"planning_depth": HORIZON, "num_iterations": budget, "parallel": 1, "timeout": 60.0}


def build_shadow_team(case_name, strategy, budget, simulator):
    """One planner per agent. The team shares its own intent bus via a stub coordinator."""
    num_agents = CASES[case_name].num_agents
    config = horizon_config(case_name, strategy, budget)
    # Strategies must not share a bus, or they would react to each other's intents.
    team_bus = SimpleNamespace(num_agents=num_agents) if num_agents > 1 else None
    hrange = simulator.agents[0]["camera"].get_hrange()
    team = []
    for agent_id in range(num_agents):
        state = initialize_agent(
            agent_id=agent_id, grid_info=GRID, start_position=(0.0, 0.0),
            action_strategy=strategy, conf_dict=simulator.conf_dict, corr_type=simulator.corr_type,
            mcts_params=legacy_mcts_params(budget) if strategy == LEGACY else {},
            optimal_alt=21.5, min_alt=None, overlap=None, seed=SEED, coordinator=team_bus,
            start_altitude=hrange[0], experiment_config=config,
        )
        team.append(state)
    return team


# ---------------------------------------------------------------------------
# What a planner believes each action is worth, for the value comparison
# ---------------------------------------------------------------------------


def planner_action_values(shadow, returned_scores):
    planner = shadow["planner"]
    if hasattr(planner, "_dec_mcts_planner"):
        values = planner._dec_mcts_planner.get_action_values()
    elif hasattr(planner, "_hierarchical_planner"):
        values = dict(planner._hierarchical_planner.llp._mcts_action_values)
    else:
        values = returned_scores or {}
    return {
        action: float(value)
        for action, value in values.items()
        if not str(action).startswith("_") and np.isfinite(value)
    }


# ---------------------------------------------------------------------------
# Lockstep driver
# ---------------------------------------------------------------------------


@dataclass
class Mismatch:
    step: int
    agent_id: int
    picked: str
    greedy_best: str
    regret: float  # fraction of greedy's best score lost by the planner's pick


@dataclass
class ShadowResult:
    decisions: int = 0
    literal_matches: int = 0  # picked exactly greedy's action, ties not forgiven
    mismatches: list = field(default_factory=list)
    value_errors: list = field(default_factory=list)  # (step, agent, action, ours, greedy)
    worst_value_error: float = 0.0  # over every compared value, within tolerance or not
    picks: dict = field(default_factory=dict)

    @property
    def agreement(self):
        return 1.0 - len(self.mismatches) / max(1, self.decisions)


def decision_orders(num_agents, mode):
    """Same agent order as Simulator._select_greedy_baseline_actions uses for IGd."""
    rng = np.random.default_rng(17) if mode == "IGd_BM" and num_agents > 1 else None
    while True:
        yield list(rng.permutation(num_agents)) if rng else list(range(num_agents))


def run_lockstep(case_name, budgets=BUDGETS, strategies=None, n_steps=None):
    case = CASES[case_name]
    strategies = strategies or [s for s in PLANNERS if case.num_agents == 1 or s != LEGACY]
    with tempfile.TemporaryDirectory() as tmp:
        simulator = build_simulator(case_name, REFERENCE, SEED, tmp, N_STEPS)
        teams = {
            (strategy, budget): build_shadow_team(case_name, strategy, budget, simulator)
            for strategy in strategies
            for budget in budgets
        }
        results = {key: ShadowResult() for key in teams}
        orders = decision_orders(case.num_agents, case.mode)
        for step in range(n_steps if n_steps is not None else N_STEPS):
            simulator._select_agent_actions(step)  # greedy decides, not yet applied
            order = next(orders)
            for key, team in teams.items():
                for agent_id in order:
                    agent = simulator.agents[agent_id]
                    _shadow_decide(key, team[agent_id], agent, step, results[key])
            simulator._update_agent_positions()
            observations = simulator._process_agent_observations()
            if simulator.coordinator is not None:
                simulator._perform_belief_fusion(observations, step)
    return results


def _shadow_decide(key, shadow, agent, step, result):
    greedy_scores = {
        a: float(v)
        for a, v in agent["info_gain_action"].items()
        if not a.startswith("_") and np.isfinite(v)
    }
    greedy_action = str(agent["_next_action"])
    best = max(greedy_scores.values())

    pos = agent["uav_pos"]
    shadow["camera"].set_position(pos.position)
    shadow["camera"].set_altitude(pos.altitude)
    picked, returned = shadow["planner"].select_action(
        agent["belief_map"].copy(), agent["uav_positions"]
    )
    picked = str(picked)

    result.decisions += 1
    result.literal_matches += picked == greedy_action
    result.picks[picked] = result.picks.get(picked, 0) + 1
    picked_score = greedy_scores.get(picked, float("-inf"))
    if picked_score < best - TIE_TOLERANCE * max(abs(best), 1.0):
        regret = (best - picked_score) / best if best > 0 else 0.0
        result.mismatches.append(Mismatch(step, agent["agent_id"], picked, greedy_action, regret))

    for action, value in planner_action_values(shadow, returned).items():
        reference = greedy_scores.get(action)
        if reference is None:
            continue
        error = abs(value - reference) / max(abs(reference), 1.0)
        result.worst_value_error = max(result.worst_value_error, error)
        if error > VALUE_TOLERANCE:
            result.value_errors.append((step, agent["agent_id"], action, value, reference))


def closed_loop_actions(case_name, strategy=REFERENCE, budget=0):
    """Executed actions per step from a full Simulator run. Greedy's run is cached."""
    key = (case_name, strategy, budget)
    if key in _CLOSED_LOOP:
        return _CLOSED_LOOP[key]
    config = None if strategy == REFERENCE else horizon_config(case_name, strategy, budget)
    with tempfile.TemporaryDirectory() as tmp:
        records = build_simulator(case_name, strategy, SEED, tmp, N_STEPS, config=config).run_measured()
    actions = [tuple(agent["action"] for agent in record["agents"]) for record in records]
    if strategy == REFERENCE:
        _CLOSED_LOOP[key] = actions
    return actions


_CLOSED_LOOP = {}
_CACHE = {}


def lockstep(case_name):
    """Each case runs once; all planners and budgets share that one trajectory."""
    if case_name not in _CACHE:
        _CACHE[case_name] = run_lockstep(case_name)
    return _CACHE[case_name]


def describe(result):
    lines = [
        f"agreed with greedy on {result.decisions - len(result.mismatches)}"
        f"/{result.decisions} decisions; picks: {result.picks}"
    ]
    for m in result.mismatches[:5]:
        lines.append(
            f"  step {m.step} agent {m.agent_id}: picked {m.picked!r}, greedy {m.greedy_best!r}, "
            f"lost {m.regret:.1%} of best score"
        )
    return "\n".join(lines)


def _cases(budgets):
    for case_name, case in CASES.items():
        for strategy in PLANNERS:
            if case.num_agents > 1 and strategy == LEGACY:
                continue
            for budget in budgets:
                yield case_name, strategy, budget


# ---------------------------------------------------------------------------
# Direct checks: no search, no RNG, no wall-clock dependence
# ---------------------------------------------------------------------------

ACTIONS = ["up", "down", "front", "back", "right", "left", "hover"]


def direct_setup():
    camera = Camera(
        GRID, 60, camera_altitude=None, f_overlap=None, s_overlap=None, seed=SEED,
        a=1.0, b=0.015,
    )
    conf_dict = camera.theoretical_conf_dict()
    occupied = np.random.default_rng(SEED).uniform(0.02, 0.98, GRID.shape)
    belief = np.stack([1.0 - occupied, occupied], axis=2)
    return camera, conf_dict, belief


def sample_states(camera):
    """Corners, edges and the centre at every altitude level."""
    altitudes = [camera.h_range[0] + k * camera.h_step for k in range(6)]
    positions = [(-25.0, -25.0), (25.0, 25.0), (-25.0, 0.0), (0.0, 25.0), (0.0, 0.0), (6.25, -3.125)]
    return [(position, altitude) for position in positions for altitude in altitudes]


def greedy_action_data(camera, conf_dict, belief, position, altitude):
    planner = GreedyIGPlanner(0, camera, GRID, conf_dict=conf_dict)
    return planner.score_admissible_actions(position, altitude, belief)


def assert_close(ours, greedy, label):
    assert abs(ours - greedy) <= VALUE_TOLERANCE * max(abs(greedy), 1.0), (
        f"{label}: planner {ours:.6f}, greedy {greedy:.6f}"
    )


def igd_scores_with_teammate(camera, conf_dict, belief, position, altitude, teammate):
    """Greedy IGd scores with one teammate staying at the given position."""
    planner = GreedyIGPlanner(0, camera, GRID, conf_dict=conf_dict, enable_discounting=True)
    planner.update_belief(belief)
    planner.update_teammate_states({1: (teammate, altitude)})
    planner.plan(position, altitude)
    return {a: v for a, v in planner.get_action_scores().items() if np.isfinite(v)}


def footprint_bounds(camera, position, altitude):
    [[imin, imax], [jmin, jmax]] = camera.get_range(
        position=position, altitude=altitude, index_form=True
    )
    return imin, imax, jmin, jmax


def test_llp_ig_footprint_and_sensor_match_greedy():
    camera, conf_dict, belief = direct_setup()
    llp = LowLevelPlanner(0, camera, GRID, conf_dict=conf_dict, horizon=HORIZON)
    llp.update_belief(belief)
    no_discount = np.ones(GRID.shape)
    for position, altitude in sample_states(camera):
        for action, data in greedy_action_data(camera, conf_dict, belief, position, altitude).items():
            ig, footprint = llp._compute_ig(data["next_position"], data["next_altitude"], no_discount)
            label = f"{action} from {position} at {altitude:.2f}"
            assert footprint == data["footprint"], label
            assert_close(ig, data["ig"], label)


def test_dec_mcts_ig_footprint_and_sensor_match_greedy():
    camera, conf_dict, belief = direct_setup()
    for position, altitude in sample_states(camera):
        for action, data in greedy_action_data(camera, conf_dict, belief, position, altitude).items():
            state = {
                "uav_pos": uav_position((data["next_position"], data["next_altitude"])),
                "belief": belief,
            }
            node = DecMCTSNode(state, camera, conf_dict=conf_dict, seed=0)
            bounds = footprint_bounds(camera, data["next_position"], data["next_altitude"])
            label = f"{action} from {position} at {altitude:.2f}"
            assert bounds == data["footprint"], label
            assert_close(node.compute_ig_reward(state, *bounds), data["ig"], label)


def test_legacy_mcts_ig_footprint_and_sensor_match_greedy():
    camera, conf_dict, belief = direct_setup()
    for position, altitude in sample_states(camera):
        for action, data in greedy_action_data(camera, conf_dict, belief, position, altitude).items():
            future = uav_position((data["next_position"], data["next_altitude"]))
            node = MCTSNode({"uav_pos": future, "belief": belief}, camera, conf_dict=conf_dict, seed=0)
            imin, imax, jmin, jmax = footprint_bounds(camera, future.position, future.altitude)
            prior = belief[imin:imax, jmin:jmax, 1]
            s0, s1 = node.sensor_model(future)
            ig = node.compute_reward(prior, *expected_posterior(prior, s0, s1))
            assert_close(ig, data["ig"], f"{action} from {position} at {future.altitude:.2f}")


def tie_belief(camera, conf_dict, position, altitude):
    """
    Greedy scores hover and back as an exact tie on this belief: all real uncertainty
    is in a patch both footprints see. Cells only hover sees are settled at p=1e-9,
    which greedy's IG treats as zero.
    """
    occupied = np.full(GRID.shape, 1e-15)
    uniform = np.full((*GRID.shape, 2), 0.5)
    data = greedy_action_data(camera, conf_dict, uniform, position, altitude)
    h_imin, h_imax, h_jmin, h_jmax = data["hover"]["footprint"]
    b_imin, b_imax, b_jmin, b_jmax = data["back"]["footprint"]
    occupied[h_imin:h_imax, h_jmin:h_jmax] = 1e-9
    occupied[b_imin:b_imax, b_jmin:b_jmax] = 1e-15
    occupied[max(h_imin, b_imin):min(h_imax, b_imax), max(h_jmin, b_jmin):min(h_jmax, b_jmax)] = 0.3
    return np.stack([1.0 - occupied, occupied], axis=2)


TIE_POSITION, TIE_ALTITUDE_LEVEL = (0.0, 0.0), 2


def test_greedy_scores_hover_and_back_as_a_tie_on_settled_cells():
    camera, conf_dict, _ = direct_setup()
    altitude = camera.h_range[0] + TIE_ALTITUDE_LEVEL * camera.h_step
    belief = tie_belief(camera, conf_dict, TIE_POSITION, altitude)
    data = greedy_action_data(camera, conf_dict, belief, TIE_POSITION, altitude)
    assert abs(data["hover"]["ig"] - data["back"]["ig"]) <= 1e-9


def test_llp_keeps_greedy_tie_on_settled_cells():
    camera, conf_dict, _ = direct_setup()
    altitude = camera.h_range[0] + TIE_ALTITUDE_LEVEL * camera.h_step
    belief = tie_belief(camera, conf_dict, TIE_POSITION, altitude)
    data = greedy_action_data(camera, conf_dict, belief, TIE_POSITION, altitude)
    llp = LowLevelPlanner(0, camera, GRID, conf_dict=conf_dict, horizon=HORIZON)
    llp.update_belief(belief)
    no_discount = np.ones(GRID.shape)
    hover, _ = llp._compute_ig(data["hover"]["next_position"], data["hover"]["next_altitude"], no_discount)
    back, _ = llp._compute_ig(data["back"]["next_position"], data["back"]["next_altitude"], no_discount)
    assert abs(hover - back) <= 1e-9, f"hover {hover:.9f}, back {back:.9f}"


def test_dec_mcts_keeps_greedy_tie_on_settled_cells():
    camera, conf_dict, _ = direct_setup()
    altitude = camera.h_range[0] + TIE_ALTITUDE_LEVEL * camera.h_step
    belief = tie_belief(camera, conf_dict, TIE_POSITION, altitude)
    data = greedy_action_data(camera, conf_dict, belief, TIE_POSITION, altitude)
    rewards = {}
    for action in ("hover", "back"):
        state = {
            "uav_pos": uav_position((data[action]["next_position"], data[action]["next_altitude"])),
            "belief": belief,
        }
        node = DecMCTSNode(state, camera, conf_dict=conf_dict, seed=0)
        rewards[action] = node.compute_ig_reward(state, *data[action]["footprint"])
    assert abs(rewards["hover"] - rewards["back"]) <= 1e-9, rewards


def test_llp_action_set_matches_greedy_after_up_and_down_moves():
    """Altitudes reached by going up then down carry float drift."""
    camera, conf_dict, _ = direct_setup()
    llp = LowLevelPlanner(0, camera, GRID, conf_dict=conf_dict, horizon=HORIZON)
    camera.set_position((0.0, 0.0))
    camera.set_altitude(camera.h_range[0])
    for move in ["up"] * 5 + ["down"] * 5 + ["up"] * 3 + ["down"] * 2 + ["up"]:
        camera.set_altitude(camera.x_future(move)[1])
        altitude = camera.altitude
        greedy = {a for a in ACTIONS if camera.x_future(a) is not None}
        planner = {a for a in ACTIONS if llp._apply_action((0.0, 0.0), altitude, a) is not None}
        assert planner == greedy, f"at altitude {altitude!r} after moving {move}"


def test_llp_discount_equals_igd_for_overlapping_teammate():
    camera, conf_dict, belief = direct_setup()
    position, altitude = (0.0, 0.0), camera.h_range[0] + 2 * camera.h_step
    teammate = (camera.xy_step, 0.0)
    expected = igd_scores_with_teammate(camera, conf_dict, belief, position, altitude, teammate)

    llp = LowLevelPlanner(0, camera, GRID, conf_dict=conf_dict, horizon=HORIZON)
    llp.update_belief(belief)
    llp.update_teammate_intents(
        {1: LLIntent(agent_id=1, footprint_sequence=[footprint_bounds(camera, teammate, altitude)])}, {}
    )
    discount = llp._compute_teammate_coverage_mask()
    for action, data in greedy_action_data(camera, conf_dict, belief, position, altitude).items():
        ig, _ = llp._compute_ig(data["next_position"], data["next_altitude"], discount)
        assert_close(ig, expected[action], f"LLP {action} next to a teammate")


def test_dec_mcts_discount_equals_igd_for_overlapping_teammate():
    camera, conf_dict, belief = direct_setup()
    position, altitude = (0.0, 0.0), camera.h_range[0] + 2 * camera.h_step
    teammate = (camera.xy_step, 0.0)
    expected = igd_scores_with_teammate(camera, conf_dict, belief, position, altitude, teammate)

    planner = DecMCTSPlanner(0, camera, GRID, conf_dict=conf_dict, config={"horizon": HORIZON})
    planner.update_state(position, altitude, belief)
    planner.update_teammate_intents(
        {1: DecMCTSIntent(agent_id=1, footprint_sequence=[footprint_bounds(camera, teammate, altitude)])}
    )
    mask = planner._compute_teammate_mask()
    for action, data in greedy_action_data(camera, conf_dict, belief, position, altitude).items():
        state = {
            "uav_pos": uav_position((data["next_position"], data["next_altitude"])),
            "belief": belief,
            "teammate_mask": mask,
        }
        node = DecMCTSNode(state, camera, conf_dict=conf_dict, config=planner.config, seed=0)
        bounds = footprint_bounds(camera, data["next_position"], data["next_altitude"])
        reward = node.compute_ig_reward(state, *bounds) - node.compute_overlap_penalty(state, *bounds)
        assert_close(reward, expected[action], f"Dec-MCTS {action} next to a teammate")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name,strategy,budget", list(_cases(BUDGETS)))
def test_horizon1_picks_greedy_action_every_step(case_name, strategy, budget):
    result = lockstep(case_name)[(strategy, budget)]
    assert not result.mismatches, describe(result)


@pytest.mark.parametrize("case_name,strategy,budget", list(_cases([b for b in BUDGETS if b])))
def test_horizon1_action_values_equal_greedy_scores(case_name, strategy, budget):
    """Every root action value the planner holds must equal greedy's score for it."""
    result = lockstep(case_name)[(strategy, budget)]
    assert not result.value_errors, (
        f"{len(result.value_errors)} action values differ from greedy; worst relative "
        f"error {result.worst_value_error:.3g}; first (step, agent, action, planner, "
        f"greedy): {result.value_errors[0]}"
    )


@pytest.mark.parametrize("strategy", list(STRATEGY_CONFIGS))
def test_horizon1_closed_loop_repeats_greedy_run(strategy):
    reference = closed_loop_actions("case2")
    actions = closed_loop_actions("case2", strategy, budget=30)
    differing = [t for t, (ours, theirs) in enumerate(zip(actions, reference)) if ours != theirs]
    assert not differing, (
        f"{len(differing)}/{len(reference)} steps differ from greedy's run, first at step {differing[0]}"
    )


def main():
    print(f"horizon={HORIZON}  steps={N_STEPS}  seed={SEED}  use_g2={USE_G2}\n")
    header = f"{'case':<7}{'planner':<24}{'iters':>6}{'agree':>9}{'literal':>9}{'value err':>11}  picks"
    print(header)
    print("-" * len(header))
    failed = 0
    for case_name in CASES:
        for (strategy, budget), result in lockstep(case_name).items():
            failed += bool(result.mismatches)
            picks = " ".join(f"{a}:{n}" for a, n in sorted(result.picks.items(), key=lambda i: -i[1]))
            error = f"{result.worst_value_error:.3g}" if budget else "-"
            literal = result.literal_matches / max(1, result.decisions)
            print(f"{case_name:<7}{strategy:<24}{budget:>6}{result.agreement:>9.1%}{literal:>9.1%}{error:>11}  {picks}")
    print(f"\n{failed} planner/budget combinations disagreed with greedy at least once")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
