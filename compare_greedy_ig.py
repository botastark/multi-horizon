#!/usr/bin/env python3
"""
Validation of MH Greedy IG against PA-Dev (ground truth).

PA-Dev is the reference implementation; MH is the baseline under test.
Reports per-step deltas as (MH − PA_ref) so negative = MH is better.

Configurations compared:
  1. IG_BS  / R=inf  (PA: planner_type="selfish"                  | MH: news_mode="IG_BS")
  2. IGd_BM / R=5    (PA: planner_type="mine_IoU_async_no_pred"   | MH: news_mode="IGd_BM")

Usage:
  python compare_greedy_ig.py [--steps 100] [--seed 42] [--out plots/comparison]
"""

import argparse
import importlib
import os
import sys
import shutil
import tempfile
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
MH_SRC = os.path.join(os.path.dirname(__file__), "src")
PA_SRC = "/home/bota/repos/Precision-Agriculture-Dev"

# MH imports (src/ must be on path)
sys.path.insert(0, MH_SRC)

from helper import gaussian_random_field
from orthomap import Field
from mapper_LBP import OccupancyMap as OM
from planner import planning
from uav_camera import Camera
from multi_agent_coordinator import MultiAgentCoordinator
from experiment_utils import initialize_agent
from simulator import Simulator
from config_loader import load_config


# ─────────────────────────────────────────────────────────────────────────────
# Grid / camera constants shared by MH
# ─────────────────────────────────────────────────────────────────────────────
class MHGrid:
    x = 50
    y = 50
    length = 0.125
    shape = (int(50 / 0.125), int(50 / 0.125))  # (400, 400)
    center = True


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_mh_config(path: str) -> dict:
    """Load and fully resolve a standalone MH config (handles 'shared' merging)."""
    configs = load_config(path)
    return configs[0]


def _make_tmp_dir():
    d = tempfile.mkdtemp(prefix="mh_compare_")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# MH runner
# ─────────────────────────────────────────────────────────────────────────────


def run_mh(
    config: dict,
    seed: int,
    n_steps: int,
    news_mode: str,
    results_folder: str,
    debug: bool = False,
) -> dict:
    """
    Run one MH greedy-IG iteration and return step-wise metrics.

    Returns dict with keys: fused_entropy, fused_mse, coverage,
    start_positions, actions_per_agent
    """
    grid_info = MHGrid()
    grf_r = config.get("cluster_radius", 4)

    camera1 = Camera(
        grid_info,
        60,
        camera_altitude=None,
        f_overlap=None,
        s_overlap=None,
        seed=seed,
        a=1.0,
        b=0.015,
    )
    camera_hrange = camera1.get_hrange()

    map_obj = Field(
        grid_info,
        grf_r,
        sweep=config.get("action_strategy", "greedy_ig"),
        h_range=camera_hrange,
        seed=seed,
    )
    map_obj.reset(seed=seed)
    ground_truth_map = map_obj.get_ground_truth()

    conf_dict = camera1.theoretical_conf_dict()

    # Determine news-sharing flags from news_mode
    dec_config = config.setdefault("decentralized", {})
    greedy_cfg = config.setdefault("greedy_ig", {})

    if news_mode == "IG_BS":
        dec_config["position_sharing"] = False
        dec_config["news_sharing"] = True
        ma_config = config.setdefault("multi_agent", {})
        ma_config["news_mode"] = "BS"
        ma_config["news_inference_type"] = "OG"
        ma_config["fusion_eps"] = 0.0
        ma_config["metric_aggregation"] = "fused_mean"
        ma_config["clip_metric_beliefs"] = False
        coord_news_mode = "BS"
        dec_config.setdefault("radius_multiplier", -1)  # infinite comm
    elif news_mode == "IGd_BM":
        dec_config["position_sharing"] = True
        dec_config["news_sharing"] = False
        ma_config = config.setdefault("multi_agent", {})
        ma_config["news_inference_type"] = "Bypass"
        ma_config["fusion_eps"] = 0.0
        ma_config["metric_aggregation"] = "fused_mean"
        ma_config["clip_metric_beliefs"] = False
        coord_news_mode = "BM"
        greedy_cfg["enable_discounting"] = True
        dec_config.setdefault("radius_multiplier", 5)  # R=5
    else:
        raise ValueError(f"Unknown news_mode: {news_mode}")

    num_agents = config.get("num_agents", 4)
    action_strategy = config.get("action_strategy", "greedy_ig")

    coordinator = MultiAgentCoordinator(
        grid_shape=grid_info.shape,
        config=config,
        conf_dict=conf_dict,
        correlation_type="pairwise",
        news_mode=coord_news_mode,
        mode=news_mode,
        grid_info=grid_info,
        debug_logs=debug,
    )

    start_positions = coordinator.reset_start_position(
        grid_info=grid_info,
        start_position=config.get("start_position", "corner"),
        min_distance=10.0,
        seed=seed,
        camera_hrange=camera_hrange,
    )

    agents = []
    for agent_id in range(num_agents):
        sp = start_positions[agent_id]
        start_xy = (sp[0], sp[1])
        start_z = sp[2]

        agent_state = initialize_agent(
            agent_id=agent_id,
            grid_info=grid_info,
            start_position=start_xy,
            action_strategy=action_strategy,
            conf_dict=conf_dict,
            corr_type="pairwise",
            mcts_params={},
            optimal_alt=21.5,
            min_alt=None,
            overlap=None,
            seed=seed,
            coordinator=coordinator,
            start_altitude=start_z,
            debug_logs=debug,
        )
        agents.append(agent_state)

        coordinator.update_agent_state(
            agent_id=agent_id,
            position=start_xy,
            altitude=start_z,
        )

    # PA uses the same RNG object for camera observations and greedy
    # tie-breaking. Share one MH camera RNG so draw order stays aligned.
    shared_obs_rng = np.random.default_rng(seed)
    map_obj.rng = shared_obs_rng
    for agent_state in agents:
        agent_state["camera"].rng = shared_obs_rng

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    os.makedirs(results_folder, exist_ok=True)

    simulator = Simulator(
        agents=agents,
        map_obj=map_obj,
        ground_truth_map=ground_truth_map,
        conf_dict=conf_dict,
        grid_info=grid_info,
        n_steps=n_steps,
        results_folder=results_folder,
        corr_type="pairwise",
        e_margin=None,
        grf_r=grf_r,
        iter_idx=0,
        enable_stepwise_plotting=False,
        enable_logging=False,
        action_strategy=action_strategy,
        coordinator=coordinator,
        multi_agent_logger=None,
        run_id=run_id,
        debug_logs=debug,
    )

    result = simulator.run()

    # Collect per-agent step histories (agent["actions"] is list of action names)
    agent_actions = {a["agent_id"]: a["actions"] for a in result["agents"]}
    start_pos_real = [(sp[0], sp[1]) for sp in start_positions]

    return {
        "fused_entropy": simulator.fused_entropy_history,  # len = n_steps
        "fused_mse": simulator.fused_mse_history,  # len = n_steps
        "coverage": simulator.combined_coverage_history,  # len = n_steps
        "start_positions": start_pos_real,
        "actions_per_agent": agent_actions,
        "ground_truth": ground_truth_map,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PA runner
# ─────────────────────────────────────────────────────────────────────────────


def run_pa(
    planner_type: str,
    cluster_radius: int,
    n_agents: int,
    n_steps: int,
    seed: int,
    radius_multiplier: int = -1,
    news_inference_type: str = "Bypass",
    debug: bool = False,
) -> dict:
    """
    Run one PA-Dev greedy-IG iteration (selfish or mine_IoU_async_no_pred)
    and return step-wise metrics.

    NOTE: PA uses a 50×50 coarse grid (1 m/cell) vs MH's 400×400 fine grid.
          Both use Gaussian r=cluster_radius with seed=seed for map generation.
          Starting positions are forced to MH's 4 outer corners so both systems
          begin from identical (x, y) locations.

    Returns dict with keys: entropy_per_agent, mse_per_agent, fused_entropy,
    fused_mse, coverage, start_positions, actions_per_agent
    """
    # Lazy import using spec so PA's simulator.py is loaded explicitly,
    # avoiding collision with MH's src/simulator.py which is already on sys.path.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pa_simulator", os.path.join(PA_SRC, "simulator.py")
    )
    pa_sim = importlib.util.module_from_spec(spec)
    # Temporarily add PA_SRC so PA's intra-module imports resolve correctly
    sys.path.insert(0, PA_SRC)
    spec.loader.exec_module(pa_sim)

    MappingEnv = pa_sim.MappingEnv
    Mapper = pa_sim.Mapper
    Planner = pa_sim.Planner
    H = pa_sim.H

    # Build experiment params matching the PA convention for N=4, r=4
    iep = dict(
        a0=1.0,
        a1=1.0,
        b0=0.015,
        b1=0.015,
        inference_type="LBP_cts_vectorized",
        cluster_radius=cluster_radius,
        news_inference_type=news_inference_type,
        map_type="gaussian",
        planner_type=planner_type,
        env_type="adhoc",
        x="BL",
        y="BL",
        altitude=0,
        n_agents=n_agents,
        centralized=False,
        n_runs=1,
        n_steps=n_steps,
        render=False,
        weights_type="adaptive",
        radius_multiplier=radius_multiplier,
    )

    # Override map RNG seed so it matches our chosen seed
    # PA normally uses default_rng(123) - we'll set it to the given seed
    pa_sim.MappingEnv.__init__  # just to check it exists

    env = MappingEnv(
        field_len=50.0,
        fov=np.pi / 3,
        **iep,
    )
    # Patch map_rng and agent_position_rng to use our seed
    env.map_rng = np.random.default_rng(seed)
    env.agent_position_rng = np.random.default_rng(seed)

    mapper = Mapper(
        env.n_cell,
        env.min_space_z,
        env.max_space_z,
        **iep,
    )

    planner = Planner(
        env.action_to_direction,
        env.altitude_to_size,
        env.position_graph,
        env.position_to_data,
        env.regions_limits,
        env.optimal_altitude,
        **iep,
    )

    # ── single run ──────────────────────────────────────────────────────────
    map_ground_truth = env.generate_map()  # uses env.map_rng (patched above)
    env.reset_map_beliefs()
    env.reset_agents_position(**iep)  # initial call required to set state shape
    planner.reset_sweep()

    # Override agent start positions to MH's 4 outer corners so both systems
    # start from exactly the same (x, y) locations.
    _corners = [
        (-25.0, -25.0),
        (25.0, -25.0),
        (-25.0, 25.0),
        (25.0, 25.0),
    ]
    for agent in env.agents:
        x, y = _corners[agent.id]
        z = (iep["altitude"] + 1) * env.v_displacement
        agent.state.set_position(np.array([x, y, z]))
        pa_sim.states[agent.id, :] = agent.state.position
    if debug:
        print(f"[PA] overriding positions to corners: {_corners[:n_agents]}")

    # Share a single observation RNG across all PA agents (matches MH's single-RNG draw order)
    shared_obs_rng = np.random.default_rng(seed)
    for agent in env.agents:
        agent.rng = shared_obs_rng

    # Record starting positions before any movement
    start_positions = [(a.state.position[0], a.state.position[1]) for a in env.agents]
    # if debug:
    # print(f"[PA] start_positions = {start_positions}")

    # Per-step histories
    entropy_per_agent = [[] for _ in range(n_agents)]  # sum entropy per agent per step
    mse_per_agent = [[] for _ in range(n_agents)]
    fused_entropy_hist = []
    fused_mse_hist = []
    coverage_hist = []
    actions_per_agent = {i: [] for i in range(n_agents)}

    # ── Step loop matching MH Simulator.run() exactly ────────────────────────
    # MH per step k:
    #   1. record metrics (pure prior at k=0; post-obs-from-k-1 at k>0)
    #   2. observe at current position (start pos at k=0; moved pos at k>0)
    #   3. plan  (with post-obs beliefs)
    #   4. move
    observations = []  # populated inside loop on first iteration

    for step_index in range(n_steps):
        # 1. Record metrics with CURRENT beliefs (prior at step 0)
        planner.compute_map_belief_entropies()
        fused_belief = pa_sim.map_beliefs.mean(axis=2)
        fused_entropy_hist.append(float(np.sum(H(fused_belief))))
        fused_mse_hist.append(
            float(np.mean(np.square(map_ground_truth - fused_belief)))
        )
        decided = (fused_belief > 0.55) | (fused_belief < 0.45)
        coverage_hist.append(float(decided.mean()))

        # 2. Observe at current position (start pos at step 0, moved pos at step>0)
        observations = env.get_observations(map_ground_truth)
        mapper.set_pairwise_potential_z(env.agents, observations)
        mapper.update_map_beliefs(env.agents, observations)
        mapper.update_news_and_fuse_map_beliefs(env.agents, observations)

        # 3. Plan using post-obs beliefs
        planner.compute_map_belief_entropies()
        actions, actions_data = planner.get_actions(env.agents, observations)
        for agent in env.agents:
            aid = agent.id
            ent = float(np.sum(pa_sim.map_belief_entropies[:, :, aid]))
            mse = float(
                np.mean(np.square(map_ground_truth - pa_sim.map_beliefs[:, :, aid]))
            )
            entropy_per_agent[aid].append(ent)
            mse_per_agent[aid].append(mse)
            actions_per_agent[aid].append(actions[aid])

        # 4. Move
        env.step(actions)

    return {
        "entropy_per_agent": entropy_per_agent,
        "mse_per_agent": mse_per_agent,
        "fused_entropy": fused_entropy_hist,
        "fused_mse": fused_mse_hist,
        "coverage": coverage_hist,
        "start_positions": start_positions,
        "actions_per_agent": actions_per_agent,
        "ground_truth": map_ground_truth,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


def plot_comparison(
    pa_data: dict,
    mh_data: dict,
    label: str,
    out_dir: str,
):
    steps_pa = list(range(len(pa_data["fused_entropy"])))
    steps_mh = list(range(len(mh_data["fused_entropy"])))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"MH (baseline) vs PA-Dev (reference)  —  {label}", fontsize=13)

    # Entropy
    ax = axes[0]
    ax.plot(
        steps_pa,
        pa_data["fused_entropy"],
        label="PA-Dev (reference)",
        color="steelblue",
        lw=2,
    )
    ax.plot(
        steps_mh,
        mh_data["fused_entropy"],
        label="MH (baseline)",
        color="darkorange",
        ls="--",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Total fused entropy")
    ax.set_title("Entropy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # MSE
    ax = axes[1]
    ax.plot(
        steps_pa, pa_data["fused_mse"], label="PA-Dev (ref)", color="steelblue", lw=2
    )
    ax.plot(
        steps_mh,
        mh_data["fused_mse"],
        label="MH (baseline)",
        color="darkorange",
        ls="--",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE (fused belief vs GT)")
    ax.set_title("MSE")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Coverage
    ax = axes[2]
    ax.plot(
        steps_pa, pa_data["coverage"], label="PA-Dev (ref)", color="steelblue", lw=2
    )
    ax.plot(
        steps_mh,
        mh_data["coverage"],
        label="MH (baseline)",
        color="darkorange",
        ls="--",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage (fraction decided cells)")
    ax.set_title("Coverage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"comparison_{label.replace('/', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_table(pa_data: dict, mh_data: dict, label: str, every: int = 10):
    """Print a step-wise comparison table at every `every` steps."""
    n = min(len(pa_data["fused_entropy"]), len(mh_data["fused_entropy"]))
    header = (
        f"\n{'─'*90}\n"
        f"  {label}   (printed every {every} steps)\n"
        f"{'─'*90}\n"
        f"{'Step':>5}  {'PA-ref entropy':>14}  {'MH-base entropy':>15}  "
        f"{'PA-ref MSE':>11}  {'MH-base MSE':>11}  {'PA-ref cov':>10}  {'MH-base cov':>11}\n"
        f"{'─'*90}"
    )
    print(header)
    for step in range(0, n, every):
        print(
            f"{step:>5}  "
            f"{pa_data['fused_entropy'][step]:>14.4f}  "
            f"{mh_data['fused_entropy'][step]:>15.4f}  "
            f"{pa_data['fused_mse'][step]:>11.6f}  "
            f"{mh_data['fused_mse'][step]:>11.6f}  "
            f"{pa_data['coverage'][step]:>10.4f}  "
            f"{mh_data['coverage'][step]:>11.4f}"
        )
    print(f"{'─'*90}")


def print_start_position_diff(pa_pos, mh_pos, label):
    print(f"\n  [{label}] Starting positions:")
    print(f"  {'Agent':>5}  {'PA-ref (x,y)':>18}  {'MH-base (x,y)':>18}")
    for i, (pp, mp) in enumerate(zip(pa_pos, mh_pos)):
        match = "✓" if np.allclose(pp, mp) else "✗"
        print(f"  {i:>5}  {str(pp):>18}  {str(mp):>18}  {match}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Compare PA-Dev vs MH Greedy IG")
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of simulation steps"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed (map + start positions)"
    )
    parser.add_argument("--agents", type=int, default=4, help="Number of agents")
    parser.add_argument("--radius", type=int, default=4, help="GRF cluster radius")
    parser.add_argument(
        "--out", type=str, default="plots/comparison", help="Output dir for plots"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  MH Greedy IG Validation  (reference: PA-Dev)")
    print(f"  seed={args.seed}  steps={args.steps}  N={args.agents}  r={args.radius}")
    print(f"{'='*70}\n")

    # ── Load MH configs ───────────────────────────────────────────────────────
    cfg_bs = _load_mh_config(
        os.path.join(
            os.path.dirname(__file__), "configs/baseline_greedy_ig_bs_rinf.json"
        )
    )
    cfg_igd = _load_mh_config(
        os.path.join(
            os.path.dirname(__file__), "configs/baseline_greedy_igd_bm_r5.json"
        )
    )
    # Force our args into the configs
    for cfg in [cfg_bs, cfg_igd]:
        cfg["num_agents"] = args.agents
        cfg["cluster_radius"] = args.radius
        cfg["n_steps"] = args.steps
        cfg["seed"] = args.seed
        cfg["enable_plotting"] = False
        cfg["enable_logging"] = False

    tmp_dir = _make_tmp_dir()

    # ──────────────────────────────────────────────────────────────────────────
    # 1.  IG_BS / R=inf
    # ──────────────────────────────────────────────────────────────────────────
    print("┌─ [1/4] Running PA-Dev  IG_BS / R=inf  (planner_type='selfish') ...")
    pa_igbs = run_pa(
        planner_type="selfish",
        cluster_radius=args.radius,
        n_agents=args.agents,
        n_steps=args.steps,
        seed=args.seed,
        radius_multiplier=1000,
        news_inference_type="OG_single",
        debug=args.debug,
    )
    print(f"└─ Done. Start pos: {pa_igbs['start_positions']}\n")

    print("┌─ [2/4] Running MH      IG_BS / R=inf ...")
    mh_igbs = run_mh(
        config=cfg_bs,
        seed=args.seed,
        n_steps=args.steps,
        news_mode="IG_BS",
        results_folder=os.path.join(tmp_dir, "mh_igbs"),
        debug=args.debug,
    )
    print(f"└─ Done. Start pos: {mh_igbs['start_positions']}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 2.  IGd_BM / R=5
    # ──────────────────────────────────────────────────────────────────────────
    print(
        "┌─ [3/4] Running PA-Dev  IGd_BM / R=5  (planner_type='mine_IoU_async_no_pred') ..."
    )
    pa_igd = run_pa(
        planner_type="mine_IoU_async_no_pred",
        cluster_radius=args.radius,
        n_agents=args.agents,
        n_steps=args.steps,
        seed=args.seed,
        radius_multiplier=5,
        news_inference_type="Bypass",
        debug=args.debug,
    )
    print(f"└─ Done. Start pos: {pa_igd['start_positions']}\n")

    print("┌─ [4/4] Running MH      IGd_BM / R=5 ...")
    mh_igd = run_mh(
        config=cfg_igd,
        seed=args.seed,
        n_steps=args.steps,
        news_mode="IGd_BM",
        results_folder=os.path.join(tmp_dir, "mh_igd"),
        debug=args.debug,
    )
    print(f"└─ Done. Start pos: {mh_igd['start_positions']}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 3.  Compare
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STARTING POSITION COMPARISON")
    print("=" * 70)
    print_start_position_diff(
        pa_igbs["start_positions"], mh_igbs["start_positions"], "IG_BS"
    )
    print_start_position_diff(
        pa_igd["start_positions"], mh_igd["start_positions"], "IGd_BM"
    )

    print("\n" + "=" * 70)
    print("  STEPWISE METRIC COMPARISON")
    print("=" * 70)

    print_table(pa_igbs, mh_igbs, label="IG_BS / R=inf", every=10)
    print_table(pa_igd, mh_igd, label="IGd_BM / R=5", every=10)

    # ──────────────────────────────────────────────────────────────────────────
    # 4.  Plots
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\nSaving plots to {args.out}/")
    plot_comparison(pa_igbs, mh_igbs, label="IG_BS_Rinf", out_dir=args.out)
    plot_comparison(pa_igd, mh_igd, label="IGd_BM_R5", out_dir=args.out)

    # Summary delta at final step  (positive = MH is worse than PA-ref)
    def _delta(pa, mh, key):
        return mh[key][-1] - pa[key][-1]

    print("\n" + "=" * 70)
    print("  FINAL-STEP DELTAS  (MH − PA_ref)   positive = MH worse than reference")
    print(f"{'Config':>15}  {'Δentropy':>12}  {'Δmse':>12}  {'Δcoverage':>12}")
    print("─" * 55)
    for label, pa_d, mh_d in [
        ("IG_BS/Rinf", pa_igbs, mh_igbs),
        ("IGd_BM/R5", pa_igd, mh_igd),
    ]:
        print(
            f"{label:>15}  "
            f"{_delta(pa_d, mh_d, 'fused_entropy'):>+12.4f}  "
            f"{_delta(pa_d, mh_d, 'fused_mse'):>+12.6f}  "
            f"{_delta(pa_d, mh_d, 'coverage'):>+12.4f}"
        )
    print("=" * 70)

    # cleanup temp
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
