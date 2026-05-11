#!/usr/bin/env python3
"""
Detailed step-by-step diagnostic: PA-Dev vs MH for IG_BS/Rinf (N=4, r=4, seed=42).

Prints per-step, per-agent:
  - action chosen
  - footprint (ij range)
  - cells observed
  - entropy before/after observation
  - entropy before/after fusion
  - sigma used
  - ground truth map stats (to verify same map)

Usage:
  conda run -n active_sensing python compare_detail.py [--steps 3]
"""

import os, sys, argparse, importlib.util
import numpy as np

MH_SRC = os.path.join(os.path.dirname(__file__), "src")
PA_SRC = "/home/bota/repos/Precision-Agriculture-Dev"
sys.path.insert(0, MH_SRC)

from helper import gaussian_random_field
from orthomap import Field
from mapper_LBP import OccupancyMap as OM
from uav_camera import Camera
from multi_agent_coordinator import MultiAgentCoordinator
from experiment_utils import initialize_agent
from config_loader import load_config
from simulator import Simulator


# ──────────────────────────────────────────────────────────────────────────────
class MHGrid:
    x = 50
    y = 50
    length = 0.125
    shape = (400, 400)
    center = True


def H_binary(p):
    """Binary entropy of array p (clipped to avoid log(0))."""
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ──────────────────────────────────────────────────────────────────────────────
# PA loader
# ──────────────────────────────────────────────────────────────────────────────
def load_pa(
    seed,
    n_steps,
    n_agents=4,
    cluster_radius=4,
    radius_multiplier=1000,
    planner_type="selfish",
    news_inference_type="OG_single",
):
    """Load PA environment.

    IG_BS  / R=inf : planner_type='selfish',                news_inference_type='OG_single', radius_multiplier=1000
    IGd_BM / R=5   : planner_type='mine_IoU_async_no_pred', news_inference_type='Bypass',   radius_multiplier=5
    """
    spec = importlib.util.spec_from_file_location(
        "pa_simulator", os.path.join(PA_SRC, "simulator.py")
    )
    pa_sim = importlib.util.module_from_spec(spec)
    sys.path.insert(0, PA_SRC)
    spec.loader.exec_module(pa_sim)

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

    env = pa_sim.MappingEnv(field_len=50.0, fov=np.pi / 3, **iep)
    env.map_rng = np.random.default_rng(seed)
    env.agent_position_rng = np.random.default_rng(seed)
    mapper = pa_sim.Mapper(env.n_cell, env.min_space_z, env.max_space_z, **iep)
    planner = pa_sim.Planner(
        env.action_to_direction,
        env.altitude_to_size,
        env.position_graph,
        env.position_to_data,
        env.regions_limits,
        env.optimal_altitude,
        **iep,
    )

    gt = env.generate_map()
    env.reset_map_beliefs()
    env.reset_agents_position(**iep)
    planner.reset_sweep()

    # Override to corners
    corners = [(-25.0, -25.0), (25.0, -25.0), (-25.0, 25.0), (25.0, 25.0)]
    for ag in env.agents:
        x, y = corners[ag.id]
        z = (iep["altitude"] + 1) * env.v_displacement
        ag.state.set_position(np.array([x, y, z]))
        pa_sim.states[ag.id, :] = ag.state.position

    # Share a single observation RNG across all PA agents, seeded identically to
    # MH's Field.rng (which is reset to np.random.default_rng(seed) with zero
    # draws after reset()).  All agents share the same object so they draw from
    # one sequential stream matching MH's single-RNG draw order.
    shared_obs_rng = np.random.default_rng(seed)
    for ag in env.agents:
        ag.rng = shared_obs_rng

    return pa_sim, env, mapper, planner, gt, iep


# ──────────────────────────────────────────────────────────────────────────────
# MH loader
# ──────────────────────────────────────────────────────────────────────────────
def load_mh(seed, n_steps, cluster_radius=4, mode="IG_BS"):
    """Load MH environment.

    mode='IG_BS'  : baseline_greedy_ig_bs_rinf.json,  news_sharing=True,  coord_news_mode='BS', radius=-1 (inf)
    mode='IGd_BM' : baseline_greedy_igd_bm_r5.json,   news_sharing=False, coord_news_mode='BM', radius=5, enable_discounting=True
    """
    if mode == "IGd_BM":
        cfg_name = "baseline_greedy_igd_bm_r5.json"
    else:
        cfg_name = "baseline_greedy_ig_bs_rinf.json"
    cfg_path = os.path.join(os.path.dirname(__file__), "configs", cfg_name)
    config = load_config(cfg_path)[0]
    config.update(
        {
            "num_agents": 4,
            "cluster_radius": cluster_radius,
            "n_steps": n_steps,
            "seed": seed,
            "enable_plotting": False,
            "enable_logging": False,
        }
    )
    dec = config.setdefault("decentralized", {})
    if mode == "IGd_BM":
        dec["position_sharing"] = True
        dec["news_sharing"] = False
        dec["radius_multiplier"] = 5
        config.setdefault("multi_agent", {})["news_mode"] = "BM"
        config["multi_agent"]["news_inference_type"] = "Bypass"
        config.setdefault("greedy_ig", {})["enable_discounting"] = True
        coord_news_mode = "BM"
    else:  # IG_BS
        dec["position_sharing"] = False
        dec["news_sharing"] = True
        dec.setdefault("radius_multiplier", -1)
        config.setdefault("multi_agent", {})["news_mode"] = "BS"
        config["multi_agent"]["news_inference_type"] = "OG"
        coord_news_mode = "BS"
    config["multi_agent"]["fusion_eps"] = 0.0
    config["multi_agent"]["metric_aggregation"] = "fused_mean"
    config["multi_agent"]["clip_metric_beliefs"] = False

    grid_info = MHGrid()
    cam1 = Camera(
        grid_info,
        60,
        camera_altitude=None,
        f_overlap=None,
        s_overlap=None,
        seed=seed,
        a=1.0,
        b=0.015,
    )
    hrange = cam1.get_hrange()
    map_obj = Field(
        grid_info, cluster_radius, sweep="greedy_ig", h_range=hrange, seed=seed
    )
    map_obj.reset(seed=seed)
    gt = map_obj.get_ground_truth()
    conf_dict = cam1.theoretical_conf_dict()

    coordinator = MultiAgentCoordinator(
        grid_shape=grid_info.shape,
        config=config,
        conf_dict=conf_dict,
        correlation_type="pairwise",
        news_mode=coord_news_mode,
        mode=mode,
        grid_info=grid_info,
        debug_logs=False,
    )

    start_positions = coordinator.reset_start_position(
        grid_info=grid_info,
        start_position="corner",
        min_distance=10.0,
        seed=seed,
        camera_hrange=hrange,
    )

    agents = []
    for aid in range(4):
        sp = start_positions[aid]
        agent_state = initialize_agent(
            agent_id=aid,
            grid_info=grid_info,
            start_position=(sp[0], sp[1]),
            action_strategy="greedy_ig",
            conf_dict=conf_dict,
            corr_type="pairwise",
            mcts_params={},
            optimal_alt=21.5,
            min_alt=None,
            overlap=None,
            seed=seed,
            coordinator=coordinator,
            start_altitude=sp[2],
            debug_logs=False,
        )
        agents.append(agent_state)
        cam = agent_state["camera"]
        uv = agent_state["uav_pos"]
        coordinator.update_agent_state(
            agent_id=aid,
            position=(uv.position[0], uv.position[1]),
            altitude=uv.altitude,
        )

    shared_obs_rng = np.random.default_rng(seed)
    map_obj.rng = shared_obs_rng
    for agent_state in agents:
        agent_state["camera"].rng = shared_obs_rng

    simulator = Simulator(
        agents=agents,
        map_obj=map_obj,
        ground_truth_map=gt,
        conf_dict=conf_dict,
        grid_info=grid_info,
        n_steps=n_steps,
        results_folder="/tmp/compare_detail",
        corr_type="pairwise",
        e_margin=None,
        grf_r=cluster_radius,
        iter_idx=0,
        enable_stepwise_plotting=False,
        enable_logging=False,
        action_strategy="greedy_ig",
        coordinator=coordinator,
        multi_agent_logger=None,
        run_id="compare_detail",
        debug_logs=False,
    )

    return config, grid_info, coordinator, agents, gt, conf_dict, hrange, map_obj, simulator


# ──────────────────────────────────────────────────────────────────────────────
# Step helpers
# ──────────────────────────────────────────────────────────────────────────────
def pa_observe_and_fuse(pa_sim, env, mapper, gt):
    """One round of: observe → set_pairwise → update_map_beliefs → fuse."""
    observations = env.get_observations(gt)
    mapper.set_pairwise_potential_z(env.agents, observations)
    mapper.update_map_beliefs(env.agents, observations)
    mapper.update_news_and_fuse_map_beliefs(env.agents, observations)
    return observations


def mh_observe(coordinator, agents, map_obj, conf_dict):
    """Local observation only (OG update + LBP). No fusion."""
    agent_obs = {}
    for agent in agents:
        aid = agent["agent_id"]
        cam = agent["camera"]
        uv = agent["uav_pos"]
        bm = agent["belief_map"]
        om = agent["occupancy_map"]
        if coordinator:
            coordinator.process_messages(aid)

        s0, s1 = conf_dict[np.round(uv.altitude, decimals=2)]
        fp_ij, submap = map_obj.get_observations(uv, [s0, s1])
        om.update_belief_OG(fp_ij, submap, uv)
        om.propagate_messages(fp_ij, submap, max_iterations=1, reset_msgs=True)
        bm[:, :, 1] = om.get_belief().copy()
        bm[:, :, 0] = 1 - bm[:, :, 1]
        agent["belief_map"] = bm
        agent["local_belief_map"] = bm.copy()
        agent_obs[aid] = {
            "fp_ij": fp_ij,
            "submap": submap,
            "sigmas": [s0, s1],
            "camera": cam,
            "uav_pos": uv,
        }
    return agent_obs


def mh_fuse(coordinator, agents, agent_obs, news_sharing=True):
    """News sharing + fusion only (after local observation is done)."""
    if not coordinator:
        return
    mmap = coordinator.map
    for agent in agents:
        aid = agent["agent_id"]
        mmap.maps[aid].map_beliefs = agent["occupancy_map"].get_belief().copy()
    if news_sharing:
        nbr = {
            aid: coordinator.get_neighbors_in_range(aid) for aid in range(len(agents))
        }
        mmap.update_news_and_fuse(agent_obs, nbr)
    for agent in agents:
        aid = agent["agent_id"]
        fb = mmap.get_agent_belief(aid)
        if fb is not None:
            agent["belief_map"][:, :, 1] = fb
            agent["belief_map"][:, :, 0] = 1 - fb
            agent["occupancy_map"].map_beliefs = fb.copy()


def mh_select_actions(agents, coordinator, grid_info, simulator=None, step=0):
    """One round of greedy IG planning for all agents."""
    from helper import uav_position

    if simulator is not None:
        simulator._select_agent_actions(step)
        actions = {}
        for agent in agents:
            aid = agent["agent_id"]
            action = agent.get("_next_action")
            agent["actions"].append(action)
            actions[aid] = action
        return actions

    actions = {}
    for agent in agents:
        aid = agent["agent_id"]
        planner = agent["planner"]
        om = agent["occupancy_map"]
        bm = agent["belief_map"]
        uv = agent["uav_pos"]
        # Update planner belief with fused map
        belief2d = bm[:, :, 1]
        action, scores = planner.select_action(belief2d, None)
        agent["actions"].append(action)
        actions[aid] = action
    return actions


def pa_step(env, actions):
    env.step(actions)


def mh_step(agents, actions, coordinator, grid_info):
    from helper import uav_position

    for agent in agents:
        aid = agent["agent_id"]
        act = actions[aid]
        cam = agent["camera"]

        uav_pos = uav_position(cam.x_future(act))
        agent["uav_pos"] = uav_pos
        agent["uav_positions"].append(uav_pos)
        cam.set_altitude(uav_pos.altitude)
        cam.set_position(uav_pos.position)

        if coordinator:
            coordinator.update_agent_state(
                agent_id=aid,
                position=(uav_pos.position[0], uav_pos.position[1]),
                altitude=uav_pos.altitude,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic printer
# ──────────────────────────────────────────────────────────────────────────────
def print_step_header(step):
    print(f"\n{'═'*80}")
    print(f"  STEP {step}")
    print(f"{'═'*80}")


def pa_entropy_total(pa_sim):
    """Sum of H(map_beliefs) across all agents (fused mean)."""
    fused = pa_sim.map_beliefs.mean(axis=2)
    return float(np.sum(H_binary(fused)))


def mh_entropy_total(agents):
    """Fused entropy using mean of agent beliefs (matching Simulator._compute_metrics)."""
    beliefs = np.stack([a["belief_map"][:, :, 1] for a in agents], axis=0)
    fused = beliefs.mean(axis=0)
    return float(np.sum(H_binary(fused)))


def print_agent_comparison(
    step, pa_sim, env, mh_agents, pa_actions, mh_actions, pa_obs, mh_obs, gt
):
    for aid in range(len(env.agents)):
        pa_ag = env.agents[aid]
        mh_ag = mh_agents[aid]
        pa_pos = pa_ag.state.position
        mh_uv = mh_ag["uav_pos"]

        # PA footprint from observation
        pa_fp = pa_obs[aid]["fp"] if pa_obs else None
        mh_fp = mh_obs.get(aid, {}).get("fp_ij") if mh_obs else None

        # Per-agent entropy (own belief)
        pa_ent = float(np.sum(H_binary(pa_sim.map_beliefs[:, :, aid])))
        mh_ent = float(np.sum(H_binary(mh_ag["belief_map"][:, :, 1])))

        print(f"\n  Agent {aid}")
        print(
            f"    PA pos: ({pa_pos[0]:+6.2f},{pa_pos[1]:+6.2f},z={pa_pos[2]:.3f})  "
            f"action={pa_actions.get(aid,'?') if pa_actions else 'N/A'}"
        )
        print(
            f"    MH pos: ({mh_uv.position[0]:+6.2f},{mh_uv.position[1]:+6.2f},z={mh_uv.altitude:.3f})  "
            f"action={mh_actions.get(aid,'?') if mh_actions else 'N/A'}"
        )
        if pa_fp is not None:
            print(f"    PA footprint: {pa_fp}")
        if mh_fp is not None:
            _r0, _r1 = mh_fp["ul"][0], mh_fp["bl"][0]
            _c0, _c1 = mh_fp["ul"][1], mh_fp["ur"][1]
            print(
                f"    MH footprint: rows [{_r0},{_r1}] cols [{_c0},{_c1}]  "
                f"({_r1-_r0}×{_c1-_c0} cells)"
            )
        print(
            f"    PA belief entropy: {pa_ent:10.2f}   MH belief entropy: {mh_ent:10.2f}   Δ={mh_ent-pa_ent:+.2f}"
        )
        if pa_obs and aid in pa_obs:
            obs_data = pa_obs[aid]
            print(
                f"    PA sigma: ({obs_data['s0']:.4f},{obs_data['s1']:.4f})   "
                f"cells observed: {obs_data['n_cells']}   "
                f"GT mean in fp: {obs_data['gt_mean']:.4f}"
            )
        if mh_obs and aid in mh_obs:
            mo = mh_obs[aid]
            print(
                f"    MH sigma: ({mo['sigmas'][0]:.4f},{mo['sigmas'][1]:.4f})   "
                f"cells observed: {mo.get('n_cells','?')}   "
                f"GT mean in fp: {mo.get('gt_mean','?')}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
# PA settings per mode
_PA_MODE_SETTINGS = {
    "IG_BS":  dict(planner_type="selfish",                news_inference_type="OG_single", radius_multiplier=1000),
    "IGd_BM": dict(planner_type="mine_IoU_async_no_pred", news_inference_type="Bypass",   radius_multiplier=5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["IG_BS", "IGd_BM"], default="IG_BS",
                    help="IG_BS=selfish/OG_single/Rinf  IGd_BM=mine_IoU/Bypass/R5")
    args = ap.parse_args()

    pa_settings = _PA_MODE_SETTINGS[args.mode]
    print(f"Loading PA-Dev  [{args.mode}]  {pa_settings}")
    pa_sim, env, mapper, planner, pa_gt, iep = load_pa(
        args.seed, args.steps, **pa_settings
    )

    print(f"Loading MH  [{args.mode}]...")
    (
        config,
        grid_info,
        coordinator,
        mh_agents,
        mh_gt,
        conf_dict,
        hrange,
        map_obj,
        mh_simulator,
    ) = load_mh(
        args.seed, args.steps, mode=args.mode
    )

    # ── Map comparison ────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("  MAP COMPARISON (ground truth)")
    print(f"{'═'*80}")
    print(
        f"  PA GT shape={pa_gt.shape}  sum={pa_gt.sum():.1f}  mean={pa_gt.mean():.4f}"
    )
    print(
        f"  MH GT shape={mh_gt.shape}  sum={mh_gt.sum():.1f}  mean={mh_gt.mean():.4f}"
    )
    print(f"  Maps identical: {np.allclose(pa_gt, mh_gt)}")
    if not np.allclose(pa_gt, mh_gt):
        diff = np.abs(pa_gt - mh_gt)
        print(f"  Max diff: {diff.max():.6f}  mean diff: {diff.mean():.6f}")
        print(f"  PA GT unique values: {np.unique(pa_gt)}")
        print(f"  MH GT unique values: {np.unique(mh_gt)}")

    # ── Sigma at starting altitude ────────────────────────────────────────────
    alt = hrange[0]
    s0_mh, s1_mh = conf_dict[np.round(alt, decimals=2)]
    a0, b0 = iep["a0"], iep["b0"]
    a1, b1 = iep["a1"], iep["b1"]
    s0_pa = a0 * (1 - np.exp(-b0 * alt))
    s1_pa = a1 * (1 - np.exp(-b1 * alt))
    print(f"\n  Sensor model at z={alt:.4f}m:")
    print(f"  PA  sigma=(s0={s0_pa:.6f}, s1={s1_pa:.6f})")
    print(f"  MH  sigma=(s0={s0_mh:.6f}, s1={s1_mh:.6f})")
    print(f"  Sigmas identical: {np.isclose(s0_pa,s0_mh) and np.isclose(s1_pa,s1_mh)}")

    # ── Starting positions ────────────────────────────────────────────────────
    print(f"\n  Starting positions:")
    for ag in env.agents:
        mh_ag = mh_agents[ag.id]
        pa_p = ag.state.position
        mh_p = mh_ag["uav_pos"]
        same_xy = np.isclose(pa_p[0], mh_p.position[0]) and np.isclose(
            pa_p[1], mh_p.position[1]
        )
        print(
            f"  Agent {ag.id}: PA=({pa_p[0]:+.1f},{pa_p[1]:+.1f},z={pa_p[2]:.3f})  "
            f"MH=({mh_p.position[0]:+.1f},{mh_p.position[1]:+.1f},z={mh_p.altitude:.3f})  "
            f"same_xy={same_xy}"
        )

    # ── Step-by-step loop ─────────────────────────────────────────────────────
    pa_observations = []  # PA observations from previous step
    mh_news_sharing = config.get("decentralized", {}).get("news_sharing", True)

    for step in range(args.steps):
        print_step_header(step)

        # ── Entropy BEFORE any sensing (pure prior at step 0) ─────────────────
        pa_ent_before = pa_entropy_total(pa_sim)
        mh_ent_before = mh_entropy_total(mh_agents)
        print(f"\n  Entropy BEFORE observe/fuse:")
        print(
            f"    PA fused={pa_ent_before:.2f}   MH fused={mh_ent_before:.2f}   Δ={mh_ent_before-pa_ent_before:+.2f}"
        )

        # ── Check per-agent belief identity before planning ───────────────────
        for ag in env.agents:
            pa_bel = pa_sim.map_beliefs[:, :, ag.id]
            mh_bel = mh_agents[ag.id]["belief_map"][:, :, 1]
            max_diff = float(np.max(np.abs(pa_bel - mh_bel)))
            if max_diff > 1e-8:
                print(
                    f"  ⚠ Agent {ag.id} per-agent belief diff before planning: max={max_diff:.6e}"
                )

        # ── PA: observe + fuse ────────────────────────────────────────────────
        obs_raw = env.get_observations(pa_gt)  # returns list of dicts per agent
        # extract PA observation details before belief update
        pa_obs_detail = {}
        for ag in env.agents:
            aid = ag.id
            fp_ij, _ = ag.camera.get_fp_vertices_ij(ag.state.position)
            rmin = fp_ij["ul"][0]
            rmax = fp_ij["bl"][0]
            cmin = fp_ij["ul"][1]
            cmax = fp_ij["ur"][1]
            patch = pa_gt[rmin:rmax, cmin:cmax]
            s0 = a0 * (1 - np.exp(-b0 * ag.state.position[2]))
            s1 = a1 * (1 - np.exp(-b1 * ag.state.position[2]))
            pa_obs_detail[aid] = {
                "fp": f"rows [{rmin},{rmax}] cols [{cmin},{cmax}] ({rmax-rmin}×{cmax-cmin} cells)",
                "n_cells": patch.size,
                "gt_mean": float(patch.mean()) if patch.size > 0 else 0.0,
                "s0": s0,
                "s1": s1,
            }
        mapper.set_pairwise_potential_z(env.agents, obs_raw)
        mapper.update_map_beliefs(env.agents, obs_raw)

        pa_ent_after_local = pa_entropy_total(pa_sim)

        mapper.update_news_and_fuse_map_beliefs(env.agents, obs_raw)
        pa_observations = obs_raw

        pa_ent_after_fuse = pa_entropy_total(pa_sim)

        # ── MH: observe + fuse ───────────────────────────────────────────────
        mh_obs_detail_raw = {}
        # capture pre-observation beliefs
        mh_beliefs_before = {
            a["agent_id"]: a["belief_map"][:, :, 1].copy() for a in mh_agents
        }

        mh_obs_raw = mh_observe(coordinator, mh_agents, map_obj, conf_dict)

        mh_ent_after_local = mh_entropy_total(mh_agents)

        mh_fuse(coordinator, mh_agents, mh_obs_raw, news_sharing=mh_news_sharing)

        mh_ent_after_fuse = mh_entropy_total(mh_agents)

        # ── Plan after observation/fusion, matching compare_greedy_ig.py ─────
        planner.compute_map_belief_entropies()
        pa_actions, pa_plan_data = planner.get_actions(env.agents, obs_raw)
        mh_actions = mh_select_actions(
            mh_agents, coordinator, grid_info, simulator=mh_simulator, step=step
        )
        mh_ig_scores = {}
        for agent in mh_agents:
            aid = agent["agent_id"]
            p = agent["planner"]
            if hasattr(p, "_greedy_ig_planner") and hasattr(
                p._greedy_ig_planner, "_action_scores"
            ):
                mh_ig_scores[aid] = dict(p._greedy_ig_planner._action_scores)

        print(f"\n  Actions chosen:")
        for aid in range(4):
            pa_act = pa_actions[aid] if pa_actions else "?"
            mh_act = mh_actions[aid] if mh_actions else "?"
            same = "✓" if pa_act == mh_act else "✗"
            print(f"    Agent {aid}: PA={pa_act:10s}  MH={mh_act:10s}  {same}")

        # ── Move after planning ───────────────────────────────────────────────
        pa_step(env, pa_actions)
        mh_step(mh_agents, mh_actions, coordinator, grid_info)

        for aid, obs in mh_obs_raw.items():
            fp = obs["fp_ij"]
            rmin, rmax = fp["ul"][0], fp["bl"][0]
            cmin, cmax = fp["ul"][1], fp["ur"][1]
            patch = mh_gt[rmin:rmax, cmin:cmax]
            mh_obs_detail_raw[aid] = {
                **obs,
                "n_cells": patch.size,
                "gt_mean": float(patch.mean()) if patch.size > 0 else 0.0,
            }

        # ── Print details ─────────────────────────────────────────────────────
        print(f"\n  After LOCAL observation (before fusion):")
        print(
            f"    PA fused={pa_ent_after_local:.2f}   MH fused={mh_ent_after_local:.2f}   Δ={mh_ent_after_local-pa_ent_after_local:+.2f}"
        )
        print(f"\n  After FUSION:")
        print(
            f"    PA fused={pa_ent_after_fuse:.2f}   MH fused={mh_ent_after_fuse:.2f}   Δ={mh_ent_after_fuse-pa_ent_after_fuse:+.2f}"
        )

        print(f"\n  Per-agent footprint + observation details:")
        for aid in range(4):
            pa_d = pa_obs_detail.get(aid, {})
            mh_d = mh_obs_detail_raw.get(aid, {})
            print(f"\n    Agent {aid}:")
            print(
                f"      PA footprint: {pa_d.get('fp','?')}  n_cells={pa_d.get('n_cells','?')}  GT_mean={pa_d.get('gt_mean',0):.4f}  sigma=({pa_d.get('s0',0):.5f},{pa_d.get('s1',0):.5f})"
            )
            mh_fp = mh_d.get("fp_ij")
            if mh_fp:
                _r0, _r1 = mh_fp["ul"][0], mh_fp["bl"][0]
                _c0, _c1 = mh_fp["ul"][1], mh_fp["ur"][1]
                print(
                    f"      MH footprint: rows [{_r0},{_r1}] cols [{_c0},{_c1}]  "
                    f"n_cells={mh_d.get('n_cells','?')}  GT_mean={mh_d.get('gt_mean',0):.4f}  "
                    f"sigma=({mh_d.get('sigmas',['?','?'])[0]:.5f},{mh_d.get('sigmas',['?','?'])[1]:.5f})"
                )
            pa_bel = pa_sim.map_beliefs[:, :, aid]
            mh_bel = mh_agents[aid]["belief_map"][:, :, 1]
            bel_diff = np.abs(pa_bel - mh_bel)
            print(
                f"      Belief diff (PA vs MH): max={bel_diff.max():.5f}  mean={bel_diff.mean():.5f}  "
                f"cells_differ(>0.01): {(bel_diff>0.01).sum()}"
            )
            # Show a 5×5 patch of beliefs around the footprint
            if pa_d.get("fp"):
                pa_ent_ag = float(np.sum(H_binary(pa_bel)))
                mh_ent_ag = float(np.sum(H_binary(mh_bel)))
                print(
                    f"      Agent entropy: PA={pa_ent_ag:.2f}  MH={mh_ent_ag:.2f}  Δ={mh_ent_ag-pa_ent_ag:+.2f}"
                )


if __name__ == "__main__":
    main()
