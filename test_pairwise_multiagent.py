"""
Test script to verify pairwise correlation factors produce different beliefs
in multi-agent settings.

Tests:
1. Single agent with equal/biased/adaptive - should show differences
2. Multi-agent with equal/biased/adaptive - should show differences
"""

import sys

sys.path.insert(0, "src")

import numpy as np
from helper import uav_position
from uav_camera import Camera
from orthomap import Field
from mapper_LBP import OccupancyMap
from multi_agent_mapper import MultiAgentMapper


def create_test_setup(seed=42):
    """Create consistent test environment."""

    # Create grid info matching main.py Gaussian config
    class GridInfo:
        def __init__(self):
            self.x = 50
            self.y = 50
            self.length = 0.125
            self.shape = (int(self.y / self.length), int(self.x / self.length))  # (400, 400)
            self.center = True  # Use center-based coordinates
            self.res = self.length

    grid_info = GridInfo()

    # Create camera first to get h_range
    camera = Camera(
        grid_info, 60, seed=seed, camera_altitude=15.0, f_overlap=0.1, s_overlap=0.1
    )

    # Create map using Field class
    # For Gaussian field, pass integer (radius) not string
    grf_r = 4  # Gaussian random field radius
    map_obj = Field(
        grid_info, field_type=grf_r, seed=seed, h_range=list(camera.get_hrange())
    )

    # Fixed starting position - corner position in world coords
    start_pos = (20.0, 20.0)  # Corner-ish position
    uav_pos = uav_position((start_pos, camera.get_hrange()[0]))
    camera.set_altitude(uav_pos.altitude)
    camera.set_position(uav_pos.position)

    # Create conf_dict (sensor model)
    conf_dict = {}
    alt_min, alt_max = camera.get_hrange()
    alt_step = camera.h_step  # Use h_step not dz
    a, b = 1.0, 0.015
    for alt in np.arange(alt_min, alt_max + alt_step, alt_step):
        sigma = a * (1 - np.exp(-b * alt))
        conf_dict[np.round(alt, decimals=2)] = (sigma, sigma)

    return grid_info, map_obj, camera, uav_pos, conf_dict


def test_single_agent_pairwise():
    """Test that pairwise factors produce different beliefs in single-agent mode."""
    print("=" * 60)
    print("TEST 1: Single Agent Pairwise Differences")
    print("=" * 60)

    grid_info, map_obj, camera, uav_pos, conf_dict = create_test_setup()

    # Get observation
    sigmas = conf_dict[np.round(uav_pos.altitude, decimals=2)]
    fp_vertices_ij, submap = map_obj.get_observations(uav_pos, sigmas)

    # Debug info
    print(f"\nFootprint: {fp_vertices_ij}")
    print(f"Submap shape: {submap.shape}")
    print(f"UAV: pos={uav_pos.position}, alt={uav_pos.altitude}")

    beliefs = {}
    for corr_type in ["equal", "biased", "adaptive"]:
        # Create fresh occupancy map
        omap = OccupancyMap(
            grid_info.shape, conf_dict=conf_dict, correlation_type=corr_type
        )

        # Update with OG (Bayesian update)
        omap.update_belief_OG(fp_vertices_ij, submap, uav_pos)

        # Run LBP propagation
        try:
            omap.propagate_messages(fp_vertices_ij, submap)
        except Exception as e:
            print(f"  LBP error ({corr_type}): {e}")

        beliefs[corr_type] = omap.get_belief().copy()

        print(f"\n{corr_type}:")
        print(f"  Belief mean: {beliefs[corr_type].mean():.6f}")
        print(f"  Belief std:  {beliefs[corr_type].std():.6f}")
        print(f"  Belief min:  {beliefs[corr_type].min():.6f}")
        print(f"  Belief max:  {beliefs[corr_type].max():.6f}")

    # Compare beliefs
    print("\n" + "-" * 40)
    print("Pairwise Comparisons (Single Agent):")

    diff_eq_bi = np.abs(beliefs["equal"] - beliefs["biased"]).sum()
    diff_eq_ad = np.abs(beliefs["equal"] - beliefs["adaptive"]).sum()
    diff_bi_ad = np.abs(beliefs["biased"] - beliefs["adaptive"]).sum()

    print(f"  |equal - biased|:   {diff_eq_bi:.6f}")
    print(f"  |equal - adaptive|: {diff_eq_ad:.6f}")
    print(f"  |biased - adaptive|: {diff_bi_ad:.6f}")

    if diff_eq_bi > 1e-6 and diff_eq_ad > 1e-6 and diff_bi_ad > 1e-6:
        print("\n✓ PASS: All pairwise types produce different beliefs!")
    else:
        print("\n✗ FAIL: Some pairwise types produce identical beliefs!")

    return beliefs


def test_multi_agent_pairwise():
    """Test that pairwise factors produce different beliefs in multi-agent mode."""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Agent Pairwise Differences")
    print("=" * 60)

    grid_info, map_obj, camera, uav_pos, conf_dict = create_test_setup()
    num_agents = 2

    # Get observation
    sigmas = conf_dict[np.round(uav_pos.altitude, decimals=2)]
    fp_vertices_ij, submap = map_obj.get_observations(uav_pos, sigmas)

    # Create second agent position (different location)
    camera2 = Camera(
        grid_info, 60, seed=43, camera_altitude=15.0, f_overlap=0.1, s_overlap=0.1
    )
    uav_pos2 = uav_position(((10.0, 10.0), camera2.get_hrange()[0]))
    camera2.set_altitude(uav_pos2.altitude)
    camera2.set_position(uav_pos2.position)
    fp_vertices_ij2, submap2 = map_obj.get_observations(uav_pos2, sigmas)

    beliefs = {}
    for corr_type in ["equal", "biased", "adaptive"]:
        # Create MultiAgentMapper
        mapper = MultiAgentMapper(
            grid_info.shape,
            num_agents,
            conf_dict,
            correlation_type=corr_type,
            news_mode="BM",
            lbp_iterations=5,
        )

        # Agent 0: local mapping
        mapper.local_mapping_update(0, fp_vertices_ij, submap, uav_pos)

        # Agent 1: local mapping
        mapper.local_mapping_update(1, fp_vertices_ij2, submap2, uav_pos2)

        # Update news beliefs
        mapper.update_news_belief(0, fp_vertices_ij, submap)
        mapper.update_news_belief(1, fp_vertices_ij2, submap2)

        # Fuse news between agents
        mapper.fuse_news_from_sender(0, [1])
        mapper.fuse_news_from_sender(1, [0])

        # Get fused belief
        beliefs[corr_type] = {
            "agent0": mapper.get_agent_belief(0).copy(),
            "agent1": mapper.get_agent_belief(1).copy(),
            "fused": mapper.get_global_fused_belief().copy(),
        }

        print(f"\n{corr_type}:")
        print(f"  Agent 0 belief mean: {beliefs[corr_type]['agent0'].mean():.6f}")
        print(f"  Agent 1 belief mean: {beliefs[corr_type]['agent1'].mean():.6f}")
        print(f"  Fused belief mean:   {beliefs[corr_type]['fused'].mean():.6f}")

    # Compare beliefs
    print("\n" + "-" * 40)
    print("Pairwise Comparisons (Multi-Agent - Agent 0):")

    diff_eq_bi = np.abs(beliefs["equal"]["agent0"] - beliefs["biased"]["agent0"]).sum()
    diff_eq_ad = np.abs(
        beliefs["equal"]["agent0"] - beliefs["adaptive"]["agent0"]
    ).sum()
    diff_bi_ad = np.abs(
        beliefs["biased"]["agent0"] - beliefs["adaptive"]["agent0"]
    ).sum()

    print(f"  |equal - biased|:   {diff_eq_bi:.6f}")
    print(f"  |equal - adaptive|: {diff_eq_ad:.6f}")
    print(f"  |biased - adaptive|: {diff_bi_ad:.6f}")

    print("\nPairwise Comparisons (Multi-Agent - Fused):")

    diff_eq_bi_f = np.abs(beliefs["equal"]["fused"] - beliefs["biased"]["fused"]).sum()
    diff_eq_ad_f = np.abs(
        beliefs["equal"]["fused"] - beliefs["adaptive"]["fused"]
    ).sum()
    diff_bi_ad_f = np.abs(
        beliefs["biased"]["fused"] - beliefs["adaptive"]["fused"]
    ).sum()

    print(f"  |equal - biased|:   {diff_eq_bi_f:.6f}")
    print(f"  |equal - adaptive|: {diff_eq_ad_f:.6f}")
    print(f"  |biased - adaptive|: {diff_bi_ad_f:.6f}")

    if diff_eq_bi > 1e-6 and diff_eq_ad > 1e-6 and diff_bi_ad > 1e-6:
        print("\n✓ PASS: All pairwise types produce different agent beliefs!")
    else:
        print("\n✗ FAIL: Some pairwise types produce identical agent beliefs!")

    return beliefs


def test_multi_agent_local_maps_only():
    """Test local maps WITHOUT fusion to isolate pairwise effect."""
    print("\n" + "=" * 60)
    print("TEST 3: Multi-Agent LOCAL Maps Only (No Fusion)")
    print("=" * 60)

    grid_info, map_obj, camera, uav_pos, conf_dict = create_test_setup()
    num_agents = 2

    # Get observation
    sigmas = conf_dict[np.round(uav_pos.altitude, decimals=2)]
    fp_vertices_ij, submap = map_obj.get_observations(uav_pos, sigmas)

    beliefs = {}
    for corr_type in ["equal", "biased", "adaptive"]:
        # Create MultiAgentMapper
        mapper = MultiAgentMapper(
            grid_info.shape,
            num_agents,
            conf_dict,
            correlation_type=corr_type,
            news_mode="BM",
            lbp_iterations=5,
        )

        # Agent 0: local mapping ONLY (no fusion)
        mapper.local_mapping_update(0, fp_vertices_ij, submap, uav_pos)

        beliefs[corr_type] = mapper.get_agent_belief(0).copy()

        print(f"\n{corr_type}:")
        print(f"  Belief mean: {beliefs[corr_type].mean():.6f}")
        print(f"  Belief std:  {beliefs[corr_type].std():.6f}")

    # Compare beliefs
    print("\n" + "-" * 40)
    print("Pairwise Comparisons (Local Maps Only):")

    diff_eq_bi = np.abs(beliefs["equal"] - beliefs["biased"]).sum()
    diff_eq_ad = np.abs(beliefs["equal"] - beliefs["adaptive"]).sum()
    diff_bi_ad = np.abs(beliefs["biased"] - beliefs["adaptive"]).sum()

    print(f"  |equal - biased|:   {diff_eq_bi:.6f}")
    print(f"  |equal - adaptive|: {diff_eq_ad:.6f}")
    print(f"  |biased - adaptive|: {diff_bi_ad:.6f}")

    if diff_eq_bi > 1e-6 and diff_eq_ad > 1e-6 and diff_bi_ad > 1e-6:
        print("\n✓ PASS: Local maps show pairwise differences!")
    else:
        print("\n✗ FAIL: Local maps are identical across pairwise types!")

    return beliefs


def test_occupancy_map_directly():
    """Test OccupancyMap directly to verify LBP uses pairwise."""
    print("\n" + "=" * 60)
    print("TEST 4: OccupancyMap Direct (LBP Pairwise Check)")
    print("=" * 60)

    grid_info, map_obj, camera, uav_pos, conf_dict = create_test_setup()

    # Get observation
    sigmas = conf_dict[np.round(uav_pos.altitude, decimals=2)]
    fp_vertices_ij, submap = map_obj.get_observations(uav_pos, sigmas)

    # Test with different LBP iterations
    for lbp_iters in [0, 1, 5, 10]:
        print(f"\n--- LBP iterations: {lbp_iters} ---")
        beliefs = {}

        for corr_type in ["equal", "biased", "adaptive"]:
            omap = OccupancyMap(
                grid_info.shape, conf_dict=conf_dict, correlation_type=corr_type
            )

            # OG update
            omap.update_belief_OG(fp_vertices_ij, submap, uav_pos)

            # LBP propagation with explicit iterations
            if lbp_iters > 0:
                omap.propagate_messages(
                    fp_vertices_ij, submap, max_iterations=lbp_iters
                )

            beliefs[corr_type] = omap.get_belief().copy()

        diff_eq_bi = np.abs(beliefs["equal"] - beliefs["biased"]).sum()
        diff_eq_ad = np.abs(beliefs["equal"] - beliefs["adaptive"]).sum()

        print(f"  |equal - biased|:   {diff_eq_bi:.6f}")
        print(f"  |equal - adaptive|: {diff_eq_ad:.6f}")

        if lbp_iters == 0:
            if diff_eq_bi < 1e-6 and diff_eq_ad < 1e-6:
                print("  ✓ Expected: No LBP = identical beliefs")
            else:
                print("  ✗ Unexpected: No LBP but different beliefs!")
        else:
            if diff_eq_bi > 1e-6 or diff_eq_ad > 1e-6:
                print("  ✓ Expected: LBP shows pairwise differences")
            else:
                print("  ✗ Problem: LBP but identical beliefs!")


if __name__ == "__main__":
    test_single_agent_pairwise()
    test_multi_agent_local_maps_only()
    test_multi_agent_pairwise()
    test_occupancy_map_directly()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
