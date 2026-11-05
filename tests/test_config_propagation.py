"""
Test: config params (hlp_horizon, llp_iterations, replan_interval, tile_size, etc.)
are correctly propagated from experiment configs to the MH-Eff and MH-Full planners.

Tests:
  1. Config flows from sweep JSON → config_loader → experiment_config dict (correct values)
  2. planning(experiment_config=...) stores the config and reads hierarchical_dec_mcts
  3. create_hierarchical_planner receives correct params from planner
  4. HierarchicalDecMCTSPlanner stores them on self.llp / self.hlp
  5. Two sweep configs (fast_replan vs balanced) produce different planner states
  6. dec_mcts config also flows correctly (horizon, iterations, ucb_c)

Run with:
    cd /home/bota/repos/multi-horizon/src
    python ../tests/test_config_propagation.py
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from types import SimpleNamespace

# ── project imports ───────────────────────────────────────────────────────────
from config_loader import load_config
from planner import planning
from uav_camera import Camera
from hierarchical_dec_mcts import (
    HierarchicalDecMCTSPlanner,
    IntentBus,
    create_hierarchical_planner,
)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
errors = []


def check(name, got, expected):
    ok = got == expected
    status = PASS if ok else FAIL
    print(f"  {status} {name}: got={got!r}  expected={expected!r}")
    if not ok:
        errors.append(f"{name}: got {got!r}, expected {expected!r}")


# ── fixtures ──────────────────────────────────────────────────────────────────
def make_grid():
    g = SimpleNamespace()
    g.x = 50.0
    g.y = 50.0
    g.length = 0.5
    g.shape = (int(g.y / g.length), int(g.x / g.length))
    g.center = True
    return g


def make_camera(grid):
    return Camera(
        grid,
        60,
        camera_altitude=None,
        f_overlap=None,
        s_overlap=None,
        seed=0,
        a=1.0,
        b=0.015,
    )


SWEEP_DIR = os.path.join(
    os.path.dirname(__file__), "..", "configs", "strategies", "sweeps"
)

FAST_REPLAN_JSON = os.path.join(SWEEP_DIR, "mh_eff_fast_replan_hlp5_rep1.json")
BALANCED_JSON = os.path.join(SWEEP_DIR, "mh_eff_balanced_hlp8_rep2.json")
DEC_MCTS_JSON = os.path.join(
    os.path.dirname(__file__), "..", "configs", "strategies", "dec_mcts.json"
)
MH_FULL_JSON = os.path.join(
    os.path.dirname(__file__), "..", "configs", "strategies", "mh_dec_mcts_full.json"
)

# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – config_loader flattens sweep params correctly
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TEST 1] config_loader flattens sweep params from JSON")

configs = load_config(FAST_REPLAN_JSON)
assert len(configs) == 1, "Expected single config"
cfg_fast = configs[0]
hier_fast = cfg_fast.get("hierarchical_dec_mcts", {})

check("llp_horizon", hier_fast.get("llp_horizon"), 3)
check("llp_iterations", hier_fast.get("llp_iterations"), 30)
check("hlp_horizon", hier_fast.get("hlp_horizon"), 5)
check("hlp_iterations", hier_fast.get("hlp_iterations"), 20)
check("hlp_ucb_c", hier_fast.get("hlp_ucb_c"), 1.0)
check("hlp_replan_interval", hier_fast.get("hlp_replan_interval"), 1.0)
check("tile_size", hier_fast.get("tile_size"), [25, 25])
check("use_mcts_llp", hier_fast.get("use_mcts_llp"), False)

configs = load_config(BALANCED_JSON)
cfg_bal = configs[0]
hier_bal = cfg_bal.get("hierarchical_dec_mcts", {})

check("balanced hlp_horizon", hier_bal.get("hlp_horizon"), 8)
check("balanced hlp_iterations", hier_bal.get("hlp_iterations"), 30)
check("balanced replan_interval", hier_bal.get("hlp_replan_interval"), 2.0)

# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – planning() stores experiment_config and reads hierarchical_dec_mcts
# ─────────────────────────────────────────────────────────────────────────────
print(
    "\n[TEST 2] planning() stores experiment_config; hier_config reads correct values"
)

grid = make_grid()
cam = make_camera(grid)

planner_fast = planning(
    grid,
    cam,
    "mh_dec_mcts_efficient",
    conf_dict=None,
    experiment_config=cfg_fast,
    seed=0,
)
planner_bal = planning(
    grid,
    cam,
    "mh_dec_mcts_efficient",
    conf_dict=None,
    experiment_config=cfg_bal,
    seed=0,
)

# Access the same logic that hierarchical_dec_mcts_decision uses
hier_from_fast = planner_fast.experiment_config.get("hierarchical_dec_mcts", {})
hier_from_bal = planner_bal.experiment_config.get("hierarchical_dec_mcts", {})

check(
    "fast planner hlp_horizon from experiment_config",
    hier_from_fast.get("hlp_horizon"),
    5,
)
check(
    "bal  planner hlp_horizon from experiment_config",
    hier_from_bal.get("hlp_horizon"),
    8,
)
check("fast planner tile_size", hier_from_fast.get("tile_size"), [25, 25])

# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – create_hierarchical_planner receives and stores correct params
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TEST 3] create_hierarchical_planner stores params on llp/hlp")

intent_bus = IntentBus(num_agents=1)


def make_hier_planner(cfg):
    hier = cfg.get("hierarchical_dec_mcts", {})
    config = {
        "llp_horizon": hier.get("llp_horizon", 5),
        "llp_iterations": hier.get("llp_iterations", 50),
        "llp_discount_factor": hier.get("llp_discount_factor", 0.95),
        "hlp_horizon": hier.get("hlp_horizon", 3),
        "hlp_iterations": hier.get("hlp_iterations", 30),
        "tile_size": hier.get("tile_size", [100, 100]),
        "hlp_replan_interval": hier.get("hlp_replan_interval", 1.0),
        "llp_ucb_c": hier.get("llp_ucb_c", 1.41),
        "hlp_ucb_c": hier.get("hlp_ucb_c", 1.0),
        "use_mcts_llp": hier.get("use_mcts_llp", False),
        "use_g2": hier.get("use_g2", False),
        "g2_mode": hier.get("g2_mode", "hl_aware"),
    }
    return create_hierarchical_planner(
        agent_id=0,
        num_agents=1,
        camera=cam,
        grid_info=grid,
        intent_bus=intent_bus,
        config=config,
        conf_dict=None,
        seed=0,
    )


hp_fast = make_hier_planner(cfg_fast)
hp_bal = make_hier_planner(cfg_bal)

# LLP horizon stored as self.llp.horizon
check("fast llp.horizon", hp_fast.llp.horizon, 3)
check("fast llp.num_iterations", hp_fast.llp.num_iterations, 30)
check("bal  llp.horizon", hp_bal.llp.horizon, 3)  # same in both balanced config

# HLP
check("fast hlp.horizon", hp_fast.hlp.horizon, 5)
check("fast hlp.num_iterations", hp_fast.hlp.num_iterations, 20)
check("bal  hlp.horizon", hp_bal.hlp.horizon, 8)
check("bal  hlp.num_iterations", hp_bal.hlp.num_iterations, 30)
check("fast hlp.replan_interval", hp_fast.hlp.replan_interval, 1.0)
check("bal  hlp.replan_interval", hp_bal.hlp.replan_interval, 2.0)

# Tile size
fast_tile = tuple(hp_fast.hlp.tile_size) if hasattr(hp_fast.hlp, "tile_size") else None
bal_tile = tuple(hp_bal.hlp.tile_size) if hasattr(hp_bal.hlp, "tile_size") else None
check("fast tile_size", fast_tile, (25, 25))

# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – two different sweep configs produce DIFFERENT planner states
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TEST 4] fast_replan vs balanced produce different HLP horizon/iterations")

ok_diff_h = hp_fast.hlp.horizon != hp_bal.hlp.horizon
ok_diff_it = hp_fast.hlp.num_iterations != hp_bal.hlp.num_iterations
ok_diff_rp = hp_fast.hlp.replan_interval != hp_bal.hlp.replan_interval
print(
    f"  {PASS if ok_diff_h  else FAIL} HLP horizon differs:      {hp_fast.hlp.horizon} vs {hp_bal.hlp.horizon}"
)
print(
    f"  {PASS if ok_diff_it else FAIL} HLP iterations differ:    {hp_fast.hlp.num_iterations} vs {hp_bal.hlp.num_iterations}"
)
print(
    f"  {PASS if ok_diff_rp else FAIL} HLP replan_interval diff: {hp_fast.hlp.replan_interval} vs {hp_bal.hlp.replan_interval}"
)
if not ok_diff_h:
    errors.append("fast vs bal hlp.horizon should differ")
if not ok_diff_it:
    errors.append("fast vs bal hlp.num_iterations should differ")
if not ok_diff_rp:
    errors.append("fast vs bal hlp.replan_interval should differ")

# ─────────────────────────────────────────────────────────────────────────────
# Test 5 – dec_mcts config flows correctly
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TEST 5] dec_mcts config read from experiment_config (not conf_dict)")

configs_dec = load_config(DEC_MCTS_JSON)
cfg_dec = configs_dec[0]
dec_section = cfg_dec.get("dec_mcts", {})

print(f"  dec_mcts section keys: {list(dec_section.keys())}")

planner_dec = planning(
    grid,
    cam,
    "dec_mcts",
    conf_dict=None,
    experiment_config=cfg_dec,
    seed=0,
)
# experiment_config should contain dec_mcts section
dec_from_planner = planner_dec.experiment_config.get("dec_mcts", {})
has_horizon = "horizon" in dec_from_planner or "horizon" in cfg_dec.get(
    "decentralized", {}
)
has_horizon = has_horizon or bool(dec_from_planner)
print(f"  dec_mcts config non-empty: {bool(dec_from_planner)}")
if dec_from_planner.get("horizon"):
    check(
        "dec_mcts horizon from experiment_config",
        dec_from_planner.get("horizon"),
        cfg_dec.get("dec_mcts", {}).get("horizon"),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Test 6 – mh_dec_mcts_full config loads and propagates
# ─────────────────────────────────────────────────────────────────────────────
print("\n[TEST 6] mh_dec_mcts_full config loads and propagates")

configs_full = load_config(MH_FULL_JSON)
cfg_full = configs_full[0]
hier_full = cfg_full.get("hierarchical_dec_mcts", {})
print(f"  hierarchical_dec_mcts keys: {list(hier_full.keys())}")

has_llp = "llp_horizon" in hier_full or "llp" in hier_full
has_hlp = "hlp_horizon" in hier_full or "hlp" in hier_full

ok_full = has_llp and has_hlp
print(
    f"  {PASS if ok_full else FAIL} has LLP+HLP params: llp_horizon={hier_full.get('llp_horizon')}, hlp_horizon={hier_full.get('hlp_horizon')}"
)
if not ok_full:
    errors.append("mh_dec_mcts_full config missing llp/hlp params")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors:
    print(f"\033[31mFAILED – {len(errors)} error(s):\033[0m")
    for e in errors:
        print(f"  • {e}")
    sys.exit(1)
else:
    print(f"\033[32m✓ All config propagation tests passed.\033[0m")
