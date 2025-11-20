# MCTS diagnostics module
# See previous explanation for usage.
from typing import List, Dict, Any
import math
import numpy as np


def _root_children(planner):
    return sorted(list(planner.root.children.items()), key=lambda kv: str(kv[0]))


def _compute_ucb_terms(planner):
    c = getattr(planner, "ucb1_c", 1.4)
    N_root = max(1, planner.root.visit_count)
    out = []
    for action, child in _root_children(planner):
        N = child.visit_count
        if N == 0:
            Q, U, total = float("nan"), float("inf"), float("inf")
        else:
            Q = child.value / N
            U = 0.0 if c == 0 else c * math.sqrt(2.0 * math.log(N_root) / N)
            total = Q + U
        out.append({"action": action, "Q": Q, "N": N, "U": U, "Q+U": total})
    return out


def _print_table(rows, title=""):
    if title:
        print(f"\n=== {title} ===")
    if not rows:
        print("(no children)")
        return
    cols = ["action", "N", "Q", "U", "Q+U"]
    print("{:<16} {:>8} {:>12} {:>12} {:>12}".format(*cols))
    for r in rows:

        def fmt(x):
            if isinstance(x, float):
                if math.isinf(x):
                    return "inf"
                if math.isnan(x):
                    return "nan"
                return f"{x:>12.6f}"
            return str(x)

        print(
            "{:<16} {:>8} {:>12} {:>12} {:>12}".format(
                str(r["action"])[:16],
                r["N"],
                fmt(r["Q"]),
                fmt(r["U"]),
                fmt(r["Q+U"]),
            )
        )


def _spearman_rank_corr(xs, ys):
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(v), dtype=float)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = 0.5 * (i + j)
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = ranks(np.array(xs)), ranks(np.array(ys))
    rx_mean, ry_mean = rx.mean(), ry.mean()
    num = float(((rx - rx_mean) * (ry - ry_mean)).sum())
    den = float(np.sqrt(((rx - rx_mean) ** 2).sum() * ((ry - ry_mean) ** 2).sum()))
    return 0.0 if den == 0 else num / den


def _root_rank_reports(planner, top_k=5):
    kids = _root_children(planner)
    if not kids:
        print("\n(no root children to rank)")
        return
    visits = np.array([child.visit_count for _, child in kids], dtype=float)
    means = np.array(
        [child.value / max(1, child.visit_count) for _, child in kids], dtype=float
    )
    actions = [a for a, _ in kids]
    idx_vis = list(np.argsort(-visits))
    idx_q = list(np.argsort(-means))

    def top_list(idx_list):
        return [
            {"action": actions[i], "N": int(visits[i]), "Q": float(means[i])}
            for i in idx_list[:top_k]
        ]

    print("\n--- Top by VISITS ---")
    _print_table(
        [
            {
                "action": r["action"],
                "N": r["N"],
                "Q": r["Q"],
                "U": float("nan"),
                "Q+U": float("nan"),
            }
            for r in top_list(idx_vis)
        ]
    )
    print("\n--- Top by AVERAGE VALUE (Q) ---")
    _print_table(
        [
            {
                "action": r["action"],
                "N": r["N"],
                "Q": r["Q"],
                "U": float("nan"),
                "Q+U": float("nan"),
            }
            for r in top_list(idx_q)
        ]
    )
    rho = _spearman_rank_corr(list(-visits), list(-means))
    print(f"\nSpearman rank correlation between VISITS and Q: {rho:.3f}")


def _final_policy_comparison(planner, n_min=0):
    kids = _root_children(planner)
    if not kids:
        print("\n(no root children to choose from)")
        return
    most_visited = max(kids, key=lambda kv: kv[1].visit_count)[0]
    avg_value = max(kids, key=lambda kv: (kv[1].value / max(1, kv[1].visit_count)))[0]
    print(f"\nFinal policy: MOST-VISITED = {most_visited} | AVG-VALUE = {avg_value}")
    if n_min > 0:
        if all(child.visit_count >= n_min for _, child in kids):
            print(f"(All children ≥ {n_min} visits)")
        else:
            print(f"(Some children < {n_min} visits)")


def _run_one_iteration_serial(planner):
    node, path = planner.tree_policy()
    reward = node.rollout(
        rng=planner._rng,
        discount_factor=planner.discount_factor,
        max_depth=planner.max_depth,
    )
    node.backpropagate(reward)


def run_mcts_diagnostics(
    planner,
    num_iterations=500,
    checkpoints=[0.1, 0.5, 1.0],
    top_k=5,
    min_visit_threshold=10,
):
    planned = sorted(set(int(num_iterations * f) for f in checkpoints if 0 < f <= 1))
    next_report = planned.pop(0) if planned else None
    for it in range(1, num_iterations + 1):
        _run_one_iteration_serial(planner)
        if next_report is not None and it >= next_report:
            rows = _compute_ucb_terms(planner)
            _print_table(rows, title=f"Root UCB snapshot at {it}/{num_iterations}")
            _root_rank_reports(planner, top_k=top_k)
            next_report = planned.pop(0) if planned else None
    rows = _compute_ucb_terms(planner)
    _print_table(rows, title=f"FINAL Root UCB snapshot at {num_iterations}")
    _root_rank_reports(planner, top_k=top_k)
    _final_policy_comparison(planner, n_min=min_visit_threshold)
