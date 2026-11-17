"""
Discounted UCT (D-UCT) statistics, as used by Dec-MCTS / MH-MCTS.

Seiler et al. 2024, Eq. (4)-(5) (after Best et al. 2019): when teammates' or
the other planning level's intents change while a tree keeps growing, the
reward function drifts. Older episodes are therefore discounted:

    C_t(s) = sum_u gamma^(t-u) 1_s(u)                 discounted visit count
    V_t(s) = sum_u gamma^(t-u) r_u 1_s(u)             discounted reward sum
    D-UCT(s, s') = V_t(s')/C_t(s') + c * sqrt(ln C_t(s) / C_t(s'))

`t` counts the iterations of one tree. Stats are decayed lazily: each node
remembers the iteration of its last update.
"""

import math


class DUCTStats:
    """Discounted count/value of one tree node (or node-action edge)."""

    __slots__ = ("count", "value", "t")

    def __init__(self):
        self.count = 0.0
        self.value = 0.0
        self.t = 0

    def add(self, t: int, gamma: float, reward: float) -> None:
        """Record an episode with `reward` passing through this node at iteration t."""
        if self.count > 0.0:
            decay = gamma ** (t - self.t)
            self.count *= decay
            self.value *= decay
        self.count += 1.0
        self.value += reward
        self.t = t

    def count_at(self, t: int, gamma: float) -> float:
        """Discounted visit count C_t."""
        return self.count * gamma ** (t - self.t) if self.count > 0.0 else 0.0

    def mean(self) -> float:
        """Discounted mean reward V_t / C_t (the decay cancels)."""
        return self.value / self.count if self.count > 0.0 else 0.0


def d_uct_score(child: DUCTStats, parent_count: float, t: int, gamma: float, c: float) -> float:
    """D-UCT score of a child; unvisited children score +inf."""
    child_count = child.count_at(t, gamma)
    if child_count <= 1e-12:
        return float("inf")
    return child.mean() + c * math.sqrt(math.log(max(parent_count, 1.0)) / child_count)
