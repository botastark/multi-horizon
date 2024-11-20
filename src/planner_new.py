import math
import random
import numpy as np

from helper import observed_m_ids, uav_position


class planning:
    def __init__(self, belief, uav):
        self.M = belief
        self.uav = uav
        self.last_action = None
        # self.visited_x = []

    def info_gain(self, m_i_id, x_future, mexgen=False):
        # print("IG check: H- ", self._entropy_mi(m_i_id))
        # print("IG check: H+ ", self._expected_entropy(m_i_id, x_future))
        ig = self._entropy_mi(m_i_id) - self._expected_entropy(
            m_i_id, x_future, mexgen=mexgen
        )
        return ig

    def _entropy_mi(self, m_i_id):
        eps = 1e-10
        prior_m_i_0 = self.M[m_i_id[0], m_i_id[1], 0]  # prob- of m_i = 0
        prior_m_i_1 = self.M[m_i_id[0], m_i_id[1], 1]  # prob- of m_i = 1
        sum_ = prior_m_i_0 + prior_m_i_1
        prior_m_i_0 = prior_m_i_0 / sum_
        prior_m_i_1 = prior_m_i_1 / sum_

        prior_m_i_0 = np.clip(prior_m_i_0, eps, 1.0)
        prior_m_i_1 = np.clip(prior_m_i_1, eps, 1.0)
        entropy0 = -prior_m_i_0 * math.log2(prior_m_i_0) if prior_m_i_0 > 0 else 0
        entropy1 = -prior_m_i_1 * math.log2(prior_m_i_1) if prior_m_i_1 > 0 else 0

        return entropy0 + entropy1

    def _entropy(self):
        total_entropy = 0
        for i, j in np.ndindex(self.M.shape):
            m_i_id = (i, j)
            total_entropy += self._entropy_mi(m_i_id)
        return total_entropy

    def sensor_model(self, z, x):
        a = 1
        b = 0.015
        sigma = a * (1 - np.exp(-b * x.altitude))

        if z == 0:
            return np.array([1 - sigma, sigma])  # [P(z=0|m=0), P(z=0|m=1)]
        elif z == 1:
            return np.array([sigma, 1 - sigma])  # [P(z=1|m=0), P(z=1|m=1)]
        return np.array([[1 - sigma, sigma], [sigma, 1 - sigma]])

    def observation_prob(self, m_i_id, x_future):
        P_z_zero = 0
        # p(z = 0) = p(z = 0|m = 0)p(m = 0) + p(z = 0|m = 1)p(m = 1)
        P_z_zero = np.sum(
            self.sensor_model(0, x_future) * self.M[m_i_id[0], m_i_id[1], :]
        )
        # p(z = 1) = 1 - p(z = 0)
        P_z = [P_z_zero, 1 - P_z_zero]
        return P_z

    def _expected_entropy(self, m_i_id, x_future, mexgen=False):
        expected_entropy = 0
        if mexgen:
            sampled_Z = self.uav.mex_gen_observation(self.M, x=x_future)
        P_z = self.observation_prob(m_i_id, x_future)

        for z_i in range(2):
            # posterior distribution probabilities
            # p(m = M|z = Z) = (p(z = Z|m = M)p(m = M))/p(z = Z)

            [p_0z, p_1z] = (
                self.sensor_model(z_i, x_future) * self.M[m_i_id[0], m_i_id[1], :]
            )

            entropy_posterior_mi = -p_0z * math.log2(p_0z) - p_1z * math.log2(p_1z)
            expected_entropy += P_z[z_i] * entropy_posterior_mi
        return expected_entropy

    """
    Upd starting here
    """

    def select_action(self, belief, visited_x, strategy="ig"):
        self.M = belief

        info_gain_action = {}
        permitted_actions = self.uav.permitted_actions(self.uav)  # at UAV position x
        print(f"permitted actions {permitted_actions}")
        if strategy == "random":
            if not permitted_actions:
                raise ValueError("The permitted_actions list is empty.")
            return random.choice(permitted_actions)
        if strategy == "sweep":
            optimal_altitude = 21.6
            if (
                self.uav.get_x().altitude < optimal_altitude
                and "up" in permitted_actions
            ):
                return "up"

            # visited_x = [s_[0] for s_ in self.s]

            for action in permitted_actions:
                x_future = uav_position(self.uav.x_future(action))
                if x_future not in visited_x and action != "up" and action != "down":
                    return action
            return random.choice(permitted_actions)
        # IG based IPP strategy
        if strategy == "ig_with_mexgen":
            mexgen = True
        else:
            mexgen = False
        for action in permitted_actions:
            print(f"action :{action}")
            # UAV position after taking action a
            x_future = uav_position(self.uav.x_future(action))
            print(f" future pos {x_future.position}-{x_future.altitude}")
            info_gain_action_a = 0
            m_s = observed_m_ids(uav=self.uav, uav_pos=x_future)
            for m_i_id in m_s:  # observed m cells
                info_gain_action_a += self.info_gain(m_i_id, x_future, mexgen=mexgen)

            info_gain_action[action] = info_gain_action_a

        # Find the maximum information gain
        max_gain = max(info_gain_action.values())

        # Collect actions with the maximum info gain
        max_gain_actions = [
            action for action, gain in info_gain_action.items() if gain == max_gain
        ]

        # Prefer previous action if it's among the max-gain actions, otherwise choose randomly
        if self.last_action in max_gain_actions:
            next_action = self.last_action
        else:
            next_action = random.choice(max_gain_actions)

        # Update previous action for the next step
        self.last_action = next_action

        print(info_gain_action)
        # next_action = max(info_gain_action, key=info_gain_action.get)
        print(next_action)
        return next_action
