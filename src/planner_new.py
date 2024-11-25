import math
import random
import numpy as np
from typing import Dict, List, Tuple, Union
from helper import observed_m_ids, uav_position


class planning:
    def __init__(self, belief, uav):
        self.M = belief
        self.uav = uav
        self.last_action = None

    # def info_gain(self, m_i_id, x_future, mexgen=False):

    #     var = self.M[m_i_id[0], m_i_id[1], 0]
    #     ig = self.H(var) - self._expected_entropy(var, x_future, mexgen=mexgen)
    #     print("IG check: H- ", self.H(var))
    #     print("IG check: H+ ", self._expected_entropy(var, x_future, mexgen=mexgen))

    #     return ig
    def info_gain(self, var, x_future, mexgen=False):

        # var = self.M[m_i_id[0], m_i_id[1], 0]
        ig = self.H(var) - self._expected_entropy(var, x_future, mexgen=mexgen)
        # print("IG check: H- ", self.H(var))
        # print("IG check: H+ ", self._expected_entropy(var, x_future, mexgen=mexgen))

        return ig

    def H(self, var: [np.ndarray, float]) -> [np.ndarray, float]:  # type: ignore
        """
        Entropy of a binary random variable. Remember that each cell
        in the map is a binary r.v.

        Args:
            var : the map belief or a patch of the map belief or a single cell belief

        Returns:
            array or float: the map belief entropy or a single cell belief entropy
        """

        # I could not make it work - problems with extreme values
        # entropy = -(var * np.log2(var, where=var > 0.0)
        #       + (1.0 - var) * np.log2((1.0 - var), where=(1.0 - var) > 0.0))

        assert np.all(np.greater_equal(var, 0.0)), f"{var[np.isnan(var)]}"
        assert np.all(np.less_equal(var, 1.0)), f"{var[np.isnan(var)]}"

        v1 = var
        v2 = 1.0 - var

        if isinstance(var, np.ndarray):
            v1 = np.where(v1 == 0.0, 1.0, v1)
            v2 = np.where(v2 == 0.0, 1.0, v2)
        else:
            if v1 == 0.0:
                v1 = 1.0
            if v2 == 0.0:
                v2 = 1.0

        l1 = np.log2(v1)
        l2 = np.log2(v2)

        assert np.all(np.less_equal(l1, 0.0))
        assert np.all(np.less_equal(l2, 0.0))

        entropy = -(v1 * l1 + v2 * l2)

        assert np.all(np.greater_equal(entropy, 0.0))

        return entropy

    def cH(self, var: np.ndarray, sigma0: float, sigma1: float) -> np.ndarray:
        """
        Conditional entropy of a binary random variable. Remember that
        each cell in the map is a binary r.v.

        Args:
            var : the map belief or a patch of the map belief or a single cell belief
            sigma0 : likelihood (the probabilistic altitude dependent sensor model FP rate) p(z = 1|m = 0, h)
            sigma1 : likelihood (the probabilistic altitude dependent sensor model FN rate) p(z = 0|m = 1, h)

        Returns:
            array: the map belief conditional entropy
            or a patch of the map belief conditional entropy
            or a single cell belief conditional entropy

        """

        # probability of the evidence
        # p(z = 0) = p(z = 0|m = 0)p(m = 0) + p(z = 0|m = 1)p(m = 1)
        a = (1.0 - sigma0) * (1.0 - var) + (sigma1 * var)
        # p(z = 1) = 1 - p(z = 0)
        b = 1.0 - a

        assert np.all(np.greater_equal(var, 0.0)), f"{var[np.isnan(var)]}"
        assert np.all(np.less_equal(var, 1.0)), f"{var[np.isnan(var)]}"

        # posterior distribution probabilities
        # p(m = 1|z = 0) = (p(z = 0|m = 1)p(m = 1))/p(z = 0)
        p10 = (sigma1 * var) / a
        # p(m = 1|z = 1) = (p(z = 1|m = 1)p(m = 1))/p(z = 1)
        p11 = ((1.0 - sigma1) * var) / b

        assert np.all(np.greater_equal(p10, 0.0)) and np.all(
            np.less_equal(p10, 1.0)
        ), f"{p10}"
        assert np.all(np.greater_equal(p11, 0.0)) and np.all(
            np.less_equal(p11, 1.0)
        ), f"{sigma1}-{var[np.greater(p11, 1.0)]}-{b[np.greater(p11, 1.0)]}"

        # conditional entropy: average of the entropy of the posterior distribution probabilities
        # H(m|z) = p(z = 0)H(p(m = 1|z = 0)) + p(z = 1)H(p(m = 1|z = 1))
        cH = a * self.H(p10) + b * self.H(p11)

        assert np.all(np.greater_equal(cH, 0.0))

        return cH

    def _entropy_mi(self, m_i_id):
        eps = 1e-10
        prior_m_i_0 = self.M[m_i_id[0], m_i_id[1], 0]  # prob- of m_i = 0
        prior_m_i_1 = self.M[m_i_id[0], m_i_id[1], 1]  # prob- of m_i = 1
        # sum_ = prior_m_i_0 + prior_m_i_1
        # prior_m_i_0 = prior_m_i_0 / sum_
        # prior_m_i_1 = prior_m_i_1 / sum_

        prior_m_i_0 = np.clip(prior_m_i_0, eps, 1.0)
        prior_m_i_1 = np.clip(prior_m_i_1, eps, 1.0)
        entropy0 = -prior_m_i_0 * math.log2(prior_m_i_0) if prior_m_i_0 > 0 else 0
        entropy1 = -prior_m_i_1 * math.log2(prior_m_i_1) if prior_m_i_1 > 0 else 0

        return entropy0 + entropy1

    def _entropy(self):
        total_entropy = 0
        return self.H(self.M)
        # for i, j in np.ndindex(self.M.shape):
        #     m_i_id = (i, j)
        #     total_entropy += self._entropy_mi(m_i_id)
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
        # p(z = 0) = p(z = 0|m)*p(m) =  [p(z = 0|m = 0), p(z = 0|m = 1)]* [p(m = 0); p(m = 1)]
        P_z_zero = np.dot(
            self.sensor_model(0, x_future), self.M[m_i_id[0], m_i_id[1], :]
        )
        # p(z = 1) = 1 - p(z = 0)
        P_z = [P_z_zero, 1 - P_z_zero]
        return P_z

    def _expected_entropy(self, var, x_future, mexgen=False):
        expected_entropy = 0
        a = 1
        b = 0.015
        sigma = a * (1 - np.exp(-b * x_future.altitude))
        expected_entropy = self.cH(var, sigma, sigma)

        # # if mexgen:
        # #     sampled_Z = self.uav.mex_gen_observation(self.M, x=x_future)
        # P_z = self.observation_prob(m_i_id, x_future)

        # for z_i in range(2):
        #     # posterior distribution probabilities
        #     # p(m = M|z = Z) = (p(z = Z|m = M)p(m = M))/p(z = Z)

        #     [p_0z, p_1z] = (
        #         self.sensor_model(z_i, x_future) * self.M[m_i_id[0], m_i_id[1], :]
        #     ) / P_z[z_i]

        #     entropy_posterior_mi = -p_0z * math.log2(p_0z) - p_1z * math.log2(p_1z)
        #     expected_entropy += P_z[z_i] * entropy_posterior_mi
        return expected_entropy

    """
    Upd starting here
    """

    def random_action(self, permitted_actions):
        if not permitted_actions:
            raise ValueError("The permitted_actions list is empty.")
        return random.choice(permitted_actions)

    def sweep(self, permitted_actions, visited_x):
        optimal_altitude = 21.6
        if self.uav.get_x().altitude < optimal_altitude and "up" in permitted_actions:
            return "up"

        # visited_x = [s_[0] for s_ in self.s]
        for action in permitted_actions:
            x_future = uav_position(self.uav.x_future(action))
            if x_future not in visited_x and action != "up" and action != "down":
                return action
        return random.choice(permitted_actions)

    def ig_based(self, permitted_actions, mexgen):
        info_gain_action = {}
        for action in permitted_actions:
            # UAV position after taking action a
            x_future = uav_position(self.uav.x_future(action))
            info_gain_action_a = 0
            [[obsd_m_i_min, obsd_m_i_max], [obsd_m_j_min, obsd_m_j_max]] = (
                self.uav.get_range(
                    position=x_future.position,
                    altitude=x_future.altitude,
                    index_form=True,
                )
            )
            obs_M = self.M[obsd_m_i_min:obsd_m_i_max, obsd_m_j_min:obsd_m_j_max, 0]
            # print(f"obs M shape {obs_M.shape}")
            info_gain_action_a = np.sum(self.info_gain(obs_M, x_future))
            # m_s = observed_m_ids(uav=self.uav, uav_pos=x_future)
            # for m_i_id in m_s:  # observed m cells
            # info_gain_action_a += self.info_gain(m_i_id, x_future, mexgen=mexgen)

            # print(
            #     f"action :{action} | future pos {x_future.position} {x_future.altitude} |  ig {info_gain_action_a}"
            # )
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

    def select_action(self, belief, visited_x, strategy="ig"):
        self.M = belief

        permitted_actions = self.uav.permitted_actions(self.uav)  # at UAV position x
        print(f"permitted actions {permitted_actions}")

        if strategy == "random":
            return self.random_action(permitted_actions)
        if strategy == "sweep":
            return self.sweep(permitted_actions, visited_x)

        # IG based IPP strategy
        if strategy == "ig_with_mexgen":
            mexgen = True
        else:
            mexgen = False
        return self.ig_based(permitted_actions, mexgen)
