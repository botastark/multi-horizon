import numpy as np
import math
import random
from helper import (
    argmax_event_matrix,
    pairwise_factor_,
    id_converter,
    get_neighbors,
    normalize_probabilities,
    observed_m_ids,
    sample_event_matrix,
    uav_position,
    point,
)


class planning:
    def __init__(self, true_map, uav_camera):
        self.M = true_map.copy()
        self.M.probability = np.array(  # P(m|s)
            [0.5 * np.ones_like(self.M.map), 0.5 * np.ones_like(self.M.map)]
        )
        self.M.map = sample_event_matrix(self.M.probability)
        self.curr_entropy = 0
        self.s = []  # history of observations
        self.uav = uav_camera
        self.z = true_map.copy()  # last measurement at x
        self.last_observation = ([], [])
        self.last_action = ""
        self.pairwise_factor_type = "equal"

    def info_gain(self, m_i_id, x_future, mexgen=False):
        # print("IG check: H- ", self._entropy_mi(m_i_id))
        # print("IG check: H+ ", self._expected_entropy(m_i_id, x_future, z_future))
        ig = self._entropy_mi(m_i_id) - self._expected_entropy(
            m_i_id, x_future, mexgen=mexgen
        )
        return ig

    def _entropy_mi(self, m_i_id):
        eps = 1e-10
        prior_m_i_0 = self.M.probability[0, m_i_id[0], m_i_id[1]]  # prob- of m_i = 0
        prior_m_i_1 = self.M.probability[1, m_i_id[0], m_i_id[1]]  # prob- of m_i = 1
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
        for i, j in np.ndindex(self.M.map.shape):
            m_i_id = (i, j)
            total_entropy += self._entropy_mi(m_i_id)
        return total_entropy

    def _CRF_elementwise(self, z_future, x_future, m_i_pos, Z):
        posterior_mi_given_s = []
        for m_i in range(self.M.probability.shape[0]):  # m_i=0 and m_i=1
            pairwise_product = 1
            evidence_factors = self.uav.sensor_model( z_future.z, m_i,  x_future)
            edges = get_neighbors(self.M.map, m_i_pos)

            for m_j_pos in edges:
                pairwise_product *= pairwise_factor_(
                    m_i,
                    self.M.map[m_j_pos],
                    obs_map=Z,
                    type=self.pairwise_factor_type,
                )
            posterior_mi_given_s.append(pairwise_product * evidence_factors)
        posterior_mi_given_s = np.array(posterior_mi_given_s)
        posterior_mi_given_s *= self.M.probability[:, m_i_pos[0], m_i_pos[1]]
        total = posterior_mi_given_s[0] + posterior_mi_given_s[1]
        posterior_mi_given_s = posterior_mi_given_s / total
        return posterior_mi_given_s

    def _expected_entropy(self, m_i_id, x_future, mexgen=False):
        expected_entropy = 0
        if mexgen:
            sampled_Z = self.uav.mex_gen_observation(self.M, x=x_future)
        P_z = self.observation_prob(m_i_id, x_future)
        for z_i in range(2):
            # posterior distribution probabilities
            # p(m = M|z = Z) = (p(z = Z|m = M)p(m = M))/p(z = Z)
            p_0z = (
                self.uav.sensor_model(z_i, 0, x_future)
                * self.M.probability[0, m_i_id[0], m_i_id[1]]
                / P_z[z_i]
            )
            p_1z = (
                self.uav.sensor_model(z_i, 1, x_future)
                * self.M.probability[1, m_i_id[0], m_i_id[1]]
                / P_z[z_i]
            )

            entropy_posterior_mi = -p_0z * math.log2(p_0z) - p_1z * math.log2(p_1z)

            expected_entropy += P_z[z_i] * entropy_posterior_mi
        return expected_entropy

    def select_action(self, strategy="ig"):

        info_gain_action = {}
        permitted_actions = self.uav.permitted_actions(self.uav)  # at UAV position x
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

            visited_x = [s_[0] for s_ in self.s]
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
            # UAV position after taking action a
            x_future = uav_position(self.uav.x_future(action))

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

    def mapping(self, x, z):
        self.uav.set_position(x.position)
        self.uav.set_altitude(x.altitude)

        # collect z_{t+1} observation from camera (done in main)
        self.z = z

        # CRF to update belief probabilities
        m_s = observed_m_ids(uav=self.uav, uav_pos=x)

        for m_i_id in m_s:  # observed m cells
            z_i_id = id_converter(self.M, m_i_id, self.z)
            z_future = point(
                z=self.z.map[z_i_id],
                x=self.z.x[z_i_id],
                y=self.z.y[z_i_id],
                p=self.z.probability[self.z.map[z_i_id], z_i_id[0], z_i_id[1]],
                # p=0,  # we do not care
            )
            posterior_mi = self._CRF_elementwise(z_future, x, m_i_id, Z=self.z)
            self.M.probability[:, m_i_id[0], m_i_id[1]] = posterior_mi

        # Store observations
        self.last_observation = (self.uav.get_x(), self.z)
        self.s.append(self.last_observation)

        # update belief matrix M
        self.curr_entropy = self._entropy()
        # self.M.map = sample_event_matrix(self.M.probability)

        self.M.map = argmax_event_matrix(self.M.probability)

    def get_current_state(self):
        return self.M

    def get_uav_current_pos(self):
        return (self.uav.position, self.uav.altitude)

    def get_belief(self):
        return self.M.map

    def get_prob(self):
        return self.M.probability

    def get_entropy(self):
        return self.curr_entropy

    def get_last_observation(self):
        return self.last_observation

    def observation_prob(self, m_i_id, x_future):
        P_z_zero = 0
        # p(z = 0) = p(z = 0|m = 0)p(m = 0) + p(z = 0|m = 1)p(m = 1)

        P_z_zero = (
            self.uav.sensor_model(0, 0, x_future)
            * self.M.probability[0, m_i_id[0], m_i_id[1]]
            + self.uav.sensor_model(1, 0, x_future)
            * self.M.probability[1, m_i_id[0], m_i_id[1]]
        )
        # p(z = 1) = 1 - p(z = 0)
        P_z = [P_z_zero, 1 - P_z_zero]
        return P_z
