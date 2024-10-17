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

    def info_gain(self, m_i_id, x_future):
        # print("IG check: H- ", self._entropy_mi(m_i_id))
        # print("IG check: H+ ", self._expected_entropy(m_i_id, x_future, z_future))
        ig = self._entropy_mi(m_i_id) - self._expected_entropy(m_i_id, x_future)
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
            prior_m_i_0 = self.M.probability[
                0, m_i_id[0], m_i_id[1]
            ]  # prob- of m_i = 0
            prior_m_i_1 = self.M.probability[
                1, m_i_id[0], m_i_id[1]
            ]  # prob- of m_i = 1
            sum_ = prior_m_i_0 + prior_m_i_1
            prior_m_i_0 = prior_m_i_0 / sum_
            prior_m_i_1 = prior_m_i_1 / sum_

            total_entropy += self._entropy_mi(m_i_id)

            # print(
            #     "m:{} entropy: {}, total:{}".format(
            #         m_i_id, self._entropy_mi(m_i_id), total_entropy
            #     )
            # )
        return total_entropy

    def _CRF_elementwise(self, z_future, x_future, m_i_pos, Z):
        posterior_mi_given_s = []
        for m_i in range(self.M.probability.shape[0]):  # m_i=0 and m_i=1
            pairwise_product = 1
            evidence_factors = self.uav.sensor_model(m_i, z_future.z, x_future)
            edges = get_neighbors(self.M.map, m_i_pos)

            for m_j_pos in edges:
                pairwise_product *= pairwise_factor_(
                    m_i,
                    self.M.map[m_j_pos],
                    obs_map=Z,
                    type="equal",
                )
            posterior_mi_given_s.append(pairwise_product * evidence_factors)
        posterior_mi_given_s = np.array(posterior_mi_given_s)
        posterior_mi_given_s *= self.M.probability[:, m_i_pos[0], m_i_pos[1]]
        total = posterior_mi_given_s[0] + posterior_mi_given_s[1]
        posterior_mi_given_s = posterior_mi_given_s / total
        return posterior_mi_given_s

    def _expected_entropy(self, m_i_id, x_future):
        expected_entropy = 0
        sampled_Z = self.uav.sample_observation(self.M, x=x_future, noise=True)
        z_i_id = id_converter(self.M, m_i_id, sampled_Z)

        for z_i in range(2):
            z_future = point(
                z=z_i,
                x=sampled_Z.x[z_i_id],
                y=sampled_Z.y[z_i_id],
                p=sampled_Z.probability[z_i, z_i_id[0], z_i_id[1]],
            )

            posterior_mi = self._CRF_elementwise(
                z_future, x_future, m_i_id, Z=sampled_Z
            )

            entropy_posterior_mi = -posterior_mi[0] * math.log2(
                posterior_mi[0]
            ) - posterior_mi[1] * math.log2(posterior_mi[1])

            expected_entropy += (
                self.uav.prob_future_observation(x_future, z_future)
                * entropy_posterior_mi
            )
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

        for action in permitted_actions:
            # UAV position after taking action a
            x_future = uav_position(self.uav.x_future(action))

            info_gain_action_a = 0
            m_s = observed_m_ids(uav=self.uav, uav_pos=x_future)
            for m_i_id in m_s:  # observed m cells
                info_gain_action_a += self.info_gain(m_i_id, x_future)
            info_gain_action[action] = info_gain_action_a
        print(info_gain_action)
        next_action = max(info_gain_action, key=info_gain_action.get)
        print(next_action)
        return next_action

    def mapping(self, x, z):
        self.uav.set_position(x.position)
        self.uav.set_altitude(x.altitude)

        # collect z_{t+1} observation @TODO add sensor model
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
