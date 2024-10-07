import numpy as np
import math
from helper import (
    pairwise_factor_,
    id_converter,
    get_neighbors,
    normalize_probabilities,
    observed_m_ids,
    uav_position,
    point,
)


class planning:
    def __init__(self, true_map, uav_camera):
        self.m = true_map.copy()
        self.P_m_given_s = true_map.copy()  # prob of m_i = 1
        self.P_m_given_s.map = 0.5 * np.ones_like(self.P_m_given_s.map)
        self.s = []  # history of observations
        self.uav = uav_camera
        self.z = []  # last measurement at x
        self.last_observation = ()

    def info_gain(self, m_i_id, x_future, z_future):
        return self._entropy_mi(m_i_id) - self._expected_entropy(
            m_i_id, x_future, z_future
        )

    def _entropy_mi(self, m_i_id):
        prior_m_i = self.P_m_given_s.map[m_i_id[0], m_i_id[1]]  # prob- of m_i = 1
        return -prior_m_i * math.log2(prior_m_i) - (1 - prior_m_i) * math.log2(
            1 - prior_m_i
        )

    def _CRF(self, z_future, x_future):
        posterior_m_given_s = self.P_m_given_s.copy()
        observed_m = observed_m_ids(z_future, self.P_m_given_s)

        for m_i_pos in observed_m:
            pairwise_product = 1
            m_i = self.m.map[m_i_pos]  # value of m_i
            z_i_future_pos = id_converter(self.P_m_given_s, m_i_pos, z_future)

            evidence_factors = self.uav.sensor_model(
                m_i, z_future.map[z_i_future_pos], x_future
            )

            edges = get_neighbors(self.m.map, m_i_pos)

            for m_j_pos in edges:
                pairwise_product *= pairwise_factor_(
                    m_i,  # m_i value
                    self.m.map[m_j_pos],
                    obs_map=z_future,
                    type="equal",
                )
            posterior_m_given_s.map[m_i_pos] = pairwise_product * evidence_factors
        posterior_m_given_s.set_map(normalize_probabilities(posterior_m_given_s.map))
        return posterior_m_given_s

    def _expected_entropy(self, m_i_id, x_future, z_futures):

        posterior_m = self._CRF(z_futures, x_future)
        posterior_mi = posterior_m.map[m_i_id]

        entropy_posterior_mi = -posterior_mi * math.log2(posterior_mi) - (
            1 - posterior_mi
        ) * math.log2(1 - posterior_mi)
        z_future_i = z_futures.map[id_converter(self.P_m_given_s, m_i_id, z_futures)]
        z_future_i = point()
        z_i_id = id_converter(self.P_m_given_s, m_i_id, z_futures)
        z_future_i.z = z_futures.map[z_i_id]
        z_future_i.x = z_futures.x[z_i_id]
        z_future_i.y = z_futures.y[z_i_id]
        expected_entropy = (
            self.uav.prob_future_observation(x_future, z_future_i)
            * entropy_posterior_mi
        )
        return expected_entropy

    def select_action(self):

        info_gain_action = {}
        permitted_actions = self.uav.permitted_actions(self.uav)  # at UAV position x

        for action in permitted_actions:
            x_future = uav_position()
            # UAV position after taking action a
            x_future.position, x_future.altitude = self.uav.x_future(action)
            z_futures = self.uav.sample_observation(self.P_m_given_s, x_future)

            info_gain_action_a = 0
            m_s = observed_m_ids(z_futures, self.P_m_given_s)
            for m_i_id in m_s:  # observed m cells
                info_gain_action_a += self.info_gain(m_i_id, x_future, z_futures)
            info_gain_action[action] = info_gain_action_a
        print(info_gain_action)
        next_action = max(info_gain_action, key=info_gain_action.get)
        print(next_action)
        return next_action

    def take_action(self, next_action, truth_map):
        x_future = uav_position()
        # UAV position after taking action a
        x_future.position, x_future.altitude = self.uav.x_future(next_action)
        self.uav.set_position(x_future.position)
        self.uav.set_altitude(x_future.altitude)
        self.z = self.uav.sample_observation(self.P_m_given_s, x_future)
        self.z = self.uav.get_observation(truth_map)
        self.P_m_given_s = self._CRF(self.z, x_future)
