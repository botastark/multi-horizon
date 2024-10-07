import numpy as np
import math
from helper import (
    pairwise_factor_,
    id_converter,
    get_neighbors,
    normalize_probabilities,
    observed_m_ids,
    sensor_model,
)

from uav_camera import camera


class planning:
    def __init__(self, true_map, uav_camera):
        self.m = true_map
        self.P_m_given_s = np.zeros_like(true_map)  # prob of m_i = 1
        self.s = []  # history of observations
        self.x = (0, 0, 5)  # uav position, default 0, 0 at alt 5m
        self.uav = uav_camera
        self.z = []  # last measurement at x
        self.last_observation = (self.x, self.z)

    def info_gain(self, m_i_id, x_future, z_future):
        return self.entropy_mi(m_i_id) - self.expected_entropy(
            m_i_id, x_future, z_future
        )

    def entropy_mi(self, m_i_id):
        prior_m_i = self.P_m_given_s.map[m_i_id[0], m_i_id[1]]  # prob- of m_i = 1
        return -prior_m_i * math.log2(prior_m_i) - (1 - prior_m_i) * math.log2(
            1 - prior_m_i
        )

    def CRF(self, z_future, x_future):
        posterior_m_given_s = self.P_m_given_s
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
        posterior_m_given_s = normalize_probabilities(posterior_m_given_s.map)
        return posterior_m_given_s

    def expected_entropy(self, m_i_id, x_future, z_futures):

        posterior_m = self.CRF(z_futures, x_future)
        posterior_mi = posterior_m.map[m_i_id]

        entropy_posterior_mi = -posterior_mi * math.log2(posterior_mi) - (
            1 - posterior_mi
        ) * math.log2(1 - posterior_mi)
        z_future_i = z_futures.map[id_converter(self.P_m_given_s, m_i_id, z_futures)]
        expected_entropy = (
            self.uav.prob_future_observation(x_future, z_future_i)
            * entropy_posterior_mi
        )
        return expected_entropy

    def select_action(self):
        for action in self.uav.actions:
            # self.uav.

            # z_futures = self.uav.sample_observation(self.P_m_given_s, x_future)

            self.info_gain(m_i_id, x_future, z_future)
