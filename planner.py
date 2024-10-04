import itertools
import numpy as np
import math
from helper import (
    pairwise_factor_,
    id_converter,
    get_neighbors,
    normalize_probabilities,
)


class planning:
    def __init__(self, true_map):
        self.m = true_map
        self.P_m_given_s = np.zeros_like(true_map)  # prob of m_i = 1
        self.s = []  # history of observations
        self.x = (0, 0, 5)  # uav position, default 0, 0 at alt 5m
        self.z = []  # last measurement at x
        self.last_observation = (self.x, self.z)
        self.a = 1
        self.b = 0.015

    def info_gain(self, m_i_id, x_future):
        return self.entropy_mi(m_i_id) - self.expected_entropy(m_i_id, x_future)

    def entropy_mi(self, m_i_id):
        prior_m_i = self.P_m_given_s.map[m_i_id[0], m_i_id[1]]  # prob- of m_i = 1
        return -prior_m_i * math.log2(prior_m_i) - (1 - prior_m_i) * math.log2(
            1 - prior_m_i
        )

    def sensor_model(self, m_i, z_i, x):
        sigma = self.a * (1 - np.exp(-self.b * x.altitude))
        if z_i == m_i:
            return 1 - sigma(
                x.altitude
            )  # Get the probability of observing the true state value at this altitude
        else:
            return sigma(x.altitude)

    def observed_m_ids(self, new_z):
        [obsd_m_i_min, obsd_m_j_min] = id_converter(new_z, [0, 0], self.P_m_given_s)
        [obsd_m_i_max, obsd_m_j_max] = id_converter(
            new_z, [new_z.map.shape[0] - 1, new_z.map.shape[1] - 1], self.P_m_given_s
        )
        observed_m = []
        for i_b in range(obsd_m_i_min, obsd_m_i_max):
            for j_b in range(obsd_m_j_min, obsd_m_j_max):
                observed_m.append((i_b, j_b))
        return observed_m

    def CRF(self, z_future, x_future):
        posterior_m_given_s = self.P_m_given_s
        observed_m = self.observed_m_ids(z_future)

        for m_i_pos in observed_m:
            pairwise_product = 1
            m_i = self.m.map[m_i_pos]  # value of m_i
            z_i_future_pos = id_converter(self.P_m_given_s, m_i_pos, z_future)

            evidence_factors = self.sensor_model(
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

    def expected_entropy(self, m_i_id, x_future):
        # z_{i+1} or z_i_future = 0
        posterior_m = self.CRF(z_future, x_future)
        posterior_mi = posterior_m.map[m_i_id]

        entropy_posterior_mi = -posterior_mi * math.log2(posterior_mi) - (
            1 - posterior_mi
        ) * math.log2(1 - posterior_mi)

        # z_{i+1} = 1


list(itertools.product(range(1, k), repeat=2))
