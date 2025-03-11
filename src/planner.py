import random
import numpy as np

# from typing import Dict, List, Tuple, Union
from helper import uav_position


class planning:
    def __init__(self, grid_info, uav, strategy, conf_dict=None, optimal_alt=21.6):
        self.M = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
        self.uav = uav
        self.last_action = None
        self.strategy = strategy
        self.conf_dict = conf_dict
        self.optimal_altitude = optimal_alt
        self.sweep_direction = None

    def reset(self, conf_dict=None):
        self.uav.reset()
        self.conf_dict = conf_dict
        self.last_action = None
        self.M = np.ones_like(self.M) * 0.5

    def info_gain(self, var, x_future, mexgen=False):
        if mexgen == False:
            ig = self.H(var) - self._expected_entropy(var, x_future)
        else:
            sampled_observation = self.sample_binary_observations(
                var, x_future.altitude
            )
            expected_entropy = self.compute_future_entropy(var, sampled_observation)

            ig = np.sum(self.H(var)) - expected_entropy

        return ig

    def get_entropy(self, belief):
        return np.sum(self.H(belief))

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
        # Check for NaN values in var
        assert not np.any(np.isnan(var)), f"NaN detected in var: {var}"
        var = np.clip(var, 0.0, 1.0)  # Clamps values to the range [0, 1]

        # assert np.all(np.greater_equal(var, 0.0)), f"{var[np.isnan(var)]}"
        # assert np.all(np.less_equal(var, 1.0)), f"{var[np.isnan(var)]}"

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
        sigma0 = np.clip(sigma0, 0.0, 1.0)
        sigma1 = np.clip(sigma1, 0.0, 1.0)
        a = (1.0 - sigma0) * (1.0 - var) + (sigma1 * var)
        # p(z = 1) = 1 - p(z = 0)
        b = 1.0 - a + 1e-6

        assert np.all(np.greater_equal(var, 0.0)), f"{var[np.isnan(var)]}"
        assert np.all(np.less_equal(var, 1.0)), f"{var[np.isnan(var)]}"

        # posterior distribution probabilities
        # p(m = 1|z = 0) = (p(z = 0|m = 1)p(m = 1))/p(z = 0)
        p10 = (sigma1 * var) / a
        # p(m = 1|z = 1) = (p(z = 1|m = 1)p(m = 1))/p(z = 1)
        p11 = ((1.0 - sigma1) * var) / b

        assert np.all(np.greater_equal(np.round(p10, decimals=2), 0.0)) and np.all(
            np.less_equal(np.round(p10, decimals=2), 1.0)
        ), f"{p10}"
        assert np.all(np.greater_equal(p11, 0.0)) and np.all(
            np.less_equal(p11, 1.0)
        ), f"{sigma1}-{var[np.greater(p11, 1.0)]}-{b[np.greater(p11, 1.0)]}"

        # conditional entropy: average of the entropy of the posterior distribution probabilities
        # H(m|z) = p(z = 0)H(p(m = 1|z = 0)) + p(z = 1)H(p(m = 1|z = 1))
        cH = a * self.H(p10) + b * self.H(p11)

        assert np.all(np.greater_equal(cH, 0.0))

        return cH

    def _expected_entropy(self, var, x_future):
        expected_entropy = 0
        a = 1
        b = 0.015
        sigma = a * (1 - np.exp(-b * x_future.altitude))

        if self.conf_dict is not None:
            s0, s1 = self.conf_dict[np.round(x_future.altitude, decimals=2)]
        else:
            s0, s1 = sigma, sigma

        expected_entropy = self.cH(var, s0, s1)

        return expected_entropy

    """
    Upd starting here
    """

    def random_action(self, permitted_actions):
        if not permitted_actions:
            raise ValueError("The permitted_actions list is empty.")
        return random.choice(permitted_actions)

    def sweep(self, permitted_actions, visited_x):
        # print(f"uav alt {self.uav.get_x().altitude}\n")
        if (
            self.uav.get_x().altitude < self.optimal_altitude
            and "up" in permitted_actions
        ):
            self.last_action = "up"
            return "up", None

        sweep_actions = []
        for action in permitted_actions:
            x_future = uav_position(self.uav.x_future(action))
            if x_future not in visited_x and action != "up" and action != "down":
                sweep_actions.append(action)

        if self.sweep_direction is None:
            if len(sweep_actions) == 1:
                if sweep_actions[0] == "left" or sweep_actions[0] == "right":
                    self.sweep_direction = "LeftRight"
                else:
                    self.sweep_direction = "BackFront"
            else:
                self.sweep_direction = random.choice(["LeftRight", "BackFront"])
        # print(
        #     f"Current UAV postion: {self.uav.get_x().position}, {self.uav.get_x().altitude}"
        # )
        # print(f"front pos:", uav_position(self.uav.x_future("front")).position)
        # print(f"back pos:", uav_position(self.uav.x_future("back")).position)
        # print(f"left pos:", uav_position(self.uav.x_future("left")).position)
        # print(f"right pos:", uav_position(self.uav.x_future("right")).position)

        # print(f"Permitted actions: {permitted_actions}")
        # print(f"Sweep actions available: {sweep_actions}")
        # print(f"Last action: {self.last_action}")
        # print(f"Determined sweep direction: {self.sweep_direction}")
        # print(f"Visited x: {[v.position for v in visited_x]}")
        self.sweep_direction = "LeftRight"
        if self.sweep_direction == "LeftRight":
            # give propriority to left or right (if one of them is present in sweep_actions, only one can be present at a time)
            if "left" in sweep_actions:
                self.last_action = "left"
            elif "right" in sweep_actions:
                self.last_action = "right"
            elif "front" in sweep_actions:
                self.last_action = "front"
            elif "back" in sweep_actions:
                self.last_action = "back"
            else:
                self.last_action = "hover"
        if self.sweep_direction == "BackFront":
            # give propriority to back or front (if one of them is present in sweep_actions, only one can be present at a time)
            if "back" in sweep_actions:
                self.last_action = "back"
            elif "front" in sweep_actions:
                self.last_action = "front"
            elif "left" in sweep_actions:
                self.last_action = "left"
            elif "right" in sweep_actions:
                self.last_action = "right"
            else:
                self.last_action = "hover"
        # print(f"Selected action: {self.last_action}")
        # if self.last_action == "hover":
        #     print("hovering")
        # self.last_action = random.choice(sweep_actions)
        return self.last_action, None

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
            obs_M = self.M[obsd_m_i_min:obsd_m_i_max, obsd_m_j_min:obsd_m_j_max, 1]
            info_gain_action_a = np.sum(self.info_gain(obs_M, x_future, mexgen=mexgen))
            info_gain_action[action] = info_gain_action_a

        # Find the maximum information gain
        max_gain = max(info_gain_action.values())

        # Collect actions with the maximum info gain
        max_gain_actions = [
            action for action, gain in info_gain_action.items() if gain == max_gain
        ]

        next_action = random.choice(max_gain_actions)
        # Update previous action for the next step
        self.last_action = next_action
        return next_action, info_gain_action

    def compute_future_entropy(
        self, prior: np.ndarray, sampled_observation: np.ndarray
    ) -> float:
        """
        Compute expected future entropy given sampled observation
        """
        # Clip values for numerical stability
        prior = np.clip(prior, 1e-10, 1 - 1e-10)
        sampled_observation = np.clip(sampled_observation, 1e-10, 1 - 1e-10)
        # print(sampled_observation)
        # Estimate posterior using sampled observation
        # P(m|z) ∝ P(z|m)P(m)
        # P(z=1|m=1)
        likelihood_ratio = sampled_observation
        P_z1 = likelihood_ratio * prior + (1 - likelihood_ratio) * (1 - prior)

        P_z0 = 1 - P_z1
        P_m1_given_z1 = likelihood_ratio * prior / P_z1
        P_m1_given_z0 = (1 - likelihood_ratio) * prior / P_z0
        P_m1_given_z1 = np.clip(P_m1_given_z1, 1e-10, 1 - 1e-10)
        P_m1_given_z0 = np.clip(P_m1_given_z0, 1e-10, 1 - 1e-10)

        # Entropy for P(m|z=1)
        H_m_given_z1 = self.H(P_m1_given_z1)
        H_m_given_z0 = self.H(P_m1_given_z0)
        entropy = P_z0 * H_m_given_z0 + P_z1 * H_m_given_z1

        # posterior = (likelihood_ratio * prior) / (likelihood_ratio * prior + (1-likelihood_ratio)*(1 - prior))

        # Compute entropy of estimated posterior
        # print(posterior)
        # posterior = (likelihood_ratio * prior) / (likelihood_ratio * prior + (1 - prior))
        # entropy = self.H(posterior)
        return np.sum(entropy)

    def select_action(self, belief, visited_x):
        self.M = belief

        permitted_actions = self.uav.permitted_actions(self.uav)  # at UAV position x
        # if self.strategy == "random":
        # return self.random_action(permitted_actions)
        if self.strategy == "sweep":
            return self.sweep(permitted_actions, visited_x)

        # IG based IPP strategy
        if self.strategy == "ig_with_mexgen":
            mexgen = True
        else:
            mexgen = False
        return self.ig_based(permitted_actions, mexgen)

    def sample_binary_observations(self, belief_map, altitude, num_samples=5):
        """
        Samples binary observations from a belief map with noise based on altitude.

        Args:
            belief_map (np.ndarray): Belief map of shape (m, n, 2), where belief_map[..., 1] is P(m=1).
            altitude (float): UAV altitude affecting noise level.
            num_samples (int): Number of samples for averaging.
            noise_factor (float): Base noise factor scaled with altitude.

        Returns:
            np.ndarray: Averaged binary observation map of shape (m, n).
        """
        m, n = belief_map.shape
        sampled_observations = np.zeros((m, n, num_samples))
        a = 0.2
        b = 0.05
        var = a * (1 - np.exp(-b * altitude))
        noise_std = np.sqrt(var)
        # noise_std = noise_factor * altitude  # Noise increases with altitude

        for i in range(num_samples):
            # Sample from the probability map with added Gaussian noise
            noise = np.random.normal(loc=0.0, scale=noise_std, size=(m, n))
            noisy_prob = belief_map + noise  # Add noise to P(m=1)
            noisy_prob = np.clip(noisy_prob, 0, 1)  # Ensure probabilities are valid

            # Sample binary observation
            sampled_observations[..., i] = np.random.binomial(1, noisy_prob)

        # Return the averaged observation map
        return np.mean(sampled_observations, axis=-1)
