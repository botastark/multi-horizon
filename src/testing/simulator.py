import time
import numpy as np
import copy
from itertools import product
from typing import Dict, List, Tuple, Union

from matplotlib import pyplot as plt, patches, colormaps as cmaps, colors
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.path import Path
import pyglet

np.set_printoptions(floatmode="fixed")

X, Y, Z = 0, 1, 2
I, J = 0, 1

states = np.array([], dtype=float)  # agents positions global container
map_beliefs = np.array([], dtype=float)  # agents map beliefs global container
map_belief_entropies = np.array(
    [], dtype=float
)  # agents map belief entropies global container
agg_map_belief = np.array([], dtype=float)
agg_map_belief_entropy = np.array([], dtype=float)

news_map_beliefs = np.array([], dtype=float)  # agents news map beliefs global container


def H(var: [np.ndarray, float]) -> [np.ndarray, float]:
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


def cH(var: np.ndarray, sigma0: float, sigma1: float) -> np.ndarray:
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
    cH = a * H(p10) + b * H(p11)

    assert np.all(np.greater_equal(cH, 0.0))

    return cH


def MSE(agent_id: int, gt: np.ndarray) -> float:
    return (np.square(gt - map_beliefs[:, :, agent_id])).mean()


def IoU(box1: Dict, box2: Dict) -> float:
    x_left = max(box1["ul"][0], box2["ul"][0])
    y_top = max(box1["ul"][1], box2["ul"][1])
    x_right = min(box1["br"][0], box2["br"][0])
    y_bottom = min(box1["br"][1], box2["br"][1])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (box1["br"][0] - box1["ul"][0]) * (box1["br"][1] - box1["ul"][1])
    box2_area = (box2["br"][0] - box2["ul"][0]) * (box2["br"][1] - box2["ul"][1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


# def E(var1, var2):
#     return np.count_nonzero(np.where(var1 > 0.5, 1, 0).astype(int) - var2)


class State:
    def __init__(
        self,
        space_clip_constraints: Dict,
        position: np.array(
            3,
        ) = np.array([0.0, 0.0, 0.0]),
    ):
        self.space_clip_constraints = space_clip_constraints
        self.position = np.clip(
            position,
            self.space_clip_constraints["min"],
            self.space_clip_constraints["max"],
        )

    def set_position(
        self,
        position: np.array(
            3,
        ),
    ):
        self.position = np.clip(
            position,
            self.space_clip_constraints["min"],
            self.space_clip_constraints["max"],
        )


class Camera:
    def __init__(
        self,
        fov=np.pi / 6,
        a0=1.0,
        b0=1.0,
        a1=1.0,
        b1=1.0,
        field_data: Dict = None,
        region_limits: Tuple = None,
    ):
        self.fov = fov
        self.field_data = field_data
        self.a0, self.b0 = a0, b0
        self.a1, self.b1 = a1, b1
        # self.sigma = lambda a, b, h: a * (1 - np.exp(-b * h))  # sigma**2
        self.region_limits = region_limits

    def set_sensor_params(self, kwargs):
        self.a0, self.b0 = kwargs.get("a0"), kwargs.get("b0")
        self.a1, self.b1 = kwargs.get("a1"), kwargs.get("b1")

    def _xy_to_ij(self, xy):
        field_cell_len = self.field_data["field_cell_len"]
        field_len = self.field_data["field_len"]

        x, y = xy[X], xy[Y]

        imgframe_x = x - (-field_len / 2)
        imgframe_y = -(y - field_len / 2)
        # print(imgframe_x, imgframe_y)
        imgframe_x_in_cells = imgframe_x / field_cell_len
        imgframe_y_in_cells = imgframe_y / field_cell_len

        i, j = int(np.round(imgframe_y_in_cells)), int(np.round(imgframe_x_in_cells))

        # print(np.round(imgframe_y_in_cells),np.round(imgframe_x_in_cells))

        return np.array([i, j])

    def get_fp_vertices_ij(
        self,
        position: np.array(
            3,
        ),
    ):
        fp_d = 2 * position[Z] * np.tan(self.fov / 2)

        if self.region_limits is None:
            f_min = self.field_data["field_clip_constraints"]["min"]
            f_max = self.field_data["field_clip_constraints"]["max"]
        else:
            f_min = [self.region_limits[X][0], self.region_limits[Y][0]]
            f_max = [self.region_limits[X][1], self.region_limits[Y][1]]

        fp_vertices_xy = {
            "ul": np.clip(
                np.array([position[X] - fp_d / 2, position[Y] + fp_d / 2]), f_min, f_max
            ),
            "bl": np.clip(
                np.array([position[X] - fp_d / 2, position[Y] - fp_d / 2]), f_min, f_max
            ),
            "ur": np.clip(
                np.array([position[X] + fp_d / 2, position[Y] + fp_d / 2]), f_min, f_max
            ),
            "br": np.clip(
                np.array([position[X] + fp_d / 2, position[Y] - fp_d / 2]), f_min, f_max
            ),
        }

        fp_vertices_ij = {
            "ul": self._xy_to_ij(fp_vertices_xy["ul"]),
            "bl": self._xy_to_ij(fp_vertices_xy["bl"]),
            "ur": self._xy_to_ij(fp_vertices_xy["ur"]),
            "br": self._xy_to_ij(fp_vertices_xy["br"]),
        }

        return fp_vertices_ij, fp_vertices_xy

    def _generate_observation(
        self, fp_vertices_ij: Dict, sigmas: Tuple[float, float], rng, map_ground_truth
    ):

        m = map_ground_truth[
            fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
            fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
        ]

        # remember that sigma is the p(error)
        # i.e.: p(z=1 | m=0, h), p(z=0 | m=1, h)
        # note that in general when the sensor model
        # is not symmetric these quantities are different
        # however we use a symmetric sensor model in our experiments
        sigma0, sigma1 = sigmas[0], sigmas[1]
        random_values = rng.random(m.shape)
        success0 = random_values <= 1.0 - sigma0
        success1 = random_values <= 1.0 - sigma1
        z0 = np.where(np.logical_and(success0, m == 0), 0, 1)
        z1 = np.where(np.logical_and(success1, m == 1), 1, 0)
        z = np.where(m == 0, z0, z1)

        return z

    def get_sigmas(
        self,
        position: np.array(
            3,
        ),
    ):
        # p(z=1 | m=0, h)
        sigma0 = lambda h: self.a0 * (1 - np.exp(-self.b0 * h))
        # p(z=0 | m=1, h)
        sigma1 = lambda h: self.a1 * (1 - np.exp(-self.b1 * h))
        return sigma0(position[Z]), sigma1(position[Z])

    def get_measurements(
        self,
        position: np.array(
            3,
        ),
        rng,
        map_ground_truth,
    ):

        fp_vertices_ij, fp_vertices_xy = self.get_fp_vertices_ij(position)
        sigmas = self.get_sigmas(position)
        z = self._generate_observation(fp_vertices_ij, sigmas, rng, map_ground_truth)

        return {
            "z": z,
            "fp_ij": fp_vertices_ij,
            "sigmas": sigmas,
            "fp_xy": fp_vertices_xy,
        }


class Proximity:
    def __init__(
        self, h_displacement: float, v_displacement: float, radius_multiplier: int = 1
    ):

        self.max_distances = radius_multiplier * np.array(
            [h_displacement, h_displacement, v_displacement], dtype=float
        )

    def get_measurements(
        self,
        position: [
            np.array(
                3,
            ),
            List,
        ],
        id: int,
    ):
        global states

        if isinstance(position, List):
            position = np.array(position)

        check = np.prod(
            np.where(np.abs(states - position) > self.max_distances, 0, 1), axis=1
        )

        neighbors_ids = [e for e in np.flatnonzero(check) if e != id]
        neighbors_positions = [states[n_id, :] for n_id in neighbors_ids]

        return {
            "neighbors_ids": neighbors_ids,
            "neighbors_positions": neighbors_positions,
        }

    def get_predicted_measurements(
        self,
        position: [
            np.array(
                3,
            ),
            List,
        ],
        id: int,
        predicted_states: np.array,
    ):

        if isinstance(position, List):
            position = np.array(position)

        check = np.prod(
            np.where(np.abs(predicted_states - position) > self.max_distances, 0, 1),
            axis=1,
        )

        neighbors_ids = [e for e in np.flatnonzero(check) if e != id]
        neighbors_positions = [predicted_states[n_id, :] for n_id in neighbors_ids]

        return {
            "neighbors_ids": neighbors_ids,
            "neighbors_positions": neighbors_positions,
        }


class Agent:
    def __init__(
        self,
        identity: int,
        state: State,
        camera: Camera,
        proximity: Proximity,
        seed=None,
    ):

        self.id = identity
        self.state = state
        self.camera = camera
        self.proximity = proximity

        # agent random number generator
        if seed is None:
            seed = identity
        self.rng = np.random.default_rng(seed)

        # local map belief
        # self.map_belief = np.ones((n_cell, n_cell), dtype=float) * 0.5
        # local map likelihoods (one per agent)
        # self.map_likelihoods_m_zero = np.ones((n_agents, n_cell, n_cell), dtype=float)
        # self.map_likelihoods_m_one = np.ones((n_agents, n_cell, n_cell), dtype=float)

        # local incoming messages cache
        # self.msg_cache = []

    def get_measurements(self, map_ground_truth):
        camera_measurements = self.camera.get_measurements(
            self.state.position, self.rng, map_ground_truth
        )
        proximity_measurements = self.proximity.get_measurements(
            self.state.position, self.id
        )
        measurements = {**camera_measurements, **proximity_measurements}
        return measurements


class MappingEnv:

    def __init__(self, field_len=50.0, fov=np.pi / 3, **kwargs):

        self.n_agents = kwargs.get("n_agents", 1)
        self.field_len = field_len

        # Camera sensor parameters
        self.fov = fov
        self.a0 = kwargs.get("a0", 0.2)
        self.b0 = kwargs.get("b0", 0.1)
        self.a1 = kwargs.get("a1", 0.2)
        self.b1 = kwargs.get("b1", 0.1)

        # Currently used planner (needed for fixed_regions type to clip the fp_vertices_ij for the observation)
        self.planner_type = kwargs.get("planner_type")

        # Proximity sensor parameters
        self.radius_multiplier = kwargs.get("radius_multiplier", 5)

        self.map_type = kwargs.get("map_type")
        assert self.map_type in ["circle", "random", "gaussian", "split", "uniform"]

        self.env_type = kwargs.get("env_type", "normal")
        assert self.env_type in ["normal", "adhoc"]

        if self.env_type == "normal":
            self.n_h_act, self.n_v_act = 5, 6
        elif self.env_type == "adhoc":
            self.n_h_act, self.n_v_act = 8, 6

        # given field_len, n_h_act
        # we have h_displacement = (field_len/2)/n_h_act
        self.h_displacement = (self.field_len / 2) / self.n_h_act
        # so from d = 2*h*tan(fov*0.5)
        # we can find h such that d/2 = (field_len/2)/n_h_act
        # d = 2*h*tan(fov*0.5) -> d/2 = h*tan(fov*0.5) ->
        # (field_len/2)/n_h_act = h*tan(fov*0.5) - > h = ((field_len/2)/n_h_act)/tan(fov*0.5) ->
        # h = h_displacement/tan(fov*0.5)

        if self.env_type == "normal":
            self.min_space_z = self.h_displacement
        elif self.env_type == "adhoc":
            self.min_space_z = self.h_displacement / np.tan(self.fov * 0.5)

        self.v_displacement = self.min_space_z
        print(f"h_disp {self.h_displacement}: v disp {self.v_displacement}")
        self.max_space_z = self.n_v_act * self.min_space_z

        divisors = [
            i for i in np.arange(1, self.field_len + 1, 1) if self.field_len % i == 0
        ]
        self.field_cell_len = 0.0
        for d in divisors:
            if self.h_displacement / d < 0.1:
                break
            self.field_cell_len = self.h_displacement / d

        self.n_cell = int(self.field_len / self.field_cell_len)

        # Field xy constraints
        self.min_field_x, self.min_field_y = -self.field_len / 2, -self.field_len / 2
        self.max_field_x, self.max_field_y = self.field_len / 2, self.field_len / 2

        # Space xy constraints
        self.min_space_x = -self.field_len / 2
        self.max_space_x = self.field_len / 2
        self.min_space_y = -self.field_len / 2
        self.max_space_y = self.field_len / 2

        field_data = {
            "field_clip_constraints": {
                "min": [self.min_field_x, self.min_field_y],
                "max": [self.max_field_x, self.max_field_y],
            },
            "field_cell_len": self.field_cell_len,
            "field_len": self.field_len,
        }

        space_clip_constraints = {
            "min": [self.min_space_x, self.min_space_y, self.min_space_z],
            "max": [self.max_space_x, self.max_space_y, self.max_space_z],
        }

        # Actions and action space
        # Z - up
        self.action_to_direction = {
            "up": np.array([0, 0, self.v_displacement], dtype=float),
            "down": np.array([0, 0, -self.v_displacement], dtype=float),
            "front": np.array([0, self.h_displacement, 0], dtype=float),
            "back": np.array([0, -self.h_displacement, 0], dtype=float),
            "right": np.array([self.h_displacement, 0, 0], dtype=float),
            "left": np.array([-self.h_displacement, 0, 0], dtype=float),
            "hover": np.array([0, 0, 0], dtype=float),
        }

        self.action_to_id = {
            k: index for index, (k, v) in enumerate(self.action_to_direction.items())
        }
        self.id_to_action = {v: k for k, v in self.action_to_id.items()}

        self.position_graph = {}
        for z in np.arange(self.min_space_z, self.max_space_z + 1, self.v_displacement):
            for x in np.arange(
                self.min_space_x, self.max_space_x + 1, self.h_displacement
            ):
                for y in np.arange(
                    self.min_space_x, self.max_space_x + 1, self.h_displacement
                ):
                    for action, direction in self.action_to_direction.items():
                        next_position = np.clip(
                            np.array([x, y, z], dtype=float) + direction,
                            space_clip_constraints["min"],
                            space_clip_constraints["max"],
                        )

                        if (
                            np.any(
                                np.not_equal(
                                    np.array([x, y, np.round(z, 8)]),
                                    np.array(
                                        [
                                            next_position[X],
                                            next_position[Y],
                                            np.round(next_position[Z], 8),
                                        ]
                                    ),
                                )
                            )
                            or action == "hover"
                        ):

                            if (x, y, int(z)) not in self.position_graph:
                                self.position_graph[(x, y, int(z))] = {
                                    action: next_position
                                }
                            else:
                                self.position_graph[(x, y, int(z))][
                                    action
                                ] = next_position

        # Region splitted field limits per agent
        n_agents_to_n_regions = {
            1: [1, 1],
            2: [2, 1],
            4: [2, 2],
            6: [3, 2],
            8: [4, 2],
            10: [5, 2],
        }
        x_positions = np.arange(
            self.min_space_x, self.max_space_x + 1, self.h_displacement
        )
        y_positions = np.arange(
            self.min_space_y, self.max_space_y + 1, self.h_displacement
        )

        x_region_width = int(
            (len(x_positions) - 1) / n_agents_to_n_regions[self.n_agents][0]
        )
        x_remainder = int(
            (len(x_positions) - 1) % n_agents_to_n_regions[self.n_agents][0]
        )
        y_region_width = int(
            (len(y_positions) - 1) / n_agents_to_n_regions[self.n_agents][1]
        )
        y_remainder = int(
            (len(y_positions) - 1) % n_agents_to_n_regions[self.n_agents][1]
        )

        region_x_index_limits = list(range(0, len(x_positions), x_region_width))
        region_x_index_limits[-1] += x_remainder
        region_y_index_limits = list(range(0, len(y_positions), y_region_width))
        region_y_index_limits[-1] += y_remainder

        region_x_limits = []
        region_y_limits = []

        for i in range(len(region_x_index_limits) - 1):
            region_x_limits.append(
                [
                    x_positions[region_x_index_limits[i]],
                    x_positions[region_x_index_limits[i + 1]],
                ]
            )

        for i in range(len(region_y_index_limits) - 1):
            region_y_limits.append(
                [
                    y_positions[region_y_index_limits[i]],
                    y_positions[region_y_index_limits[i + 1]],
                ]
            )

        self.regions_limits = list(product(region_x_limits, region_y_limits))

        self.regions_limits = [
            list(self.regions_limits[i]) for i in range(len(self.regions_limits))
        ]

        for rl in self.regions_limits:
            n_tot_positions = (((rl[X][1] - rl[X][0]) / self.h_displacement) + 1) * (
                ((rl[Y][1] - rl[Y][0]) / self.h_displacement) + 1
            )
            rl.append(n_tot_positions)


        # Agents
        # self.agents = [Agent(id,
        #                      State(space_clip_constraints),
        #                      Camera(field_data=field_data,
        #                             region_limits=self.regions_limits[id]
        #                             if self.planner_type == "fixed_regions" or self.planner_type == "sweep"
        #                             else None,
        #                             a0=self.a0, b0=self.b0,
        #                             a1=self.a1, b1=self.b1,
        #                             fov=self.fov),
        #                      Proximity(self.h_displacement,
        #                                self.v_displacement,
        #                                radius_multiplier=self.radius_multiplier),
        #                      seed = 0
        #                      )
        #                for id in range(self.n_agents)]

        # Agents
        self.agents = []
        for id in range(self.n_agents):
            state = State(space_clip_constraints)

            region_limits = (
                self.regions_limits[id]
                if self.planner_type == "fixed_regions" or self.planner_type == "sweep"
                else None
            )

            camera = Camera(
                field_data=field_data,
                a0=self.a0,
                b0=self.b0,
                a1=self.a1,
                b1=self.b1,
                fov=self.fov,
                region_limits=region_limits,
            )

            proximity = Proximity(
                self.h_displacement,
                self.v_displacement,
                radius_multiplier=self.radius_multiplier,
            )

            self.agents.append(Agent(id, state, camera, proximity, seed=0))

        global states, map_beliefs, map_belief_entropies, agg_map_belief, agg_map_belief_entropy, news_map_beliefs
        states = np.zeros((self.n_agents, 3), dtype=float)
        map_beliefs = (
            np.ones((self.n_cell, self.n_cell, self.n_agents), dtype=float) * 0.5
        )
        map_belief_entropies = np.ones(
            (self.n_cell, self.n_cell, self.n_agents), dtype=float
        )
        agg_map_belief = np.ones((self.n_cell, self.n_cell), dtype=float) * 0.5
        agg_map_belief_entropy = np.ones((self.n_cell, self.n_cell), dtype=float)
        news_map_beliefs = (
            np.ones(
                (self.n_agents, self.n_agents, self.n_cell, self.n_cell), dtype=float
            )
            * 0.5
        )

        # Altitudes
        self.altitude_to_size = {}

        for z in np.arange(self.min_space_z, self.max_space_z + 1, self.v_displacement):
            fp_vertices_ij, _ = self.agents[0].camera.get_fp_vertices_ij([0.0, 0.0, z])
            n_cell = fp_vertices_ij["ur"][1] - fp_vertices_ij["ul"][1]
            self.altitude_to_size[int(z)] = n_cell**2

        self.optimal_altitude, optimal_ig = -1, -1
        xc = (self.regions_limits[0][X][1] + self.regions_limits[0][X][0]) / 2
        yc = (self.regions_limits[0][Y][1] + self.regions_limits[0][Y][0]) / 2
        for z in np.arange(self.min_space_z, self.max_space_z + 1, self.v_displacement):
            fp_vertices_ij, _ = self.agents[0].camera.get_fp_vertices_ij([xc, yc, z])

            fp_map_belief = map_beliefs[
                fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                0,
            ]

            # H(M_fp) - entropy of the prior = sum of the entropy of each cell in the footprint
            H_M_fp = H(fp_map_belief)

            # H(M_fp|Z) - conditional entropy of the posterior = sum of the averaged entropy of each cell in the footprint
            sigma0, sigma1 = self.agents[0].camera.get_sigmas([xc, yc, z])
            H_M_fp_Z = cH(fp_map_belief, sigma0, sigma1)

            diff = H_M_fp - H_M_fp_Z
            diff = np.where(np.isclose(H_M_fp, H_M_fp_Z), 0.0, diff)

            ig = np.sum(diff)

            if ig >= optimal_ig:
                optimal_ig = ig
                self.optimal_altitude = z

        # position to fp_ij, sigmas dictionary
        self.position_to_data = {}
        for z in np.arange(self.min_space_z, self.max_space_z + 1, self.v_displacement):
            for x in np.arange(
                self.min_space_x, self.max_space_x + 1, self.h_displacement
            ):
                for y in np.arange(
                    self.min_space_x, self.max_space_x + 1, self.h_displacement
                ):
                    fp_vertices_ij, _ = self.agents[0].camera.get_fp_vertices_ij(
                        [x, y, z]
                    )
                    sigmas = self.agents[0].camera.get_sigmas([x, y, z])
                    self.position_to_data[(x, y, int(z))] = {
                        "fp_vertices_ij": fp_vertices_ij,
                        "sigmas": sigmas,
                    }

        # Map
        if self.map_type == "gaussian":
            self.cluster_radius = kwargs.get("cluster_radius", 1)
        self.mul = kwargs.get("mul", 1)
        self.patch_pos = kwargs.get("patch_pos")
        self.cluster_radius_to_amplitude = {}  # container for generate_grf

        # environment rng map ground truth
        self.map_rng = np.random.default_rng(123)

        # environment rng agent position
        self.agent_position_rng = np.random.default_rng(12)

        # Render 2D
        # fig1 = plt.figure(figsize=(10,4))
        # mosaic = [["ground_truth", "ground_truth",
        #            f"agent_map_entropy_{a.id}", f"agent_map_entropy_{a.id}",
        #            #f"agent_map_ig_{a.id}", f"agent_map_ig_{a.id}",
        #            #f"agent_allucinated_map_belief_{a.id}", f"agent_allucinated_map_belief_{a.id}",
        #            f"agent_map_belief_{a.id}", f"agent_map_belief_{a.id}",
        #
        #            ] for a in self.agents]
        # self.ax_dict = fig1.subplot_mosaic(mosaic)

        # render 3D
        # fig2 = plt.figure(figsize=(9, 9))
        # self.ax = fig2.add_subplot(projection='3d', computed_zorder=False)
        # self.ax.view_init(20, -45)

    def step(self, actions: List):
        global states

        assert len(actions) == self.n_agents

        for agent, action in zip(self.agents, actions):
            direction = self.action_to_direction[action]
            agent.state.set_position(agent.state.position + direction)
            states[agent.id, :] = agent.state.position

    def get_observations(self, map_ground_truth):
        observations = [a.get_measurements(map_ground_truth) for a in self.agents]
        for o in observations:
            for v in o["fp_ij"].values():
                assert (
                    v[I] >= 0
                    and v[I] <= self.n_cell
                    and v[J] >= 0
                    and v[J] <= self.n_cell
                )

        return observations

    def _get_info(self):
        return 0

    def __str__(self):

        n_cell_per_h = []

        for z in np.arange(self.min_space_z, self.max_space_z + 1, self.v_displacement):
            n_cell_min_z, _ = self.agents[0].camera.get_fp_vertices_ij([0.0, 0.0, z])
            n_cell_min_z = n_cell_min_z["ur"][1] - n_cell_min_z["ul"][1]
            n_cell_per_h.append(n_cell_min_z)

        return (
            f"\nn_agents: {self.n_agents}\n"
            f"field_len: {self.field_len} [m]\n"
            f"cell_len: {self.field_cell_len} [m]\n"
            f"n_cell: {self.n_cell}\n"
            f"v_displacement: {self.v_displacement} [m]\n"
            f"h_displacement: {self.h_displacement} [m]\n"
            f"min_field_x - max_field_x: {self.min_field_x} - {self.max_field_x} [m] \n"
            f"min_field_y - max_field_y: {self.min_field_y} - {self.max_field_y} [m] \n"
            f"min_space_x - max_space_x: {self.min_space_x} - {self.max_space_x} [m] \n"
            f"min_space_y - max_space_y: {self.min_space_y} - {self.max_space_y} [m] \n"
            f"min_space_z - max_space_z: {self.min_space_z} - {self.max_space_z} [m] \n"
            f"n_fp_cell@h: {n_cell_per_h}"
        )

    def render(self, observations: List[Dict], map_ground_truth: np.ndarray):

        x_min, x_max = int(-self.field_len / 2), int(self.field_len / 2)
        y_min, y_max = int(-self.field_len / 2), int(self.field_len / 2)

        params = {
            "ground_truth": {
                "title": "Terrain",
                "xticks": range(x_min, x_max + 1, x_max),
                "yticks": range(y_min, y_max + 1, y_max),
            },
            "agent_map_belief": {
                "title": "Map Belief",
                "xticks": range(x_min, x_max + 1, x_max),
                "yticks": range(y_min, y_max + 1, y_max),
            },
            "agent_allucinated_map_belief": {
                "title": "Map Belief",
                "xticks": range(x_min, x_max + 1, x_max),
                "yticks": range(y_min, y_max + 1, y_max),
            },
            "agent_map_ig": {
                "title": "Map Information Gain",
                "xticks": range(x_min, x_max + 1, x_max),
                "yticks": range(y_min, y_max + 1, y_max),
            },
            "agent_map_entropy": {
                "title": "Map Entropy",
                "xticks": range(x_min, x_max + 1, x_max),
                "yticks": range(y_min, y_max + 1, y_max),
            },
        }

        def ground_truth(ax, params):
            ax.set(**params)
            ax.set_yticks([])

            ax.imshow(
                map_ground_truth,
                cmap="Greens",
                extent=[x_min, x_max, y_min, y_max],
                interpolation="none",
                vmin=0,
                vmax=1,
            )

        def agent_map_belief(ax, params, agent):
            ax.set(**params)
            ax.set_yticks([])

            ax.imshow(
                map_beliefs[:, :, agent.id],
                cmap="Greens",
                extent=[x_min, x_max, y_min, y_max],
                interpolation="none",
                vmin=0,
                vmax=1,
            )

        def agent_allucinated_map_belief(ax, params, agent):
            ax.set(**params)
            ax.set_yticks([])

            if hasattr(agent, "allucinated_map_belief"):
                ax.imshow(
                    agent.allucinated_map_belief,
                    cmap="Greens",
                    extent=[x_min, x_max, y_min, y_max],
                    interpolation="none",
                    vmin=0,
                    vmax=1,
                )

        def agent_map_entropy(ax, params, agent):
            ax.set(**params)
            ax.set_yticks([])

            ax.imshow(
                H(map_beliefs[:, :, agent.id]),
                cmap="Reds",
                extent=[x_min, x_max, y_min, y_max],
                interpolation="none",
                vmin=0,
                vmax=1,
            )

        def agent(ax, agent, height, width, alpha):
            rect = patches.Rectangle(
                (
                    agent.state.position[X] - (height / 2),
                    agent.state.position[Y] - (width / 2),
                ),
                width,
                height,
                fill=True,
                alpha=alpha,
                facecolor=cmaps["Set1"](1),
                zorder=2,
            )
            ax.add_patch(rect)

        def communication(ax, observations):
            for a, o in zip(self.agents, observations):
                for neighbor_id in o["neighbors_ids"]:

                    verts = [
                        (a.state.position[X], a.state.position[Y]),
                        (
                            self.agents[neighbor_id].state.position[X],
                            self.agents[neighbor_id].state.position[Y],
                        ),
                    ]

                    codes = [
                        Path.MOVETO,
                        Path.LINETO,
                    ]

                    path = Path(verts, codes)
                    line = patches.PathPatch(path, lw=1, zorder=1)
                    ax.add_patch(line)

        def footprint(ax):
            for agent in self.agents:
                fp_ij = agent.camera.get_fp_vertices_ij(agent.state.position)[1]
                verts = [
                    fp_ij["ul"],
                    fp_ij["ur"],
                    fp_ij["br"],
                    fp_ij["bl"],
                    fp_ij["ul"],
                ]

                codes = [
                    Path.MOVETO,
                    Path.LINETO,
                    Path.LINETO,
                    Path.LINETO,
                    Path.CLOSEPOLY,
                ]

                path = Path(verts, codes)
                line = patches.PathPatch(
                    path, lw=1, zorder=1, facecolor="None", ls="--", edgecolor="red"
                )
                ax.add_patch(line)

        for label, ax in self.ax_dict.items():
            if label == "ground_truth":
                ground_truth(ax, params[label])
                for a in self.agents:
                    agent(ax, a, 3, 3, 1.0)
                footprint(ax)
            if "agent_map_belief" in label:
                agent_map_belief(ax, params[label[:-2]], self.agents[int(label[-1])])
                agent(ax, self.agents[int(label[-1])], 3, 3, 1.0)
            if "agent_allucinated_map_belief" in label:
                agent_allucinated_map_belief(
                    ax, params[label[:-2]], self.agents[int(label[-1])]
                )
                agent(ax, self.agents[int(label[-1])], 3, 3, 1.0)
            if "agent_map_ig" in label:
                agent(ax, self.agents[int(label[-1])], 3, 3, 1.0)
            if "agent_map_entropy" in label:
                agent_map_entropy(ax, params[label[:-2]], self.agents[int(label[-1])])
                agent(ax, self.agents[int(label[-1])], 3, 3, 1.0)

        plt.pause(0.3)

        # plt.savefig(f"imgs/asymmetric_GIF_FG/{step_index:02}", dpi=300)
        # plt.show(block=False)
        # plt.cla()
        #
        for label, ax in self.ax_dict.items():
            for patch in ax.patches:
                patch.remove()

    def reset_map_beliefs(self):
        global states, map_beliefs, map_belief_entropies, agg_map_belief, agg_map_belief_entropy, news_map_beliefs
        states = np.zeros((self.n_agents, 3), dtype=float)
        map_beliefs = (
            np.ones((self.n_cell, self.n_cell, self.n_agents), dtype=float) * 0.5
        )
        map_belief_entropies = np.ones(
            (self.n_cell, self.n_cell, self.n_agents), dtype=float
        )
        agg_map_belief = np.ones((self.n_cell, self.n_cell), dtype=float) * 0.5
        agg_map_belief_entropy = np.ones((self.n_cell, self.n_cell), dtype=float)
        news_map_beliefs = (
            np.ones(
                (self.n_agents, self.n_agents, self.n_cell, self.n_cell), dtype=float
            )
            * 0.5
        )

    def _generate_random_map(self, p_occupied=0.5):
        m = self.map_rng.random((self.n_cell, self.n_cell))
        m = np.where(m <= p_occupied, 1, 0).astype(int)

        return m

    def _gaussian_random_field(self, cluster_radius):
        """Generate 2D gaussian random field:
        https://andrewwalker.github.io/statefultransitions/post/gaussian-fields/"""

        def _fft_indices(n) -> List:
            a = list(range(0, int(np.floor(n / 2)) + 1))
            b = reversed(range(1, int(np.floor(n / 2))))
            b = [-i for i in b]
            return a + b

        def _pk2(kx, ky):
            if kx == 0 and ky == 0:
                return 0.0
            val = np.sqrt(np.sqrt(kx**2 + ky**2) ** (-cluster_radius))
            return val

        # memoization of amplitude (amplitude depends on n_cell, fft_indices, cluster_radius)
        # there is a one-to-one mapping between cluster_radius and amplitude
        # for each cluster_radius, what makes the maps different is just the noise
        if cluster_radius not in self.cluster_radius_to_amplitude:

            amplitude = np.zeros((self.n_cell, self.n_cell))
            fft_indices = _fft_indices(self.n_cell)

            for i, kx in enumerate(fft_indices):
                for j, ky in enumerate(fft_indices):
                    amplitude[i, j] = _pk2(kx, ky)

            self.cluster_radius_to_amplitude[cluster_radius] = np.copy(amplitude)

        noise = np.fft.fft2(self.map_rng.normal(size=(self.n_cell, self.n_cell)))
        random_field = np.fft.ifft2(
            noise * self.cluster_radius_to_amplitude[cluster_radius]
        ).real
        normalized_random_field = (random_field - np.min(random_field)) / (
            np.max(random_field) - np.min(random_field)
        )

        # Make field binary
        normalized_random_field[normalized_random_field >= 0.5] = 1
        normalized_random_field[normalized_random_field < 0.5] = 0

        return normalized_random_field.astype(np.uint8)

    def _generate_gaussian_map(self, cluster_radius):
        return self._gaussian_random_field(cluster_radius)


    def generate_map(self):

        if self.map_type == "gaussian":
            m = self._generate_gaussian_map(self.cluster_radius)
        return m

    def reset_agents_position(self, **kwargs):
        global states

        x = kwargs.get("x", None)
        y = kwargs.get("y", None)
        altitude = kwargs.get("altitude", None)

        if altitude is None:
            raise ValueError(f"Altitude cannot be {altitude}")

        planner_type = kwargs.get("planner_type")

        if x is None and y is None:
            if planner_type != "fixed_regions":
                for agent in self.agents:
                    x = self.agent_position_rng.choice(
                        np.arange(
                            self.min_field_x, self.max_field_x + 1, self.h_displacement
                        )
                    )
                    y = self.agent_position_rng.choice(
                        np.arange(
                            self.min_field_y, self.max_field_y + 1, self.h_displacement
                        )
                    )
                    agent.state.set_position(
                        np.array([x, y, (altitude + 1) * self.v_displacement])
                    )
                    states[agent.id, :] = agent.state.position
            else:
                for agent in self.agents:
                    x = self.agent_position_rng.choice(
                        np.arange(
                            self.regions_limits[agent.id][X][0],
                            self.regions_limits[agent.id][X][1] + 1,
                            self.h_displacement,
                        )
                    )
                    y = self.agent_position_rng.choice(
                        np.arange(
                            self.regions_limits[agent.id][Y][0],
                            self.regions_limits[agent.id][Y][1] + 1,
                            self.h_displacement,
                        )
                    )
                    agent.state.set_position(
                        np.array([x, y, (altitude + 1) * self.v_displacement])
                    )
                    states[agent.id, :] = agent.state.position

        # if x is not None and y is not None and type(x) != str and type(y) != str:
        #     for agent in self.agents:
        #         agent.state.set_position(np.array([x, y, (altitude + 1) * self.v_displacement]))
        #         states[agent.id, :] = agent.state.position
        #
        if x == "BL" and y == "BL":
            for agent in self.agents:
                x = self.regions_limits[agent.id][X][0]
                y = self.regions_limits[agent.id][Y][0]
                agent.state.set_position(
                    np.array([x, y, (altitude + 1) * self.v_displacement])
                )
                states[agent.id, :] = agent.state.position

    # def set_agents_position(self, positions):
    #     assert self.n_agents == len(positions)
    #
    #     for agent in self.agents:
    #         x, y, z = positions[agent.id][X], positions[agent.id][Y], positions[agent.id][Z]
    #         agent.state.set_position(np.array([x, y, (z + 1) * self.v_displacement]))
    #         states[agent.id, :] = agent.state.position

    def saturation(self):
        global map_beliefs
        return np.any(np.equal(map_beliefs, 0.0))


class Mapper:
    def __init__(self, n_cell: int, min_space_z: float, max_space_z: float, **kwargs):

        self.inference_type = kwargs.get("inference_type")
        self.weights_type = kwargs.get("weights_type")
        self.p_eq = kwargs.get("p_eq")

        self.news_inference_type = kwargs.get("news_inference_type", "Bypass")
        assert self.news_inference_type in [
            "OG_single",
            "OG_multi",
            "LBP_single",
            "LBP_multi",
            "Bypass",
        ], f"News inference type cannot be {self.news_inference_type}"

        if self.inference_type in ["LBP_cas", "LBP_cts"]:
            self._init_LBP_graph(n_cell)

        if (
            self.inference_type in ["LBP_cas_vectorized", "LBP_cts_vectorized"]
            or "LBP" in self.news_inference_type
        ):
            self._init_LBP_msgs(n_cell)

        self.centralized = kwargs.get("centralized", True)

        self.min_space_z = min_space_z
        self.max_space_z = max_space_z

    def update_map_beliefs(self, agents: List[Agent], observations: List[Dict]):
        if self.inference_type == "OG":
            self._update_belief_OG(observations, agents)
        # elif self.inference_type == "LBP_cas":
        #     self._update_belief_LBP_cas(observations, 1, agents)
        # elif self.inference_type == "LBP_cts":
        #     self._update_belief_LBP_cts(observations, 1, agents)
        # elif self.inference_type == "LBP_cas_vectorized":
        #     self._update_belief_LBP_cas_vectorized(observations, 1, agents)
        elif self.inference_type == "LBP_cts_vectorized":
            return self._update_belief_LBP_cts_vectorized_prova(observations, 1, agents)

    def update_news_and_fuse_map_beliefs(
        self, agents: List[Agent], observations: List[Dict]
    ):
        if self.news_inference_type == "OG_single":
            pass
            # self._update_news_belief_OG_and_fuse_single(agents, observations)
        # if self.news_inference_type == "OG_multi":
            # self._update_news_belief_OG_and_fuse_multi(agents, observations)
        if self.news_inference_type == "LBP_single":
            self._update_news_belief_LBP_and_fuse_single(agents, observations)
        # if self.news_inference_type == "LBP_multi":
        #     self._update_news_belief_LBP_and_fuse_multi(agents, observations)

    def reset_msgs(self):
        for k, v in self.graph.items():
            self.graph[k]["msgs"] = np.ones_like(v["msgs"])

    def reset_msgs_vectorized(self):
        self.msgs = np.ones_like(self.msgs) * 0.5
        self.msgs_buffer = np.ones_like(self.msgs) * 0.5

    def update_belief_OG(self, observations: List[Dict], agents: List[Agent]):

        global map_beliefs

        for o, a in zip(observations, agents):

            z, fp_vertices_ij, sigma0, sigma1 = (
                o["z"],
                o["fp_ij"],
                o["sigmas"][0],
                o["sigmas"][1],
            )

            likelihood_m_zero = np.where(z == 0, 1 - sigma0, sigma0)
            likelihood_m_one = np.where(z == 0, sigma1, 1 - sigma1)

            posterior_m_zero = likelihood_m_zero * (
                1.0
                - map_beliefs[
                    fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                    fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                    a.id,
                ]
            )
            posterior_m_one = (
                likelihood_m_one
                * map_beliefs[
                    fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                    fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                    a.id,
                ]
            )

            assert np.all(np.greater_equal(posterior_m_one, 0.0))

            # posterior_m_zero_norm = posterior_m_zero / (posterior_m_zero + posterior_m_one)
            posterior_m_one_norm = posterior_m_one / (
                posterior_m_zero + posterior_m_one
            )

            assert np.all(np.greater_equal(posterior_m_one_norm, 0.0)) and np.all(
                np.less_equal(posterior_m_one_norm, 1.0)
            )

            if self.centralized:
                map_beliefs[
                    fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                    fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                    :,
                ] = np.round(posterior_m_one_norm[:, :, np.newaxis], decimals=10)
            else:
                map_beliefs[
                    fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                    fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                    a.id,
                ] = np.round(posterior_m_one_norm, decimals=10)


    def _init_LBP_graph(self, n_cell: int):
        self.graph: Dict[Tuple, Dict] = dict()

        is_inside_map = lambda cell_ij: (
            cell_ij[I] >= 0
            and cell_ij[I] < n_cell
            and cell_ij[J] >= 0
            and cell_ij[J] < n_cell
        )

        for i in range(n_cell):
            for j in range(n_cell):
                self.graph[(i, j)] = {"adj": List, "msgs": np.array, "masks": np.array}

                neighbors_cell_ij = [
                    (i, j + 1),
                    (i, j - 1),
                    (i - 1, j),
                    (i + 1, j),
                ]  # r,l,u,d
                neighbors_admissible_cell_ij = [
                    cell_ij for cell_ij in neighbors_cell_ij if is_inside_map(cell_ij)
                ]
                adj = neighbors_admissible_cell_ij + [(-1, -1)]

                msgs = np.ones((2, len(neighbors_admissible_cell_ij) + 1))

                masks = np.ones(
                    (
                        len(neighbors_admissible_cell_ij) + 1,
                        2,
                        len(neighbors_admissible_cell_ij) + 1,
                    ),
                    dtype=int,
                )

                for mask_index in range(masks.shape[0]):
                    masks[mask_index, :, mask_index] = 0

                self.graph[(i, j)]["adj"] = adj
                self.graph[(i, j)]["msgs"] = msgs
                self.graph[(i, j)]["masks"] = masks

    
    def _init_LBP_msgs(self, n_cell):
        # depth_to_direction = 0123_4 -> URDL_fake
        self.msgs = np.ones((4 + 1, n_cell, n_cell), dtype=float) * 0.5
        self.msgs_buffer = np.ones_like(self.msgs) * 0.5

        # self.pairwise_potential = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)

        # (channelS, row_slice, col_slice) to product & marginalize
        # (row_slice, col_slice) to read
        # (channel, row_slice, col_slice) to write
        self.direction_to_slicing_data = {
            "up": {
                "product_slice": lambda fp_ij: (
                    (1, 2, 3, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(
                        1 if fp_ij["ul"][I] == 0 else 0, fp_ij["bl"][I] - fp_ij["ul"][I]
                    ),
                    slice(0, fp_ij["ur"][J] - fp_ij["ul"][J]),
                ),
                "write_slice": lambda fp_ij: (
                    2,
                    slice(max(0, fp_ij["ul"][I] - 1), min(n_cell, fp_ij["bl"][I] - 1)),
                    slice(max(0, fp_ij["ul"][J]), min(n_cell, fp_ij["br"][J])),
                ),
            },
            "right": {
                "product_slice": lambda fp_ij: (
                    (0, 2, 3, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(0, fp_ij["bl"][I] - fp_ij["ul"][I]),
                    slice(
                        0,
                        (
                            fp_ij["ur"][J] - fp_ij["ul"][J] - 1
                            if fp_ij["ur"][J] == n_cell
                            else fp_ij["ur"][J] - fp_ij["ul"][J]
                        ),
                    ),
                ),
                "write_slice": lambda fp_ij: (
                    3,
                    slice(max(0, fp_ij["ul"][I]), min(n_cell, fp_ij["bl"][I])),
                    slice(max(0, fp_ij["ul"][J] + 1), min(n_cell, fp_ij["br"][J] + 1)),
                ),
            },
            "down": {
                "product_slice": lambda fp_ij: (
                    (0, 1, 3, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(
                        0,
                        (
                            fp_ij["bl"][I] - fp_ij["ul"][I] - 1
                            if fp_ij["bl"][I] == n_cell
                            else fp_ij["bl"][I] - fp_ij["ul"][I]
                        ),
                    ),
                    slice(0, fp_ij["ur"][J] - fp_ij["ul"][J]),
                ),
                "write_slice": lambda fp_ij: (
                    0,
                    slice(max(0, fp_ij["ul"][I] + 1), min(n_cell, fp_ij["bl"][I] + 1)),
                    slice(max(0, fp_ij["ul"][J]), min(n_cell, fp_ij["br"][J])),
                ),
            },
            "left": {
                "product_slice": lambda fp_ij: (
                    (0, 1, 2, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(0, fp_ij["bl"][I] - fp_ij["ul"][I]),
                    slice(
                        1 if fp_ij["ul"][J] == 0 else 0, fp_ij["ur"][J] - fp_ij["ul"][J]
                    ),
                ),
                "write_slice": lambda fp_ij: (
                    1,
                    slice(max(0, fp_ij["ul"][I]), min(n_cell, fp_ij["bl"][I])),
                    slice(max(0, fp_ij["ul"][J] - 1), min(n_cell, fp_ij["br"][J] - 1)),
                ),
            },
        }
    def get_map_beliefs(self):
        global map_beliefs
        return map_beliefs

    def set_pairwise_potential_t(self, step_index):
        a = 0.7 - 0.2 * np.exp(-0.03 * step_index)
        b = 0.3 + 0.2 * np.exp(-0.03 * step_index)

        a = 0.5 - 0.2 * np.exp(-0.03 * step_index)
        b = 0.5 + 0.2 * np.exp(-0.03 * step_index)

        self.pairwise_potential = np.array([[b, a], [a, b]], dtype=float)

    def set_pairwise_potential_h(self, agents):

        if self.weights_type == "adaptive":
            for a in agents:
                p_eq = 0.5 + 0.1 * ((a.state.position[Z] - 5.4) / (32.4 - 5.4))
                p_neq = 0.5 - 0.1 * ((a.state.position[Z] - 5.4) / (32.4 - 5.4))

                if a.state.position[Z] <= 21.65:
                    p_eq = 0.6 - 0.1 * ((a.state.position[Z] - 5.4) / (21.65 - 5.4))
                    p_neq = 0.4 + 0.1 * ((a.state.position[Z] - 5.4) / (21.65 - 5.4))

                else:
                    p_eq = 0.5
                    p_neq = 0.5

                a.pairwise_potential = np.array(
                    [[p_eq, p_neq], [p_neq, p_eq]], dtype=float
                )
        if self.weights_type == "equal":
            for a in agents:
                a.pairwise_potential = np.array(
                    [
                        [self.p_eq, 1 - self.p_eq],
                        [
                            1 - self.p_eq,
                            self.p_eq,
                        ],
                    ],
                    dtype=float,
                )
            # print(f"pairwise is {a.pairwise_potential}")

    def set_pairwise_potential_z(self, agents, observations):
        win_size = 3

        for a, o in zip(agents, observations):

            observation = o["z"]
            # print(observation.shape)

            if (
                observation.shape[0] % win_size != 0
                or observation.shape[1] % win_size != 0
            ):

                pad_rows, pad_cols = 0, 0

                if observation.shape[0] % win_size != 0:
                    current_shape = observation.shape[0]
                    while current_shape % win_size != 0:
                        current_shape += 1
                    pad_rows = current_shape - observation.shape[0]

                if observation.shape[1] % win_size != 0:
                    current_shape = observation.shape[1]
                    while current_shape % win_size != 0:
                        current_shape += 1
                    pad_cols = current_shape - observation.shape[1]

                observation = np.pad(
                    observation, ((0, pad_rows), (0, pad_cols)), mode="edge"
                )

                # print(f"reshape to: {observation.shape}")

            assert observation.shape[0] % win_size == 0
            assert observation.shape[1] % win_size == 0

            v = (
                observation.reshape(
                    (
                        max(
                            observation.shape[0] // win_size,
                            observation.shape[1] // win_size,
                        ),
                        win_size,
                        -1,
                        win_size,
                    )
                )
                .swapaxes(1, 2)
                .reshape(
                    (observation.shape[0] // win_size)
                    * (observation.shape[1] // win_size),
                    win_size,
                    win_size,
                )
            )

            m_center = v[:, 1, 1]
            m_neighbors = np.zeros((v.shape[0], 4), dtype=int)
            m_neighbors[:, 0] = v[:, 0, 1]
            m_neighbors[:, 1] = v[:, 1, 2]
            m_neighbors[:, 2] = v[:, 2, 1]
            m_neighbors[:, 3] = v[:, 1, 0]

            counts_one = np.count_nonzero(m_neighbors, axis=1)
            stacked = np.hstack((m_center.reshape(-1, 1), counts_one.reshape(-1, 1)))

            # mutual information

            # pearson
            pearson = np.corrcoef(stacked[:, 0], stacked[:, 1])[0, 1]
            sigmoid = 1 / (1 + np.exp(-pearson))

            W = np.array([[sigmoid, 1 - sigmoid], [1 - sigmoid, sigmoid]])

            # ad-hoc conditional distro
            # u = np.unique(stacked, axis=0, return_counts=True)
            #
            # #   0 1 2 3 4
            # # 0
            # # 1
            # # add-one smoothing
            # p_joint = np.ones((2, 5))
            # for i, (r, c) in enumerate(u[0]):
            #     p_joint[r, c] += u[1][i]
            #
            # p_joint /= np.sum(p_joint)
            # p_conditional = p_joint / np.sum(p_joint, axis=0)
            #
            # # W = np.array([[p_conditional[0, 0], np.max(p_conditional[:, 2])],
            # #               [np.max(p_conditional[:, 2]), p_conditional[1, 4]]])
            #
            # W = np.array([[np.mean(p_conditional[0, 1:3]), 1-np.mean(p_conditional[0, 1:3])],
            #               [1-np.mean(p_conditional[1, 2:4]), np.mean(p_conditional[1, 2:4])]])
            W = np.round(W, decimals=10)
            a.pairwise_potential = W


    def _update_belief_LBP_cts_vectorized_prova(
        self,
        observations: List[Dict],
        n_iteration: int,
        agents: List[Agent],
    ):

        # update the belief in OG fashion
        # and then use it as a local_potential(s)
        # after we're going to update the belief once more
        # self._update_belief_OG(observations, agents)

        global map_beliefs

        for o, agent in zip(observations, agents):

            # reset msgs and msgs_buffer
            self.msgs = np.ones_like(self.msgs) * 0.5
            self.msgs_buffer = np.ones_like(self.msgs) * 0.5
            self.msgs[4, :, :] = map_beliefs[
                :, :, agent.id
            ]  # set msgs last channel with current map belief
            agent.pairwise_potential=np.round(agent.pairwise_potential, decimals=10)
            fp_vertices_ij = o["fp_ij"]
            for _ in range(n_iteration):
                for direction, data in self.direction_to_slicing_data.items():
                    product_slice = data["product_slice"](fp_vertices_ij)
                    read_slice = data["read_slice"](fp_vertices_ij)
                    write_slice = data["write_slice"](fp_vertices_ij)

                    # elementwise multiplication of msgs
                    mul_0 = np.prod(1 - self.msgs[product_slice], axis=0)
                    mul_1 = np.prod(self.msgs[product_slice], axis=0)

                    # matrix-vector multiplication (factor-msg)
                    msg_0 = (
                        agent.pairwise_potential[0, 0] * mul_0
                        + agent.pairwise_potential[0, 1] * mul_1
                    )
                    msg_1 = (
                        agent.pairwise_potential[1, 0] * mul_0
                        + agent.pairwise_potential[1, 1] * mul_1
                    )

                    # normalize the first coordinate of the msg
                    norm_msg_1 = msg_1 / (msg_0 + msg_1)

                    # buffering
                    self.msgs_buffer[write_slice] = norm_msg_1[read_slice]

                # copy the first 4 channels only
                # the 5th one is the map belief
                self.msgs[:4, :, :] = self.msgs_buffer[:4, :, :]
                # return self.msgs[:4,:,:]

            bel_0 = np.prod(
                1 - self.msgs[:, product_slice[1], product_slice[2]], axis=0
            )
            bel_1 = np.prod(self.msgs[:, product_slice[1], product_slice[2]], axis=0)

            # norm_bel_0 = bel_0 / (bel_0 + bel_1)
            map_beliefs[product_slice[1], product_slice[2], agent.id] = bel_1 / (
                bel_0 + bel_1
            )

            assert np.all(
                np.greater_equal(
                    map_beliefs[product_slice[1], product_slice[2], agent.id], 0.0
                )
            ) and np.all(
                np.less_equal(
                    map_beliefs[product_slice[1], product_slice[2], agent.id], 1.0
                )
            )

    def get_msgs(self):
        return self.msgs[:4,:,:]
    def _update_news_belief_LBP_and_fuse_single(
        self, agents: List[Agent], observations: List[Dict]
    ):

        global news_map_beliefs, map_beliefs

        for agent_id in range(len(agents)):
            z, fp_vertices_ij = (
                observations[agent_id]["z"],
                observations[agent_id]["fp_ij"],
            )
            sigma0, sigma1 = (
                observations[agent_id]["sigmas"][0],
                observations[agent_id]["sigmas"][1],
            )

            likelihood_m_zero = np.where(z == 0, 1 - sigma0, sigma0)
            likelihood_m_one = np.where(z == 0, sigma1, 1 - sigma1)

            posterior_m_zero = likelihood_m_zero * (
                1.0
                - news_map_beliefs[
                    agent_id,
                    agent_id,
                    fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                    fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                ]
            )
            posterior_m_one = (
                likelihood_m_one
                * news_map_beliefs[
                    agent_id,
                    agent_id,
                    fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                    fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                ]
            )

            assert np.all(np.greater_equal(posterior_m_one, 0.0))

            posterior_m_one_norm = posterior_m_one / (
                posterior_m_zero + posterior_m_one
            )

            assert np.all(np.greater_equal(posterior_m_one_norm, 0.0)) and np.all(
                np.less_equal(posterior_m_one_norm, 1.0)
            )

            news_map_beliefs[
                agent_id,
                agent_id,
                fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
            ] = posterior_m_one_norm

        for agent_id in range(len(agents)):

            # reset msgs and msgs_buffer
            self.msgs = np.ones_like(self.msgs) * 0.5
            self.msgs_buffer = np.ones_like(self.msgs) * 0.5
            self.msgs[4, :, :] = news_map_beliefs[
                agent_id, agent_id, :, :
            ]  # set msgs last channel with current map belief

            fp_vertices_ij = observations[agent_id]["fp_ij"]

            # just 1 iteration
            for direction, data in self.direction_to_slicing_data.items():
                product_slice = data["product_slice"](fp_vertices_ij)
                read_slice = data["read_slice"](fp_vertices_ij)
                write_slice = data["write_slice"](fp_vertices_ij)

                # elementwise multiplication of msgs
                mul_0 = np.prod(1 - self.msgs[product_slice], axis=0)
                mul_1 = np.prod(self.msgs[product_slice], axis=0)
                agents[agent_id].pairwise_potential=np.round(agents[agent_id].pairwise_potential, decimals=10)

                # matrix-vector multiplication (factor-msg)
                msg_0 = (
                    agents[agent_id].pairwise_potential[0, 0] * mul_0
                    + agents[agent_id].pairwise_potential[0, 1] * mul_1
                )
                msg_1 = (
                    agents[agent_id].pairwise_potential[1, 0] * mul_0
                    + agents[agent_id].pairwise_potential[1, 1] * mul_1
                )

                # normalize the first coordinate of the msg
                norm_msg_1 = msg_1 / (msg_0 + msg_1)

                # buffering
                # self.msgs_buffer[write_slice] =  np.round(norm_msg_1[read_slice], decimals=10)

                self.msgs_buffer[write_slice] = norm_msg_1[read_slice]

            # copy the first 4 channels only
            # the 5th one is the map belief
            self.msgs[:4, :, :] = self.msgs_buffer[:4, :, :]

            bel_0 = np.prod(
                1 - self.msgs[:, product_slice[1], product_slice[2]], axis=0
            )
            bel_1 = np.prod(self.msgs[:, product_slice[1], product_slice[2]], axis=0)

            # norm_bel_0 = bel_0 / (bel_0 + bel_1)
            news_map_beliefs[agent_id, agent_id, product_slice[1], product_slice[2]] = (
                bel_1 / (bel_0 + bel_1)
            )

            assert np.all(
                np.greater_equal(
                    news_map_beliefs[
                        agent_id, agent_id, product_slice[1], product_slice[2]
                    ],
                    0.0,
                )
            ) and np.all(
                np.less_equal(
                    news_map_beliefs[
                        agent_id, agent_id, product_slice[1], product_slice[2]
                    ],
                    1.0,
                )
            )

        for agent_id in range(len(agents)):

            neighbors_ids = []
            if len(observations) != 0:
                neighbors_ids = observations[agent_id]["neighbors_ids"]

            for neighbor_id in neighbors_ids:
                mul = (
                    news_map_beliefs[agent_id, agent_id, :, :]
                    * map_beliefs[:, :, neighbor_id]
                )
                map_beliefs[:, :, neighbor_id] = mul / (
                    mul
                    + (1.0 - news_map_beliefs[agent_id, agent_id, :, :])
                    * (1.0 - map_beliefs[:, :, neighbor_id])
                )

                assert np.all(
                    np.greater_equal(map_beliefs[:, :, neighbor_id], 0.0)
                ) and np.all(np.less_equal(map_beliefs[:, :, neighbor_id], 1.0))

            if len(neighbors_ids) != 0:
                news_map_beliefs[agent_id, agent_id, :, :] = 0.5



class Planner:
    def __init__(
        self,
        action_to_direction: Dict,
        altitude_to_size: Dict,
        position_graph: Dict,
        position_data: Dict,
        regions_limits: List[float],
        optimal_altitude: float,
        **kwargs,
    ):

        self.action_to_direction = action_to_direction
        self.altitude_to_size = altitude_to_size
        # self.v_displacement = action_to_direction["up"][Z]
        self.regions_limits = regions_limits
        self.optimal_altitude = optimal_altitude

        self.position_graph = position_graph
        self.position_data = position_data
        self.action_to_id = {
            k: index for index, (k, v) in enumerate(self.action_to_direction.items())
        }
        self.id_to_action = {v: k for k, v in self.action_to_id.items()}

        self.planner_type = kwargs.get("planner_type")
        assert self.planner_type in [
            "map_loop_closing",
            "selfish",
            "IoU",
            "mine_IoU_sync",
            "mine_IoU_async",
            "mine_IoU_async_no_pred",
            "fixed_regions",
            "sweep",
            "random",
            "weighted_async_no_pred",
        ], f"Planner type cannot be {self.planner_type}"

        if self.planner_type == "targeted":
            self.interest_th = kwargs.get("interest_th")

        self.centralized = kwargs.get("centralized", True)

        self.action_to_direction_vec = np.zeros((len(self.action_to_direction), 3))
        for k, v in self.id_to_action.items():
            self.action_to_direction_vec[k, :] = self.action_to_direction[v]

        # planner agent decision order (for async routines)
        self.agent_decision_order_rng = np.random.default_rng(17)

        # _sweep helpers direction memory
        self.sweep_left_right = ["left"] * kwargs.get("n_agents")
        self.last_action = ["up"] * kwargs.get("n_agents")
        self.n_visited_positions = [0] * kwargs.get("n_agents")

        # _non_targeted_mini_weighted_async_no_prediction helper z_buffer and n_buffer
        self.z_buffer = np.ones((400, 400), dtype=float)
        self.n_buffer = np.zeros((400, 400), dtype=int)

    def get_actions(self, agents: List[Agent], observations: List[Dict]):

        if self.planner_type == "selfish":
            return self._non_targeted_mini(agents)
        elif self.planner_type == "IoU":
            return self._non_targeted_mini_IoU(agents, observations)
        elif self.planner_type == "mine_IoU_sync":
            return self._non_targeted_mini_mine_IoU_sync(agents, observations)
        elif self.planner_type == "mine_IoU_async":
            return self._non_targeted_mini_mine_IoU_async(agents, observations)
        elif self.planner_type == "mine_IoU_async_no_pred":
            return self._non_targeted_mini_mine_IoU_async_no_prediction_true_neighbors(
                agents, observations
            )
        elif self.planner_type == "weighted_async_no_pred":
            return self._non_targeted_mini_weighted_async_no_prediction_rnd(
                agents, observations
            )
        elif self.planner_type == "fixed_regions":
            return self._fixed_regions(agents)
        elif self.planner_type == "sweep":
            return self._sweep(agents)
        elif self.planner_type == "random":
            return self._random(agents)

    def _sweep(self, agents: List[Agent]):
        actions, data = [], {}

        for agent in agents:
            if (
                self.n_visited_positions[agent.id] % self.regions_limits[agent.id][2]
                == 0
            ):
                if self.sweep_left_right[agent.id] == "right":
                    self.sweep_left_right[agent.id] = "left"
                else:
                    self.sweep_left_right[agent.id] = "right"

        for agent in agents:
            if (
                self.last_action[agent.id] == "up"
                and agent.state.position[Z] != self.optimal_altitude
            ):
                actions.append("up")

            if (
                self.last_action[agent.id] == "up"
                and agent.state.position[Z] == self.optimal_altitude
            ):
                actions.append("front")

            if (
                self.last_action[agent.id] == "front"
                and agent.state.position[Y] != self.regions_limits[agent.id][Y][1]
            ):
                actions.append("front")

            if (
                self.last_action[agent.id] == "front"
                and agent.state.position[Y] == self.regions_limits[agent.id][Y][1]
            ):
                actions.append(self.sweep_left_right[agent.id])

            if (
                self.last_action[agent.id] == "right"
                or self.last_action[agent.id] == "left"
            ) and agent.state.position[Y] == self.regions_limits[agent.id][Y][1]:
                actions.append("back")

            if (
                self.last_action[agent.id] == "right"
                or self.last_action[agent.id] == "left"
            ) and agent.state.position[Y] == self.regions_limits[agent.id][Y][0]:
                actions.append("front")

            if (
                self.last_action[agent.id] == "back"
                and agent.state.position[Y] != self.regions_limits[agent.id][Y][0]
            ):
                actions.append("back")

            if (
                self.last_action[agent.id] == "back"
                and agent.state.position[Y] == self.regions_limits[agent.id][Y][0]
            ):
                actions.append(self.sweep_left_right[agent.id])

        self.last_action = list(actions)

        for agent in agents:
            self.n_visited_positions[agent.id] += 1

        return actions, data

    def _random(self, agents: List[Agent]):
        actions, data = [], {}

        for agent in agents:
            future_action_position = self.position_graph[
                (
                    agent.state.position[X],
                    agent.state.position[Y],
                    int(agent.state.position[Z]),
                )
            ]

            random_action = agent.rng.choice(list(future_action_position.keys()))

            actions.append(random_action)

        return actions, data

    def _fixed_regions(self, agents: List[Agent]):

        global map_beliefs, map_belief_entropies
        actions, data = [], []

        for agent in agents:

            admissible_action_to_IG = {}
            admissible_action_to_fp_ij = {}

            future_action_position = self.position_graph[
                (
                    agent.state.position[X],
                    agent.state.position[Y],
                    int(agent.state.position[Z]),
                )
            ]

            for action, position in future_action_position.items():
                if (
                    position[X] >= self.regions_limits[agent.id][X][0]
                    and position[X] <= self.regions_limits[agent.id][X][1]
                    and position[Y] >= self.regions_limits[agent.id][Y][0]
                    and position[Y] <= self.regions_limits[agent.id][Y][1]
                ):

                    fp_vertices_ij, _ = agent.camera.get_fp_vertices_ij(position)

                    fp_map_belief = map_beliefs[
                        fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                        fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                        agent.id,
                    ]

                    # H(M_fp) - entropy of the prior = sum of the entropy of each cell in the footprint
                    current_H = map_belief_entropies[
                        fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                        fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                        agent.id,
                    ]

                    # H(M_fp|Z) - conditional entropy of the posterior = sum of the averaged entropy of each cell in the footprint
                    sigma0, sigma1 = agent.camera.get_sigmas(position)
                    current_cH = cH(
                        fp_map_belief, np.round(sigma0, 7), np.round(sigma1, 7)
                    )

                    # assert non-negativity
                    diff = current_H - current_cH
                    diff = np.where(np.isclose(current_H, current_cH), 0.0, diff)
                    assert np.all(np.greater_equal(diff, 0.0)), (
                        f"{position}"
                        f"{action}"
                        f"{current_H[np.less(diff, 0.0)]}"
                        f"{fp_map_belief[np.less(diff, 0.0)]}"
                    )

                    admissible_action_to_IG[action] = [
                        # np.round(np.sum(diff) / cost, 8),
                        np.sum(diff)
                    ]

                    admissible_action_to_fp_ij[action] = fp_vertices_ij

            best_admissible_actions = []
            sorted_action_to_IG = sorted(
                admissible_action_to_IG.items(), key=lambda x: x[1][0], reverse=True
            )

            best_action, best_IG = sorted_action_to_IG.pop(0)

            best_admissible_actions.append(best_action)

            for action, IG in sorted_action_to_IG:
                if IG[0] == best_IG[0]:
                    best_admissible_actions.append(action)

            actions.append(agent.rng.choice(best_admissible_actions))

            data.append(
                {
                    "admissible_action_to_IG": admissible_action_to_IG,
                    "admissible_action_to_fp_ij": admissible_action_to_fp_ij,
                    "length_actions": len(best_admissible_actions),
                }
            )

        return actions, data

    def _non_targeted_mini(self, agents: List[Agent]):

        global map_beliefs, map_belief_entropies
        actions, data = [], []

        for agent in agents:

            admissible_action_to_IG = {}
            admissible_action_to_fp_ij = {}

            future_action_position = self.position_graph[
                (
                    agent.state.position[X],
                    agent.state.position[Y],
                    int(agent.state.position[Z]),
                )
            ]

            for action, position in future_action_position.items():

                fp_vertices_ij, _ = agent.camera.get_fp_vertices_ij(position)

                fp_map_belief = map_beliefs[
                    fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                    fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                    agent.id,
                ]

                # H(M_fp) - entropy of the prior = sum of the entropy of each cell in the footprint
                current_H = map_belief_entropies[
                    fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                    fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                    agent.id,
                ]

                # H(M_fp|Z) - conditional entropy of the posterior = sum of the averaged entropy of each cell in the footprint
                sigma0, sigma1 = agent.camera.get_sigmas(position)
                current_cH = cH(fp_map_belief, np.round(sigma0, 7), np.round(sigma1, 7))

                # assert non-negativity
                diff = current_H - current_cH
                diff = np.where(np.isclose(current_H, current_cH), 0.0, diff)
                assert np.all(np.greater_equal(diff, 0.0)), (
                    f"{position}"
                    f"{action}"
                    f"{current_H[np.less(diff, 0.0)]}"
                    f"{fp_map_belief[np.less(diff, 0.0)]}"
                )

                admissible_action_to_IG[action] = [
                    # np.round(np.sum(diff) / cost, 8),
                    np.sum(diff)
                ]

                admissible_action_to_fp_ij[action] = fp_vertices_ij

            best_admissible_actions = []
            sorted_action_to_IG = sorted(
                admissible_action_to_IG.items(), key=lambda x: x[1][0], reverse=True
            )

            best_action, best_IG = sorted_action_to_IG.pop(0)

            best_admissible_actions.append(best_action)

            for action, IG in sorted_action_to_IG:
                if IG[0] == best_IG[0]:
                    best_admissible_actions.append(action)

            actions.append(agent.rng.choice(best_admissible_actions))

            data.append(
                {
                    "admissible_action_to_IG": admissible_action_to_IG,
                    "admissible_action_to_fp_ij": admissible_action_to_fp_ij,
                    "length_actions": len(best_admissible_actions),
                }
            )

        return actions, data

    def _non_targeted_mini_IoU(self, agents: List[Agent], observations: List[Dict]):
        selfish_actions, data = self._non_targeted_mini(agents)

        synchronized_actions = []

        for agent in agents:
            # print("PRIMA")
            # print("----------", agent.id)
            # print(selfish_actions[agent.id])
            # pprint(data[agent.id]["admissible_action_to_IG"])

            neighbors_ids = []
            if len(observations) != 0:
                neighbors_ids = observations[agent.id]["neighbors_ids"]

            # print(neighbors_ids)

            for neighbor_id in neighbors_ids:
                neighbor = agents[neighbor_id]

                admissible_action_to_IG = {}
                admissible_action_to_fp_ij = {}

                future_action_position = self.position_graph[
                    (
                        neighbor.state.position[X],
                        neighbor.state.position[Y],
                        int(neighbor.state.position[Z]),
                    )
                ]

                for action, position in future_action_position.items():

                    fp_vertices_ij, _ = neighbor.camera.get_fp_vertices_ij(position)

                    # H(M_fp) - entropy of the prior = sum of the entropy of each cell in the footprint
                    current_H = map_belief_entropies[
                        fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                        fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                        agent.id,
                    ]

                    # H(M_fp|Z) - conditional entropy of the posterior = sum of the averaged entropy of each cell in the footprint
                    sigma0, sigma1 = neighbor.camera.get_sigmas(position)
                    fp_map_belief = map_beliefs[
                        fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                        fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                        agent.id,
                    ]
                    current_cH = cH(
                        fp_map_belief, np.round(sigma0, 7), np.round(sigma1, 7)
                    )

                    # assert non-negativity
                    diff = current_H - current_cH
                    diff = np.where(np.isclose(current_H, current_cH), 0.0, diff)
                    assert np.all(np.greater_equal(diff, 0.0)), (
                        f"{position}"
                        f"{action}"
                        f"{current_H[np.less(diff, 0.0)]}"
                        f"{fp_map_belief[np.less(diff, 0.0)]}"
                    )

                    # alpha = 1.0
                    # for index, fp_neighbor_ij in enumerate(selfish_action_fp_ij):
                    #     if agent.id != index:
                    #         alpha *= (1.0 - IoU(fp_vertices_ij, fp_neighbor_ij))

                    admissible_action_to_IG[action] = [np.sum(diff)]
                    admissible_action_to_fp_ij[action] = fp_vertices_ij

                neighbor_action = self.__argmin_action(
                    neighbor, admissible_action_to_IG
                )

                # print(neighbor.id, neighbor_action)

                neighbor_fp_ij = admissible_action_to_fp_ij[neighbor_action]
                for action, agent_fp_ij in data[agent.id][
                    "admissible_action_to_fp_ij"
                ].items():
                    # print(action, (1.0 - IoU(neighbor_fp_ij, agent_fp_ij)))
                    data[agent.id]["admissible_action_to_IG"][action][0] *= 1.0 - IoU(
                        neighbor_fp_ij, agent_fp_ij
                    )

            agent_action = self.__argmin_action(
                agent, data[agent.id]["admissible_action_to_IG"]
            )

            # print("DOPO")
            # print(agent.id)
            # print(agent_action)
            # pprint(data[agent.id]["admissible_action_to_IG"])

            synchronized_actions.append(agent_action)

        return synchronized_actions, data

    def _non_targeted_mini_mine_IoU_sync(
        self, agents: List[Agent], observations: List[Dict]
    ):
        selfish_actions, data = self._non_targeted_mini(agents)

        synchronized_actions = []

        for agent in agents:

            # print("PRIMA")
            # print("----------", agent.id)
            # print(selfish_actions[agent.id])
            # pprint(data[agent.id]["admissible_action_to_IG"])

            neighbors_ids = []
            if len(observations) != 0:
                neighbors_ids = observations[agent.id]["neighbors_ids"]

            for neighbor_id in neighbors_ids:
                neighbor_action = selfish_actions[neighbor_id]
                neighbor_fp_ij = data[neighbor_id]["admissible_action_to_fp_ij"][
                    neighbor_action
                ]
                for action, agent_fp_ij in data[agent.id][
                    "admissible_action_to_fp_ij"
                ].items():
                    # print(action, (1.0 - IoU(neighbor_fp_ij, agent_fp_ij)))
                    data[agent.id]["admissible_action_to_IG"][action][0] *= 1.0 - IoU(
                        neighbor_fp_ij, agent_fp_ij
                    )

            agent_action = self.__argmin_action(
                agent, data[agent.id]["admissible_action_to_IG"]
            )

            # print("DOPO")
            # print(agent.id)
            # print(agent_action)
            # pprint(data[agent.id]["admissible_action_to_IG"])

            synchronized_actions.append(agent_action)

        return synchronized_actions, data

    # def _non_targeted_mini_mine_IoU_async(self, agents: List[Agent], observations: List[Dict]):
    #     selfish_actions, data = self._non_targeted_mini(agents)
    #
    #     synchronized_async_actions = list(selfish_actions)
    #     decision_order = self.agent_decision_order_rng.permutation(len(agents))
    #
    #     for agent in agents:
    #
    #         # print("PRIMA")
    #         # print("----------", agent.id)
    #         # print(selfish_actions[agent.id])
    #         # pprint(data[agent.id]["admissible_action_to_IG"])
    #
    #         neighbors_ids = []
    #         if len(observations) != 0:
    #             neighbors_ids = observations[agent.id]["neighbors_ids"]
    #
    #         for neighbor_id in neighbors_ids:
    #             neighbor_action = synchronized_async_actions[neighbor_id]
    #             neighbor_fp_ij = data[neighbor_id]["admissible_action_to_fp_ij"][neighbor_action]
    #             for action, agent_fp_ij in data[agent.id]["admissible_action_to_fp_ij"].items():
    #                 # print(action, (1.0 - IoU(neighbor_fp_ij, agent_fp_ij)))
    #                 data[agent.id]["admissible_action_to_IG"][action][0] *= (1.0 - IoU(neighbor_fp_ij, agent_fp_ij))
    #
    #         agent_action = self.__argmin_action(agent, data[agent.id]["admissible_action_to_IG"])
    #
    #         synchronized_async_actions[agent.id] = agent_action
    #
    #         # print("DOPO")
    #         # print(agent.id)
    #         # print(agent_action)
    #         # pprint(data[agent.id]["admissible_action_to_IG"])
    #
    #     return synchronized_async_actions, data

    def _non_targeted_mini_mine_IoU_async(
        self, agents: List[Agent], observations: List[Dict]
    ):
        selfish_actions, data = self._non_targeted_mini(agents)

        synchronized_async_actions = list(selfish_actions)
        decision_order = self.agent_decision_order_rng.permutation(len(agents))

        for id in decision_order:

            # print("PRIMA")
            # print("----------", agent.id)
            # print(selfish_actions[agent.id])
            # pprint(data[agent.id]["admissible_action_to_IG"])

            neighbors_ids = []
            if len(observations) != 0:
                neighbors_ids = observations[id]["neighbors_ids"]

            for neighbor_id in neighbors_ids:
                neighbor_action = synchronized_async_actions[neighbor_id]
                neighbor_fp_ij = data[neighbor_id]["admissible_action_to_fp_ij"][
                    neighbor_action
                ]
                for action, agent_fp_ij in data[id][
                    "admissible_action_to_fp_ij"
                ].items():
                    # print(action, (1.0 - IoU(neighbor_fp_ij, agent_fp_ij)))
                    data[id]["admissible_action_to_IG"][action][0] *= 1.0 - IoU(
                        neighbor_fp_ij, agent_fp_ij
                    )

            agent_action = self.__argmin_action(
                agents[id], data[id]["admissible_action_to_IG"]
            )

            synchronized_async_actions[id] = agent_action

            # print("DOPO")
            # print(agent.id)
            # print(agent_action)
            # pprint(data[agent.id]["admissible_action_to_IG"])

        return synchronized_async_actions, data

    def _non_targeted_mini_mine_IoU_async_no_prediction(
        self, agents: List[Agent], observations: List[Dict]
    ):
        _, data = self._non_targeted_mini(agents)

        synchronized_async_actions_no_predictions = ["hover"] * len(agents)
        decision_order = self.agent_decision_order_rng.permutation(len(agents))

        for id in decision_order:

            # print("PRIMA")
            # print("----------", agent.id)
            # print(selfish_actions[agent.id])
            # pprint(data[agent.id]["admissible_action_to_IG"])

            neighbors_ids = []
            if len(observations) != 0:
                neighbors_ids = observations[id]["neighbors_ids"]

            for neighbor_id in neighbors_ids:
                neighbor_action = synchronized_async_actions_no_predictions[neighbor_id]
                neighbor_fp_ij = data[neighbor_id]["admissible_action_to_fp_ij"][
                    neighbor_action
                ]
                for action, agent_fp_ij in data[id][
                    "admissible_action_to_fp_ij"
                ].items():
                    # print(action, (1.0 - IoU(neighbor_fp_ij, agent_fp_ij)))
                    data[id]["admissible_action_to_IG"][action][0] *= 1.0 - IoU(
                        neighbor_fp_ij, agent_fp_ij
                    )

            agent_action = self.__argmin_action(
                agents[id], data[id]["admissible_action_to_IG"]
            )

            synchronized_async_actions_no_predictions[id] = agent_action

            # print("DOPO")
            # print(agent.id)
            # print(agent_action)
            # pprint(data[agent.id]["admissible_action_to_IG"])

        return synchronized_async_actions_no_predictions, data

    def _non_targeted_mini_mine_IoU_async_no_prediction_true_neighbors(
        self, agents: List[Agent], observations: List[Dict]
    ):

        global states

        # print(states)

        predicted_actions, data = self._non_targeted_mini(agents)

        if len(agents) > 1:

            predicted_states = np.copy(states)
            predicted_actions = ["hover"] * len(agents)
            decision_order = self.agent_decision_order_rng.permutation(len(agents))

            for id in decision_order:

                # print("PRIMA")
                # print("----------", agent.id)
                # print(selfish_actions[agent.id])
                # pprint(data[agent.id]["admissible_action_to_IG"])

                promity_measurement = agents[id].proximity.get_predicted_measurements(
                    agents[id].state.position, id, predicted_states
                )
                neighbors_ids = promity_measurement["neighbors_ids"]

                for neighbor_id in neighbors_ids:
                    neighbor_fp_ij, _ = agents[neighbor_id].camera.get_fp_vertices_ij(
                        predicted_states[neighbor_id, :]
                    )
                    for action, agent_fp_ij in data[id][
                        "admissible_action_to_fp_ij"
                    ].items():
                        # print(action, (1.0 - IoU(neighbor_fp_ij, agent_fp_ij)))
                        data[id]["admissible_action_to_IG"][action][0] *= 1.0 - IoU(
                            neighbor_fp_ij, agent_fp_ij
                        )

                agent_action = self.__argmin_action(
                    agents[id], data[id]["admissible_action_to_IG"]
                )

                predicted_actions[id] = agent_action

                predicted_states[id, :] += self.action_to_direction[agent_action]

                # print("DOPO")
                # print(agent.id)
                # print(agent_action)
                # pprint(data[agent.id]["admissible_action_to_IG"])

            # print(predicted_actions)
            # print(predicted_states)

        return predicted_actions, data

    def _non_targeted_mini_weighted_async_no_prediction(
        self, agents: List[Agent], observations: List[Dict]
    ):

        actions, data = ["" for _ in range(len(agents))], []
        positions = [agent.state.position for agent in agents]

        for agent in agents:

            admissible_action_to_IG = {}

            neighbors_ids = []
            if len(observations) != 0:
                neighbors_ids = observations[agent.id]["neighbors_ids"]

            for agent_action, agent_future_position in self.position_graph[
                (
                    agent.state.position[X],
                    agent.state.position[Y],
                    int(agent.state.position[Z]),
                )
            ].items():

                agent_future_fp_ij = self.position_data[
                    (
                        agent_future_position[X],
                        agent_future_position[Y],
                        int(agent_future_position[Z]),
                    )
                ]["fp_vertices_ij"]

                self.n_buffer[:] = 1
                self.z_buffer[:] = agent_future_position[Z]

                for neighbor_id in neighbors_ids:
                    if positions[neighbor_id][Z] <= agent_future_position[Z]:
                        neighbor_fp_ij = self.position_data[
                            (
                                positions[neighbor_id][X],
                                positions[neighbor_id][Y],
                                int(positions[neighbor_id][Z]),
                            )
                        ]["fp_vertices_ij"]

                        row_slice = slice(
                            neighbor_fp_ij["ul"][I], neighbor_fp_ij["bl"][I]
                        )
                        col_slice = slice(
                            neighbor_fp_ij["ul"][J], neighbor_fp_ij["ur"][J]
                        )

                        upd_n_buffer_slice = np.where(
                            positions[neighbor_id][Z] == agent_future_position[Z],
                            self.n_buffer[row_slice, col_slice] + 1,
                            self.n_buffer[row_slice, col_slice],
                        )

                        upd_z_buffer_slice = np.where(
                            positions[neighbor_id][Z]
                            <= self.z_buffer[row_slice, col_slice],
                            positions[neighbor_id][Z],
                            self.z_buffer[row_slice, col_slice],
                        )

                        self.n_buffer[row_slice, col_slice] = upd_n_buffer_slice
                        self.z_buffer[row_slice, col_slice] = upd_z_buffer_slice

                row_slice = slice(
                    agent_future_fp_ij["ul"][I], agent_future_fp_ij["bl"][I]
                )
                col_slice = slice(
                    agent_future_fp_ij["ul"][J], agent_future_fp_ij["ur"][J]
                )

                cells_weights = np.where(
                    self.z_buffer[row_slice, col_slice] < agent_future_position[Z],
                    0,
                    1 / self.n_buffer[row_slice, col_slice],
                )

                fp_map_belief = map_beliefs[row_slice, col_slice, agent.id]

                # H(M_fp) - entropy of the prior = sum of the entropy of each cell in the footprint
                current_H = map_belief_entropies[row_slice, col_slice, agent.id]

                # H(M_fp|Z) - conditional entropy of the posterior = sum of the averaged entropy of each cell in the footprint
                sigma0, sigma1 = self.position_data[
                    (
                        agent_future_position[X],
                        agent_future_position[Y],
                        int(agent_future_position[Z]),
                    )
                ]["sigmas"]
                current_cH = cH(fp_map_belief, np.round(sigma0, 7), np.round(sigma1, 7))

                # assert non-negativity
                diff = current_H - current_cH
                diff = np.where(np.isclose(current_H, current_cH), 0.0, diff)
                diff *= cells_weights
                assert np.all(np.greater_equal(diff, 0.0))

                admissible_action_to_IG[agent_action] = [
                    # np.round(np.sum(diff) / cost, 8),
                    np.sum(diff)
                ]

            selected_agent_action = self.__argmin_action(agent, admissible_action_to_IG)
            actions[agent.id] = selected_agent_action
            positions[agent.id] = self.position_graph[
                (
                    agent.state.position[X],
                    agent.state.position[Y],
                    int(agent.state.position[Z]),
                )
            ][selected_agent_action]

        return actions, data

    def _non_targeted_mini_weighted_async_no_prediction_rnd(
        self, agents: List[Agent], observations: List[Dict]
    ):

        actions, data = ["" for _ in range(len(agents))], []
        positions = [agent.state.position for agent in agents]

        decision_order = self.agent_decision_order_rng.permutation(len(agents))

        for id in decision_order:

            admissible_action_to_IG = {}

            neighbors_ids = []
            if len(observations) != 0:
                neighbors_ids = observations[id]["neighbors_ids"]

            for agent_action, agent_future_position in self.position_graph[
                (
                    agents[id].state.position[X],
                    agents[id].state.position[Y],
                    int(agents[id].state.position[Z]),
                )
            ].items():

                agent_future_fp_ij = self.position_data[
                    (
                        agent_future_position[X],
                        agent_future_position[Y],
                        int(agent_future_position[Z]),
                    )
                ]["fp_vertices_ij"]

                self.n_buffer[:] = 1
                self.z_buffer[:] = agent_future_position[Z]

                for neighbor_id in neighbors_ids:
                    if positions[neighbor_id][Z] <= agent_future_position[Z]:
                        neighbor_fp_ij = self.position_data[
                            (
                                positions[neighbor_id][X],
                                positions[neighbor_id][Y],
                                int(positions[neighbor_id][Z]),
                            )
                        ]["fp_vertices_ij"]

                        row_slice = slice(
                            neighbor_fp_ij["ul"][I], neighbor_fp_ij["bl"][I]
                        )
                        col_slice = slice(
                            neighbor_fp_ij["ul"][J], neighbor_fp_ij["ur"][J]
                        )

                        upd_n_buffer_slice = np.where(
                            positions[neighbor_id][Z] == agent_future_position[Z],
                            self.n_buffer[row_slice, col_slice] + 1,
                            self.n_buffer[row_slice, col_slice],
                        )

                        upd_z_buffer_slice = np.where(
                            positions[neighbor_id][Z]
                            <= self.z_buffer[row_slice, col_slice],
                            positions[neighbor_id][Z],
                            self.z_buffer[row_slice, col_slice],
                        )

                        self.n_buffer[row_slice, col_slice] = upd_n_buffer_slice
                        self.z_buffer[row_slice, col_slice] = upd_z_buffer_slice

                row_slice = slice(
                    agent_future_fp_ij["ul"][I], agent_future_fp_ij["bl"][I]
                )
                col_slice = slice(
                    agent_future_fp_ij["ul"][J], agent_future_fp_ij["ur"][J]
                )

                cells_weights = np.where(
                    self.z_buffer[row_slice, col_slice] < agent_future_position[Z],
                    0,
                    1 / self.n_buffer[row_slice, col_slice],
                )

                fp_map_belief = map_beliefs[row_slice, col_slice, id]

                # H(M_fp) - entropy of the prior = sum of the entropy of each cell in the footprint
                current_H = map_belief_entropies[row_slice, col_slice, id]

                # H(M_fp|Z) - conditional entropy of the posterior = sum of the averaged entropy of each cell in the footprint
                sigma0, sigma1 = self.position_data[
                    (
                        agent_future_position[X],
                        agent_future_position[Y],
                        int(agent_future_position[Z]),
                    )
                ]["sigmas"]
                current_cH = cH(fp_map_belief, np.round(sigma0, 7), np.round(sigma1, 7))

                # assert non-negativity
                diff = current_H - current_cH
                diff = np.where(np.isclose(current_H, current_cH), 0.0, diff)
                diff *= cells_weights
                assert np.all(np.greater_equal(diff, 0.0))

                admissible_action_to_IG[agent_action] = [
                    # np.round(np.sum(diff) / cost, 8),
                    np.sum(diff)
                ]

            selected_agent_action = self.__argmin_action(
                agents[id], admissible_action_to_IG
            )
            actions[id] = selected_agent_action
            positions[id] = self.position_graph[
                (
                    agents[id].state.position[X],
                    agents[id].state.position[Y],
                    int(agents[id].state.position[Z]),
                )
            ][selected_agent_action]

        return actions, data

    # def _map_loop_closing(self, agents: List[Agent], observations: List[Dict]):
    #
    #     selfish_actions = self._non_targeted_mini_new(agents, map_belief=map_belief)
    #
    #     if len(observations) == 0: return selfish_actions
    #     n_neighbors_per_agent = [len(o["neighbors_ids"]) for o in observations]
    #     if np.all(np.array(n_neighbors_per_agent) == 0):
    #         return selfish_actions
    #
    #     selfish_actions = [sa[0] for sa in selfish_actions]
    #     actions = []
    #
    #     current_map_belief = map_belief
    #
    #     for o, agent, sa in zip(observations, agents, selfish_actions):
    #
    #         if map_belief is None:
    #             current_map_belief = agent.map_belief
    #
    #         current_map_belief_copy = np.copy(current_map_belief)
    #
    #         for neighbor_id, neighbor_position in zip(o["neighbors_ids"], o["neighbors_positions"]):
    #             neighbor_selfish_action = selfish_actions[neighbor_id]
    #             neighbor_next_position = neighbor_position + self.action_to_direction[neighbor_selfish_action]
    #             fp_vertices_ij, _ = agent.camera.get_fp_vertices_ij(neighbor_next_position)
    #             sigma0, sigma1 = agent.camera.get_sigmas(neighbor_next_position)
    #
    #             fp_map_belief = current_map_belief_copy[fp_vertices_ij["ul"][I]:fp_vertices_ij["bl"][I],
    #                             fp_vertices_ij["ul"][J]:fp_vertices_ij["ur"][J]]
    #
    #             # posterior predictive p(z_t+1|z_0:t) = SUM_m p(z_t+1|m)*p(m|z_0:t)
    #             p1h = (1.0-sigma1)*fp_map_belief + sigma0*(1.0-fp_map_belief)
    #             random_obs = np.where(np.random.random(p1h.shape) >= 0.5, 1, 0)
    #
    #             # most probable observation
    #             z = np.where(p1h > 0.5, 1, p1h)
    #             z = np.where(z < 0.5, 0, z)
    #             z = np.where(z == 0.5, random_obs, z)
    #
    #             likelihood_m_zero = np.where(z == 0, 1.0 - sigma0, sigma0)
    #             likelihood_m_one = np.where(z == 0, sigma1, 1.0 - sigma1)
    #
    #             posterior_m_zero = likelihood_m_zero * (1.0 - fp_map_belief)
    #             posterior_m_one = likelihood_m_one * fp_map_belief
    #
    #             assert np.all(np.greater_equal(posterior_m_one, 0.0))
    #
    #             # posterior_m_zero_norm = posterior_m_zero / (posterior_m_zero + posterior_m_one)
    #             posterior_m_one_norm = posterior_m_one / (posterior_m_zero + posterior_m_one)
    #
    #             assert np.all(np.greater_equal(posterior_m_one_norm, 0.0)) and \
    #                    np.all(np.less_equal(posterior_m_one_norm, 1.0))
    #
    #             current_map_belief_copy[fp_vertices_ij["ul"][I]:fp_vertices_ij["bl"][I],
    #                                     fp_vertices_ij["ul"][J]:fp_vertices_ij["ur"][J]] = posterior_m_one_norm
    #
    #
    #         agent.allucinated_map_belief = np.copy(current_map_belief_copy)
    #
    #         admissible_action_to_IG = {}
    #
    #         map_H = H(current_map_belief)
    #
    #         future_action_position = self.position_graph[(agent.state.position[X],
    #                                                       agent.state.position[Y],
    #                                                       int(agent.state.position[Z]))]
    #
    #         for action, position in future_action_position.items():
    #             fp_vertices_ij, _ = agent.camera.get_fp_vertices_ij(position)
    #
    #             fp_map_belief = current_map_belief_copy[fp_vertices_ij["ul"][I]:fp_vertices_ij["bl"][I],
    #                             fp_vertices_ij["ul"][J]:fp_vertices_ij["ur"][J]]
    #
    #             # H(M_fp) - entropy of the prior = sum of the entropy of each cell in the footprint
    #             current_H = H(current_map_belief_copy[fp_vertices_ij["ul"][I]:fp_vertices_ij["bl"][I],
    #                         fp_vertices_ij["ul"][J]:fp_vertices_ij["ur"][J]])
    #             # H_M_fp = np.sum(current_H)
    #
    #             # H(M_fp|Z) - conditional entropy of the posterior = sum of the averaged entropy of each cell in the footprint
    #             sigma0, sigma1 = agent.camera.get_sigmas(position)
    #             current_cH = cH(fp_map_belief, np.round(sigma0, 7), np.round(sigma1, 7))
    #             # H_M_fp_Z = np.sum(current_cH)
    #
    #             # assert non-negativity
    #             diff = current_H - current_cH
    #             diff = np.where(np.isclose(current_H, current_cH), 0.0, diff)
    #             assert np.all(np.greater_equal(diff, 0.0)), \
    #                 f"{position}" \
    #                 f"{action}" \
    #                 f"{current_H[np.less(diff, 0.0)]}" \
    #                 f"{fp_map_belief[np.less(diff, 0.0)]}"
    #
    #             cost = 1 if action in ["up", "down"] else 1
    #             admissible_action_to_IG[action] = [
    #                 # np.round(np.sum(diff) / cost, 8),
    #                 np.sum(diff) / cost
    #             ]
    #
    #         best_admissible_actions = []
    #         sorted_action_to_IG = sorted(admissible_action_to_IG.items(),
    #                                      key=lambda x: x[1][0],
    #                                      reverse=True)
    #
    #         best_action, best_IG = sorted_action_to_IG.pop(0)
    #
    #         best_admissible_actions.append(best_action)
    #
    #         for action, IG in sorted_action_to_IG:
    #             if IG[0] == best_IG[0]:
    #                 best_admissible_actions.append(action)
    #
    #         actions.append((agent.rng.choice(best_admissible_actions),
    #                         admissible_action_to_IG,
    #                         len(best_admissible_actions),
    #                         np.sum(map_H)))
    #
    #     return actions

    def compute_map_belief_entropies(self):
        global map_beliefs, map_belief_entropies

        if self.centralized:
            map_belief_entropies[:, :, :] = H(map_beliefs[:, :, 0])[:, :, np.newaxis]
        else:
            map_belief_entropies = H(map_beliefs)

    def compute_agg_map_belief(self):
        global map_beliefs, map_belief_entropies, agg_map_belief, agg_map_belief_entropy

        if self.centralized:
            # map_belief_entropies[:, :, :] = H(map_beliefs[:, :, 0])[:,:,np.newaxis]
            agg_map_belief = map_beliefs[:, :, 0]
            agg_map_belief_entropy = map_belief_entropies[:, :, 0]
        else:
            # argmin = np.argmin(map_belief_entropies, axis = 2, keepdims = True)
            # agg_map_belief = np.squeeze(np.take_along_axis(map_beliefs, argmin, axis=2))
            # agg_map_belief_entropy = np.squeeze(np.take_along_axis(map_belief_entropies, argmin, axis=2))

            agg_map_belief = np.prod(map_beliefs, axis=2) / (
                np.prod(map_beliefs, axis=2) + np.prod(1.0 - map_beliefs, axis=2)
            )
            agg_map_belief_entropy = H(agg_map_belief)

    def __argmin_action(self, agent: Agent, admissible_action_to_IG: Dict):
        best_admissible_actions = []
        sorted_action_to_IG = sorted(
            admissible_action_to_IG.items(), key=lambda x: x[1][0], reverse=True
        )

        best_action, best_IG = sorted_action_to_IG.pop(0)

        best_admissible_actions.append(best_action)

        for action, IG in sorted_action_to_IG:
            if IG[0] == best_IG[0]:
                best_admissible_actions.append(action)

        return agent.rng.choice(best_admissible_actions)

    def reset_sweep(self):
        self.sweep_left_right = ["left"] * len(self.sweep_left_right)
        self.last_action = ["up"] * len(self.last_action)
        self.n_visited_positions = [0] * len(self.n_visited_positions)


class Viewer:
    def __init__(self, w, h, min_space_z, max_space_z, n_agents):
        self.window = pyglet.window.Window(width=w, height=h)

        self.agents = [
            pyglet.shapes.Circle(x=0, y=0, radius=10, color=(0, 0, 0))
            for _ in range(n_agents)
        ]

        self.footprints = [
            pyglet.shapes.Box(0, 0, 20, 20, thickness=2, color=(180, 0, 0))
            for _ in range(n_agents)
        ]

        self.step_index = pyglet.text.Label(
            "",
            font_name="Times New Roman",
            font_size=36,
            x=self.window.width // 2,
            y=self.window.height // 2,
            anchor_x="center",
            anchor_y="center",
        )

        self.n_agents = n_agents
        self.min_space_z = min_space_z
        self.max_space_z = max_space_z

        # pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
        # pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)

    def set_image(self, image):
        w, h = image.shape[:2]
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        mask = image.astype(bool)
        image = np.where(mask, np.array([0, 100, 0]), np.array([255, 255, 255]))
        image = np.flipud(image)
        img = image.astype(np.uint8).reshape(-1)
        tex_data = (pyglet.gl.GLubyte * img.size)(*img)
        pyg_img = pyglet.image.ImageData(
            h,
            w,
            "RGB",
            tex_data,
            # pitch=h * w * 1,  # width x channels x bytes per pixel
        )

        self.map_ground_truth = pyglet.sprite.Sprite(pyg_img)

    def render(self, agents: List[Agent], step_index, sleep):
        pyglet.gl.glClearColor(1, 1, 1, 1)

        pyglet.clock.tick()
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self.map_ground_truth.draw()

        self.step_index.text = str(step_index)
        self.step_index.draw()

        # for i in range(self.n_agents):
        #     w,h = map_beliefs[:,:,i].shape[:2]
        #     image = np.zeros((w,h,4))
        #     image[:] = np.array([0, 100, 0, 0])
        #     image[:,:,3] = np.flipud(map_beliefs[:,:,i]) * 255
        #     img = image.astype(np.uint8).reshape(-1)
        #     tex_data = (pyglet.gl.GLubyte * img.size)(*img)
        #     pyg_img = pyglet.image.ImageData(
        #         h,
        #         w,
        #         "RGBA",
        #         tex_data,
        #         #pitch=w*h,  # width x channels x bytes per pixel
        #     )
        #
        #     #pyg_img.blit(w*(i+1), 0)
        #     sp = pyglet.sprite.Sprite(x=w*(i+1), y=0, img=pyg_img)
        #     sp.update(scale=0.3)
        #     sp.draw()

        for i in range(self.n_agents):
            ij = agents[i].camera._xy_to_ij(states[i][:2])
            z = states[i][Z]

            self.agents[i].x, self.agents[i].y = ij[J], self.window.height - ij[I]
            self.agents[i].radius = (
                4
                + ((z - self.min_space_z) / (self.max_space_z - self.min_space_z)) * 10
            )

            fp_ij, _ = agents[i].camera.get_fp_vertices_ij(states[i])
            w = fp_ij["br"][J] - fp_ij["bl"][J]
            h = fp_ij["bl"][I] - fp_ij["ul"][I]

            self.footprints[i].x = fp_ij["bl"][J]
            self.footprints[i].y = self.window.height - fp_ij["bl"][I]
            self.footprints[i].width = w
            self.footprints[i].height = h

            self.footprints[i].draw()
            self.agents[i].draw()

        self.window.flip()

        time.sleep(sleep)


# ------------------------------------------------
# for future expansions - currently not used
# ------------------------------------------------


class Communication:
    def __init__(self, n_cell: int, **kwargs):
        self.inference_type = kwargs.get("inference_type")

        self.fusion_type = kwargs.get("fusion_type")
        assert self.fusion_type in ["naive", "CF_LBP", "CF_OG"]

        self.n_agents = kwargs.get("n_agents", 1)

        self.n_cell = n_cell

        # depth_to_direction = 0123_4 -> URDL_fake
        self.msgs = np.ones((4 + 1, n_cell, n_cell), dtype=float) * 0.5
        self.msgs_buffer = np.ones_like(self.msgs) * 0.5

        self.pairwise_potential = np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)

        # (channelS, row_slice, col_slice) to product & marginalize
        # (row_slice, col_slice) to read
        # (channel, row_slice, col_slice) to write
        self.direction_to_slicing_data = {
            "up": {
                "product_slice": lambda fp_ij: (
                    (1, 2, 3, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(
                        1 if fp_ij["ul"][I] == 0 else 0, fp_ij["bl"][I] - fp_ij["ul"][I]
                    ),
                    slice(0, fp_ij["ur"][J] - fp_ij["ul"][J]),
                ),
                "write_slice": lambda fp_ij: (
                    2,
                    slice(max(0, fp_ij["ul"][I] - 1), min(n_cell, fp_ij["bl"][I] - 1)),
                    slice(max(0, fp_ij["ul"][J]), min(n_cell, fp_ij["br"][J])),
                ),
            },
            "right": {
                "product_slice": lambda fp_ij: (
                    (0, 2, 3, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(0, fp_ij["bl"][I] - fp_ij["ul"][I]),
                    slice(
                        0,
                        (
                            fp_ij["ur"][J] - fp_ij["ul"][J] - 1
                            if fp_ij["ur"][J] == n_cell
                            else fp_ij["ur"][J] - fp_ij["ul"][J]
                        ),
                    ),
                ),
                "write_slice": lambda fp_ij: (
                    3,
                    slice(max(0, fp_ij["ul"][I]), min(n_cell, fp_ij["bl"][I])),
                    slice(max(0, fp_ij["ul"][J] + 1), min(n_cell, fp_ij["br"][J] + 1)),
                ),
            },
            "down": {
                "product_slice": lambda fp_ij: (
                    (0, 1, 3, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(
                        0,
                        (
                            fp_ij["bl"][I] - fp_ij["ul"][I] - 1
                            if fp_ij["bl"][I] == n_cell
                            else fp_ij["bl"][I] - fp_ij["ul"][I]
                        ),
                    ),
                    slice(0, fp_ij["ur"][J] - fp_ij["ul"][J]),
                ),
                "write_slice": lambda fp_ij: (
                    0,
                    slice(max(0, fp_ij["ul"][I] + 1), min(n_cell, fp_ij["bl"][I] + 1)),
                    slice(max(0, fp_ij["ul"][J]), min(n_cell, fp_ij["br"][J])),
                ),
            },
            "left": {
                "product_slice": lambda fp_ij: (
                    (0, 1, 2, 4),
                    slice(fp_ij["ul"][I], fp_ij["bl"][I]),
                    slice(fp_ij["ul"][J], fp_ij["ur"][J]),
                ),
                "read_slice": lambda fp_ij: (
                    slice(0, fp_ij["bl"][I] - fp_ij["ul"][I]),
                    slice(
                        1 if fp_ij["ul"][J] == 0 else 0, fp_ij["ur"][J] - fp_ij["ul"][J]
                    ),
                ),
                "write_slice": lambda fp_ij: (
                    1,
                    slice(max(0, fp_ij["ul"][I]), min(n_cell, fp_ij["bl"][I])),
                    slice(max(0, fp_ij["ul"][J] - 1), min(n_cell, fp_ij["br"][J] - 1)),
                ),
            },
        }

    def _send_belief(self, observations: List[Dict], agents: List[Agent]):

        for o, a in zip(observations, agents):
            neighbors_ids = o["neighbors_ids"]
            fp_vertices_ij = o["fp_ij"]
            fp_map_belief = a.map_belief[
                fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
            ]

            for neighbor_id in neighbors_ids:
                agents[neighbor_id].msg_cache.append(
                    {
                        "id": a.id,
                        "fp_ij": fp_vertices_ij,
                        "fp_map_belief": fp_map_belief,
                    }
                )

    def _integrate_belief(self, agents: List[Agent]):
        for a in agents:
            for msg in a.msg_cache:
                fp_vertices_ij = msg["fp_ij"]

                sender_fp_map_belief_1 = msg["fp_map_belief"]

                my_fp_map_belief_1 = a.map_belief[
                    fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                    fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                ]

                fused_1 = my_fp_map_belief_1 * sender_fp_map_belief_1
                fused_0 = (1.0 - my_fp_map_belief_1) * (1.0 - sender_fp_map_belief_1)

                den = fused_0 + fused_1

                fused_1[den == 0.0] = 0.5
                den = np.where(den == 0.0, 1.0, den)

                a.map_belief[
                    fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                    fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
                ] = (
                    fused_1 / den
                )

    def _send_likelihoods(self, observations: List[Dict], agents: List[Agent]):
        for o, a in zip(observations, agents):

            z, fp_vertices_ij, sigma0, sigma1 = (
                o["z"],
                o["fp_ij"],
                o["sigmas"][0],
                o["sigmas"][1],
            )

            likelihood_m_zero = np.where(z == 0, 1 - sigma0, sigma0)
            likelihood_m_one = np.where(z == 0, sigma1, 1 - sigma1)

            a.map_likelihoods_m_zero[
                :,
                fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
            ] *= likelihood_m_zero

            a.map_likelihoods_m_one[
                :,
                fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
                fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
            ] *= likelihood_m_one

            neighbors_ids = o["neighbors_ids"]

            for neighbor_id in neighbors_ids:
                agents[neighbor_id].msg_cache.append(
                    {
                        "id": a.id,
                        "likelihood_m_zero": np.copy(
                            a.map_likelihoods_m_zero[neighbor_id, :, :]
                        ),
                        "likelihood_m_one": np.copy(
                            a.map_likelihoods_m_one[neighbor_id, :, :]
                        ),
                    }
                )

                a.map_likelihoods_m_zero[neighbor_id, :, :] = 1.0
                a.map_likelihoods_m_one[neighbor_id, :, :] = 1.0

    def _integrate_likelihood_OG(self, agents: List[Agent]):

        for a in agents:
            if len(a.msg_cache) != 0:

                likelihood_m_zero = np.ones_like(a.map_belief)
                likelihood_m_one = np.ones_like(a.map_belief)

                for msg in a.msg_cache:

                    likelihood_m_zero *= msg["likelihood_m_zero"]
                    likelihood_m_one *= msg["likelihood_m_one"]

                assert np.all(np.not_equal(likelihood_m_zero, 0.0))
                assert np.all(np.not_equal(likelihood_m_one, 0.0))

                posterior_m_zero = likelihood_m_zero * (1.0 - a.map_belief)
                posterior_m_one = likelihood_m_one * a.map_belief

                assert np.all(np.greater_equal(posterior_m_one, 0.0))

                # posterior_m_zero_norm = posterior_m_zero / (posterior_m_zero + posterior_m_one)
                posterior_m_one_norm = posterior_m_one / (
                    posterior_m_zero + posterior_m_one
                )

                assert np.all(np.greater_equal(posterior_m_one_norm, 0.0)) and np.all(
                    np.less_equal(posterior_m_one_norm, 1.0)
                )

                a.map_belief = np.copy(posterior_m_one_norm)

    def _integrate_likelihood_LBP(self, agents: List[Agent], n_iteration: int = 1):
        self._integrate_likelihood_OG(agents)

        for a in agents:
            if len(a.msg_cache) != 0:
                # reset msgs and msgs_buffer
                self.msgs = np.ones_like(self.msgs) * 0.5
                self.msgs_buffer = np.ones_like(self.msgs) * 0.5
                self.msgs[4, :, :] = (
                    a.map_belief
                )  # set msgs last channel with current map belief

                # the whole map!
                fp_vertices_ij = {
                    "ul": (0, 0),
                    "bl": (self.n_cell, 0),
                    "ur": (0, self.n_cell),
                    "br": (self.n_cell, self.n_cell),
                }

                for _ in range(n_iteration):
                    for direction, data in self.direction_to_slicing_data.items():
                        product_slice = data["product_slice"](fp_vertices_ij)
                        read_slice = data["read_slice"](fp_vertices_ij)
                        write_slice = data["write_slice"](fp_vertices_ij)

                        # elementwise multiplication of msgs
                        mul_0 = np.prod(1 - self.msgs[product_slice], axis=0)
                        mul_1 = np.prod(self.msgs[product_slice], axis=0)

                        # matrix-vector multiplication (factor-msg)
                        msg_0 = (
                            self.pairwise_potential[0, 0] * mul_0
                            + self.pairwise_potential[0, 1] * mul_1
                        )
                        msg_1 = (
                            self.pairwise_potential[1, 0] * mul_0
                            + self.pairwise_potential[1, 1] * mul_1
                        )

                        # normalize the first coordinate of the msg
                        norm_msg_1 = msg_1 / (msg_0 + msg_1)

                        # buffering
                        self.msgs_buffer[write_slice] = norm_msg_1[read_slice]

                    # copy the first 4 channels only
                    # the 5th one is the map belief
                    self.msgs[:4, :, :] = self.msgs_buffer[:4, :, :]

                bel_0 = np.prod(
                    1 - self.msgs[:, product_slice[1], product_slice[2]], axis=0
                )
                bel_1 = np.prod(
                    self.msgs[:, product_slice[1], product_slice[2]], axis=0
                )

                # norm_bel_0 = bel_0 / (bel_0 + bel_1)
                a.map_belief[product_slice[1], product_slice[2]] = bel_1 / (
                    bel_0 + bel_1
                )

                assert np.all(
                    np.greater_equal(
                        a.map_belief[product_slice[1], product_slice[2]], 0.0
                    )
                ) and np.all(
                    np.less_equal(a.map_belief[product_slice[1], product_slice[2]], 1.0)
                )

    def fuse_belief(self, observations: List[Dict], agents: List[Agent]):
        if self.fusion_type == "naive":
            self._send_belief(observations, agents)
            self._integrate_belief(agents)
        if self.fusion_type == "CF_OG":
            self._send_likelihoods(observations, agents)
            self._integrate_likelihood_OG(agents)
        if self.fusion_type == "CF_LBP":
            self._send_likelihoods(observations, agents)
            self._integrate_likelihood_LBP(agents)

        for a in agents:
            a.msg_cache = []


class Inference2:
    def __init__(self):
        n_row, n_col = 5, 5
        f_row = lambda i, j: {0, 2, 3} if i == 0 else {}
        f_col = lambda i, j: {0, 1, 3} if j == 0 else {}
        l_row = lambda i, j: {0, 1, 2} if i == n_row - 1 else {}
        l_col = lambda i, j: {1, 2, 3} if j == n_col - 1 else {}

        a_to_dir = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}

        self._dynamics_factor = {}

        for i in range(n_row):
            for j in range(n_col):
                condition = [f_row(i, j), f_col(i, j), l_row(i, j), l_col(i, j)]
                check_condition = [c for c in condition if len(c) > 0]
                if len(check_condition) == 0:
                    check_condition.append({0, 1, 2, 3})
                f = set.intersection(*check_condition)
                self._dynamics_factor[(i, j)] = {
                    a: (i + a_to_dir[a][0], j + a_to_dir[a][1]) for a in list(f)
                }

    def dynamics_factor(self, _x, _a, x):
        try:
            x_ = self._dynamics_factor[_x][_a]
        except:
            return 0.0

        if x != x_:
            return 0.0

        return 1.0

    def collision_factor(self, xA, xB):
        min_d = np.sqrt(2)
        d = np.linalg.norm(np.array([xA[I] - xB[I], xA[J] - xB[J]]))
        return min(1.0, d // min_d)

    def goal_factor(self, x, x_goal):
        d = np.linalg.norm(np.array([x[I] - x_goal[I], x[J] - x_goal[J]]))

        return 1.0 / (d + 1.0)


class StateNode:
    def __init__(self, name: str):
        self.V_list = {}
        self.parent: ActionNode = None
        self.children = []
        self.P = 0.0
        self.name = name

    def send(self):
        V = max(list(self.V_list.values()))
        self.parent.average += self.P * V


class ActionNode:
    def __init__(self, name: str):
        self.R = 0.0
        self.average = 0.0
        self.parent: StateNode = None
        self.children = []
        self.name = name

    def send(self):
        self.parent.V_list[self.name] = self.R + self.average


class Tree:

    def __init__(self, aid_to_albl, dynamics_dict):
        self.aid_to_albl = aid_to_albl
        self.dynamics_dict = dynamics_dict
        self.root_history = []

    # def set_reward_dependencies(self, goal, walls):
    #     self.goal = goal
    #     self.walls = walls
    #
    # def _reward(self, node):
    #     d = np.linalg.norm(np.array([node[0] - self.goal[0],
    #                              node[1] - self.goal[1]]))
    #
    #     return 1.0 / (d + 1.0) if node not in self.walls else -10

    def _reward(self, node, agent, map_belief):

        agent_future_state = copy.deepcopy(agent.state)
        agent_future_state.set_position(np.array(node))

        fp_vertices_ij = agent.camera.get_fp_vertices_ij(agent_future_state.position)

        fp_map_belief = map_belief[
            fp_vertices_ij["ul"][I] : fp_vertices_ij["bl"][I],
            fp_vertices_ij["ul"][J] : fp_vertices_ij["ur"][J],
        ]

        # H(M_fp) - entropy of the prior = sum of the entropy of each cell in the footprint
        current_H = H(fp_map_belief)

        # H(M_fp|Z) - conditional entropy of the posterior = sum of the averaged entropy of each cell in the footprint
        sigma0, sigma1 = agent.camera.get_sigmas(agent_future_state.position)
        current_cH = cH(fp_map_belief, sigma0, sigma1)

        # assert non-negativity
        diff = current_H - current_cH
        diff = np.where(np.isclose(current_H, current_cH), 0.0, diff)
        assert np.all(np.greater_equal(diff, 0.0))

        ig = np.sum(diff)

        return ig

    def _add_node(self, parent_name, node, depth):
        if parent_name != None:
            if parent_name not in self.tree:
                raise ValueError(f"Parent {parent_name} not in tree")

        node_name = self._to_str(node)
        if isinstance(node, tuple):
            self.tree[node_name] = {
                "parent_name": parent_name if parent_name != None else None,
                "depth": depth,
                "n_message": 0,
                "children_name": [],
                "name": node_name,
                "V_dict": {},
                "P": 1.0,
                "visited": False,
            }

        elif isinstance(node, str):
            self.tree[node_name] = {
                "parent_name": parent_name if parent_name != None else None,
                "depth": depth,
                "n_message": 0,
                "children_name": [],
                "name": node_name,
                "average": 0.0,
                "R": 0.0,
            }

        if parent_name != None:
            self.tree[parent_name]["children_name"].append(node_name)

        self.id += 1

        return node_name

    def _to_str(self, n):
        if isinstance(n, tuple):
            return f"{n[0]}{n[1]}{n[2]}_s_{self.id}"
        elif isinstance(n, str) == 1:
            return f"{n}_a_{self.id}"

    def expand_tree(self, agent, map_belief, max_depth):

        self.id = 0
        self.tree = {}

        root = tuple(agent.state.position)
        root = (root[0], root[1], int(root[2]))
        s_name = self._add_node(None, root, 0)
        # self.root_history.append(root)
        frontier = [(s_name, root)]
        leaves = []

        while len(frontier) != 0:
            _s_name, _s = frontier.pop(0)
            for _a, s in self.dynamics_dict[_s].items():
                # if s not in self.root_history:
                _a_name = self._add_node(_s_name, _a, self.tree[_s_name]["depth"])
                self.tree[_s_name]["V_dict"][_a_name] = 0.0

                if self.tree[_s_name]["depth"] + 1 <= max_depth:
                    s_name = self._add_node(_a_name, s, self.tree[_s_name]["depth"] + 1)
                    frontier.append((s_name, s))

                # per calcolare la V degli stati terminali
                # la V di uno stato terminale è solamente la sua R
                # perchè sarebbe V = R + P1*0.0+P2*0.0+P3*0.0
                self.tree[_a_name]["R"] = self._reward(s, agent, map_belief)

                if self.tree[_a_name]["depth"] == max_depth:
                    leaves.append(_a_name)

        return leaves

    def _action_send(self, a_name):
        action = self.tree[a_name]
        state = self.tree[action["parent_name"]]
        state["V_dict"][a_name] = action["R"] + action["average"]

        state["n_message"] += 1

        return state

    def _state_send(self, s_name):
        state = self.tree[s_name]
        action = self.tree[state["parent_name"]]
        V = max(list(state["V_dict"].values()))
        action["average"] += state["P"] * V

        action["n_message"] += 1

        return action

    def backward_induction(self, leaves):
        frontier = list(leaves)

        while len(frontier) != 0:
            node_name = frontier.pop(0)

            if "a" in node_name:
                state = self._action_send(node_name)
                if state["n_message"] == len(state["children_name"]):
                    frontier.append(state["name"])
            elif "s" in node_name:
                if self.tree[node_name]["parent_name"] != None:
                    action = self._state_send(node_name)
                    if action["n_message"] == len(action["children_name"]):
                        frontier.append(action["name"])

        # print(self.tree[node_name])
        sorted_V_dict = sorted(
            self.tree[node_name]["V_dict"].items(), key=lambda x: x[1], reverse=True
        )
        # print(sorted_V_dict)
        return (
            sorted_V_dict[0][0],
            {k[: k.find("_")]: [v] for k, v in self.tree[node_name]["V_dict"].items()},
            -1,
        )


class Variable:
    def __init__(self, dim: int, name: str, evidence: np.ndarray = None):
        self.name = name
        self.dim = dim
        self.msgs_in = {}
        self.evidence = evidence

    def set_evidence(self, evidence: np.ndarray):
        self.evidence = evidence

    def compute(self, factor):
        msg_out = np.ones((self.dim, 1)) if self.evidence is None else self.evidence
        for factor_name, msg_in in self.msgs_in.items():
            if factor_name != factor.name:
                msg_out *= msg_in

        msg_out = msg_out.reshape((self.dim, 1))
        # msg_out = msg_out/np.sum(msg_out)

        return msg_out

    def marginal(self):
        marginal = np.ones((self.dim, 1))
        for msg_in in self.msgs_in.values():
            marginal *= msg_in

        return marginal / np.sum(marginal)


class Factor:
    def __init__(self, name: str):
        self.name = name
        self.msgs_in = {}
        self.weights = np.ones((1, 1))
        self.variable_tensor_dim = []
        self.var_name_to_tensor_data = {}

    def set_weights(self, weights: np.ndarray):
        assert np.array_equal(self.weights.shape, weights.shape)
        self.weights = weights

    def compute(self, variable):
        msg_out = np.ones_like(self.weights)
        for variable_name, msg_in in self.msgs_in.items():
            if variable_name != variable.name:
                shape = [1] * self.weights.ndim
                var_tensor_dim = self.var_name_to_tensor_data[variable_name][
                    "tensor_dim"
                ]
                var_dim = self.var_name_to_tensor_data[variable_name]["dim"]
                shape[var_tensor_dim] = var_dim
                msg_in_reshaped = msg_in.reshape(tuple(shape))

                msg_out *= msg_in_reshaped

        msg_out *= self.weights
        var_tensor_dim = self.var_name_to_tensor_data[variable.name]["tensor_dim"]
        var_dim = self.var_name_to_tensor_data[variable.name]["dim"]
        if np.count_nonzero(np.array(self.weights.shape) > 1) > 1:
            msg_out = np.sum(
                msg_out, axis=tuple(set(range(self.weights.ndim)) - {var_tensor_dim})
            )

        msg_out = msg_out.reshape((var_dim, 1))
        # msg_out = msg_out / np.sum(msg_out)

        return msg_out

    def marginal(self):
        pass


class FactorGraph:
    def __init__(self):
        self.graph = dict()

    def add_edge(self, node1: Union[Factor, Variable], node2: Union[Factor, Variable]):

        assert (
            isinstance(node1, Factor)
            and isinstance(node2, Variable)
            or isinstance(node2, Factor)
            and isinstance(node1, Variable)
        )

        if node1.name not in self.graph:
            self.graph[node1.name] = {"obj": node1, "adj": [node2.name], "n_visit": 0}
        else:
            self.graph[node1.name]["adj"].append(node2.name)

        if node2.name not in self.graph:
            self.graph[node2.name] = {"obj": node2, "adj": [node1.name], "n_visit": 0}
        else:
            self.graph[node2.name]["adj"].append(node1.name)

        if isinstance(node1, Variable):
            node1.msgs_in[node2.name] = np.ones((node1.dim, 1))
            node2.msgs_in[node1.name] = np.ones((node1.dim, 1))

            node2.variable_tensor_dim.append((node1.name, node1.dim))
            node2.weights = np.zeros(
                tuple(name_dim[1] for name_dim in node2.variable_tensor_dim)
            )
            node2.var_name_to_tensor_data = {
                name_dim[0]: {"dim": name_dim[1], "tensor_dim": index}
                for index, name_dim in enumerate(node2.variable_tensor_dim)
            }

        else:
            node1.msgs_in[node2.name] = np.ones((node2.dim, 1))
            node2.msgs_in[node1.name] = np.ones((node2.dim, 1))

            node1.variable_tensor_dim.append((node2.name, node2.dim))
            node1.weights = np.zeros(
                tuple(name_dim[1] for name_dim in node1.variable_tensor_dim)
            )
            node1.var_name_to_tensor_data = {
                name_dim[0]: {"dim": name_dim[1], "tensor_dim": index}
                for index, name_dim in enumerate(node1.variable_tensor_dim)
            }

    def reset(self):
        for node_name, node_data in self.graph.items():
            for adj_name, msg in node_data["obj"].msgs_in.items():
                node_data["obj"].msgs_in[adj_name] = np.ones_like(msg)

    def remove_edge(self, factor: Factor):
        pass
