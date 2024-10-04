def prob_uav_position(uav_pos, belief_map):
    if (
        not uav_pos.position[0] in belief_map.x_range
        or not uav_pos.position[1] in belief_map.y_range
    ):
        return 0
    else:
        return 1 / len(belief_map)[0] / len(belief_map)[1]


def count_possible_states(uav_pos, map):
    # possible_actions = {"up", "down", "front","back", "left","right", "hover"}
    count = 1  # hover
    max_h = 20
    min_h = 5
    step_h = 5
    step_xy = map.grid.length

    if uav_pos.altitude + step_h <= max_h:  # up
        count += 1
    if uav_pos.altitude - step_h >= min_h:  # down
        count += 1
    if uav_pos.position[1] + step_xy <= map.y_range[1]:  # front (+y)
        count += 1
    if uav_pos.position[1] - step_xy <= map.y_range[0]:  # back (-y)
        count += 1
    if uav_pos.position[0] + step_xy <= map.x_range[1]:  # right (+x)
        count += 1
    if uav_pos.position[0] - step_xy <= map.x_range[0]:  # left (-x)
        count += 1
    return count


def prob_candidate_uav_position(uav_pos, belief_map):
    """
    P(x_{t+1}) = P(x_{t+1} | x_{t}) * P(x_{t})
    P(x_{t+1} | x_{t}) = 1 over all possible future positions from the current position
    """
    prob_candidate_given_current = 1 / count_possible_states(uav_pos, belief_map)
    return prob_candidate_given_current * prob_uav_position(uav_pos, belief_map)


def calc_future_obs_prob(belief_map, m, z, uav):
    # terrain_area = belief_map.map.shape[0] * belief_map.map.shape[1]
    # obs_area = obs_map.map.shape[0] * obs_map.map.shape[1]
    # prob_tobe_observed = obs_area / terrain_area

    P_z_given_x = sensor_model(m, z, uav.altitude)

    return P_z_given_x * prob_candidate_uav_position(uav, belief_map)


def map_(belief_map, obs_map, uav, belief_prob=[]):
    evidence_factor = sensor_model(m_i, z_i, uav.altitude)
    prior_m_i = belief_prob.map[i_b, j_b]  # prob of m_i = 1
    prob_future_observation = evidence_factor * prob_candidate_uav_position(
        uav, belief_map
    )
    posterior_m_i = future_belief.map[i_b, j_b]

    entropy_post = -posterior_m_i * math.log2(posterior_m_i) - (
        1 - posterior_m_i
    ) * math.log2(1 - posterior_m_i)
    expected_entropy = prob_future_observation * entropy_post
