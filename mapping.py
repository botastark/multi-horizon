# pairwise_factor_weights: equal, biased, adaptive


def get_w(m_i, m_j, o, type = "equal"):
    if type=="equal":
        return 0.5
    elif type == "biased":
        if m_i==m_j:
            return 0.7
        else:
            return 0.3
    else:
        total_number_of_cells_footprint = obs_map.shape[0]*obs_map.shape[1]
        sample_set_size_h = total_number_of_cells_footprint / 9


