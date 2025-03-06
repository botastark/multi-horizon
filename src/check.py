# articles to checl:
# https://www.mdpi.com/2072-4292/11/10/1252 (overlap persentage vs effeciency and sensor resolution to increase )
# https://cdnsciencepub.com/doi/pdf/10.1139/juvs-2018-0014 might be informative


def compute_uav_step_sizes(
    flight_altitude,
    forward_overlap,
    side_overlap,
):
    """
    Computes the forward and side step sizes for a UAV flight plan based on camera parameters
    and desired overlap percentages.

    Parameters:
    - flight_altitude (m): UAV flight altitude above ground.
    - focal_length (mm): Camera focal length.
    - sensor_width (mm): Camera sensor width.
    - sensor_height (mm): Camera sensor height.
    - forward_overlap (float): Forward overlap percentage (0-1 range).
    - side_overlap (float): Side overlap percentage (0-1 range).

    Returns:
    - xy_step_f (m): Step size in the forward direction.
    - xy_step_s (m): Step size in the side direction.
    - W (m): Ground footprint width.
    - H (m): Ground footprint height.
    """
    import numpy as np

    # Compute FoV angles
    theta_w = np.deg2rad(60)
    # 2 * np.arctan(
    #     (sensor_width / 2) / focal_length
    # )  # Horizontal FoV (radians)
    theta_h = np.deg2rad(60)
    # 2 * np.arctan(
    #     (sensor_height / 2) / focal_length
    # )  # Vertical FoV (radians)

    # Compute ground footprint dimensions
    W = 2 * flight_altitude * np.tan(theta_w / 2)  # Ground width (meters)
    H = 2 * flight_altitude * np.tan(theta_h / 2)  # Ground height (meters)

    # Compute step sizes based on overlap
    xy_step_f = H * (1 - forward_overlap)  # Forward step
    xy_step_s = W * (1 - side_overlap)  # Side step

    return round(xy_step_f, 2), round(xy_step_s, 2)


# Example usage with given parameters
flight_altitude = 19.5  # meters
focal_length = 24  # mm
sensor_width = 13.2  # mm
sensor_height = 8.8  # mm
forward_overlap = 0.80  # 80%
side_overlap = 0.70  # 70%
print(
    compute_uav_step_sizes(
        flight_altitude,
        # focal_length,
        # sensor_width,
        # sensor_height,
        forward_overlap,
        side_overlap,
    )
)
