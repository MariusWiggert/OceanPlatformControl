from typing import Tuple

import numpy as np


def transform_u_dir_to_u(u_dir) -> Tuple[float, float]:
    """ Transforms the given u and v velocities to the corresponding heading and thrust.

    Args:
        u_dir:
            An nested array containing the u and v velocities. More rigorously:
            array([[u velocity (longitudinal)], [v velocity (latitudinal)]])

    Returns:
        Thrust in [] and heading in radians
    """
    thrust = np.sqrt(u_dir[0] ** 2 + u_dir[1] ** 2)  # Calculating thrust from distance formula on input u
    heading = np.arctan2(u_dir[1], u_dir[0])  # Finds heading angle from input u
    return thrust, heading
