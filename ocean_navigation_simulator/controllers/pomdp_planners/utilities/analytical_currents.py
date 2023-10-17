from typing import List

import numpy as np


# This will be on particle level!


def anlt_dg_vel_from_particle(particle_state):
    # Formatting of a particle is [x, y, t, A, eps, omega]
    a = particle_state[:, 4] * np.sin(particle_state[:, 5] * particle_state[:, 2])
    b = 1 - 2 * a
    f = a * (particle_state[:, 0] ** 2) + b * particle_state[:, 0]
    df = 2 * (a * particle_state[:, 0]) + b
    return np.array([-np.pi * particle_state[:, 3] * np.sin(np.pi * f) * np.cos(np.pi * particle_state[:, 1]),
                     np.pi * particle_state[:, 3] * np.cos(np.pi * f) * np.sin(np.pi * particle_state[:, 1]) * df])


def highway_current_analytical(particle_states: np.array, y_range_highway: List[float]) -> np.array:
    """Analytical Formula for highway velocity.
    Args:
        particle_states:  numpy array of shape (n,4) where n is the number of particles and the columns are x,y,t,u_highway
        y_range_highway:  list of length 2 with the y range of the highway e.g. [2, 4]
    Returns:
        currents:         currents as numpy array
    """
    u_currents = np.where(np.logical_and(y_range_highway[0] <= particle_states[:, 1],
                                        particle_states[:, 1] <= y_range_highway[1]),
                          particle_states[:, 3],
                          0.0)
    return np.array([u_currents, np.zeros(u_currents.shape)]).T


def highway_current_analytical2D(particle_states: np.array, y_highway_center: float) -> np.array:
    """Analytical Formula for highway velocity.
    Args:
        particle_states:  numpy array of shape (n,4) where n is the number of particles and the columns are x,y,t,u_highway
        y_range_highway:  list of length 2 with the y range of the highway e.g. [2, 4]
    Returns:
        currents:         currents as numpy array
    """

    u_currents = np.where(np.logical_and(y_highway_center - particle_states[:, 4] <= particle_states[:, 1],
                                        particle_states[:, 1] <= y_highway_center + particle_states[:, 4]),
                          particle_states[:, 3],
                          0.0)
    return np.array([u_currents, np.zeros(u_currents.shape)]).T
