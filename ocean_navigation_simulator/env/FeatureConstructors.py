import numpy as np

from ocean_navigation_simulator.env.Arena import ArenaObservation
from ocean_navigation_simulator.env.NavigationProblem import NavigationProblem

"""
Feature Constructors should take in the ArenaObservation and other relavant information, make any featurization changes,
and then convert to a numpy array that the RL model can use.
"""


def double_gyre_feature_constructor(obs: ArenaObservation, problem: NavigationProblem) -> np.ndarray:
    """
    Converts the observation to use relative positions
    Args:
        problem: class containing information about RL problem (end region, start state, etc.)
        obs: current platform observation

    Returns:
        numpy array containing relative lat pos, relative lon pos, elapsed time, u_curr, v_curr
    """
    target = problem.end_region
    lat_diff = target.lat.deg - obs.platform_state.lat.deg
    lon_diff = target.lon.deg - obs.platform_state.lon.deg
    current = obs.true_current_at_state
    time_elapsed = (obs.platform_state.date_time - problem.start_state.date_time).total_seconds()

    return np.array([lat_diff, lon_diff, time_elapsed, current.u, current.v], dtype=np.float32)


def double_gyre_simple_feature_constructor(obs: ArenaObservation, problem: NavigationProblem) -> \
        np.ndarray:
    """
    Converts the observation to use relative positions
    Args:
        problem: class containing information about RL problem (end region, start state, etc.)
        obs: current platform observation

    Returns:
        numpy array containing relative lat pos, relative lon pos, elapsed time, u_curr, v_curr
    """
    lon_diff = problem.end_region.lon.deg - obs.platform_state.lon.deg
    lat_diff = problem.end_region.lat.deg - obs.platform_state.lat.deg
    time_elapsed = (obs.platform_state.date_time - problem.start_state.date_time).total_seconds()

    return np.array([lon_diff, lat_diff, time_elapsed], dtype=np.float32)
